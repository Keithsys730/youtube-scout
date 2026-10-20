"""Tests for transcript acquisition module, subtitle format parsing, and fallback chain."""

import json
from unittest.mock import patch, MagicMock
from youtube_scout.models import TranscriptSegment
from youtube_scout.transcript import (
    parse_json3,
    parse_vtt,
    parse_srt,
    fetch_transcript_youtube_api,
    fetch_transcript_ytdlp,
    get_video_transcript
)


def test_parse_json3():
    data = json.dumps({
        "events": [
            {
                "tStartMs": 1200,
                "dDurationMs": 2500,
                "segs": [{"utf8": "Hello"}, {"utf8": " world"}]
            },
            {
                "tStartMs": 4000,
                "dDurationMs": 1800,
                "segs": [{"utf8": "Rust Tokio runtime"}]
            }
        ]
    })
    segments = parse_json3(data)
    assert len(segments) == 2
    assert segments[0].text == "Hello world"
    assert segments[0].start == 1.2
    assert segments[0].duration == 2.5
    assert segments[1].text == "Rust Tokio runtime"
    assert segments[1].start == 4.0


def test_parse_vtt():
    vtt_content = """WEBVTT
Kind: captions
Language: en

00:00:01.500 --> 00:00:03.500
Hello world

00:00:04.000 --> 00:00:06.200
<c>Zero-copy</c> buffer management
"""
    segments = parse_vtt(vtt_content)
    assert len(segments) == 2
    assert segments[0].text == "Hello world"
    assert segments[0].start == 1.5
    assert segments[0].duration == 2.0
    assert segments[1].text == "Zero-copy buffer management"
    assert segments[1].start == 4.0
    assert round(segments[1].duration, 1) == 2.2


def test_parse_srt():
    srt_content = """1
00:00:01,500 --> 00:00:03,500
Hello world

2
00:00:04,000 --> 00:00:06,200
Benchmark comparison
"""
    segments = parse_srt(srt_content)
    assert len(segments) == 2
    assert segments[0].text == "Hello world"
    assert segments[0].start == 1.5
    assert segments[1].text == "Benchmark comparison"
    assert segments[1].start == 4.0


def test_fallback_chain_primary_success():
    mock_segments = [TranscriptSegment(text="Primary transcript", start=0.0, duration=2.0)]
    with patch("youtube_scout.transcript.fetch_transcript_youtube_api", return_value=mock_segments) as mock_api:
        with patch("youtube_scout.transcript.fetch_transcript_ytdlp") as mock_ytdlp:
            result = get_video_transcript("dQw4w9WgXcQ")
            assert result == mock_segments
            mock_api.assert_called_once()
            mock_ytdlp.assert_not_called()


def test_fallback_chain_primary_fails_ytdlp_succeeds():
    mock_segments = [TranscriptSegment(text="yt-dlp transcript", start=0.0, duration=2.0)]
    with patch("youtube_scout.transcript.fetch_transcript_youtube_api", return_value=None) as mock_api:
        with patch("youtube_scout.transcript.fetch_transcript_ytdlp", return_value=mock_segments) as mock_ytdlp:
            result = get_video_transcript("dQw4w9WgXcQ")
            assert result == mock_segments
            mock_api.assert_called_once()
            mock_ytdlp.assert_called_once()


def test_fallback_chain_both_fail_graceful_none():
    with patch("youtube_scout.transcript.fetch_transcript_youtube_api", return_value=None):
        with patch("youtube_scout.transcript.fetch_transcript_ytdlp", return_value=None):
            result = get_video_transcript("dQw4w9WgXcQ")
            assert result is None


def test_fallback_chain_primary_exception_handled():
    mock_segments = [TranscriptSegment(text="yt-dlp after error", start=0.0, duration=2.0)]
    with patch("youtube_scout.transcript.fetch_transcript_youtube_api", side_effect=Exception("Bot detection error")):
        with patch("youtube_scout.transcript.fetch_transcript_ytdlp", return_value=mock_segments):
            # Must not raise; must proceed to yt-dlp fallback
            result = get_video_transcript("dQw4w9WgXcQ")
            assert result == mock_segments


def test_fetch_transcript_ytdlp_language_priority(tmp_path):
    # Simulate yt-dlp downloading both en.json3 and zh-TW.vtt
    en_file = tmp_path / "dQw4w9WgXcQ.en.json3"
    en_file.write_text(json.dumps({
        "events": [{"tStartMs": 0, "dDurationMs": 1000, "segs": [{"utf8": "English subtitle"}]}]
    }), encoding="utf-8")

    zh_file = tmp_path / "dQw4w9WgXcQ.zh-TW.vtt"
    zh_file.write_text("WEBVTT\n\n00:00:00.000 --> 00:00:01.000\n繁體中文逐字稿\n", encoding="utf-8")

    class MockTempDir:
        def __enter__(self):
            return str(tmp_path)
        def __exit__(self, *args):
            pass

    with patch("tempfile.TemporaryDirectory", return_value=MockTempDir()):
        with patch("subprocess.run", return_value=MagicMock(returncode=0, stderr="")):
            segments = fetch_transcript_ytdlp("dQw4w9WgXcQ", languages=["zh-TW", "en"])
            assert segments is not None
            assert len(segments) == 1
            # Must prioritize zh-TW over en even though en is json3 and zh is vtt
            assert segments[0].text == "繁體中文逐字稿"


def test_parse_json3_non_dict():
    assert parse_json3("[]") == []
    assert parse_json3('"not a dict"') == []
    assert parse_json3("123") == []
    assert parse_json3("") == []
    assert parse_json3("{malformed json}") == []


def test_fetch_transcript_ytdlp_iterates_on_empty_first_file(tmp_path):
    from youtube_scout.transcript import fetch_transcript_ytdlp
    # File 1: empty json3
    empty_file = tmp_path / "dQw4w9WgXcQ.zh-TW.json3"
    empty_file.write_text("{}", encoding="utf-8")

    # File 2: valid vtt fallback
    vtt_file = tmp_path / "dQw4w9WgXcQ.zh-TW.vtt"
    vtt_file.write_text("WEBVTT\n\n00:00:00.000 --> 00:00:01.000\n備用格式內容\n", encoding="utf-8")

    class MockTempDir:
        def __enter__(self):
            return str(tmp_path)
        def __exit__(self, *args):
            pass

    with patch("tempfile.TemporaryDirectory", return_value=MockTempDir()):
        with patch("subprocess.run", return_value=MagicMock(returncode=0, stderr="")):
            segments = fetch_transcript_ytdlp("dQw4w9WgXcQ", languages=["zh-TW"])
            assert segments is not None
            assert len(segments) == 1
            assert segments[0].text == "備用格式內容"

