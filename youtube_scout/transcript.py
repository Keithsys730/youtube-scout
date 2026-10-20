"""Swappable transcript acquisition module with layered fallback chain and graceful degradation."""

import os
import re
import sys
import json
import logging
import tempfile
import subprocess
from typing import List, Optional
from .models import TranscriptSegment

logger = logging.getLogger("youtube_scout.transcript")


def parse_json3(content: str) -> List[TranscriptSegment]:
    """Parse YouTube JSON3 subtitle format into TranscriptSegments."""
    try:
        data = json.loads(content)
        if not isinstance(data, dict):
            return []

        segments: List[TranscriptSegment] = []
        for event in data.get("events") or []:
            if not isinstance(event, dict):
                continue
            segs = event.get("segs") or []
            text = "".join(s.get("utf8", "") for s in segs if isinstance(s, dict)).strip()
            if not text:
                continue
            start_ms = event.get("tStartMs", 0)
            dur_ms = event.get("dDurationMs", 0)
            start = float(start_ms) / 1000.0
            dur = float(dur_ms) / 1000.0
            segments.append(TranscriptSegment(text=text, start=start, duration=dur))

        return segments
    except Exception as e:
        logger.debug(f"JSON3 parse error: {e}")
        return []


def parse_vtt(content: str) -> List[TranscriptSegment]:
    """Parse WebVTT subtitle format into TranscriptSegments."""
    segments: List[TranscriptSegment] = []
    lines = content.splitlines()
    time_re = re.compile(
        r'(?:(\d+):)?(\d{2}):(\d{2})[.,](\d{3})\s*-->\s*(?:(\d+):)?(\d{2}):(\d{2})[.,](\d{3})'
    )
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        match = time_re.search(line)
        if match:
            h1, m1, s1, ms1, h2, m2, s2, ms2 = match.groups()
            start = (int(h1 or 0) * 3600 + int(m1) * 60 + int(s1)) + int(ms1) / 1000.0
            end = (int(h2 or 0) * 3600 + int(m2) * 60 + int(s2)) + int(ms2) / 1000.0
            dur = max(0.0, end - start)
            i += 1

            text_lines = []
            while i < len(lines) and lines[i].strip() and not time_re.search(lines[i]):
                # Strip inline HTML/styling tags (e.g. <c>, </c>, <b>)
                clean_line = re.sub(r'<[^>]+>', '', lines[i]).strip()
                if clean_line and not clean_line.startswith("Align:") and not clean_line.startswith("Position:"):
                    text_lines.append(clean_line)
                i += 1

            text = " ".join(text_lines).strip()
            if text:
                segments.append(TranscriptSegment(text=text, start=start, duration=dur))
        else:
            i += 1

    return segments


def parse_srt(content: str) -> List[TranscriptSegment]:
    """Parse SubRip (SRT) subtitle format into TranscriptSegments."""
    segments: List[TranscriptSegment] = []
    blocks = re.split(r'\n\s*\n', content.strip())
    time_re = re.compile(
        r'(?:(\d+):)?(\d{2}):(\d{2})[.,](\d{3})\s*-->\s*(?:(\d+):)?(\d{2}):(\d{2})[.,](\d{3})'
    )

    for block in blocks:
        lines = [line.strip() for line in block.splitlines() if line.strip()]
        if not lines:
            continue

        match = None
        text_start_idx = 1
        for idx, line in enumerate(lines[:3]):
            m = time_re.search(line)
            if m:
                match = m
                text_start_idx = idx + 1
                break

        if match:
            h1, m1, s1, ms1, h2, m2, s2, ms2 = match.groups()
            start = (int(h1 or 0) * 3600 + int(m1) * 60 + int(s1)) + int(ms1) / 1000.0
            end = (int(h2 or 0) * 3600 + int(m2) * 60 + int(s2)) + int(ms2) / 1000.0
            dur = max(0.0, end - start)
            text = " ".join(lines[text_start_idx:]).strip()
            text = re.sub(r'<[^>]+>', '', text).strip()
            if text:
                segments.append(TranscriptSegment(text=text, start=start, duration=dur))

    return segments


def fetch_transcript_youtube_api(
    video_id: str,
    languages: Optional[List[str]] = None
) -> Optional[List[TranscriptSegment]]:
    """Fetch transcript using youtube-transcript-api library. Graceful on any failure."""
    langs = languages or ["zh-TW", "zh-Hant", "zh", "en"]
    try:
        from youtube_transcript_api import YouTubeTranscriptApi

        raw_segments = None

        # Check if legacy get_transcript classmethod exists
        if hasattr(YouTubeTranscriptApi, 'get_transcript'):
            try:
                raw_segments = YouTubeTranscriptApi.get_transcript(video_id, languages=langs)
            except Exception as e:
                logger.debug(f"YouTubeTranscriptApi.get_transcript failed for {video_id}: {e}")

        # Check if v1.x instance API exists
        if raw_segments is None:
            try:
                api = YouTubeTranscriptApi()
                raw_segments = api.fetch(video_id, languages=langs)
            except Exception as e:
                logger.debug(f"YouTubeTranscriptApi().fetch failed for {video_id}: {e}")
                # Fallback to listing transcripts
                try:
                    t_list = api.list(video_id)
                    found = None
                    for lang in langs:
                        try:
                            found = t_list.find_transcript([lang])
                            if found:
                                break
                        except Exception:
                            pass
                    if not found:
                        for t in t_list:
                            found = t
                            break
                    if found:
                        raw_segments = found.fetch()
                except Exception as list_err:
                    logger.debug(f"YouTubeTranscriptApi().list failed for {video_id}: {list_err}")

        if not raw_segments:
            return None

        segments: List[TranscriptSegment] = []
        for item in raw_segments:
            if isinstance(item, dict):
                text = item.get("text", "")
                start = float(item.get("start", 0.0))
                dur = float(item.get("duration", 0.0))
            else:
                text = getattr(item, "text", "")
                start = float(getattr(item, "start", 0.0))
                dur = float(getattr(item, "duration", 0.0))

            if text and text.strip():
                segments.append(TranscriptSegment(text=text.strip(), start=start, duration=dur))

        return segments if segments else None

    except Exception as e:
        logger.warning(f"youtube-transcript-api error for {video_id}: {e}")
        return None


def fetch_transcript_ytdlp(
    video_id: str,
    languages: Optional[List[str]] = None
) -> Optional[List[TranscriptSegment]]:
    """Fetch auto/manual subtitles via yt-dlp CLI subprocess. Graceful on any failure."""
    langs = languages or ["zh-TW", "zh-Hant", "zh", "en"]
    lang_str = ",".join(langs)
    video_url = f"https://www.youtube.com/watch?v={video_id}"

    try:
        with tempfile.TemporaryDirectory() as tmp_dir:
            out_tmpl = os.path.join(tmp_dir, "%(id)s.%(ext)s")
            cmd = [
                sys.executable, "-m", "yt_dlp",
                "--write-auto-sub",
                "--write-sub",
                "--sub-lang", lang_str,
                "--sub-format", "json3/vtt/srt",
                "--skip-download",
                "--no-warnings",
                "--quiet",
                "-o", out_tmpl,
                video_url
            ]

            logger.debug(f"Running yt-dlp fallback command for {video_id}")
            res = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=45,
                stdin=subprocess.DEVNULL
            )
            if res.returncode != 0 and res.stderr:
                logger.debug(f"yt-dlp returned non-zero code {res.returncode}: {res.stderr}")

            # Scan generated subtitle files in the temporary directory
            files = [
                os.path.join(tmp_dir, f) for f in os.listdir(tmp_dir)
                if f.endswith((".json3", ".vtt", ".srt"))
            ]

            if not files:
                logger.debug(f"yt-dlp did not produce subtitle files for {video_id}")
                return None

            # Prioritize requested language order first, then format (json3 > vtt > srt)
            def file_sort_key(file_path: str):
                base = os.path.basename(file_path)
                lang_priority = 999
                for idx, lang in enumerate(langs):
                    # match .zh-TW. or .zh. or similar
                    if f".{lang}." in base or f".{lang}-" in base:
                        lang_priority = idx
                        break
                fmt_priority = 0 if base.endswith(".json3") else (1 if base.endswith(".vtt") else 2)
                return (lang_priority, fmt_priority)

            files.sort(key=file_sort_key)

            # Try candidate files in priority order until one yields non-empty segments
            for chosen_file in files:
                try:
                    with open(chosen_file, "r", encoding="utf-8", errors="replace") as f:
                        content = f.read()

                    parsed_segments = None
                    if chosen_file.endswith(".json3"):
                        parsed_segments = parse_json3(content)
                    elif chosen_file.endswith(".vtt"):
                        parsed_segments = parse_vtt(content)
                    elif chosen_file.endswith(".srt"):
                        parsed_segments = parse_srt(content)

                    if parsed_segments:
                        return parsed_segments
                except Exception as fe:
                    logger.debug(f"Failed parsing subtitle file {chosen_file}: {fe}")
                    continue

    except Exception as e:
        logger.warning(f"yt-dlp error for {video_id}: {e}")
        return None

    return None


def get_video_transcript(
    video_id: str,
    languages: Optional[List[str]] = None
) -> Optional[List[TranscriptSegment]]:
    """Acquire transcript for a video with swappable fallback chain:
    1. youtube-transcript-api
    2. yt-dlp --write-auto-sub --skip-download
    3. give up -> return None
    """
    logger.info(f"Acquiring transcript for video {video_id} via primary backend (youtube-transcript-api)...")
    try:
        segments = fetch_transcript_youtube_api(video_id, languages=languages)
        if segments:
            logger.info(f"Acquired {len(segments)} transcript segments via youtube-transcript-api.")
            return segments
    except Exception as e:
        logger.warning(f"Unexpected error in primary transcript fetcher: {e}")

    logger.info(f"Primary backend unavailable; falling back to yt-dlp for video {video_id}...")
    try:
        segments = fetch_transcript_ytdlp(video_id, languages=languages)
        if segments:
            logger.info(f"Acquired {len(segments)} transcript segments via yt-dlp.")
            return segments
    except Exception as e:
        logger.warning(f"Unexpected error in yt-dlp fallback fetcher: {e}")

    logger.warning(f"No transcript available for video {video_id}; returning None.")
    return None
