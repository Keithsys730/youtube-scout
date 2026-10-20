"""Tests for URL extraction, duration parsing, chapter parsing, and metadata fetching."""

import pytest
import requests
from unittest.mock import patch, MagicMock
from youtube_scout.fetcher import (
    extract_video_id,
    parse_iso8601_duration,
    extract_chapters_from_description,
    YouTubeMetadataFetcher,
    MissingApiKeyError,
    InvalidUrlError,
    MetadataFetchError
)


def test_extract_video_id_valid_formats():
    valid_cases = [
        ("https://www.youtube.com/watch?v=dQw4w9WgXcQ", "dQw4w9WgXcQ"),
        ("https://www.youtube.com/watch/?v=dQw4w9WgXcQ", "dQw4w9WgXcQ"),
        ("https://youtu.be/dQw4w9WgXcQ", "dQw4w9WgXcQ"),
        ("https://youtu.be/dQw4w9WgXcQ?t=42", "dQw4w9WgXcQ"),
        ("https://www.youtube.com/embed/dQw4w9WgXcQ", "dQw4w9WgXcQ"),
        ("https://www.youtube.com/shorts/dQw4w9WgXcQ", "dQw4w9WgXcQ"),
        ("https://www.youtube.com/live/dQw4w9WgXcQ", "dQw4w9WgXcQ"),
        ("https://m.youtube.com/watch?v=dQw4w9WgXcQ&feature=shared", "dQw4w9WgXcQ"),
        ("dQw4w9WgXcQ", "dQw4w9WgXcQ"),
    ]
    for input_val, expected in valid_cases:
        assert extract_video_id(input_val) == expected


def test_extract_video_id_invalid_formats():
    invalid_cases = [
        "",
        "not_a_valid_id",
        "https://example.com/watch?v=123",
        "https://youtube.com",
        "https://youtube.com/watch?v=short",
    ]
    for invalid in invalid_cases:
        with pytest.raises(InvalidUrlError):
            extract_video_id(invalid)


def test_parse_iso8601_duration():
    assert parse_iso8601_duration("PT15M33S") == (933, "15:33")
    assert parse_iso8601_duration("PT1H2M30S") == (3750, "01:02:30")
    assert parse_iso8601_duration("PT45S") == (45, "00:45")
    assert parse_iso8601_duration("PT2H") == (7200, "02:00:00")
    assert parse_iso8601_duration("invalid") == (0, "00:00")
    assert parse_iso8601_duration("") == (0, "00:00")


def test_extract_chapters_from_description():
    desc = """
    Check out this talk!
    Chapters:
    00:00 - Introduction
    02:15 Tokio Runtime Configuration
    [06:40] Zero-Copy Buffer Management
    1:12:10 Scaling Beyond 100k QPS
    Link: https://github.com/example/repo
    """
    chapters = extract_chapters_from_description(desc)
    assert len(chapters) == 4
    assert chapters[0].timestamp_str == "00:00"
    assert chapters[0].start_seconds == 0
    assert chapters[0].title == "Introduction"

    assert chapters[1].timestamp_str == "02:15"
    assert chapters[1].start_seconds == 135
    assert chapters[1].title == "Tokio Runtime Configuration"

    assert chapters[2].timestamp_str == "06:40"
    assert chapters[2].start_seconds == 400
    assert chapters[2].title == "Zero-Copy Buffer Management"

    assert chapters[3].timestamp_str == "1:12:10"
    assert chapters[3].start_seconds == 4330
    assert chapters[3].title == "Scaling Beyond 100k QPS"


def test_metadata_fetcher_missing_key(monkeypatch):
    monkeypatch.delenv("YOUTUBE_API_KEY", raising=False)
    with pytest.raises(MissingApiKeyError) as exc_info:
        YouTubeMetadataFetcher()
    assert "Missing YOUTUBE_API_KEY environment variable" in str(exc_info.value)


def test_metadata_fetcher_success(raw_videos_list_response):
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = raw_videos_list_response

    with patch("requests.get", return_value=mock_resp) as mock_get:
        fetcher = YouTubeMetadataFetcher(api_key="test_api_key")
        metadata = fetcher.fetch_metadata("dQw4w9WgXcQ")

        assert metadata.video_id == "dQw4w9WgXcQ"
        assert metadata.title == "Building High-Throughput Event-Driven Microservices in Rust"
        assert metadata.channel_title == "Systems Engineering Hub"
        assert metadata.duration_seconds == 1305
        assert metadata.duration_str == "21:45"
        assert metadata.view_count == 142500
        assert metadata.like_count == 8920
        assert len(metadata.chapters) == 5
        assert metadata.chapters[1].title == "Tokio Runtime Configuration"
        assert metadata.chapters[1].start_seconds == 135

        # Verify API key was passed in header, not in query params
        _, kwargs = mock_get.call_args
        assert kwargs["headers"]["X-Goog-Api-Key"] == "test_api_key"
        assert "key" not in kwargs["params"]


def test_metadata_fetcher_redacts_key_in_exception():
    with patch("requests.get", side_effect=requests.RequestException("Connection error with https://api?key=SECRET_KEY_123")):
        fetcher = YouTubeMetadataFetcher(api_key="SECRET_KEY_123")
        with pytest.raises(MetadataFetchError) as exc_info:
            fetcher.fetch_metadata("dQw4w9WgXcQ")
        err_msg = str(exc_info.value)
        assert "SECRET_KEY_123" not in err_msg
        assert "[REDACTED]" in err_msg


def test_metadata_fetcher_api_error():
    mock_resp = MagicMock()
    mock_resp.status_code = 403
    mock_resp.text = "Quota exceeded"

    with patch("requests.get", return_value=mock_resp):
        fetcher = YouTubeMetadataFetcher(api_key="test_api_key")
        with pytest.raises(MetadataFetchError) as exc_info:
            fetcher.fetch_metadata("dQw4w9WgXcQ")
        assert "status 403" in str(exc_info.value)


def test_metadata_fetcher_not_found():
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {"items": []}

    with patch("requests.get", return_value=mock_resp):
        fetcher = YouTubeMetadataFetcher(api_key="test_api_key")
        with pytest.raises(MetadataFetchError) as exc_info:
            fetcher.fetch_metadata("nonexistent11")
        assert "No video found" in str(exc_info.value)


def test_search_videos_success():
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {
        "items": [
            {"id": {"videoId": "vid_1"}},
            {"id": {"videoId": "vid_2"}},
            {"id": "vid_3"}
        ]
    }

    with patch("requests.get", return_value=mock_resp) as mock_get:
        fetcher = YouTubeMetadataFetcher(api_key="test_api_key")
        video_ids = fetcher.search_videos("rust microservices", max_results=20, order="viewCount")

        assert video_ids == ["vid_1", "vid_2", "vid_3"]
        _, kwargs = mock_get.call_args
        assert kwargs["headers"]["X-Goog-Api-Key"] == "test_api_key"
        assert kwargs["params"]["q"] == "rust microservices"
        assert kwargs["params"]["type"] == "video"
        assert kwargs["params"]["part"] == "id"
        assert kwargs["params"]["maxResults"] == 20
        assert kwargs["params"]["order"] == "viewCount"


def test_search_videos_api_error_and_redaction():
    with patch("requests.get", side_effect=requests.RequestException("Timeout on https://search?key=SECRET_SEARCH_KEY")):
        fetcher = YouTubeMetadataFetcher(api_key="SECRET_SEARCH_KEY")
        with pytest.raises(MetadataFetchError) as exc_info:
            fetcher.search_videos("rust")
        assert "SECRET_SEARCH_KEY" not in str(exc_info.value)
        assert "[REDACTED]" in str(exc_info.value)

    mock_resp = MagicMock()
    mock_resp.status_code = 400
    mock_resp.text = "Bad Request"
    with patch("requests.get", return_value=mock_resp):
        fetcher = YouTubeMetadataFetcher(api_key="key")
        with pytest.raises(MetadataFetchError) as exc_info:
            fetcher.search_videos("rust")
        assert "status 400" in str(exc_info.value)


def test_fetch_metadata_batch_empty():
    fetcher = YouTubeMetadataFetcher(api_key="test_api_key")
    assert fetcher.fetch_metadata_batch([]) == []
    assert fetcher.fetch_metadata_batch(["   ", ""]) == []


def test_fetch_metadata_batch_success(raw_videos_list_response):
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = raw_videos_list_response

    with patch("requests.get", return_value=mock_resp) as mock_get:
        fetcher = YouTubeMetadataFetcher(api_key="test_api_key")
        results = fetcher.fetch_metadata_batch(["dQw4w9WgXcQ", "abc12345678"])

        assert len(results) == 1
        assert results[0].video_id == "dQw4w9WgXcQ"
        assert results[0].title == "Building High-Throughput Event-Driven Microservices in Rust"
        _, kwargs = mock_get.call_args
        assert kwargs["headers"]["X-Goog-Api-Key"] == "test_api_key"
        assert kwargs["params"]["id"] == "dQw4w9WgXcQ,abc12345678"


def test_fetch_metadata_batch_error_and_redaction():
    with patch("requests.get", side_effect=requests.RequestException("Failed with https://videos?key=SECRET_BATCH_KEY")):
        fetcher = YouTubeMetadataFetcher(api_key="SECRET_BATCH_KEY")
        with pytest.raises(MetadataFetchError) as exc_info:
            fetcher.fetch_metadata_batch(["dQw4w9WgXcQ"])
        assert "SECRET_BATCH_KEY" not in str(exc_info.value)
        assert "[REDACTED]" in str(exc_info.value)


def test_fetch_metadata_batch_filters_whitespace_before_truncating():
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {"items": []}

    # 10 whitespace entries followed by 45 valid IDs (total 55 entries)
    # If [:50] were applied before filtering, only 40 valid IDs would be fetched.
    # Filtering first ensures all 45 valid IDs are fetched.
    whitespace_entries = ["", "  ", "\t", "\n"] * 2 + ["", " "]
    valid_ids = [f"vid_{i:02d}" for i in range(45)]
    input_ids = whitespace_entries + valid_ids

    with patch("requests.get", return_value=mock_resp) as mock_get:
        fetcher = YouTubeMetadataFetcher(api_key="test_api_key")
        fetcher.fetch_metadata_batch(input_ids)

        _, kwargs = mock_get.call_args
        requested_ids = kwargs["params"]["id"].split(",")
        assert len(requested_ids) == 45
        assert requested_ids == valid_ids


def test_parse_video_item_null_subdicts():
    fetcher = YouTubeMetadataFetcher(api_key="test_api_key")
    item_with_nulls = {
        "id": "null_subdicts_id",
        "snippet": None,
        "contentDetails": None,
        "statistics": None,
    }
    meta = fetcher._parse_video_item(item_with_nulls)
    assert meta.video_id == "null_subdicts_id"
    assert meta.title == ""
    assert meta.channel_title == ""
    assert meta.duration_seconds == 0
    assert meta.view_count == 0
    assert meta.like_count == 0
    assert meta.chapters == []


