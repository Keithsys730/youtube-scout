"""Metadata fetching and URL parsing via YouTube Data API v3."""

import os
import re
import logging
from typing import Tuple, List, Optional, Dict, Any
from urllib.parse import urlparse, parse_qs
import requests

from .models import VideoMetadata, VideoChapter

logger = logging.getLogger("youtube_scout.fetcher")


class MissingApiKeyError(Exception):
    """Raised when YOUTUBE_API_KEY environment variable is not set."""


class InvalidUrlError(Exception):
    """Raised when a provided URL or ID is not a valid YouTube identifier."""


class MetadataFetchError(Exception):
    """Raised when YouTube Data API fails or returns an error."""


def extract_video_id(url_or_id: str) -> str:
    """Extract standard 11-character YouTube video ID from various URL formats or raw ID."""
    if not url_or_id or not isinstance(url_or_id, str):
        raise InvalidUrlError("Video URL or ID must be a non-empty string.")

    cleaned = url_or_id.strip()

    # Raw 11-character video ID
    if re.fullmatch(r'[a-zA-Z0-9_-]{11}', cleaned):
        return cleaned

    # Parse as URL
    if not (cleaned.startswith("http://") or cleaned.startswith("https://")):
        cleaned = f"https://{cleaned}"

    try:
        parsed = urlparse(cleaned)
    except Exception as e:
        raise InvalidUrlError(f"Cannot parse URL '{url_or_id}': {e}")

    hostname = parsed.hostname.lower() if parsed.hostname else ""

    # Shortened: youtu.be/<id>
    if "youtu.be" in hostname:
        path = parsed.path.strip("/")
        vid = path.split("/")[0] if path else ""
        if re.fullmatch(r'[a-zA-Z0-9_-]{11}', vid):
            return vid

    # Standard: youtube.com/watch?v=<id> or youtube.com/watch/?v=<id>
    if "youtube.com" in hostname or "youtube-nocookie.com" in hostname:
        if parsed.path.rstrip("/") == "/watch":
            query_params = parse_qs(parsed.query)
            v_list = query_params.get("v")
            if v_list and re.fullmatch(r'[a-zA-Z0-9_-]{11}', v_list[0]):
                return v_list[0]

        # Embeds, Shorts, Videos, or Live: youtube.com/embed/<id>, youtube.com/shorts/<id>, youtube.com/live/<id>
        parts = parsed.path.strip("/").split("/")
        if len(parts) >= 2 and parts[0] in ("embed", "shorts", "v", "live"):
            vid = parts[1]
            if re.fullmatch(r'[a-zA-Z0-9_-]{11}', vid):
                return vid

    raise InvalidUrlError(f"Could not extract a valid 11-character YouTube video ID from: {url_or_id}")


def parse_iso8601_duration(duration_str: str) -> Tuple[int, str]:
    """Parse ISO 8601 duration (e.g. PT1H2M30S) into (seconds, formatted_string)."""
    if not duration_str or not isinstance(duration_str, str):
        return 0, "00:00"

    pattern = r'^P(?:(\d+)D)?(?:T(?:(\d+)H)?(?:(\d+)M)?(?:(\d+)S)?)?$'
    match = re.match(pattern, duration_str.strip())
    if not match:
        return 0, "00:00"

    days = int(match.group(1) or 0)
    hours = int(match.group(2) or 0)
    minutes = int(match.group(3) or 0)
    seconds = int(match.group(4) or 0)

    total_seconds = days * 86400 + hours * 3600 + minutes * 60 + seconds

    if hours > 0 or days > 0:
        total_hours = days * 24 + hours
        formatted = f"{total_hours:02d}:{minutes:02d}:{seconds:02d}"
    else:
        formatted = f"{minutes:02d}:{seconds:02d}"

    return total_seconds, formatted


def parse_timestamp_to_seconds(ts_str: str) -> int:
    """Convert timestamp string (e.g., '01:23' or '1:02:30') to seconds."""
    parts = [int(p) for p in ts_str.strip().split(":")]
    if len(parts) == 2:
        return parts[0] * 60 + parts[1]
    elif len(parts) == 3:
        return parts[0] * 3600 + parts[1] * 60 + parts[2]
    return 0


def extract_chapters_from_description(description: str) -> List[VideoChapter]:
    """Extract timestamped chapter markers from video description."""
    if not description:
        return []

    chapters: List[VideoChapter] = []
    pattern = re.compile(r'(?:^|\n)\s*\[?(\d{1,2}:\d{2}(?::\d{2})?)\]?\s*[-–—:]?\s*([^\n\r]+)')

    for line in description.splitlines():
        match = pattern.search(line)
        if match:
            ts_str = match.group(1)
            title = match.group(2).strip()
            # Exclude short meaningless titles or URLs
            if title and not title.lower().startswith("http"):
                seconds = parse_timestamp_to_seconds(ts_str)
                chapters.append(VideoChapter(start_seconds=seconds, timestamp_str=ts_str, title=title))

    return chapters


class YouTubeMetadataFetcher:
    """Fetches video metadata and search discovery using YouTube Data API v3."""

    BASE_URL = "https://www.googleapis.com/youtube/v3/videos"
    SEARCH_URL = "https://www.googleapis.com/youtube/v3/search"

    def __init__(self, api_key: Optional[str] = None):
        key = api_key or os.environ.get("YOUTUBE_API_KEY")
        if not key or not key.strip():
            raise MissingApiKeyError(
                "Missing YOUTUBE_API_KEY environment variable. "
                "Please set YOUTUBE_API_KEY=<your_api_key> before running youtube-scout."
            )
        self.api_key = key.strip()

    def _parse_video_item(self, item: Dict[str, Any], fallback_id: str = "") -> VideoMetadata:
        """Parse raw YouTube video resource item into VideoMetadata dataclass."""
        snippet = item.get("snippet") or {}
        content_details = item.get("contentDetails") or {}
        statistics = item.get("statistics") or {}

        video_id = item.get("id") or fallback_id
        if isinstance(video_id, dict):
            video_id = video_id.get("videoId", fallback_id)

        title = snippet.get("title", "")
        channel_title = snippet.get("channelTitle", "")
        published_at = snippet.get("publishedAt", "")
        description = snippet.get("description", "")

        raw_duration = content_details.get("duration", "")
        duration_seconds, duration_str = parse_iso8601_duration(raw_duration)

        view_count = int(statistics.get("viewCount") or 0)
        like_count = int(statistics.get("likeCount") or 0)

        chapters = extract_chapters_from_description(description)

        return VideoMetadata(
            video_id=str(video_id),
            title=title,
            channel_title=channel_title,
            published_at=published_at,
            duration_seconds=duration_seconds,
            duration_str=duration_str,
            view_count=view_count,
            like_count=like_count,
            description=description,
            chapters=chapters
        )

    def _sanitize_error_text(self, text: str) -> str:
        safe = re.sub(r'([?&]key=)[^&\s]+', r'\1[REDACTED]', str(text))
        if self.api_key:
            safe = safe.replace(self.api_key, "[REDACTED]")
        return safe

    def search_videos(self, query: str, max_results: int = 50, order: str = "relevance") -> List[str]:
        """Search YouTube for videos matching a query, returning a list of video IDs."""
        headers = {
            "X-Goog-Api-Key": self.api_key
        }
        params = {
            "part": "id",
            "type": "video",
            "q": query,
            "maxResults": min(max(max_results, 1), 50),
            "order": order
        }

        try:
            response = requests.get(self.SEARCH_URL, headers=headers, params=params, timeout=15)
        except requests.RequestException as e:
            safe_err = self._sanitize_error_text(str(e))
            raise MetadataFetchError(f"Network error communicating with YouTube Data API search: {safe_err}")

        if response.status_code != 200:
            safe_text = self._sanitize_error_text(response.text)
            raise MetadataFetchError(
                f"YouTube Data API search failed with status {response.status_code}: {safe_text}"
            )

        data = response.json()
        items = data.get("items", [])
        video_ids = []
        for item in items:
            id_info = item.get("id")
            if isinstance(id_info, dict):
                vid = id_info.get("videoId")
            elif isinstance(id_info, str):
                vid = id_info
            else:
                vid = None
            if vid and isinstance(vid, str):
                video_ids.append(vid)

        return video_ids

    def fetch_metadata(self, video_id: str) -> VideoMetadata:
        """Fetch snippet, contentDetails, and statistics for a given video ID."""
        headers = {
            "X-Goog-Api-Key": self.api_key
        }
        params = {
            "part": "snippet,contentDetails,statistics",
            "id": video_id
        }

        try:
            response = requests.get(self.BASE_URL, headers=headers, params=params, timeout=15)
        except requests.RequestException as e:
            safe_err = self._sanitize_error_text(e)
            raise MetadataFetchError(f"Network error communicating with YouTube Data API: {safe_err}")

        if response.status_code != 200:
            safe_text = self._sanitize_error_text(response.text)
            raise MetadataFetchError(
                f"YouTube Data API request failed with status {response.status_code}: {safe_text}"
            )

        data = response.json()
        items = data.get("items", [])
        if not items:
            raise MetadataFetchError(f"No video found for ID '{video_id}' (video may be private or deleted).")

        return self._parse_video_item(items[0], fallback_id=video_id)

    def fetch_metadata_batch(self, video_ids: List[str]) -> List[VideoMetadata]:
        """Fetch snippet, contentDetails, and statistics for up to 50 video IDs in a single batch request."""
        if not video_ids:
            return []

        clean_ids = list(dict.fromkeys(vid.strip() for vid in video_ids if vid and vid.strip()))[:50]
        if not clean_ids:
            return []

        headers = {
            "X-Goog-Api-Key": self.api_key
        }
        params = {
            "part": "snippet,contentDetails,statistics",
            "id": ",".join(clean_ids)
        }

        try:
            response = requests.get(self.BASE_URL, headers=headers, params=params, timeout=15)
        except requests.RequestException as e:
            safe_err = self._sanitize_error_text(e)
            raise MetadataFetchError(f"Network error communicating with YouTube Data API batch: {safe_err}")

        if response.status_code != 200:
            safe_text = self._sanitize_error_text(response.text)
            raise MetadataFetchError(
                f"YouTube Data API batch request failed with status {response.status_code}: {safe_text}"
            )

        data = response.json()
        items = data.get("items", [])
        return [self._parse_video_item(item) for item in items]

