"""Test fixtures and offline test setup for youtube-scout."""

import os
import json
import pytest
from typing import Dict, Any, List
from youtube_scout.models import VideoMetadata, TranscriptSegment, VideoChapter


@pytest.fixture
def fixtures_dir() -> str:
    return os.path.join(os.path.dirname(__file__), "fixtures")


@pytest.fixture
def raw_videos_list_response(fixtures_dir: str) -> Dict[str, Any]:
    path = os.path.join(fixtures_dir, "videos_list_response.json")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


@pytest.fixture
def sample_transcript_segments(fixtures_dir: str) -> List[TranscriptSegment]:
    path = os.path.join(fixtures_dir, "sample_transcript.json")
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return [
        TranscriptSegment(
            text=item["text"],
            start=float(item["start"]),
            duration=float(item["duration"])
        )
        for item in data
    ]


@pytest.fixture
def sample_video_metadata() -> VideoMetadata:
    return VideoMetadata(
        video_id="dQw4w9WgXcQ",
        title="Building High-Throughput Event-Driven Microservices in Rust",
        channel_title="Systems Engineering Hub",
        published_at="2024-03-15T14:30:00Z",
        duration_seconds=1305,
        duration_str="21:45",
        view_count=142500,
        like_count=8920,
        description="In this technical deep dive, we explore how to architect high-throughput event loops.\n\nChapters:\n00:00 Introduction\n02:15 Tokio Configuration",
        chapters=[
            VideoChapter(start_seconds=0, timestamp_str="00:00", title="Introduction"),
            VideoChapter(start_seconds=135, timestamp_str="02:15", title="Tokio Configuration")
        ]
    )


@pytest.fixture(autouse=True)
def isolate_test_ledger(tmp_path, monkeypatch):
    """Ensure tests that use the default data/history.db run with an isolated temporary database."""
    from youtube_scout.deduplicator import SQLiteLedger

    test_db = str(tmp_path / "isolated_history.db")
    original_init = SQLiteLedger.__init__

    def patched_init(self, db_path: str = "data/history.db"):
        if db_path == "data/history.db":
            db_path = test_db
        return original_init(self, db_path=db_path)

    monkeypatch.setattr(SQLiteLedger, "__init__", patched_init)

