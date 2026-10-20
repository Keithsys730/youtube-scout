"""Unit tests for SQLiteLedger and Deduplicator candidate filtering."""

import os
import sqlite3
import pytest
from datetime import datetime, timezone, timedelta
from unittest.mock import patch, MagicMock

from youtube_scout.deduplicator import SQLiteLedger, Deduplicator
from youtube_scout.models import (
    VideoMetadata,
    VideoAnalysis,
    ScoredVideoCandidate,
    SignalScores,
    VideoChapter,
)


@pytest.fixture
def temp_db_path(tmp_path) -> str:
    """Fixture providing a temporary SQLite database path."""
    return str(tmp_path / "test_history.db")


@pytest.fixture
def ledger(temp_db_path: str) -> SQLiteLedger:
    """Fixture providing an initialized SQLiteLedger instance."""
    return SQLiteLedger(db_path=temp_db_path)


@pytest.fixture
def sample_metadata() -> VideoMetadata:
    """Sample video metadata fixture."""
    return VideoMetadata(
        video_id="video123456",
        title="Async Rust in Production",
        channel_title="Rustacean",
        published_at="2026-08-01T10:00:00Z",
        duration_seconds=600,
        duration_str="10:00",
        view_count=50000,
        like_count=2500,
        description="Async Rust overview",
    )


@pytest.fixture
def sample_candidate(sample_metadata: VideoMetadata) -> ScoredVideoCandidate:
    """Sample scored candidate fixture."""
    return ScoredVideoCandidate(
        metadata=sample_metadata,
        signals=SignalScores(
            view_velocity=10.0,
            engagement_ratio=0.05,
            recency_decay=0.9,
            composite_score=8.5,
        ),
        rank=1,
    )


def test_sqlite_ledger_initialization(temp_db_path: str):
    """Verify table schema, indices, WAL mode, and pragmas upon initialization (B1)."""
    ledger = SQLiteLedger(db_path=temp_db_path)
    assert os.path.exists(temp_db_path)

    with ledger._get_connection() as conn:
        cursor = conn.cursor()
        # Verify journal mode is WAL
        cursor.execute("PRAGMA journal_mode;")
        row = cursor.fetchone()
        assert row[0].upper() == "WAL"

        # Verify busy timeout is 5000 ms
        cursor.execute("PRAGMA busy_timeout;")
        row = cursor.fetchone()
        assert row[0] == 5000

        # Verify videos table schema
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='videos';")
        assert cursor.fetchone() is not None

        # Verify indices
        cursor.execute("SELECT name FROM sqlite_master WHERE type='index' AND name='idx_videos_last_recommended';")
        assert cursor.fetchone() is not None

        cursor.execute("SELECT name FROM sqlite_master WHERE type='index' AND name='idx_videos_query';")
        assert cursor.fetchone() is not None


def test_sqlite_ledger_record_and_upsert(ledger: SQLiteLedger, sample_metadata: VideoMetadata):
    """Verify atomic insertion of new videos and upsert of existing videos (B3)."""
    # 1. Insert new video
    recorded = ledger.record_videos(
        [sample_metadata],
        recorded_at="2026-08-10",
        status="ANALYZED",
        query="Rust",
    )
    assert recorded is True

    with ledger._get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM videos WHERE video_id = ?", (sample_metadata.video_id,))
        row = cursor.fetchone()
        assert row["video_id"] == sample_metadata.video_id
        assert row["title"] == sample_metadata.title
        assert row["channel_title"] == sample_metadata.channel_title
        assert row["first_seen_date"] == "2026-08-10"
        assert row["last_recommended_date"] == "2026-08-10"
        assert row["recommend_count"] == 1
        assert row["status"] == "ANALYZED"
        assert row["query"] == "Rust"
        assert row["view_count"] == 50000

    # 2. Upsert same video on later date with updated view count
    updated_meta = VideoMetadata(
        video_id=sample_metadata.video_id,
        title="Async Rust in Production - Updated Title",
        channel_title=sample_metadata.channel_title,
        published_at=sample_metadata.published_at,
        duration_seconds=sample_metadata.duration_seconds,
        duration_str=sample_metadata.duration_str,
        view_count=75000,
        like_count=3500,
        description=sample_metadata.description,
    )
    upserted = ledger.record_videos(
        [updated_meta],
        recorded_at="2026-08-20",
        status="RECOMMENDED",
    )
    assert upserted is True

    with ledger._get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM videos WHERE video_id = ?", (sample_metadata.video_id,))
        row = cursor.fetchone()
        assert row["first_seen_date"] == "2026-08-10"  # preserved
        assert row["last_recommended_date"] == "2026-08-20"  # updated
        assert row["recommend_count"] == 2  # incremented
        assert row["status"] == "RECOMMENDED"
        assert row["view_count"] == 75000
        assert row["like_count"] == 3500
        assert row["title"] == "Async Rust in Production - Updated Title"
        assert row["query"] == "Rust"  # preserved via COALESCE


def test_sqlite_ledger_record_various_types(ledger: SQLiteLedger, sample_metadata: VideoMetadata, sample_candidate: ScoredVideoCandidate):
    """Verify recording of VideoAnalysis, ScoredVideoCandidate, VideoMetadata, and dict items."""
    analysis = VideoAnalysis(
        metadata=sample_metadata,
        status="SUCCESS",
        overview="High performance Rust",
        key_points=[],
        takeaways=[],
    )
    dict_item = {
        "video_id": "dict_vid_01",
        "title": "Dict Video Title",
        "channel_title": "Dict Channel",
        "video_url": "https://youtu.be/dict_vid_01",
        "duration_seconds": 300,
        "view_count": 1000,
        "like_count": 100,
        "status": "ANALYZED",
    }

    assert ledger.record_videos([analysis, sample_candidate, dict_item], recorded_at="2026-08-25") is True

    with ledger._get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM videos")
        # sample_metadata and sample_candidate share "video123456", so 1 + 1 (dict) = 2 distinct rows
        assert cursor.fetchone()[0] == 2


def test_sqlite_ledger_is_in_cooldown(ledger: SQLiteLedger, sample_metadata: VideoMetadata):
    """Verify cooldown boundary calculations, zero cooldown, and non-existent videos (B2)."""
    # Not in DB -> False
    assert ledger.is_in_cooldown(sample_metadata.video_id, cooldown_days=30) is False

    # Record video at 2026-08-10
    ledger.record_videos([sample_metadata], recorded_at="2026-08-10")

    # 5 days later (2026-08-15) -> In cooldown
    assert ledger.is_in_cooldown(sample_metadata.video_id, cooldown_days=30, as_of="2026-08-15") is True

    # 29 days later (2026-09-08) -> In cooldown (< 30)
    assert ledger.is_in_cooldown(sample_metadata.video_id, cooldown_days=30, as_of="2026-09-08") is True

    # Exact 30 days later (2026-09-09) -> Cooldown expired (30 is not < 30)
    assert ledger.is_in_cooldown(sample_metadata.video_id, cooldown_days=30, as_of="2026-09-09") is False

    # 40 days later (2026-09-19) -> False
    assert ledger.is_in_cooldown(sample_metadata.video_id, cooldown_days=30, as_of="2026-09-19") is False

    # Cooldown days = 0 -> always False
    assert ledger.is_in_cooldown(sample_metadata.video_id, cooldown_days=0, as_of="2026-08-15") is False
    assert ledger.is_in_cooldown(sample_metadata.video_id, cooldown_days=-5, as_of="2026-08-15") is False


def test_sqlite_ledger_inspect_history(ledger: SQLiteLedger):
    """Verify history summary statistics and recent records inspection (B12)."""
    # Empty DB
    empty_res = ledger.inspect_history(cooldown_days=30)
    assert empty_res["total_records"] == 0
    assert empty_res["cooldown_count"] == 0
    assert empty_res["recent_entries"] == []

    # Insert two videos: one recent, one older than 30 days
    today_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    old_date = (datetime.now(timezone.utc) - timedelta(days=45)).strftime("%Y-%m-%d")

    v1 = {"video_id": "recent_vid", "title": "Recent Video", "channel_title": "Ch1"}
    v2 = {"video_id": "old_vid", "title": "Old Video", "channel_title": "Ch2"}

    ledger.record_videos([v1], recorded_at=today_str, status="ANALYZED", query="Query1")
    ledger.record_videos([v2], recorded_at=old_date, status="RECOMMENDED", query="Query2")

    res = ledger.inspect_history(limit=10, cooldown_days=30)
    assert res["total_records"] == 2
    assert res["cooldown_count"] == 1  # only recent_vid is within 30 days
    assert res["cooldown_days"] == 30
    assert len(res["recent_entries"]) == 2

    # Verify descending ordering: recent_vid should come first
    assert res["recent_entries"][0]["video_id"] == "recent_vid"
    assert res["recent_entries"][0]["in_cooldown"] is True
    assert res["recent_entries"][1]["video_id"] == "old_vid"
    assert res["recent_entries"][1]["in_cooldown"] is False


def test_sqlite_ledger_fail_open_resilience(tmp_path):
    """Verify fail-open error resilience on corrupted/locked database (B13, B18)."""
    corrupt_db = str(tmp_path / "corrupted.db")
    with open(corrupt_db, "wb") as f:
        f.write(b"NOT A VALID SQLITE DB FILE HEADER")

    ledger = SQLiteLedger(db_path=corrupt_db)

    # All methods must gracefully degrade without raising unhandled exceptions
    assert ledger.is_in_cooldown("vid123", cooldown_days=30) is False
    assert ledger.record_videos([{"video_id": "vid123"}]) is False
    history = ledger.inspect_history()
    assert history["total_records"] == 0
    assert history["cooldown_count"] == 0
    assert history["recent_entries"] == []


def test_deduplicator_filter_candidates(ledger: SQLiteLedger, sample_candidate: ScoredVideoCandidate):
    """Verify candidate partitioning into fresh and in-cooldown sets (B4)."""
    cand2 = ScoredVideoCandidate(
        metadata=VideoMetadata(
            video_id="video_fresh_99",
            title="Fresh Video",
            channel_title="Fresh Channel",
            published_at="2026-08-20T10:00:00Z",
            duration_seconds=500,
            duration_str="08:20",
            view_count=10000,
            like_count=500,
            description="Fresh",
        ),
        signals=SignalScores(view_velocity=5.0, engagement_ratio=0.05, recency_decay=0.9, composite_score=6.0),
        rank=2,
    )

    # Record sample_candidate as seen today
    ledger.record_videos([sample_candidate], status="ANALYZED")

    deduplicator = Deduplicator(ledger=ledger, cooldown_days=30)
    fresh, cooldown = deduplicator.filter_candidates([sample_candidate, cand2])

    assert len(cooldown) == 1
    assert cooldown[0].metadata.video_id == sample_candidate.metadata.video_id
    assert len(fresh) == 1
    assert fresh[0].metadata.video_id == "video_fresh_99"


def test_deduplicator_has_exhaustion(ledger: SQLiteLedger):
    """Verify topic exhaustion detection (B5)."""
    deduplicator = Deduplicator(ledger=ledger, cooldown_days=30)

    # Non-empty total, empty fresh -> Exhausted
    assert deduplicator.has_exhaustion(fresh_candidates=[], total_candidates=["cand1", "cand2"]) is True

    # Non-empty fresh -> Not exhausted
    assert deduplicator.has_exhaustion(fresh_candidates=["cand1"], total_candidates=["cand1", "cand2"]) is False

    # Empty total -> Not exhausted
    assert deduplicator.has_exhaustion(fresh_candidates=[], total_candidates=[]) is False


def test_deduplicator_bypass_when_cooldown_zero(ledger: SQLiteLedger, sample_candidate: ScoredVideoCandidate):
    """Verify deduplication is bypassed when cooldown_days=0."""
    ledger.record_videos([sample_candidate], status="ANALYZED")

    deduplicator = Deduplicator(ledger=ledger, cooldown_days=0)
    fresh, cooldown = deduplicator.filter_candidates([sample_candidate])

    assert len(fresh) == 1
    assert len(cooldown) == 0


def test_sqlite_ledger_record_empty_or_invalid_items(ledger: SQLiteLedger):
    """Verify empty list and invalid items are safely handled."""
    assert ledger.record_videos([]) is True
    assert ledger.record_videos([None, {}, {"title": "no video id"}]) is True
    assert ledger.inspect_history()["total_records"] == 0


def test_sqlite_ledger_security_boundary_no_secrets(ledger: SQLiteLedger):
    """Verify API keys are never stored in database columns (B15)."""
    fake_key = "AIzaSySecretApiKeyForYouTube12345"
    meta = {
        "video_id": "clean_id_01",
        "title": "Clean Title",
        "channel_title": "Clean Channel",
        "api_key": fake_key,  # accidental extra key
    }
    ledger.record_videos([meta], status="ANALYZED")

    with ledger._get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM videos WHERE video_id = 'clean_id_01'")
        row = dict(cursor.fetchone())
        # All columns
        for col_name, val in row.items():
            assert fake_key not in str(val)

