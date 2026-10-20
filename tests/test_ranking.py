"""Tests for filtering and candidate ranking in youtube_scout.ranking."""

from datetime import datetime, timezone, timedelta
import pytest
from youtube_scout.models import VideoMetadata
from youtube_scout.ranking import filter_by_duration, rank_candidates


def make_test_video(
    video_id: str,
    duration_seconds: int,
    view_count: int = 1000,
    like_count: int = 50,
    published_at: str = "2026-08-01T00:00:00Z"
) -> VideoMetadata:
    return VideoMetadata(
        video_id=video_id,
        title=f"Title {video_id}",
        channel_title="Test Channel",
        published_at=published_at,
        duration_seconds=duration_seconds,
        duration_str=f"{duration_seconds // 60:02d}:{duration_seconds % 60:02d}",
        view_count=view_count,
        like_count=like_count,
        description="Test description",
        chapters=[]
    )


def test_filter_by_duration():
    v1 = make_test_video("v1", 30)
    v2 = make_test_video("v2", 119)
    v3 = make_test_video("v3", 120)
    v4 = make_test_video("v4", 600)

    # Default 120s
    filtered = filter_by_duration([v1, v2, v3, v4], min_duration_seconds=120)
    assert len(filtered) == 2
    assert [v.video_id for v in filtered] == ["v3", "v4"]

    # Custom 60s
    filtered_60 = filter_by_duration([v1, v2, v3, v4], min_duration_seconds=60)
    assert len(filtered_60) == 3
    assert [v.video_id for v in filtered_60] == ["v2", "v3", "v4"]

    # Empty list
    assert filter_by_duration([], 120) == []


def test_rank_candidates_empty():
    assert rank_candidates([], {}) == []


def test_rank_candidates_single_candidate():
    v = make_test_video("single", 300, view_count=5000, like_count=250)
    config = {
        "ranking": {
            "weights": {"velocity": 0.5, "engagement": 0.3, "recency": 0.2},
            "recency_half_life_days": 30.0
        }
    }
    ranked = rank_candidates([v], config)
    assert len(ranked) == 1
    assert ranked[0].rank == 1
    # Velocity and engagement normalize to 1.0; recency decay is naturally bounded in [0.0, 1.0]
    expected_composite = round(0.5 * 1.0 + 0.3 * 1.0 + 0.2 * ranked[0].signals.recency_decay, 4)
    assert ranked[0].signals.composite_score == expected_composite
    assert ranked[0].signals.engagement_ratio == 0.05


def test_rank_candidates_signals_and_sorting():
    fixed_now = datetime(2026, 8, 27, 0, 0, 0, tzinfo=timezone.utc)
    # v_fast: published 10 days ago, 100,000 views (velocity = 10,000/day)
    pub_10d = (fixed_now - timedelta(days=10)).isoformat()
    v_fast = make_test_video("v_fast", 500, view_count=100000, like_count=5000, published_at=pub_10d)

    # v_slow: published 100 days ago, 1,000 views (velocity = 10/day)
    pub_100d = (fixed_now - timedelta(days=100)).isoformat()
    v_slow = make_test_video("v_slow", 500, view_count=1000, like_count=10, published_at=pub_100d)

    config = {
        "ranking": {
            "weights": {"velocity": 0.5, "engagement": 0.3, "recency": 0.2},
            "recency_half_life_days": 30.0
        }
    }

    ranked = rank_candidates([v_slow, v_fast], config, now=fixed_now)
    assert len(ranked) == 2
    assert ranked[0].metadata.video_id == "v_fast"
    assert ranked[0].rank == 1
    assert ranked[1].metadata.video_id == "v_slow"
    assert ranked[1].rank == 2
    assert ranked[0].signals.composite_score > ranked[1].signals.composite_score


def test_rank_candidates_deterministic_tie_breaking():
    # Candidates with identical metrics except tie-breakers
    # Case 1: Tie-breaker by view count
    v1 = make_test_video("v_more_views", 300, view_count=2000, like_count=100, published_at="2026-08-01T00:00:00Z")
    v2 = make_test_video("v_less_views", 300, view_count=1000, like_count=50, published_at="2026-08-01T00:00:00Z")
    # Set weights to 0 for velocity and 1 for engagement (both have 0.05 engagement ratio)
    config = {
        "weights": {"velocity": 0.0, "engagement": 1.0, "recency": 0.0},
        "recency_half_life_days": 30.0
    }
    ranked = rank_candidates([v2, v1], config)
    assert ranked[0].metadata.video_id == "v_more_views"
    assert ranked[1].metadata.video_id == "v_less_views"

    # Case 2: Tie-breaker by published date (newer first)
    v_newer = make_test_video("v_newer", 300, view_count=1000, like_count=50, published_at="2026-08-10T00:00:00Z")
    v_older = make_test_video("v_older", 300, view_count=1000, like_count=50, published_at="2026-08-01T00:00:00Z")
    ranked_date = rank_candidates([v_older, v_newer], config)
    assert ranked_date[0].metadata.video_id == "v_newer"
    assert ranked_date[1].metadata.video_id == "v_older"

    # Case 3: Tie-breaker by video_id ascending
    v_a = make_test_video("aaa", 300, view_count=1000, like_count=50, published_at="2026-08-01T00:00:00Z")
    v_z = make_test_video("zzz", 300, view_count=1000, like_count=50, published_at="2026-08-01T00:00:00Z")
    ranked_id = rank_candidates([v_z, v_a], config)
    assert ranked_id[0].metadata.video_id == "aaa"
    assert ranked_id[1].metadata.video_id == "zzz"


def test_rank_candidates_zero_views_and_fallback_date():
    fixed_now = datetime(2026, 8, 27, 0, 0, 0, tzinfo=timezone.utc)
    v_zero = make_test_video("v_zero", 300, view_count=0, like_count=0, published_at="invalid-date")
    config = {
        "ranking": {
            "weights": {"velocity": 0.5, "engagement": 0.3, "recency": 0.2},
            "recency_half_life_days": 30.0
        }
    }
    ranked = rank_candidates([v_zero], config, now=fixed_now)
    assert len(ranked) == 1
    assert ranked[0].signals.engagement_ratio == 0.0
    assert ranked[0].signals.view_velocity == 0.0


def test_rank_candidates_none_config_and_null_subkeys():
    v1 = make_test_video("v1", 300, view_count=1000, like_count=50)
    v2 = make_test_video("v2", 400, view_count=2000, like_count=100)

    null_configs = [
        None,
        {},
        {"ranking": None},
        {"weights": None},
        {"ranking": {"weights": None}},
        {"ranking": {"weights": None, "recency_half_life_days": None}},
        {"ranking": {"weights": {"velocity": None, "engagement": None, "recency": None}}},
    ]

    for cfg in null_configs:
        ranked = rank_candidates([v1, v2], cfg)
        assert len(ranked) == 2
        assert ranked[0].rank == 1
        assert ranked[1].rank == 2
        assert ranked[0].signals.composite_score >= 0.0


def test_rank_candidates_corrupt_date_neutral_age():
    fixed_now = datetime(2026, 8, 27, 0, 0, 0, tzinfo=timezone.utc)
    v_corrupt = make_test_video("v_corrupt", 300, view_count=3000, like_count=150, published_at="invalid-iso-string")
    v_none = make_test_video("v_none", 300, view_count=3000, like_count=150, published_at="")

    config = {
        "ranking": {
            "weights": {"velocity": 0.5, "engagement": 0.3, "recency": 0.2},
            "recency_half_life_days": 30.0
        }
    }

    # For single video, min == max so normalized composite = 1.0, but raw signals are preserved in SignalScores
    ranked_corrupt = rank_candidates([v_corrupt], config, now=fixed_now)
    assert len(ranked_corrupt) == 1
    # velocity = view_count / 30.0 = 3000 / 30.0 = 100.0 (NOT 3000 / 0.1 = 30000.0)
    assert ranked_corrupt[0].signals.view_velocity == 100.0
    # recency = 2 ** (-30.0 / 30.0) = 0.5 (NOT 2 ** (-0.1 / 30.0) = 0.9977)
    assert ranked_corrupt[0].signals.recency_decay == 0.5

    ranked_none = rank_candidates([v_none], config, now=fixed_now)
    assert len(ranked_none) == 1
    assert ranked_none[0].signals.view_velocity == 100.0
    assert ranked_none[0].signals.recency_decay == 0.5


def test_rank_candidates_tie_breaking_reaches_tier_2_view_count():
    fixed_now = datetime(2026, 8, 27, 0, 0, 0, tzinfo=timezone.utc)
    pub_dt = (fixed_now - timedelta(days=10)).isoformat()

    # Both videos have identical normalized velocity and engagement and recency,
    # resulting in identical rounded composite_score (e.g. 1.0000).
    # However v_popular has 500,000 views while v_smaller has 50,000 views.
    v_popular = make_test_video("popular", 300, view_count=500000, like_count=25000, published_at=pub_dt)
    v_smaller = make_test_video("smaller", 300, view_count=50000, like_count=2500, published_at=pub_dt)

    config = {
        "ranking": {
            "weights": {"velocity": 0.5, "engagement": 0.3, "recency": 0.2},
            "recency_half_life_days": 30.0
        }
    }

    # Pass in order: [smaller, popular]
    ranked = rank_candidates([v_smaller, v_popular], config, now=fixed_now)
    assert len(ranked) == 2
    # Tier 2 (view_count) ensures v_popular is ranked #1
    assert ranked[0].metadata.video_id == "popular"
    assert ranked[1].metadata.video_id == "smaller"


def test_rank_candidates_micro_variance_epsilon():
    fixed_now = datetime(2026, 8, 27, 0, 0, 0, tzinfo=timezone.utc)
    pub_dt = (fixed_now - timedelta(days=10)).isoformat()

    # Two videos with difference in view count of 0.000001 (diff < 1e-5)
    v1 = make_test_video("v1", 300, view_count=1000, like_count=50, published_at=pub_dt)
    v2 = make_test_video("v2", 300, view_count=1000, like_count=50, published_at=pub_dt)

    config = {
        "ranking": {
            "weights": {"velocity": 0.5, "engagement": 0.3, "recency": 0.2},
            "recency_half_life_days": 30.0
        }
    }

    ranked = rank_candidates([v1, v2], config, now=fixed_now)
    assert len(ranked) == 2
    assert ranked[0].signals.composite_score == ranked[1].signals.composite_score


