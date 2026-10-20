"""Signal calculation, shorts filtering, and candidate ranking for YouTube Scout."""

from datetime import datetime, timezone, timedelta
from typing import List, Dict, Any, Optional
from .models import VideoMetadata, SignalScores, ScoredVideoCandidate


def filter_by_duration(videos: List[VideoMetadata], min_duration_seconds: int = 120) -> List[VideoMetadata]:
    """Filter out videos with duration less than min_duration_seconds (e.g. YouTube Shorts)."""
    return [v for v in videos if v.duration_seconds >= min_duration_seconds]


def rank_candidates(
    videos: List[VideoMetadata],
    config: Dict[str, Any],
    now: Optional[datetime] = None
) -> List[ScoredVideoCandidate]:
    """Rank candidate videos using a configurable composite signal formula.
    
    Formula components:
      1. View velocity = view_count / days_since_published
      2. Engagement ratio = like_count / view_count
      3. Recency decay = 2 ** (-days_since_published / half_life_days)
    
    Each component is batch Min-Max normalized to [0.0, 1.0] across all candidates.
    If min == max for a feature, all candidates receive 1.0.
    
    Composite score = w_v * norm_v + w_e * norm_e + w_r * norm_r.
    
    Deterministic sorting:
      Descending by composite score, then view count, then published timestamp, then video ID.
    """
    if not videos:
        return []

    # Parse config with fallback defaults
    safe_config = config if isinstance(config, dict) else {}
    ranking_cfg = safe_config.get("ranking")
    if ranking_cfg is None:
        ranking_cfg = safe_config if ("weights" in safe_config or "recency_half_life_days" in safe_config) else {}
    ranking_cfg = ranking_cfg or {}
    if not isinstance(ranking_cfg, dict):
        ranking_cfg = {}

    weights_cfg = (ranking_cfg.get("weights") or {}) if isinstance(ranking_cfg, dict) else {}
    if not isinstance(weights_cfg, dict):
        weights_cfg = {}

    raw_v = weights_cfg.get("velocity")
    w_v = float(raw_v) if raw_v is not None else 0.5

    raw_e = weights_cfg.get("engagement")
    w_e = float(raw_e) if raw_e is not None else 0.3

    raw_r = weights_cfg.get("recency")
    w_r = float(raw_r) if raw_r is not None else 0.2

    raw_hl = ranking_cfg.get("recency_half_life_days")
    half_life_days = float(raw_hl) if raw_hl is not None else 30.0
    if half_life_days <= 0:
        half_life_days = 30.0

    if now is None:
        now = datetime.now(timezone.utc)
    elif now.tzinfo is None:
        now = now.replace(tzinfo=timezone.utc)

    # Step 1: Compute raw metrics for each candidate
    raw_data = []
    for video in videos:
        pub_dt = None
        if video.published_at:
            try:
                clean_iso = video.published_at.replace("Z", "+00:00")
                pub_dt = datetime.fromisoformat(clean_iso)
                if pub_dt.tzinfo is None:
                    pub_dt = pub_dt.replace(tzinfo=timezone.utc)
            except Exception:
                pub_dt = None

        if pub_dt is not None:
            pub_timestamp = pub_dt.timestamp()
            diff_seconds = (now - pub_dt).total_seconds()
            days = max(diff_seconds / 86400.0, 0.1)
        else:
            days = float(half_life_days)
            pub_timestamp = (now - timedelta(days=days)).timestamp()

        velocity = video.view_count / days
        engagement = (video.like_count / video.view_count) if video.view_count > 0 else 0.0
        recency = 2.0 ** (-days / half_life_days)

        raw_data.append({
            "video": video,
            "pub_timestamp": pub_timestamp,
            "velocity": velocity,
            "engagement": engagement,
            "recency": recency,
        })

    # Step 2: Normalization per feature
    v_vals = [d["velocity"] for d in raw_data]
    min_v, max_v = min(v_vals), max(v_vals)

    e_vals = [d["engagement"] for d in raw_data]
    min_e, max_e = min(e_vals), max(e_vals)

    for d in raw_data:
        norm_v = 1.0 if (max_v - min_v) < 1e-5 else (d["velocity"] - min_v) / (max_v - min_v)
        norm_e = 1.0 if (max_e - min_e) < 1e-5 else (d["engagement"] - min_e) / (max_e - min_e)
        # Recency decay is already an absolute mathematical metric bounded in [0.0, 1.0]
        norm_r = max(0.0, min(1.0, d["recency"]))

        composite = (w_v * norm_v) + (w_e * norm_e) + (w_r * norm_r)
        d["composite"] = composite

    # Step 3: Deterministic sorting
    # Descending by rounded composite (4 decimals), then view_count, then pub_timestamp, then video_id ascending
    raw_data.sort(
        key=lambda d: (
            -round(d["composite"], 4),
            -d["video"].view_count,
            -d["pub_timestamp"],
            d["video"].video_id
        )
    )

    # Step 4: Construct ScoredVideoCandidate with 1-based rank
    ranked_candidates: List[ScoredVideoCandidate] = []
    for idx, d in enumerate(raw_data, start=1):
        signals = SignalScores(
            view_velocity=round(d["velocity"], 2),
            engagement_ratio=round(d["engagement"], 4),
            recency_decay=round(d["recency"], 4),
            composite_score=round(d["composite"], 4),
        )
        ranked_candidates.append(
            ScoredVideoCandidate(
                metadata=d["video"],
                signals=signals,
                rank=idx
            )
        )

    return ranked_candidates
