"""Transactional SQLite deduplication ledger and candidate filtering for YouTube Scout."""

import os
import sqlite3
import logging
from contextlib import contextmanager
from datetime import datetime, timezone, timedelta
from typing import Optional, Dict, Any, List, Tuple

from .models import VideoAnalysis, ScoredVideoCandidate, VideoMetadata

logger = logging.getLogger("youtube_scout.deduplicator")


class SQLiteLedger:
    """SQLite-backed persistent history ledger with WAL mode and fail-open resilience."""

    def __init__(self, db_path: str = "data/history.db"):
        self.db_path = db_path
        self._init_db()

    @contextmanager
    def _get_connection(self):
        """Context manager yielding SQLite connection with WAL pragmas and busy timeout."""
        conn = sqlite3.connect(self.db_path, timeout=5.0)
        conn.row_factory = sqlite3.Row
        try:
            conn.execute("PRAGMA journal_mode=WAL;")
            conn.execute("PRAGMA busy_timeout=5000;")
            conn.execute("PRAGMA synchronous=NORMAL;")
            yield conn
        finally:
            conn.close()

    def _init_db(self) -> None:
        """Initialize SQLite database directory, table schema, and query indices."""
        try:
            db_dir = os.path.dirname(self.db_path)
            if db_dir:
                os.makedirs(db_dir, exist_ok=True)
            with self._get_connection() as conn:
                with conn:
                    conn.execute(
                        """
                        CREATE TABLE IF NOT EXISTS videos (
                            video_id TEXT PRIMARY KEY,
                            title TEXT,
                            channel_title TEXT,
                            video_url TEXT,
                            first_seen_date TEXT,
                            last_recommended_date TEXT,
                            recommend_count INTEGER DEFAULT 1,
                            status TEXT,
                            query TEXT,
                            duration_seconds INTEGER,
                            view_count INTEGER,
                            like_count INTEGER
                        );
                        """
                    )
                    conn.execute(
                        "CREATE INDEX IF NOT EXISTS idx_videos_last_recommended ON videos(last_recommended_date);"
                    )
                    conn.execute(
                        "CREATE INDEX IF NOT EXISTS idx_videos_query ON videos(query);"
                    )
        except Exception as e:
            logger.warning(f"Failed to initialize SQLite ledger at {self.db_path}: {e}")

    def is_in_cooldown(
        self,
        video_id: str,
        cooldown_days: int = 30,
        as_of: Optional[str] = None
    ) -> bool:
        """Check if video was recommended within cooldown period.
        
        Returns False if cooldown_days <= 0, video not found, or on database error.
        """
        if cooldown_days <= 0 or not video_id:
            return False

        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    "SELECT last_recommended_date FROM videos WHERE video_id = ?",
                    (video_id,)
                )
                row = cursor.fetchone()
                if not row or not row["last_recommended_date"]:
                    return False

                last_str = str(row["last_recommended_date"])[:10]
                last_date = datetime.strptime(last_str, "%Y-%m-%d").date()

                if as_of is None:
                    as_of_date = datetime.now(timezone.utc).date()
                else:
                    as_of_date = datetime.strptime(str(as_of)[:10], "%Y-%m-%d").date()

                diff_days = (as_of_date - last_date).days
                return diff_days < cooldown_days
        except Exception as e:
            logger.warning(f"Failed to check cooldown for video {video_id}: {e}")
            return False

    def record_videos(
        self,
        videos: List[Any],
        recorded_at: Optional[str] = None,
        status: str = "ANALYZED",
        query: Optional[str] = None
    ) -> bool:
        """Atomically insert or upsert video records inside a transaction."""
        if not videos:
            return True

        if recorded_at is None:
            record_date = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        else:
            record_date = str(recorded_at)[:10]

        try:
            with self._get_connection() as conn:
                with conn:
                    cursor = conn.cursor()
                    for item in videos:
                        info = self._extract_video_info(item, default_status=status)
                        if not info or not info.get("video_id"):
                            continue

                        v_id = info["video_id"]
                        cursor.execute("SELECT video_id FROM videos WHERE video_id = ?", (v_id,))
                        existing = cursor.fetchone()

                        if existing:
                            cursor.execute(
                                """
                                UPDATE videos
                                SET last_recommended_date = ?,
                                    recommend_count = recommend_count + 1,
                                    status = ?,
                                    view_count = ?,
                                    like_count = ?,
                                    title = CASE WHEN ? != '' THEN ? ELSE title END,
                                    channel_title = CASE WHEN ? != '' THEN ? ELSE channel_title END,
                                    query = COALESCE(?, query)
                                WHERE video_id = ?
                                """,
                                (
                                    record_date,
                                    info["status"],
                                    info["view_count"],
                                    info["like_count"],
                                    info["title"],
                                    info["title"],
                                    info["channel_title"],
                                    info["channel_title"],
                                    query,
                                    v_id
                                )
                            )
                        else:
                            cursor.execute(
                                """
                                INSERT INTO videos (
                                    video_id, title, channel_title, video_url,
                                    first_seen_date, last_recommended_date, recommend_count,
                                    status, query, duration_seconds, view_count, like_count
                                ) VALUES (?, ?, ?, ?, ?, ?, 1, ?, ?, ?, ?, ?)
                                """,
                                (
                                    v_id,
                                    info["title"],
                                    info["channel_title"],
                                    info["video_url"],
                                    record_date,
                                    record_date,
                                    info["status"],
                                    query,
                                    info["duration_seconds"],
                                    info["view_count"],
                                    info["like_count"]
                                )
                            )
            return True
        except Exception as e:
            logger.warning(f"Failed to record videos to ledger: {e}")
            return False

    @staticmethod
    def _extract_video_info(item: Any, default_status: str = "ANALYZED") -> Dict[str, Any]:
        """Extract standardized video fields across analysis models and dictionaries."""
        if isinstance(item, VideoAnalysis):
            meta = item.metadata
            return {
                "video_id": meta.video_id,
                "title": meta.title,
                "channel_title": meta.channel_title,
                "video_url": meta.video_url,
                "duration_seconds": meta.duration_seconds,
                "view_count": meta.view_count,
                "like_count": meta.like_count,
                "status": item.status or default_status,
            }
        if isinstance(item, ScoredVideoCandidate):
            meta = item.metadata
            return {
                "video_id": meta.video_id,
                "title": meta.title,
                "channel_title": meta.channel_title,
                "video_url": meta.video_url,
                "duration_seconds": meta.duration_seconds,
                "view_count": meta.view_count,
                "like_count": meta.like_count,
                "status": default_status,
            }
        if isinstance(item, VideoMetadata):
            return {
                "video_id": item.video_id,
                "title": item.title,
                "channel_title": item.channel_title,
                "video_url": item.video_url,
                "duration_seconds": item.duration_seconds,
                "view_count": item.view_count,
                "like_count": item.like_count,
                "status": default_status,
            }
        if isinstance(item, dict):
            meta = item.get("metadata", {}) if isinstance(item.get("metadata"), dict) else {}
            v_id = item.get("video_id") or meta.get("video_id")
            title = item.get("title") or meta.get("title", "")
            channel = item.get("channel_title") or meta.get("channel_title", "")
            url = item.get("video_url") or meta.get("video_url", f"https://youtu.be/{v_id}" if v_id else "")
            duration = item.get("duration_seconds") if "duration_seconds" in item else meta.get("duration_seconds", 0)
            views = item.get("view_count") if "view_count" in item else meta.get("view_count", 0)
            likes = item.get("like_count") if "like_count" in item else meta.get("like_count", 0)
            status_val = item.get("status", default_status)
            return {
                "video_id": v_id,
                "title": title,
                "channel_title": channel,
                "video_url": url,
                "duration_seconds": int(duration or 0),
                "view_count": int(views or 0),
                "like_count": int(likes or 0),
                "status": str(status_val or default_status),
            }
        return {}

    def inspect_history(self, limit: int = 20, cooldown_days: int = 30) -> Dict[str, Any]:
        """Query summary statistics and recent records from the ledger."""
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT COUNT(*) FROM videos")
                total_records = cursor.fetchone()[0]

                if cooldown_days <= 0:
                    cooldown_count = 0
                    cutoff_date = ""
                else:
                    today = datetime.now(timezone.utc).date()
                    cutoff_date = (today - timedelta(days=cooldown_days - 1)).strftime("%Y-%m-%d")
                    cursor.execute(
                        "SELECT COUNT(*) FROM videos WHERE last_recommended_date IS NOT NULL AND substr(last_recommended_date, 1, 10) >= ?",
                        (cutoff_date,)
                    )
                    cooldown_count = cursor.fetchone()[0]

                cursor.execute(
                    "SELECT * FROM videos ORDER BY last_recommended_date DESC, rowid DESC LIMIT ?",
                    (max(1, limit),)
                )
                rows = cursor.fetchall()
                recent_entries = []
                for row in rows:
                    entry = dict(row)
                    if cooldown_days <= 0:
                        entry["in_cooldown"] = False
                    else:
                        rec_date_str = str(entry.get("last_recommended_date") or "")[:10]
                        entry["in_cooldown"] = bool(rec_date_str and rec_date_str >= cutoff_date)
                    recent_entries.append(entry)

                return {
                    "total_records": total_records,
                    "cooldown_count": cooldown_count,
                    "cooldown_days": cooldown_days,
                    "recent_entries": recent_entries,
                }
        except Exception as e:
            logger.warning(f"Failed to inspect history ledger: {e}")
            return {
                "total_records": 0,
                "cooldown_count": 0,
                "cooldown_days": cooldown_days,
                "recent_entries": [],
            }


class Deduplicator:
    """Candidate deduplicator utilizing SQLiteLedger with exhaustion value floor."""

    def __init__(
        self,
        ledger: Optional[SQLiteLedger] = None,
        cooldown_days: int = 30
    ):
        self.ledger = ledger
        self.cooldown_days = cooldown_days

    def filter_candidates(
        self,
        candidates: List[ScoredVideoCandidate],
        cooldown_days: Optional[int] = None
    ) -> Tuple[List[ScoredVideoCandidate], List[ScoredVideoCandidate]]:
        """Partition candidates into (fresh_candidates, in_cooldown_candidates)."""
        cd_days = self.cooldown_days if cooldown_days is None else cooldown_days
        if not self.ledger or cd_days <= 0 or not candidates:
            return list(candidates), []

        fresh_candidates: List[ScoredVideoCandidate] = []
        in_cooldown_candidates: List[ScoredVideoCandidate] = []

        for cand in candidates:
            v_id = (
                cand.metadata.video_id
                if hasattr(cand, "metadata")
                else (cand.video_id if hasattr(cand, "video_id") else cand.get("video_id", ""))
            )
            if self.ledger.is_in_cooldown(v_id, cooldown_days=cd_days):
                in_cooldown_candidates.append(cand)
            else:
                fresh_candidates.append(cand)

        return fresh_candidates, in_cooldown_candidates

    def has_exhaustion(self, fresh_candidates: List[Any], total_candidates: List[Any]) -> bool:
        """True if total_candidates is non-empty and fresh_candidates is empty."""
        return bool(total_candidates and not fresh_candidates)
