"""Pipeline orchestrator for YouTube Scout single-video processing."""

import os
import yaml
import logging
from typing import Optional, Dict, Any, Callable, List
from .models import VideoAnalysis, TopicSearchResult, TopicRadarSynthesis
from .fetcher import YouTubeMetadataFetcher, extract_video_id
from .transcript import get_video_transcript
from .analyzer import VideoAnalyzer, MissingApiKeyError
from .reporter import VideoReporter
from .ranking import filter_by_duration, rank_candidates
from .deduplicator import SQLiteLedger, Deduplicator

logger = logging.getLogger("youtube_scout.runner")


class ScoutRunner:
    """End-to-end orchestrator for single-video and topic-search radar pipelines."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        custom_llm_handler: Optional[Callable[..., Any]] = None,
        config_path: str = "config.yaml",
        db_path: Optional[str] = None,
        cooldown_days: Optional[int] = None,
    ):
        self.config = self._load_config(config_path)
        self.custom_llm_handler = custom_llm_handler
        self.fetcher = YouTubeMetadataFetcher(api_key=api_key)

        # History & Deduplication setup
        if db_path is not None:
            self.history_file = db_path
        else:
            self.history_file = str(self.config.get("history_file") or "data/history.db")

        if cooldown_days is not None:
            self.cooldown_days = cooldown_days
        else:
            raw_cd = self.config.get("cooldown_days")
            try:
                self.cooldown_days = int(raw_cd) if raw_cd is not None else 30
            except (ValueError, TypeError):
                self.cooldown_days = 30

        self.ledger = SQLiteLedger(db_path=self.history_file)
        self.deduplicator = Deduplicator(ledger=self.ledger, cooldown_days=self.cooldown_days)

        search_cfg = self.config.get("search", {}) if isinstance(self.config, dict) else {}
        raw_top_n = self.config.get("top_n") if isinstance(self.config, dict) else None
        if raw_top_n is None and isinstance(search_cfg, dict):
            raw_top_n = search_cfg.get("top_n")
        try:
            self.config_top_n = max(1, int(raw_top_n)) if raw_top_n is not None else 3
        except (ValueError, TypeError):
            self.config_top_n = 3

        llm_cfg = self.config.get("llm", {})
        model_name = llm_cfg.get("model", "gpt-4o-mini")
        temperature = float(llm_cfg.get("temperature", 0.2))
        max_chars = self.config.get("max_transcript_chars", 100000)
        self.preferred_languages = self.config.get(
            "preferred_caption_languages", ["zh-TW", "zh-Hant", "zh", "en"]
        )
        self.analyzer = VideoAnalyzer(
            custom_llm_handler=custom_llm_handler,
            model_name=model_name,
            temperature=temperature,
            max_transcript_chars=max_chars
        )

    @classmethod
    def _deep_merge(cls, source: Dict[str, Any], target: Dict[str, Any]) -> Dict[str, Any]:
        """Recursively merge source defaults into target dictionary."""
        for key, value in source.items():
            if key not in target or target[key] is None:
                target[key] = value
            elif isinstance(value, dict) and isinstance(target[key], dict):
                cls._deep_merge(value, target[key])
        return target

    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load YAML configuration or return defaults."""
        defaults: Dict[str, Any] = {
            "reports_dir": "reports",
            "history_file": "data/history.db",
            "cooldown_days": 30,
            "default_language": "zh-TW",
            "preferred_caption_languages": ["zh-TW", "zh-Hant", "zh", "en"],
            "max_transcript_chars": 100000,
            "top_n": 3,
            "search": {
                "order": "relevance",
                "max_results": 50,
                "top_n": 3,
            },
            "filter": {
                "min_duration_seconds": 120,
            },
            "ranking": {
                "weights": {
                    "velocity": 0.5,
                    "engagement": 0.3,
                    "recency": 0.2,
                },
                "recency_half_life_days": 30.0,
            },
            "llm": {"model": "gpt-4o-mini", "temperature": 0.2}
        }
        if os.path.exists(config_path):
            try:
                with open(config_path, "r", encoding="utf-8") as f:
                    loaded = yaml.safe_load(f) or {}
                    if isinstance(loaded, dict):
                        return self._deep_merge(defaults, loaded)
                    return defaults
            except Exception as e:
                logger.warning(f"Failed to read config file {config_path}: {e}")
        return defaults

    def run_video(self, url_or_id: str, dry_run: bool = False) -> VideoAnalysis:
        """Execute single-video tracer bullet pipeline."""
        video_id = extract_video_id(url_or_id)
        logger.info(f"Processing video ID: {video_id}")

        # 1. Fetch metadata via YouTube Data API
        metadata = self.fetcher.fetch_metadata(video_id)

        # 2. Acquire transcript via fallback chain with configured language preferences
        transcript = get_video_transcript(video_id, languages=self.preferred_languages)

        # 3. Analyze with LLM or emit NO_TRANSCRIPT card
        analysis = self.analyzer.analyze(metadata, transcript)

        # 4. Record to ledger unless dry_run (B7)
        if not dry_run and getattr(self, "ledger", None) is not None:
            self.ledger.record_videos([analysis], status=analysis.status)

        return analysis

    def run_topic(
        self,
        query: str,
        limit: int = 10,
        top_n: Optional[int] = None,
        rank_only: bool = False,
        no_dedup: bool = False,
        force: bool = False,
        dry_run: bool = False,
    ) -> TopicSearchResult:
        """Execute topic search, batch fetch metadata, filter shorts, rank candidates, and run Map-Reduce radar."""
        safe_limit = max(1, int(limit)) if limit is not None else 10
        cfg_n = getattr(self, "config_top_n", 3)
        effective_top_n = min(safe_limit, max(1, int(top_n if top_n is not None else cfg_n)))
        logger.info(
            f"Running topic search for: {query} (limit={safe_limit}, top_n={effective_top_n}, rank_only={rank_only}, no_dedup={no_dedup}, force={force}, dry_run={dry_run})"
        )

        search_cfg = self.config.get("search") if isinstance(self.config, dict) else {}
        if not isinstance(search_cfg, dict):
            search_cfg = {}
        filter_cfg = self.config.get("filter") if isinstance(self.config, dict) else {}
        if not isinstance(filter_cfg, dict):
            filter_cfg = {}

        order = str(search_cfg.get("order") or "relevance")
        raw_max = search_cfg.get("max_results")
        max_results = int(raw_max) if raw_max is not None else 50

        raw_min = filter_cfg.get("min_duration_seconds")
        min_duration = int(raw_min) if raw_min is not None else 120

        # 1. Quota-guarded search (single search.list call, max 50 items)
        video_ids = self.fetcher.search_videos(query=query, max_results=max_results, order=order)

        # 2. Batch fetch metadata in single videos.list call
        raw_videos = self.fetcher.fetch_metadata_batch(video_ids)
        total_found = len(raw_videos)

        # 3. Filter short videos / noise
        filtered_videos = filter_by_duration(raw_videos, min_duration_seconds=min_duration)
        filtered_count = len(filtered_videos)

        # 4. Rank candidates using composite signal formula
        ranked_candidates = rank_candidates(filtered_videos, config=self.config)

        # 5. History Deduplication Filtering (B4, B5, B10)
        cooldown_filtered_count = 0
        exhaustion_notice = False
        dedup = getattr(self, "deduplicator", None)
        ledger = getattr(self, "ledger", None)

        if not no_dedup and not force and dedup is not None:
            fresh_cands, cooldown_cands = dedup.filter_candidates(ranked_candidates)
            cooldown_filtered_count = len(cooldown_cands)
            if dedup.has_exhaustion(fresh_cands, ranked_candidates):
                # Topic exhaustion value floor (B5)
                active_candidates = ranked_candidates
                exhaustion_notice = True
            else:
                active_candidates = fresh_cands
                exhaustion_notice = False
        else:
            active_candidates = ranked_candidates
            cooldown_filtered_count = 0
            exhaustion_notice = False

        # 6. Stage 1 (Map) and Stage 2 (Reduce)
        video_analyses: List[VideoAnalysis] = []
        radar_synthesis: Optional[TopicRadarSynthesis] = None

        if not rank_only and not exhaustion_notice and active_candidates:
            map_candidates = active_candidates[:effective_top_n]
            for cand in map_candidates:
                meta = cand.metadata
                logger.info(f"Stage 1 (Map) analyzing #{cand.rank} candidate: {meta.title} ({meta.video_id})")
                try:
                    transcript = get_video_transcript(meta.video_id, languages=self.preferred_languages)
                    analysis = self.analyzer.analyze(meta, transcript)
                except MissingApiKeyError as exc:
                    logger.warning(f"OpenAI API key missing; preserving radar results with metadata fallback: {exc}")
                    analysis = VideoAnalysis(
                        metadata=meta,
                        status="NO_TRANSCRIPT",
                        overview="未設定 OPENAI_API_KEY，僅展示候選雷達與元數據資訊。",
                        key_points=[],
                        takeaways=[],
                    )
                except Exception as exc:
                    logger.warning(f"Stage 1 candidate analysis failed for {meta.video_id}: {exc}; degrading to NO_TRANSCRIPT.")
                    analysis = VideoAnalysis(
                        metadata=meta,
                        status="NO_TRANSCRIPT",
                        overview="影片逐字稿分析異常，已為您保留基本元數據與描述欄章節導覽。",
                        key_points=[],
                        takeaways=[],
                    )
                video_analyses.append(analysis)

            if video_analyses:
                logger.info(f"Stage 2 (Reduce) synthesizing topic radar across {len(video_analyses)} videos.")
                radar_synthesis = self.analyzer.synthesize_topic(query=query, analyses=video_analyses)

        displayed_candidates = active_candidates[:safe_limit]

        # 7. Persist analyzed videos to ledger unless dry_run (B6)
        if not dry_run and ledger is not None and video_analyses:
            ledger.record_videos(video_analyses, status="ANALYZED", query=query)

        return TopicSearchResult(
            query=query,
            total_candidates_found=total_found,
            filtered_candidates_count=filtered_count,
            candidates=displayed_candidates,
            radar_synthesis=radar_synthesis,
            video_analyses=video_analyses,
            top_analysis=video_analyses[0] if video_analyses else None,
            exhaustion_notice=exhaustion_notice,
            cooldown_filtered_count=cooldown_filtered_count,
        )



