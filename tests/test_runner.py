"""Tests for ScoutRunner end-to-end orchestration."""

import os
from unittest.mock import patch, MagicMock
from youtube_scout.runner import ScoutRunner
from youtube_scout.models import VideoMetadata, VideoAnalysis, TranscriptSegment
from youtube_scout.analyzer import MissingApiKeyError


def test_scout_runner_success(sample_video_metadata, sample_transcript_segments):
    mock_llm_payload = {
        "overview": "Rust 微服務架構分析",
        "key_points": [
            {
                "timestamp_seconds": 135,
                "point": "Tokio runtime 優化",
                "quote": "configure Tokio"
            }
        ],
        "takeaways": ["零拷貝很重要"]
    }

    def dummy_llm(prompt, meta, transcript):
        return mock_llm_payload

    with patch("youtube_scout.runner.YouTubeMetadataFetcher.fetch_metadata", return_value=sample_video_metadata):
        with patch("youtube_scout.runner.get_video_transcript", return_value=sample_transcript_segments):
            runner = ScoutRunner(api_key="test_key", custom_llm_handler=dummy_llm)
            analysis = runner.run_video("dQw4w9WgXcQ")

            assert analysis.status == "SUCCESS"
            assert len(analysis.key_points) == 1
            assert analysis.key_points[0].deep_link == "https://youtu.be/dQw4w9WgXcQ?t=135"


def test_scout_runner_no_transcript(sample_video_metadata):
    with patch("youtube_scout.runner.YouTubeMetadataFetcher.fetch_metadata", return_value=sample_video_metadata):
        with patch("youtube_scout.runner.get_video_transcript", return_value=None):
            runner = ScoutRunner(api_key="test_key")
            analysis = runner.run_video("https://youtu.be/dQw4w9WgXcQ")

            assert analysis.status == "NO_TRANSCRIPT"
            assert len(analysis.key_points) == 0
            assert "無可用逐字稿" in analysis.overview or "字幕已停用" in analysis.overview


def test_scout_runner_default_config_fallback(tmp_path):
    runner = ScoutRunner(api_key="test_key", config_path=str(tmp_path / "nonexistent.yaml"))
    assert runner.config["llm"]["model"] == "gpt-4o-mini"
    assert runner.config["search"]["order"] == "relevance"
    assert runner.config["filter"]["min_duration_seconds"] == 120
    assert runner.config["ranking"]["weights"]["velocity"] == 0.5


def test_scout_runner_run_topic_success(sample_video_metadata, sample_transcript_segments):
    # Create 3 videos: v1 (long, high views), v2 (short 60s), v3 (long, moderate views)
    v1 = VideoMetadata(
        video_id="vid_top",
        title="Top Video",
        channel_title="Channel A",
        published_at="2026-08-20T00:00:00Z",
        duration_seconds=600,
        duration_str="10:00",
        view_count=50000,
        like_count=3000,
        description="Top video description",
        chapters=[]
    )
    v2_short = VideoMetadata(
        video_id="vid_short",
        title="Short Video",
        channel_title="Channel B",
        published_at="2026-08-21T00:00:00Z",
        duration_seconds=45,
        duration_str="00:45",
        view_count=100000,
        like_count=5000,
        description="Short description",
        chapters=[]
    )
    v3 = VideoMetadata(
        video_id="vid_second",
        title="Second Video",
        channel_title="Channel C",
        published_at="2026-08-10T00:00:00Z",
        duration_seconds=900,
        duration_str="15:00",
        view_count=10000,
        like_count=500,
        description="Second description",
        chapters=[]
    )

    mock_llm_payload = {
        "overview": "Top video summary",
        "key_points": [],
        "takeaways": []
    }

    with patch("youtube_scout.runner.YouTubeMetadataFetcher.search_videos", return_value=["vid_top", "vid_short", "vid_second"]) as mock_search:
        with patch("youtube_scout.runner.YouTubeMetadataFetcher.fetch_metadata_batch", return_value=[v1, v2_short, v3]) as mock_batch:
            with patch("youtube_scout.runner.get_video_transcript", return_value=sample_transcript_segments) as mock_transcript:
                runner = ScoutRunner(
                    api_key="test_key",
                    custom_llm_handler=lambda p, m, t: mock_llm_payload
                )
                result = runner.run_topic(query="rust architecture", limit=5, rank_only=False)

                # Quota boundary: exactly 1 search call and 1 batch metadata call
                mock_search.assert_called_once_with(query="rust architecture", max_results=50, order="relevance")
                mock_batch.assert_called_once_with(["vid_top", "vid_short", "vid_second"])

                assert result.total_candidates_found == 3
                # v2_short (45s) filtered out by min_duration_seconds=120
                assert result.filtered_candidates_count == 2
                assert len(result.candidates) == 2
                assert result.candidates[0].metadata.video_id == "vid_top"
                assert result.candidates[0].rank == 1

                # Stage 1 Map ran on top N candidates (here 2 candidates)
                assert mock_transcript.call_count == 2
                assert len(result.video_analyses) == 2
                assert result.top_analysis is not None
                assert result.top_analysis.status == "SUCCESS"
                assert result.top_analysis.metadata.video_id == "vid_top"
                # Stage 2 Reduce synthesized radar
                assert result.radar_synthesis is not None



def test_scout_runner_run_topic_rank_only(tmp_path, sample_video_metadata):
    db_path = str(tmp_path / "rank_only.db")
    with patch("youtube_scout.runner.YouTubeMetadataFetcher.search_videos", return_value=["vid_top"]):
        with patch("youtube_scout.runner.YouTubeMetadataFetcher.fetch_metadata_batch", return_value=[sample_video_metadata]):
            with patch("youtube_scout.runner.get_video_transcript") as mock_transcript:
                runner = ScoutRunner(api_key="test_key", db_path=db_path)
                result = runner.run_topic(query="rust", limit=5, rank_only=True)

                # Transcript and LLM are NOT called when rank_only=True
                mock_transcript.assert_not_called()
                assert len(result.candidates) == 1
                assert result.top_analysis is None
                # D-02: rank_only mode does NOT record to the ledger
                assert runner.ledger.inspect_history()["total_records"] == 0


def test_scout_runner_run_topic_empty():
    with patch("youtube_scout.runner.YouTubeMetadataFetcher.search_videos", return_value=[]):
        with patch("youtube_scout.runner.YouTubeMetadataFetcher.fetch_metadata_batch", return_value=[]):
            runner = ScoutRunner(api_key="test_key")
            result = runner.run_topic(query="empty query", limit=10)

            assert result.total_candidates_found == 0
            assert result.filtered_candidates_count == 0
            assert result.candidates == []
            assert result.top_analysis is None


def test_scout_runner_run_topic_limit_slices_candidates():
    videos = [
        VideoMetadata(
            video_id=f"vid_{i}",
            title=f"Video {i}",
            channel_title="Channel",
            published_at="2026-08-20T00:00:00Z",
            duration_seconds=300,
            duration_str="05:00",
            view_count=1000 * (10 - i),
            like_count=50 * (10 - i),
            description="",
            chapters=[]
        )
        for i in range(4)
    ]
    with patch("youtube_scout.runner.YouTubeMetadataFetcher.search_videos", return_value=[f"vid_{i}" for i in range(4)]):
        with patch("youtube_scout.runner.YouTubeMetadataFetcher.fetch_metadata_batch", return_value=videos):
            runner = ScoutRunner(api_key="test_key")
            result = runner.run_topic(query="test query", limit=2, rank_only=True)

            assert result.total_candidates_found == 4
            assert result.filtered_candidates_count == 4
            assert len(result.candidates) == 2
            assert result.candidates[0].rank == 1
            assert result.candidates[1].rank == 2
            # Verify JSON dictionary representation honors limit as well
            json_dict = result.to_dict()
            assert len(json_dict["candidates"]) == 2


def test_scout_runner_run_topic_openai_key_missing_graceful_fallback():
    raw_video = VideoMetadata(
        video_id="safe_id",
        title="Title",
        channel_title="Channel",
        published_at="2026-08-20T00:00:00Z",
        duration_seconds=300,
        duration_str="05:00",
        view_count=1000,
        like_count=50,
        description="",
        chapters=[]
    )
    with patch("youtube_scout.runner.YouTubeMetadataFetcher.search_videos", return_value=["safe_id"]):
        with patch("youtube_scout.runner.YouTubeMetadataFetcher.fetch_metadata_batch", return_value=[raw_video]):
            with patch("youtube_scout.runner.get_video_transcript", return_value=[{"start": 0.0, "duration": 5.0, "text": "test"}]):
                with patch("youtube_scout.runner.VideoAnalyzer.analyze", side_effect=MissingApiKeyError("Missing OPENAI_API_KEY")):
                    runner = ScoutRunner(api_key="test_key")
                    result = runner.run_topic(query="test", limit=5, rank_only=False)

                    # Does not crash, candidate table intact, top_analysis degraded to NO_TRANSCRIPT
                    assert len(result.candidates) == 1
                    assert result.top_analysis is not None
                    assert result.top_analysis.status == "NO_TRANSCRIPT"
                    assert "OPENAI_API_KEY" in result.top_analysis.overview


def test_scout_runner_run_topic_null_config_sections():
    runner = ScoutRunner.__new__(ScoutRunner)
    runner.config = {"search": None, "filter": None, "ranking": None}
    runner.fetcher = MagicMock()
    runner.fetcher.search_videos.return_value = []
    runner.fetcher.fetch_metadata_batch.return_value = []

    result = runner.run_topic("test", limit=5)
    assert result.candidates == []
    assert result.total_candidates_found == 0


def test_scout_runner_run_topic_top_n_slices_stage1():
    videos = [
        VideoMetadata(
            video_id=f"vid_{i}",
            title=f"Video {i}",
            channel_title="Channel",
            published_at="2026-08-20T00:00:00Z",
            duration_seconds=300,
            duration_str="05:00",
            view_count=1000 * (10 - i),
            like_count=50 * (10 - i),
            description="",
            chapters=[]
        )
        for i in range(4)
    ]
    with patch("youtube_scout.runner.YouTubeMetadataFetcher.search_videos", return_value=[f"vid_{i}" for i in range(4)]):
        with patch("youtube_scout.runner.YouTubeMetadataFetcher.fetch_metadata_batch", return_value=videos):
            with patch("youtube_scout.runner.get_video_transcript", return_value=[TranscriptSegment(text="t", start=0.0, duration=5.0)]) as mock_transcript:
                runner = ScoutRunner(
                    api_key="test_key",
                    custom_llm_handler=lambda p, m, t: {"overview": "test", "key_points": [], "takeaways": []}
                )
                # limit is 4, but top_n is 1
                result = runner.run_topic(query="test", limit=4, top_n=1, rank_only=False)

                # Exactly 1 transcript call and 1 analysis in Stage 1
                assert mock_transcript.call_count == 1
                assert len(result.video_analyses) == 1
                assert len(result.candidates) == 4
                assert result.radar_synthesis is not None


def test_scout_runner_run_topic_mixed_transcripts():
    v1 = VideoMetadata(
        video_id="v1_with_ts",
        title="Video with TS",
        channel_title="Channel 1",
        published_at="2026-08-20T00:00:00Z",
        duration_seconds=300,
        duration_str="05:00",
        view_count=2000,
        like_count=100,
        description="Desc 1",
        chapters=[]
    )
    v2 = VideoMetadata(
        video_id="v2_no_ts",
        title="Video without TS",
        channel_title="Channel 2",
        published_at="2026-08-19T00:00:00Z",
        duration_seconds=300,
        duration_str="05:00",
        view_count=1000,
        like_count=50,
        description="Desc 2",
        chapters=[]
    )

    def mock_transcript_func(vid, languages):
        if vid == "v1_with_ts":
            return [TranscriptSegment(text="Hello world", start=0.0, duration=5.0)]
        return None

    with patch("youtube_scout.runner.YouTubeMetadataFetcher.search_videos", return_value=["v1_with_ts", "v2_no_ts"]):
        with patch("youtube_scout.runner.YouTubeMetadataFetcher.fetch_metadata_batch", return_value=[v1, v2]):
            with patch("youtube_scout.runner.get_video_transcript", side_effect=mock_transcript_func):
                runner = ScoutRunner(
                    api_key="test_key",
                    custom_llm_handler=lambda p, m, t: {"overview": "test", "key_points": [], "takeaways": []}
                )
                result = runner.run_topic(query="test", limit=5, top_n=2, rank_only=False)

                assert len(result.video_analyses) == 2
                # v1 had transcript -> SUCCESS
                assert result.video_analyses[0].status == "SUCCESS"
                assert result.video_analyses[0].metadata.video_id == "v1_with_ts"
                # v2 had no transcript -> NO_TRANSCRIPT preserved in rank order
                assert result.video_analyses[1].status == "NO_TRANSCRIPT"
                assert result.video_analyses[1].metadata.video_id == "v2_no_ts"
                # Stage 2 radar generated
                assert result.radar_synthesis is not None


def test_scout_runner_init_config_top_n_defensive_handling(tmp_path):
    import yaml

    # Case 1: negative top_n clamped to 1
    cfg1 = tmp_path / "cfg1.yaml"
    cfg1.write_text(yaml.dump({"top_n": -5}), encoding="utf-8")
    runner1 = ScoutRunner(api_key="test", config_path=str(cfg1))
    assert runner1.config_top_n == 1

    # Case 2: non-numeric string top_n defaults to 3
    cfg2 = tmp_path / "cfg2.yaml"
    cfg2.write_text(yaml.dump({"top_n": "invalid_num"}), encoding="utf-8")
    runner2 = ScoutRunner(api_key="test", config_path=str(cfg2))
    assert runner2.config_top_n == 3

    # Case 3: positive top_n respected
    cfg3 = tmp_path / "cfg3.yaml"
    cfg3.write_text(yaml.dump({"top_n": 7}), encoding="utf-8")
    runner3 = ScoutRunner(api_key="test", config_path=str(cfg3))
    assert runner3.config_top_n == 7


def test_scout_runner_run_topic_effective_top_n_edge_cases(sample_video_metadata):
    vids = [
        VideoMetadata(
            video_id=f"vid_{i}",
            title=f"Video {i}",
            channel_title="Channel",
            published_at="2026-08-20T00:00:00Z",
            duration_seconds=300,
            duration_str="05:00",
            view_count=1000 * (10 - i),
            like_count=50,
            description="Desc",
            chapters=[]
        )
        for i in range(5)
    ]

    with patch("youtube_scout.runner.YouTubeMetadataFetcher.search_videos", return_value=[v.video_id for v in vids]):
        with patch("youtube_scout.runner.YouTubeMetadataFetcher.fetch_metadata_batch", return_value=vids):
            with patch("youtube_scout.runner.get_video_transcript", return_value=None):
                runner = ScoutRunner(
                    api_key="test_key",
                    custom_llm_handler=lambda p, *args: {
                        "summary": "Summary",
                        "overview": "Overview",
                        "key_points": [],
                        "takeaways": [],
                        "consensus_points": [],
                        "divergence_points": [],
                        "tech_radar": []
                    }
                )
                runner.config_top_n = 4

                # top_n=None should use config_top_n (4) bounded by limit (2)
                res_none = runner.run_topic(query="test", limit=2, top_n=None, rank_only=False, no_dedup=True)
                assert len(res_none.video_analyses) == 2

                # top_n exceeding limit (10 > 3) should clamp to limit (3)
                res_clamped = runner.run_topic(query="test", limit=3, top_n=10, rank_only=False, no_dedup=True)
                assert len(res_clamped.video_analyses) == 3

                # top_n non-positive should clamp to 1
                res_neg = runner.run_topic(query="test", limit=5, top_n=-2, rank_only=False, no_dedup=True)
                assert len(res_neg.video_analyses) == 1


def test_scout_runner_run_topic_single_candidate_unexpected_exception_preserves_pipeline():
    v1 = VideoMetadata("v1", "Title 1", "Channel", "2026-08-20T00:00:00Z", 300, "05:00", 2000, 100, "", [])
    v2 = VideoMetadata("v2", "Title 2", "Channel", "2026-08-20T00:00:00Z", 300, "05:00", 1000, 50, "", [])

    with patch("youtube_scout.runner.YouTubeMetadataFetcher.search_videos", return_value=["v1", "v2"]):
        with patch("youtube_scout.runner.YouTubeMetadataFetcher.fetch_metadata_batch", return_value=[v1, v2]):
            # v1 transcript raises unexpected exception (e.g. malformed segment or parsing crash)
            def mock_get_transcript(video_id, languages):
                if video_id == "v1":
                    raise RuntimeError("Unexpected internal crash parsing video transcript")
                return [TranscriptSegment("Valid segment", 0.0, 5.0)]

            def dummy_handler(prompt, *args):
                vid = args[0].video_id if args and hasattr(args[0], "video_id") else "topic"
                return {
                    "summary": "Topic summary",
                    "overview": f"Success for {vid}",
                    "key_points": [],
                    "takeaways": [],
                    "consensus_points": [],
                    "divergence_points": [],
                    "tech_radar": []
                }

            with patch("youtube_scout.runner.get_video_transcript", side_effect=mock_get_transcript):
                runner = ScoutRunner(
                    api_key="test_key",
                    custom_llm_handler=dummy_handler
                )
                result = runner.run_topic(query="test", limit=5, top_n=2, rank_only=False)

                # Does not crash!
                assert len(result.candidates) == 2
                assert len(result.video_analyses) == 2
                # v1 degraded to NO_TRANSCRIPT
                assert result.video_analyses[0].status == "NO_TRANSCRIPT"
                assert "影片逐字稿分析異常" in result.video_analyses[0].overview
                # v2 succeeded normally
                assert result.video_analyses[1].status == "SUCCESS"
                assert result.video_analyses[1].overview == "Success for v2"


def test_scout_runner_deep_merge_recursive():
    defaults = {
        "top_n": 3,
        "ranking": {
            "weights": {
                "velocity": 0.5,
                "engagement": 0.3,
                "recency": 0.2
            },
            "recency_half_life_days": 30.0
        }
    }
    target = {
        "ranking": {
            "weights": {
                "velocity": 0.8
            }
        }
    }
    merged = ScoutRunner._deep_merge(defaults, target)
    assert merged["top_n"] == 3
    assert merged["ranking"]["weights"]["velocity"] == 0.8
    assert merged["ranking"]["weights"]["engagement"] == 0.3
    assert merged["ranking"]["weights"]["recency"] == 0.2
    assert merged["ranking"]["recency_half_life_days"] == 30.0


def test_runner_initialization_with_custom_db_and_cooldown(tmp_path):
    custom_db = str(tmp_path / "custom_history.db")
    runner = ScoutRunner(
        api_key="test_key",
        db_path=custom_db,
        cooldown_days=14
    )
    assert runner.history_file == custom_db
    assert runner.cooldown_days == 14
    assert runner.ledger.db_path == custom_db
    assert runner.deduplicator.cooldown_days == 14


def test_runner_run_topic_cooldown_filtering_and_exhaustion(tmp_path, sample_video_metadata):
    custom_db = str(tmp_path / "test_filter.db")
    v1 = VideoMetadata(
        video_id="vid_in_cd",
        title="Video In Cooldown",
        channel_title="Channel 1",
        published_at="2026-08-01T10:00:00Z",
        duration_seconds=300,
        duration_str="05:00",
        view_count=100000,
        like_count=5000,
        description="Cool video",
    )
    v2 = VideoMetadata(
        video_id="vid_fresh",
        title="Fresh Video",
        channel_title="Channel 2",
        published_at="2026-08-10T10:00:00Z",
        duration_seconds=400,
        duration_str="06:40",
        view_count=50000,
        like_count=2500,
        description="Fresh video",
    )

    with patch("youtube_scout.runner.YouTubeMetadataFetcher.search_videos", return_value=["vid_in_cd", "vid_fresh"]):
        with patch("youtube_scout.runner.YouTubeMetadataFetcher.fetch_metadata_batch", return_value=[v1, v2]):
            with patch("youtube_scout.runner.get_video_transcript", return_value=[]):
                runner = ScoutRunner(api_key="test_key", db_path=custom_db, cooldown_days=30)

                # Pre-record v1 into ledger so it is in cooldown
                runner.ledger.record_videos([v1], status="ANALYZED")

                # 1. Normal run: v1 filtered by cooldown, only v2 is analyzed
                result = runner.run_topic(query="test query", rank_only=True)
                assert result.exhaustion_notice is False
                assert result.cooldown_filtered_count == 1
                assert len(result.candidates) == 1
                assert result.candidates[0].metadata.video_id == "vid_fresh"

                # 2. Force flag: bypasses cooldown
                force_result = runner.run_topic(query="test query", rank_only=True, force=True)
                assert force_result.cooldown_filtered_count == 0
                assert len(force_result.candidates) == 2

                # 3. No-dedup flag: bypasses cooldown
                nodedup_result = runner.run_topic(query="test query", rank_only=True, no_dedup=True)
                assert nodedup_result.cooldown_filtered_count == 0
                assert len(nodedup_result.candidates) == 2

                # 4. Now record v2 as well -> All candidates in cooldown -> Exhaustion fallback (B5)
                runner.ledger.record_videos([v2], status="ANALYZED")
                exhausted_result = runner.run_topic(query="test query", rank_only=True)
                assert exhausted_result.exhaustion_notice is True
                assert exhausted_result.cooldown_filtered_count == 2
                assert len(exhausted_result.candidates) == 2


def test_runner_persistence_and_dry_run(tmp_path, sample_video_metadata):
    custom_db = str(tmp_path / "test_persist.db")
    runner = ScoutRunner(
        api_key="test_key",
        db_path=custom_db,
        custom_llm_handler=lambda p, *args: {
            "summary": "Summary",
            "overview": "Overview",
            "key_points": [],
            "takeaways": [],
            "consensus_points": [],
            "divergence_points": [],
            "tech_radar": []
        }
    )

    # 1. run_video with dry_run=True: should NOT persist
    with patch("youtube_scout.runner.YouTubeMetadataFetcher.fetch_metadata", return_value=sample_video_metadata):
        with patch("youtube_scout.runner.get_video_transcript", return_value=None):
            runner.run_video("dQw4w9WgXcQ", dry_run=True)
            assert runner.ledger.inspect_history()["total_records"] == 0

            # 2. run_video with dry_run=False: should persist
            runner.run_video("dQw4w9WgXcQ", dry_run=False)
            assert runner.ledger.inspect_history()["total_records"] == 1

    # 3. run_topic with rank_only=True: should NOT add new records even when dry_run=False (D-02)
    v_new = VideoMetadata(
        video_id="vid_brand_new",
        title="Brand New Video",
        channel_title="New Channel",
        published_at="2026-08-20T10:00:00Z",
        duration_seconds=300,
        duration_str="05:00",
        view_count=1000,
        like_count=100,
        description="New",
    )
    with patch("youtube_scout.runner.YouTubeMetadataFetcher.search_videos", return_value=["vid_brand_new"]):
        with patch("youtube_scout.runner.YouTubeMetadataFetcher.fetch_metadata_batch", return_value=[v_new]):
            # rank_only=True and dry_run=True: should NOT add new records
            runner.run_topic(query="dry test", rank_only=True, dry_run=True)
            assert runner.ledger.inspect_history()["total_records"] == 1

            # rank_only=True and dry_run=False: should NOT record to ledger (D-02)
            runner.run_topic(query="dry test", rank_only=True, dry_run=False)
            assert runner.ledger.inspect_history()["total_records"] == 1

            # 4. run_topic with full analysis and dry_run=True: should NOT record
            with patch("youtube_scout.runner.get_video_transcript", return_value=[TranscriptSegment("segment", 0.0, 5.0)]):
                runner.run_topic(query="full topic dry", rank_only=False, dry_run=True)
                assert runner.ledger.inspect_history()["total_records"] == 1

                # 5. run_topic with full analysis and dry_run=False: SHOULD persist
                runner.run_topic(query="full topic live", rank_only=False, dry_run=False)
                assert runner.ledger.inspect_history()["total_records"] == 2
                recent = runner.ledger.inspect_history()["recent_entries"]
                assert any(e["video_id"] == "vid_brand_new" for e in recent)


def test_runner_run_topic_exhaustion_skips_llm_and_does_not_bump_ledger(tmp_path, sample_video_metadata):
    """Verify D-01 & Regression: exhaustion skips Stage 1 & 2 without burning LLM or recording, while --force re-enables them."""
    custom_db = str(tmp_path / "exhaustion_skip.db")
    v1 = VideoMetadata(
        video_id="vid_exhausted_1",
        title="Exhausted Video 1",
        channel_title="Channel 1",
        published_at="2026-08-01T10:00:00Z",
        duration_seconds=300,
        duration_str="05:00",
        view_count=100000,
        like_count=5000,
        description="Desc 1",
    )
    v2 = VideoMetadata(
        video_id="vid_exhausted_2",
        title="Exhausted Video 2",
        channel_title="Channel 2",
        published_at="2026-08-05T10:00:00Z",
        duration_seconds=400,
        duration_str="06:40",
        view_count=50000,
        like_count=2500,
        description="Desc 2",
    )

    with patch("youtube_scout.runner.YouTubeMetadataFetcher.search_videos", return_value=["vid_exhausted_1", "vid_exhausted_2"]):
        with patch("youtube_scout.runner.YouTubeMetadataFetcher.fetch_metadata_batch", return_value=[v1, v2]):
            with patch("youtube_scout.runner.get_video_transcript", return_value=[TranscriptSegment(text="segment", start=0.0, duration=5.0)]) as mock_transcript:
                runner = ScoutRunner(
                    api_key="test_key",
                    db_path=custom_db,
                    cooldown_days=30,
                    custom_llm_handler=lambda prompt, *args: {
                        "summary": "Summary",
                        "overview": "Overview",
                        "key_points": [],
                        "takeaways": [],
                        "consensus_points": [],
                        "divergence_points": [],
                        "tech_radar": []
                    }
                )

                # Pre-record both videos so they are in cooldown
                runner.ledger.record_videos([v1, v2], status="ANALYZED", query="init query")
                history_before = runner.ledger.inspect_history()
                assert history_before["total_records"] == 2
                assert history_before["cooldown_count"] == 2
                initial_entry_v1 = next(e for e in history_before["recent_entries"] if e["video_id"] == "vid_exhausted_1")
                initial_rec_date = initial_entry_v1["last_recommended_date"]
                initial_count = initial_entry_v1["recommend_count"]

                # 1. Unforced topic scan with rank_only=False and force=False
                res_exhausted = runner.run_topic(query="test exhaustion", rank_only=False, force=False)

                # Assertions:
                # - Stage 1 and Stage 2 are skipped
                assert res_exhausted.video_analyses == []
                assert res_exhausted.radar_synthesis is None
                assert res_exhausted.top_analysis is None
                # - Candidate rankings are displayed
                assert len(res_exhausted.candidates) == 2
                assert res_exhausted.exhaustion_notice is True
                # - LLM/transcript was NOT called
                mock_transcript.assert_not_called()
                # - Ledger records are NOT updated (no bump in count, no new records)
                history_after = runner.ledger.inspect_history()
                assert history_after["total_records"] == 2
                entry_v1_after = next(e for e in history_after["recent_entries"] if e["video_id"] == "vid_exhausted_1")
                assert entry_v1_after["recommend_count"] == initial_count
                assert entry_v1_after["last_recommended_date"] == initial_rec_date

                # 2. When --force is provided: Stage 1 and Stage 2 execute and analyze candidates
                res_forced = runner.run_topic(query="test exhaustion", rank_only=False, force=True)
                assert res_forced.exhaustion_notice is False
                assert len(res_forced.video_analyses) == 2
                assert res_forced.radar_synthesis is not None
                assert res_forced.top_analysis is not None
                assert mock_transcript.call_count == 2

                # And ledger is updated when forced
                history_forced = runner.ledger.inspect_history()
                entry_v1_forced = next(e for e in history_forced["recent_entries"] if e["video_id"] == "vid_exhausted_1")
                assert entry_v1_forced["recommend_count"] == initial_count + 1



def test_runner_graceful_degradation_on_broken_ledger(tmp_path, sample_video_metadata):
    """Verify runner completes successfully without crashing even if SQLite ledger fails (B13, B18)."""
    corrupt_db = str(tmp_path / "corrupted_runner.db")
    with open(corrupt_db, "wb") as f:
        f.write(b"CORRUPT")

    runner = ScoutRunner(api_key="test_key", db_path=corrupt_db)

    with patch("youtube_scout.runner.YouTubeMetadataFetcher.fetch_metadata", return_value=sample_video_metadata):
        with patch("youtube_scout.runner.get_video_transcript", return_value=None):
            # run_video must not crash
            analysis = runner.run_video("dQw4w9WgXcQ")
            assert analysis.status == "NO_TRANSCRIPT"

    with patch("youtube_scout.runner.YouTubeMetadataFetcher.search_videos", return_value=["dQw4w9WgXcQ"]):
        with patch("youtube_scout.runner.YouTubeMetadataFetcher.fetch_metadata_batch", return_value=[sample_video_metadata]):
            # run_topic must not crash
            result = runner.run_topic(query="corrupt test", rank_only=True)
            assert len(result.candidates) == 1







