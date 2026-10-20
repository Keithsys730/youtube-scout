"""Tests for CLI argument parsing, clean error handling, and exit codes."""

import os
import json
from unittest.mock import patch, MagicMock
from youtube_scout.cli import main
from youtube_scout.models import (
    VideoAnalysis,
    KeyPoint,
    SignalScores,
    ScoredVideoCandidate,
    TopicSearchResult
)


def test_cli_missing_youtube_api_key(monkeypatch, capsys):
    monkeypatch.delenv("YOUTUBE_API_KEY", raising=False)
    exit_code = main(["--url", "dQw4w9WgXcQ"])
    
    assert exit_code == 1
    captured = capsys.readouterr()
    assert "Error: Missing YOUTUBE_API_KEY environment variable" in captured.err
    # Must have no Python traceback
    assert "Traceback" not in captured.err


def test_cli_invalid_url(monkeypatch, capsys):
    monkeypatch.setenv("YOUTUBE_API_KEY", "dummy_key")
    exit_code = main(["--url", "not_valid"])

    assert exit_code == 1
    captured = capsys.readouterr()
    assert "Error: Could not extract a valid 11-character YouTube video ID" in captured.err
    assert "Traceback" not in captured.err


def test_cli_no_transcript_exits_zero(monkeypatch, sample_video_metadata, capsys):
    monkeypatch.setenv("YOUTUBE_API_KEY", "dummy_key")
    no_ts_analysis = VideoAnalysis(
        metadata=sample_video_metadata,
        status="NO_TRANSCRIPT",
        overview="此影片無逐字稿。",
        key_points=[],
        takeaways=[]
    )

    with patch("youtube_scout.runner.ScoutRunner.run_video", return_value=no_ts_analysis):
        exit_code = main(["--url", "dQw4w9WgXcQ"])
        assert exit_code == 0

        captured = capsys.readouterr()
        assert "NO_TRANSCRIPT" in captured.out
        assert sample_video_metadata.title in captured.out


def test_cli_success_with_timestamp_links(monkeypatch, sample_video_metadata, capsys):
    monkeypatch.setenv("YOUTUBE_API_KEY", "dummy_key")
    kp = KeyPoint(
        point="Tokio 多線程運行時調優。",
        timestamp_seconds=135,
        timestamp_str="02:15",
        deep_link="https://youtu.be/dQw4w9WgXcQ?t=135",
        quote="configure the multi-threaded Tokio runtime"
    )
    success_analysis = VideoAnalysis(
        metadata=sample_video_metadata,
        status="SUCCESS",
        overview="Rust 高效能架構。",
        key_points=[kp],
        takeaways=["零拷貝架構重要性。"]
    )

    with patch("youtube_scout.runner.ScoutRunner.run_video", return_value=success_analysis):
        exit_code = main(["--url", "https://www.youtube.com/watch?v=dQw4w9WgXcQ"])
        assert exit_code == 0

        captured = capsys.readouterr()
        assert "[`02:15`](https://youtu.be/dQw4w9WgXcQ?t=135)" in captured.out
        assert "Tokio 多線程運行時調優。" in captured.out


def test_cli_json_flag(monkeypatch, sample_video_metadata, capsys):
    monkeypatch.setenv("YOUTUBE_API_KEY", "dummy_key")
    analysis = VideoAnalysis(
        metadata=sample_video_metadata,
        status="NO_TRANSCRIPT",
        overview="無逐字稿。",
        key_points=[],
        takeaways=[]
    )

    with patch("youtube_scout.runner.ScoutRunner.run_video", return_value=analysis):
        exit_code = main(["--url", "dQw4w9WgXcQ", "--json"])
        assert exit_code == 0

        captured = capsys.readouterr()
        data = json.loads(captured.out)
        assert data["status"] == "NO_TRANSCRIPT"
        assert data["metadata"]["video_id"] == "dQw4w9WgXcQ"


def test_cli_output_to_file(monkeypatch, sample_video_metadata, tmp_path):
    monkeypatch.setenv("YOUTUBE_API_KEY", "dummy_key")
    analysis = VideoAnalysis(
        metadata=sample_video_metadata,
        status="NO_TRANSCRIPT",
        overview="無逐字稿。",
        key_points=[],
        takeaways=[]
    )

    out_file = tmp_path / "nested" / "subdir" / "report.md"
    with patch("youtube_scout.runner.ScoutRunner.run_video", return_value=analysis):
        exit_code = main(["--url", "dQw4w9WgXcQ", "--output", str(out_file)])
        assert exit_code == 0
        assert out_file.exists()
        content = out_file.read_text(encoding="utf-8")
        assert "NO_TRANSCRIPT" in content


def test_cli_missing_openai_api_key(monkeypatch, capsys):
    monkeypatch.setenv("YOUTUBE_API_KEY", "dummy_youtube_key")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    from youtube_scout.fetcher import MissingApiKeyError
    with patch("youtube_scout.runner.ScoutRunner.run_video", side_effect=MissingApiKeyError("Missing OPENAI_API_KEY environment variable.")):
        exit_code = main(["--url", "dQw4w9WgXcQ"])
        assert exit_code == 1
        captured = capsys.readouterr()
        assert "Error: Missing OPENAI_API_KEY environment variable." in captured.err
        assert "Traceback" not in captured.err


def test_cli_neither_url_nor_query(monkeypatch, capsys):
    monkeypatch.setenv("YOUTUBE_API_KEY", "dummy_key")
    exit_code = main([])
    assert exit_code == 1
    captured = capsys.readouterr()
    assert "Error: Please specify either --url/-u <video_url> or --query/-q <search_query>." in captured.err


def test_cli_both_url_and_query(monkeypatch, capsys):
    monkeypatch.setenv("YOUTUBE_API_KEY", "dummy_key")
    exit_code = main(["--url", "dQw4w9WgXcQ", "--query", "rust microservices"])
    assert exit_code == 1
    captured = capsys.readouterr()
    assert "Error: Cannot specify both --url and --query at the same time." in captured.err


def test_cli_query_success_markdown(monkeypatch, sample_video_metadata, capsys):
    monkeypatch.setenv("YOUTUBE_API_KEY", "dummy_key")
    signals = SignalScores(view_velocity=100.0, engagement_ratio=0.05, recency_decay=0.8, composite_score=0.9)
    candidate = ScoredVideoCandidate(metadata=sample_video_metadata, signals=signals, rank=1)
    result = TopicSearchResult(
        query="rust microservices",
        total_candidates_found=1,
        filtered_candidates_count=1,
        candidates=[candidate],
        top_analysis=None
    )

    with patch("youtube_scout.runner.ScoutRunner.run_topic", return_value=result) as mock_run_topic:
        exit_code = main(["--query", "rust microservices", "--limit", "5"])
        assert exit_code == 0
        mock_run_topic.assert_called_once_with(query="rust microservices", limit=5, top_n=None, rank_only=False)

        captured = capsys.readouterr()
        assert "# 🔍 YouTube Scout 主題偵察：rust microservices" in captured.out
        assert sample_video_metadata.title in captured.out


def test_cli_query_json(monkeypatch, sample_video_metadata, capsys):
    monkeypatch.setenv("YOUTUBE_API_KEY", "dummy_key")
    signals = SignalScores(view_velocity=100.0, engagement_ratio=0.05, recency_decay=0.8, composite_score=0.9)
    candidate = ScoredVideoCandidate(metadata=sample_video_metadata, signals=signals, rank=1)
    result = TopicSearchResult(
        query="rust microservices",
        total_candidates_found=1,
        filtered_candidates_count=1,
        candidates=[candidate],
        top_analysis=None
    )

    with patch("youtube_scout.runner.ScoutRunner.run_topic", return_value=result):
        exit_code = main(["-q", "rust microservices", "--json"])
        assert exit_code == 0

        captured = capsys.readouterr()
        data = json.loads(captured.out)
        assert data["query"] == "rust microservices"
        assert len(data["candidates"]) == 1


def test_cli_query_rank_only(monkeypatch, sample_video_metadata):
    monkeypatch.setenv("YOUTUBE_API_KEY", "dummy_key")
    result = TopicSearchResult(
        query="rust",
        total_candidates_found=0,
        filtered_candidates_count=0,
        candidates=[],
        top_analysis=None
    )

    with patch("youtube_scout.runner.ScoutRunner.run_topic", return_value=result) as mock_run_topic:
        exit_code = main(["-q", "rust", "--rank-only"])
        assert exit_code == 0
        mock_run_topic.assert_called_once_with(query="rust", limit=10, top_n=None, rank_only=True)



def test_cli_query_output_to_file(monkeypatch, sample_video_metadata, tmp_path):
    monkeypatch.setenv("YOUTUBE_API_KEY", "dummy_key")
    signals = SignalScores(view_velocity=100.0, engagement_ratio=0.05, recency_decay=0.8, composite_score=0.9)
    candidate = ScoredVideoCandidate(metadata=sample_video_metadata, signals=signals, rank=1)
    result = TopicSearchResult(
        query="rust",
        total_candidates_found=1,
        filtered_candidates_count=1,
        candidates=[candidate],
        top_analysis=None
    )

    out_file = tmp_path / "topic_report.md"
    with patch("youtube_scout.runner.ScoutRunner.run_topic", return_value=result):
        exit_code = main(["-q", "rust", "-o", str(out_file)])
        assert exit_code == 0
        assert out_file.exists()
        content = out_file.read_text(encoding="utf-8")
        assert "YouTube Scout 主題偵察" in content


def test_cli_url_with_rank_only_fails(monkeypatch, capsys):
    monkeypatch.setenv("YOUTUBE_API_KEY", "dummy_key")
    exit_code = main(["--url", "dQw4w9WgXcQ", "--rank-only"])
    assert exit_code == 1
    captured = capsys.readouterr()
    assert "Error: --rank-only can only be used with --query/-q.\n" == captured.err
    assert "Traceback" not in captured.err


def test_cli_whitespace_only_query_fails(monkeypatch, capsys):
    monkeypatch.setenv("YOUTUBE_API_KEY", "dummy_key")
    exit_code = main(["-q", "   "])
    assert exit_code == 1
    captured = capsys.readouterr()
    assert "Error: Search query cannot be empty or whitespace only." in captured.err


def test_cli_non_positive_limit_fails(monkeypatch, capsys):
    monkeypatch.setenv("YOUTUBE_API_KEY", "dummy_key")
    exit_code = main(["-q", "rust", "--limit", "0"])
    assert exit_code == 1
    captured = capsys.readouterr()
    assert "Error: --limit must be a positive integer." in captured.err


def test_cli_api_key_redaction_in_general_exception(monkeypatch, capsys):
    secret_key = "mock-live-secret-openai-key-12345"
    monkeypatch.setenv("OPENAI_API_KEY", secret_key)
    monkeypatch.setenv("YOUTUBE_API_KEY", "dummy_yt_key")

    with patch("youtube_scout.cli.ScoutRunner.run_video", side_effect=RuntimeError(f"Fatal crash leaking {secret_key}")):
        exit_code = main(["--url", "dQw4w9WgXcQ"])
        assert exit_code == 1
        captured = capsys.readouterr()
        assert secret_key not in captured.err
        assert "[REDACTED]" in captured.err


def test_cli_top_n_passed_to_runner(monkeypatch):
    monkeypatch.setenv("YOUTUBE_API_KEY", "dummy_key")
    result = TopicSearchResult(
        query="rust",
        total_candidates_found=0,
        filtered_candidates_count=0,
        candidates=[],
        top_analysis=None
    )

    with patch("youtube_scout.runner.ScoutRunner.run_topic", return_value=result) as mock_run_topic:
        exit_code = main(["-q", "rust", "--limit", "5", "--top-n", "2"])
        assert exit_code == 0
        mock_run_topic.assert_called_once_with(query="rust", limit=5, top_n=2, rank_only=False)


def test_cli_top_n_non_positive_fails(monkeypatch, capsys):
    monkeypatch.setenv("YOUTUBE_API_KEY", "dummy_key")
    exit_code = main(["-q", "rust", "--top-n", "0"])
    assert exit_code == 1
    captured = capsys.readouterr()
    assert "Error: --top-n must be a positive integer.\n" == captured.err

    exit_code2 = main(["-q", "rust", "--top-n", "-2"])
    assert exit_code2 == 1
    captured2 = capsys.readouterr()
    assert "Error: --top-n must be a positive integer.\n" == captured2.err


def test_cli_top_n_exceeds_limit_fails(monkeypatch, capsys):
    monkeypatch.setenv("YOUTUBE_API_KEY", "dummy_key")
    exit_code = main(["-q", "rust", "--limit", "5", "--top-n", "6"])
    assert exit_code == 1
    captured = capsys.readouterr()
    assert "Error: --top-n cannot exceed --limit.\n" == captured.err


def test_cli_url_with_top_n_fails(monkeypatch, capsys):
    monkeypatch.setenv("YOUTUBE_API_KEY", "dummy_key")
    exit_code = main(["--url", "dQw4w9WgXcQ", "--top-n", "3"])
    assert exit_code == 1
    captured = capsys.readouterr()
    assert "Error: --top-n can only be used with --query/-q.\n" == captured.err


def test_cli_inspect_history_markdown(capsys):
    fake_history = {
        "total_records": 5,
        "cooldown_count": 2,
        "cooldown_days": 30,
        "recent_entries": [
            {
                "video_id": "history123",
                "title": "History Video",
                "channel_title": "Hist Channel",
                "status": "ANALYZED",
                "last_recommended_date": "2026-08-27",
                "recommend_count": 1,
                "query": "Rust",
            }
        ]
    }
    with patch("youtube_scout.deduplicator.SQLiteLedger.inspect_history", return_value=fake_history):
        # Exits 0 and doesn't require --url or --query (B12)
        exit_code = main(["--inspect-history"])
        assert exit_code == 0
        captured = capsys.readouterr()
        assert "# 📜 YouTube Scout 歷史紀錄與去重狀態" in captured.out
        assert "History Video" in captured.out
        assert "history123" in captured.out


def test_cli_inspect_history_json(capsys):
    fake_history = {
        "total_records": 1,
        "cooldown_count": 1,
        "cooldown_days": 14,
        "recent_entries": []
    }
    with patch("youtube_scout.deduplicator.SQLiteLedger.inspect_history", return_value=fake_history):
        exit_code = main(["--inspect-history", "--json"])
        assert exit_code == 0
        captured = capsys.readouterr()
        data = json.loads(captured.out)
        assert data["total_records"] == 1
        assert data["cooldown_days"] == 14


def test_cli_inspect_history_output_to_file(tmp_path):
    out_file = str(tmp_path / "hist.md")
    fake_history = {"total_records": 0, "cooldown_count": 0, "cooldown_days": 30, "recent_entries": []}
    with patch("youtube_scout.deduplicator.SQLiteLedger.inspect_history", return_value=fake_history):
        exit_code = main(["--inspect-history", "-o", out_file])
        assert exit_code == 0
        assert os.path.exists(out_file)
        with open(out_file, "r", encoding="utf-8") as f:
            content = f.read()
        assert "歷史紀錄與去重狀態" in content


def test_cli_cooldown_days_negative_fails(capsys):
    exit_code = main(["--cooldown-days", "-1", "-q", "rust"])
    assert exit_code == 1
    captured = capsys.readouterr()
    assert "Error: --cooldown-days must be a non-negative integer.\n" == captured.err


def test_cli_db_path_and_cooldown_days_passed_to_runner(monkeypatch):
    monkeypatch.setenv("YOUTUBE_API_KEY", "dummy_key")
    result = TopicSearchResult(query="rust", total_candidates_found=0, filtered_candidates_count=0, candidates=[])

    with patch("youtube_scout.cli.ScoutRunner") as mock_runner_cls:
        mock_instance = MagicMock()
        mock_instance.run_topic.return_value = result
        mock_runner_cls.return_value = mock_instance

        exit_code = main(["-q", "rust", "--db-path", "data/custom.db", "--cooldown-days", "45"])
        assert exit_code == 0
        mock_runner_cls.assert_called_once_with(
            config_path="config.yaml",
            db_path="data/custom.db",
            cooldown_days=45,
        )


def test_cli_no_dedup_force_and_dry_run_flags_passed(monkeypatch):
    monkeypatch.setenv("YOUTUBE_API_KEY", "dummy_key")
    result = TopicSearchResult(query="rust", total_candidates_found=0, filtered_candidates_count=0, candidates=[])

    with patch("youtube_scout.runner.ScoutRunner.run_topic", return_value=result) as mock_run_topic:
        exit_code = main(["-q", "rust", "--no-dedup", "--force", "--dry-run"])
        assert exit_code == 0
        mock_run_topic.assert_called_once_with(
            query="rust",
            limit=10,
            top_n=None,
            rank_only=False,
            no_dedup=True,
            force=True,
            dry_run=True,
        )


def test_cli_video_dry_run_flag(monkeypatch, sample_video_metadata):
    monkeypatch.setenv("YOUTUBE_API_KEY", "dummy_key")
    analysis = VideoAnalysis(metadata=sample_video_metadata, status="SUCCESS", overview="test", key_points=[], takeaways=[])

    with patch("youtube_scout.runner.ScoutRunner.run_video", return_value=analysis) as mock_run_video:
        exit_code = main(["--url", "dQw4w9WgXcQ", "--dry-run"])
        assert exit_code == 0
        mock_run_video.assert_called_once_with("dQw4w9WgXcQ", dry_run=True)





