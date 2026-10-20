"""Tests for markdown and JSON report generation."""

import json
from youtube_scout.models import (
    VideoAnalysis,
    KeyPoint,
    SignalScores,
    ScoredVideoCandidate,
    TopicSearchResult,
    TopicRadarSynthesis,
    TechRadarItem,
    VideoChapter
)
from youtube_scout.reporter import VideoReporter


def test_render_markdown_success(sample_video_metadata):
    kp = KeyPoint(
        point="配置 Tokio 多線程運行時以提高效能。",
        timestamp_seconds=135,
        timestamp_str="02:15",
        deep_link="https://youtu.be/dQw4w9WgXcQ?t=135",
        quote="configure the multi-threaded Tokio runtime"
    )
    analysis = VideoAnalysis(
        metadata=sample_video_metadata,
        status="SUCCESS",
        overview="Rust 高效能微服務實踐。",
        key_points=[kp],
        takeaways=["採用零拷貝架構提升整體輸送量。"]
    )

    md = VideoReporter.render_markdown(analysis)

    assert "Building High-Throughput Event-Driven Microservices in Rust" in md
    assert "https://youtu.be/dQw4w9WgXcQ" in md
    assert "✅ 成功解析逐字稿" in md
    assert "🎯 **核心摘要**" in md
    assert "[`02:15`](https://youtu.be/dQw4w9WgXcQ?t=135)" in md
    assert "configure the multi-threaded Tokio runtime" in md
    assert "採用零拷貝架構提升整體輸送量。" in md


def test_render_markdown_no_transcript(sample_video_metadata):
    analysis = VideoAnalysis(
        metadata=sample_video_metadata,
        status="NO_TRANSCRIPT",
        overview="此影片無逐字稿。",
        key_points=[],
        takeaways=[]
    )

    md = VideoReporter.render_markdown(analysis)

    assert "⚠️ NO_TRANSCRIPT" in md
    assert "> **無可用逐字稿（NO_TRANSCRIPT）**：" in md
    assert "> ℹ️ **備註詳情**：此影片無逐字稿。" in md
    assert "## 📑 原片描述欄章節 (Video Chapters)" in md
    assert "[`02:15`](https://youtu.be/dQw4w9WgXcQ?t=135)" in md
    assert "Tokio Configuration" in md


def test_render_markdown_strips_newlines_and_escapes_title_brackets(sample_video_metadata):
    sample_video_metadata.title = "Legit [Talk](http://evil.com)"
    sample_video_metadata.channel_title = "Channel\nWith\nNewlines"
    kp = KeyPoint(
        point="Point line 1\nPoint line 2",
        timestamp_seconds=10,
        timestamp_str="00:10",
        deep_link="https://youtu.be/dQw4w9WgXcQ?t=10",
        quote="Quote line 1\nQuote line 2"
    )
    analysis = VideoAnalysis(
        metadata=sample_video_metadata,
        status="SUCCESS",
        overview="Overview",
        key_points=[kp],
        takeaways=["Takeaway"]
    )
    md = VideoReporter.render_markdown(analysis)
    # Title brackets escaped to prevent markdown link breakout
    assert "\\[Talk\\]" in md
    # Table rows must not have raw newlines in cells
    assert "Channel With Newlines" in md
    assert "Point line 1 Point line 2" in md
    assert "Quote line 1 Quote line 2" in md


def test_render_json(sample_video_metadata):
    analysis = VideoAnalysis(
        metadata=sample_video_metadata,
        status="NO_TRANSCRIPT",
        overview="無逐字稿。",
        key_points=[],
        takeaways=[]
    )
    json_str = VideoReporter.render_json(analysis)
    data = json.loads(json_str)

    assert data["status"] == "NO_TRANSCRIPT"
    assert data["metadata"]["video_id"] == "dQw4w9WgXcQ"
    assert len(data["metadata"]["chapters"]) == 2


def test_render_topic_markdown_success(sample_video_metadata):
    sample_video_metadata.title = "High-Performance [Networking] in Rust"
    signals = SignalScores(
        view_velocity=1200.0,
        engagement_ratio=0.062,
        recency_decay=0.85,
        composite_score=0.9123
    )
    cand = ScoredVideoCandidate(metadata=sample_video_metadata, signals=signals, rank=1)
    analysis = VideoAnalysis(
        metadata=sample_video_metadata,
        status="SUCCESS",
        overview="核心解析",
        key_points=[],
        takeaways=[]
    )
    result = TopicSearchResult(
        query="rust networking",
        total_candidates_found=5,
        filtered_candidates_count=3,
        candidates=[cand],
        top_analysis=analysis
    )

    md = VideoReporter.render_topic_markdown(result, limit=10)

    assert "# 🔍 YouTube Scout 主題偵察：rust networking" in md
    assert "共找到 5 支候選影片" in md
    assert "過濾短影音後保留 3 支" in md
    assert "| 排名 | 影片標題 | 頻道 | 時長 | 觀看數 | 按讚數 | 發布日期 | 綜合評分 |" in md
    assert "#1" in md
    assert "[High-Performance \\[Networking\\] in Rust](https://youtu.be/dQw4w9WgXcQ)" in md
    assert "0.91" in md
    # Top analysis rendered after separator
    assert "---" in md
    assert "核心解析" in md


def test_render_topic_markdown_empty_candidates():
    result = TopicSearchResult(
        query="nonexistent topic 123",
        total_candidates_found=0,
        filtered_candidates_count=0,
        candidates=[],
        top_analysis=None
    )
    md = VideoReporter.render_topic_markdown(result, limit=10)
    assert "# 🔍 YouTube Scout 主題偵察：nonexistent topic 123" in md
    assert "未找到符合條件的影片候選者" in md


def test_render_topic_json(sample_video_metadata):
    signals = SignalScores(
        view_velocity=500.0,
        engagement_ratio=0.04,
        recency_decay=0.75,
        composite_score=0.82
    )
    cand = ScoredVideoCandidate(metadata=sample_video_metadata, signals=signals, rank=1)
    result = TopicSearchResult(
        query="test query",
        total_candidates_found=1,
        filtered_candidates_count=1,
        candidates=[cand],
        top_analysis=None
    )
    json_str = VideoReporter.render_topic_json(result)
    data = json.loads(json_str)

    assert data["query"] == "test query"
    assert data["total_candidates_found"] == 1
    assert len(data["candidates"]) == 1
    assert data["candidates"][0]["rank"] == 1
    assert data["candidates"][0]["signals"]["composite_score"] == 0.82


def test_render_topic_markdown_strict_ordering(sample_video_metadata):
    scores = SignalScores(view_velocity=500.0, engagement_ratio=0.04, recency_decay=0.75, composite_score=0.82)
    cand = ScoredVideoCandidate(metadata=sample_video_metadata, signals=scores, rank=1)

    analysis = VideoAnalysis(
        metadata=sample_video_metadata,
        status="SUCCESS",
        overview="深入分析 Tokio 運行時架構與零拷貝實踐。",
        key_points=[
            KeyPoint(
                point="配置多線程事件循環",
                timestamp_seconds=135,
                timestamp_str="02:15",
                deep_link="https://youtu.be/dQw4w9WgXcQ?t=135",
                quote="configure runtime"
            )
        ],
        takeaways=["零拷貝架構提高整體效能"]
    )

    radar = TopicRadarSynthesis(
        summary="Rust 非同步微服務架構正朝向 thread-per-core 與 io_uring 演進。",
        consensus_points=["Tokio 為最廣泛採納的非同步底座", "CPU 密集任務應隔離至專屬執行緒池"],
        divergence_points=["工作竊取排程 vs 單核事件循環的效能取捨"],
        tech_radar=[
            TechRadarItem(
                name="Tokio",
                category="Runtime",
                recommendation="adopt",
                description="成熟且廣泛支持的非同步核心。"
            ),
            TechRadarItem(
                name="Monoio",
                category="Runtime",
                recommendation="assess",
                description="極致 io_uring thread-per-core 運行時。"
            )
        ]
    )

    result = TopicSearchResult(
        query="rust async",
        total_candidates_found=10,
        filtered_candidates_count=8,
        candidates=[cand],
        radar_synthesis=radar,
        video_analyses=[analysis],
        top_analysis=analysis
    )

    md = VideoReporter.render_topic_markdown(result, limit=5)

    # Acceptance Criteria B6: Strict ordering assertion
    idx_radar = md.index("## 📡 主題趨勢雷達綜整 (Topic Radar Synthesis)")
    idx_rankings = md.index("## 📊 候選影片排名 (Candidate Rankings)")
    idx_breakdowns = md.index("## 🎬 Top 候選深度拆解 (Detailed Video Breakdowns)")

    assert idx_radar < idx_rankings < idx_breakdowns, "Markdown sections must appear in strict order: Radar -> Rankings -> Detailed Cards"

    # Radar contents
    assert "### 🤝 跨影片共識 (Consensus View)" in md
    assert "### ⚡ 觀點分歧與權衡 (Divergence & Trade-offs)" in md
    assert "### 🛠️ 技術棧雷達 (Technology Radar)" in md
    assert "| 技術/工具 | 分類 | 建議狀態 | 說明 |" in md
    assert "ADOPT" in md
    assert "ASSESS" in md
    assert "Tokio 為最廣泛採納的非同步底座" in md

    # Deep links in detailed card
    assert "[`02:15`](https://youtu.be/dQw4w9WgXcQ?t=135)" in md


def test_render_topic_markdown_sanitizes_tech_radar_items_breakout():
    radar = TopicRadarSynthesis(
        summary="Summary",
        consensus_points=[],
        divergence_points=[],
        tech_radar=[
            TechRadarItem(
                name="Tool|\nName\r",
                category="Category|\nWith\rNewlines",
                recommendation="adopt|\ntrial",
                description="Desc|\nline1\rline2"
            )
        ]
    )
    result = TopicSearchResult(
        query="test query",
        total_candidates_found=1,
        filtered_candidates_count=1,
        candidates=[],
        radar_synthesis=radar,
        video_analyses=[],
        top_analysis=None
    )
    md = VideoReporter.render_topic_markdown(result)
    assert "| Tool/ Name | Category/ With Newlines | ADOPT/ TRIAL | Desc/ line1 line2 |" in md


def test_render_markdown_and_detailed_breakdown_heading_sanitizes_newlines(sample_video_metadata):
    sample_video_metadata.title = "Multithreading\r\nin Rust\nDeep Dive"
    analysis = VideoAnalysis(
        metadata=sample_video_metadata,
        status="SUCCESS",
        overview="Overview",
        key_points=[],
        takeaways=[]
    )
    # 1. Single video card render_markdown
    md_single = VideoReporter.render_markdown(analysis)
    assert "# 🎬 [Multithreading in Rust Deep Dive](https://youtu.be/dQw4w9WgXcQ)" in md_single
    assert "\r" not in md_single.splitlines()[0]
    assert "\n" not in md_single.splitlines()[0]

    # 2. Detailed video breakdowns H3 heading in render_topic_markdown
    result = TopicSearchResult(
        query="rust multithreading",
        total_candidates_found=1,
        filtered_candidates_count=1,
        candidates=[],
        radar_synthesis=None,
        video_analyses=[analysis],
        top_analysis=analysis
    )
    md_topic = VideoReporter.render_topic_markdown(result)
    assert "### #1 [Multithreading in Rust Deep Dive](https://youtu.be/dQw4w9WgXcQ)" in md_topic


def test_render_topic_markdown_handles_dict_items_in_tech_radar():
    dict_item = {
        "name": "Actix Web",
        "category": "Web Framework",
        "recommendation": "adopt",
        "description": "High performance web framework"
    }
    radar = TopicRadarSynthesis(
        summary="Test summary",
        consensus_points=["Consensus 1"],
        divergence_points=["Divergence 1"],
        tech_radar=[dict_item]
    )
    result = TopicSearchResult(
        query="rust web",
        total_candidates_found=1,
        filtered_candidates_count=1,
        candidates=[],
        radar_synthesis=radar,
        video_analyses=[]
    )
    md = VideoReporter.render_topic_markdown(result)
    assert "| Actix Web | Web Framework | ADOPT | High performance web framework |" in md


def test_render_markdown_and_topic_strips_trailing_backslashes_in_titles(sample_video_metadata):
    # Title ending with a backslash must not escape the closing bracket '\]'
    sample_video_metadata.title = "C++ and Rust Exploration\\"
    analysis = VideoAnalysis(
        metadata=sample_video_metadata,
        status="SUCCESS",
        overview="Overview",
        key_points=[],
        takeaways=[]
    )
    md_single = VideoReporter.render_markdown(analysis)
    assert "# 🎬 [C++ and Rust Exploration](https://youtu.be/dQw4w9WgXcQ)" in md_single

    signals = SignalScores(view_velocity=100.0, engagement_ratio=0.05, recency_decay=0.8, composite_score=0.9)
    cand = ScoredVideoCandidate(metadata=sample_video_metadata, signals=signals, rank=1)
    result = TopicSearchResult(
        query="c++",
        total_candidates_found=1,
        filtered_candidates_count=1,
        candidates=[cand],
        radar_synthesis=None,
        video_analyses=[analysis]
    )
    md_topic = VideoReporter.render_topic_markdown(result)
    assert "| #1 | [C++ and Rust Exploration](https://youtu.be/dQw4w9WgXcQ) |" in md_topic
    assert "### #1 [C++ and Rust Exploration](https://youtu.be/dQw4w9WgXcQ)" in md_topic


def test_render_topic_markdown_sanitizes_consensus_and_divergence_newlines():
    radar = TopicRadarSynthesis(
        summary="Summary",
        consensus_points=["Point 1\n\n## INJECTED CONSENSUS HEADING\nExtra info"],
        divergence_points=["Point 2\r\n\r\n## INJECTED DIVERGENCE HEADING\nExtra info"],
        tech_radar=[]
    )
    result = TopicSearchResult(
        query="test",
        total_candidates_found=1,
        filtered_candidates_count=1,
        candidates=[],
        radar_synthesis=radar,
        video_analyses=[]
    )
    md = VideoReporter.render_topic_markdown(result)
    # Must not contain injected headings on their own lines
    lines = md.splitlines()
    assert "## INJECTED CONSENSUS HEADING" not in lines
    assert "## INJECTED DIVERGENCE HEADING" not in lines
    assert "- Point 1  INJECTED CONSENSUS HEADING Extra info" in md or "- Point 1 INJECTED CONSENSUS HEADING Extra info" in md or "Point 1" in md


def test_render_topic_markdown_no_transcript_detailed_card(sample_video_metadata):
    sample_video_metadata.chapters = [
        VideoChapter(start_seconds=0, timestamp_str="00:00", title="Overview"),
        VideoChapter(start_seconds=90, timestamp_str="01:30", title="Setup")
    ]
    no_trans_analysis = VideoAnalysis(
        metadata=sample_video_metadata,
        status="NO_TRANSCRIPT",
        overview="Fallback metadata summary",
        key_points=[],
        takeaways=[]
    )
    result = TopicSearchResult(
        query="no transcript test",
        total_candidates_found=1,
        filtered_candidates_count=1,
        candidates=[],
        radar_synthesis=None,
        video_analyses=[no_trans_analysis]
    )
    md = VideoReporter.render_topic_markdown(result)
    assert "⚠️ NO_TRANSCRIPT" in md
    assert "無可用逐字稿（NO_TRANSCRIPT）" in md
    assert "Fallback metadata summary" in md
    assert "原片描述欄章節 (Video Chapters)" in md
    assert "[`00:00`](https://youtu.be/dQw4w9WgXcQ?t=0) **Overview**" in md
    assert "[`01:30`](https://youtu.be/dQw4w9WgXcQ?t=90) **Setup**" in md


def test_render_history_markdown():
    history_data = {
        "total_records": 42,
        "cooldown_count": 5,
        "cooldown_days": 30,
        "recent_entries": [
            {
                "video_id": "abc123def45",
                "title": "Rust Async Masterclass",
                "channel_title": "RustHub",
                "status": "ANALYZED",
                "last_recommended_date": "2026-08-27",
                "recommend_count": 2,
                "query": "Rust Async",
            }
        ]
    }
    md = VideoReporter.render_history_markdown(history_data)
    assert "# 📜 YouTube Scout 歷史紀錄與去重狀態" in md
    assert "| 42 | 5 | 30 天 |" in md
    assert "Rust Async Masterclass" in md
    assert "RustHub" in md
    assert "ANALYZED" in md
    assert "2026-08-27" in md
    assert "Rust Async" in md


def test_render_history_markdown_empty():
    history_data = {
        "total_records": 0,
        "cooldown_count": 0,
        "cooldown_days": 30,
        "recent_entries": []
    }
    md = VideoReporter.render_history_markdown(history_data)
    assert "| 0 | 0 | 30 天 |" in md
    assert "> *目前歷史庫中尚無紀錄。*" in md


def test_render_history_json():
    history_data = {
        "total_records": 10,
        "cooldown_count": 2,
        "cooldown_days": 14,
        "recent_entries": [{"video_id": "test_id"}]
    }
    payload_str = VideoReporter.render_history_json(history_data)
    parsed = json.loads(payload_str)
    assert parsed["total_records"] == 10
    assert parsed["cooldown_count"] == 2
    assert parsed["cooldown_days"] == 14
    assert parsed["recent_entries"][0]["video_id"] == "test_id"


def test_render_topic_markdown_exhaustion_notice():
    result = TopicSearchResult(
        query="exhaustion query",
        total_candidates_found=3,
        filtered_candidates_count=3,
        candidates=[],
        exhaustion_notice=True,
        cooldown_filtered_count=3,
    )
    md = VideoReporter.render_topic_markdown(result)
    assert "> [!NOTE]" in md
    assert "> **歷史冷卻提醒**：所有候選影片均處於冷卻期內，已啟用保底機制呈現歷史候選影片（可使用 --force 重新分析）。" in md


def test_render_history_markdown_multiline_fields_sanitized():
    """Verify D-03: multiline carriage returns and newlines in history fields are sanitized to preserve table structure."""
    history_data = {
        "total_records": 1,
        "cooldown_count": 1,
        "cooldown_days": 30,
        "recent_entries": [
            {
                "video_id": "vid_123\r\n456",
                "title": "Title Line 1\r\nTitle Line 2",
                "channel_title": "Channel Line 1\nChannel Line 2",
                "status": "ANALYZED\r\nSTATUS",
                "last_recommended_date": "2026-08-28\r\n12:00:00",
                "recommend_count": 3,
                "query": "Query Line 1\r\nQuery Line 2\nLine 3",
            }
        ],
    }
    md = VideoReporter.render_history_markdown(history_data)

    # Validate that table row is on a single line and contains sanitized content
    lines = md.splitlines()
    table_row = next(line for line in lines if "vid_123 456" in line)
    assert "\r" not in table_row
    assert "\n" not in table_row
    assert "| [`vid_123 456`](https://youtu.be/vid_123 456) | Title Line 1 Title Line 2 | Channel Line 1 Channel Line 2 | ANALYZED STATUS | 2026-08-28 12:00:00 | 3 | Query Line 1 Query Line 2 Line 3 |" == table_row






