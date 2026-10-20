"""Markdown and JSON report generators for YouTube Scout with sanitized output and deep links."""

import json
import logging
from typing import Optional, Dict, Any, List
from .models import VideoAnalysis, UntrustedContent, TopicSearchResult

logger = logging.getLogger("youtube_scout.reporter")


class VideoReporter:
    """Renders structured, human-readable Traditional Chinese reports and JSON payloads."""

    @classmethod
    def render_markdown(cls, analysis: VideoAnalysis) -> str:
        """Render markdown card for CLI terminal display and report files."""
        meta = analysis.metadata
        raw_title = UntrustedContent.sanitize_text(meta.title).rstrip("\\")
        clean_title = (
            raw_title
            .replace("[", "\\[")
            .replace("]", "\\]")
            .replace("\r\n", " ")
            .replace("\r", " ")
            .replace("\n", " ")
            .strip()
        )
        clean_channel = (
            UntrustedContent.sanitize_text(meta.channel_title)
            .replace("|", "/")
            .replace("\r\n", " ")
            .replace("\r", " ")
            .replace("\n", " ")
            .strip()
        )
        published_date = meta.published_at.split("T")[0] if meta.published_at else "Unknown"

        lines = []
        lines.append(f"# 🎬 [{clean_title}]({meta.video_url})")
        lines.append("")

        # Metadata overview table
        lines.append("| 頻道 | 發布日期 | 影片時長 | 觀看次數 | 按讚數 | 狀態 |")
        lines.append("|---|---|---|---|---|---|")
        status_tag = "✅ 成功解析逐字稿" if analysis.status == "SUCCESS" else "⚠️ NO_TRANSCRIPT"
        lines.append(
            f"| {clean_channel} | {published_date} | {meta.duration_str} | "
            f"{meta.view_count:,} | {meta.like_count:,} | {status_tag} |"
        )
        lines.append("")

        # VALUE FLOOR: Handling NO_TRANSCRIPT status
        if analysis.status == "NO_TRANSCRIPT":
            lines.append("> [!WARNING]")
            lines.append("> **無可用逐字稿（NO_TRANSCRIPT）**：")
            lines.append("> 此影片未提供字幕，或字幕功能已停用。已自動啟動價值底線機制，呈現基本元數據與描述欄章節導覽。")
            lines.append("")

            if analysis.overview:
                lines.append(f"> ℹ️ **備註詳情**：{analysis.overview}")
                lines.append("")

            if meta.chapters:
                lines.append("## 📑 原片描述欄章節 (Video Chapters)")
                lines.append("")
                for ch in meta.chapters:
                    chapter_url = f"https://youtu.be/{meta.video_id}?t={ch.start_seconds}"
                    clean_ch_title = UntrustedContent.sanitize_text(ch.title)
                    lines.append(f"- [`{ch.timestamp_str}`]({chapter_url}) **{clean_ch_title}**")
                lines.append("")
            else:
                lines.append("> *註記：此影片描述欄亦未包含結構化時間戳記章節。*")
                lines.append("")

            return "\n".join(lines).strip()

        # SUCCESS: High-signal timestamped summary
        if analysis.overview:
            lines.append(f"> 🎯 **核心摘要**：{analysis.overview}")
            lines.append("")

        if analysis.key_points:
            lines.append("## ⏱️ 時間戳記重點脈絡 (Timestamped Key Points)")
            lines.append("")
            lines.append("| 時間戳記 | 核心重點 | 原音佐證 (Short Quote) |")
            lines.append("|---|---|---|")
            for kp in analysis.key_points:
                link_md = f"[`{kp.timestamp_str}`]({kp.deep_link})"
                point_clean = (
                    UntrustedContent.sanitize_text(kp.point)
                    .replace("|", "/")
                    .replace("\r", " ")
                    .replace("\n", " ")
                    .strip()
                )
                if kp.quote:
                    quote_text = (
                        UntrustedContent.sanitize_text(kp.quote)
                        .replace("|", "/")
                        .replace("\r", " ")
                        .replace("\n", " ")
                        .strip()
                    )
                    quote_clean = f"「{quote_text}」"
                else:
                    quote_clean = "-"
                lines.append(f"| {link_md} | {point_clean} | {quote_clean} |")
            lines.append("")

        if analysis.takeaways:
            lines.append("## 💡 關鍵啟示與行動洞察 (Key Takeaways)")
            lines.append("")
            for t in analysis.takeaways:
                clean_takeaway = (
                    UntrustedContent.sanitize_text(t)
                    .replace("\r\n", " ")
                    .replace("\r", " ")
                    .replace("\n", " ")
                    .strip()
                )
                lines.append(f"- {clean_takeaway}")
            lines.append("")

        if meta.chapters:
            lines.append("## 📑 官方描述欄章節索引 (Original Chapters)")
            lines.append("")
            for ch in meta.chapters:
                chapter_url = f"https://youtu.be/{meta.video_id}?t={ch.start_seconds}"
                clean_ch_title = (
                    UntrustedContent.sanitize_text(ch.title)
                    .replace("\r\n", " ")
                    .replace("\r", " ")
                    .replace("\n", " ")
                    .strip()
                )
                lines.append(f"- [`{ch.timestamp_str}`]({chapter_url}) {clean_ch_title}")
            lines.append("")

        return "\n".join(lines).strip()

    @classmethod
    def render_json(cls, analysis: VideoAnalysis) -> str:
        """Render machine-readable JSON representation."""
        return json.dumps(analysis.to_dict(), indent=2, ensure_ascii=False)

    @classmethod
    def render_topic_markdown(cls, result: TopicSearchResult, limit: Optional[int] = None) -> str:
        """Render markdown report for topic search radar with strict ordering (B6)."""
        clean_query = (
            UntrustedContent.sanitize_text(result.query)
            .replace("[", "\\[")
            .replace("]", "\\]")
            .replace("\r", " ")
            .replace("\n", " ")
            .strip()
        )
        lines = []

        # 1. Header
        lines.append(f"# 🔍 YouTube Scout 主題偵察：{clean_query}")
        lines.append("")

        if getattr(result, "exhaustion_notice", False):
            lines.append("> [!NOTE]")
            lines.append("> **歷史冷卻提醒**：所有候選影片均處於冷卻期內，已啟用保底機制呈現歷史候選影片（可使用 --force 重新分析）。")
            lines.append("")

        # 2. Stage 2 Topic Radar Synthesis (if available)
        if result.radar_synthesis:
            radar = result.radar_synthesis
            lines.append("## 📡 主題趨勢雷達綜整 (Topic Radar Synthesis)")
            lines.append("")
            if radar.summary:
                lines.append(f"> 🌐 **趨勢總覽**：{radar.summary}")
                lines.append("")

            lines.append("### 🤝 跨影片共識 (Consensus View)")
            lines.append("")
            if radar.consensus_points:
                for pt in radar.consensus_points:
                    clean_pt = (
                        UntrustedContent.sanitize_text(pt)
                        .replace("\r\n", " ")
                        .replace("\r", " ")
                        .replace("\n", " ")
                        .strip()
                    )
                    lines.append(f"- {clean_pt}")
            else:
                lines.append("> *無顯著跨影片共識。*")
            lines.append("")

            lines.append("### ⚡ 觀點分歧與權衡 (Divergence & Trade-offs)")
            lines.append("")
            if radar.divergence_points:
                for pt in radar.divergence_points:
                    clean_pt = (
                        UntrustedContent.sanitize_text(pt)
                        .replace("\r\n", " ")
                        .replace("\r", " ")
                        .replace("\n", " ")
                        .strip()
                    )
                    lines.append(f"- {clean_pt}")
            else:
                lines.append("> *未發現明顯觀點分歧。*")
            lines.append("")

            lines.append("### 🛠️ 技術棧雷達 (Technology Radar)")
            lines.append("")
            if radar.tech_radar:
                lines.append("| 技術/工具 | 分類 | 建議狀態 | 說明 |")
                lines.append("|---|---|---|---|")
                for item in radar.tech_radar:
                    raw_name = item.get("name", "") if isinstance(item, dict) else getattr(item, "name", "")
                    raw_cat = item.get("category", "") if isinstance(item, dict) else getattr(item, "category", "")
                    raw_rec = item.get("recommendation", "") if isinstance(item, dict) else getattr(item, "recommendation", "")
                    raw_desc = item.get("description", "") if isinstance(item, dict) else getattr(item, "description", "")

                    c_name = (
                        UntrustedContent.sanitize_text(raw_name)
                        .replace("|", "/")
                        .replace("\r\n", " ")
                        .replace("\r", " ")
                        .replace("\n", " ")
                        .strip()
                    )
                    c_cat = (
                        UntrustedContent.sanitize_text(raw_cat)
                        .replace("|", "/")
                        .replace("\r\n", " ")
                        .replace("\r", " ")
                        .replace("\n", " ")
                        .strip()
                    )
                    rec_upper = (
                        UntrustedContent.sanitize_text(raw_rec)
                        .upper()
                        .replace("|", "/")
                        .replace("\r\n", " ")
                        .replace("\r", " ")
                        .replace("\n", " ")
                        .strip()
                    )
                    c_desc = (
                        UntrustedContent.sanitize_text(raw_desc)
                        .replace("|", "/")
                        .replace("\r\n", " ")
                        .replace("\r", " ")
                        .replace("\n", " ")
                        .strip()
                    )
                    lines.append(f"| {c_name} | {c_cat} | {rec_upper} | {c_desc} |")
            else:
                lines.append("> *本次偵察未標註特定技術棧項目。*")
            lines.append("")

            lines.append("---")
            lines.append("")

        # 3. Candidate Rankings
        lines.append("## 📊 候選影片排名 (Candidate Rankings)")
        lines.append("")

        candidates_to_render = (
            result.candidates[:limit]
            if limit is not None and limit > 0
            else result.candidates
        )
        shown_count = len(candidates_to_render)
        lines.append(
            f"> 📊 **探索概覽**：共找到 {result.total_candidates_found} 支候選影片，"
            f"過濾短影音後保留 {result.filtered_candidates_count} 支"
            f"（展示前 {shown_count} 名候選影片）。"
        )
        lines.append("")

        if not candidates_to_render:
            lines.append("> *未找到符合條件的影片候選者。*")
            lines.append("")
        else:
            lines.append("| 排名 | 影片標題 | 頻道 | 時長 | 觀看數 | 按讚數 | 發布日期 | 綜合評分 |")
            lines.append("|---|---|---|---|---|---|---|---|")
            for candidate in candidates_to_render:
                meta = candidate.metadata
                raw_title = UntrustedContent.sanitize_text(meta.title).rstrip("\\")
                clean_title = (
                    raw_title
                    .replace("[", "\\[")
                    .replace("]", "\\]")
                    .replace("|", "/")
                    .replace("\r\n", " ")
                    .replace("\r", " ")
                    .replace("\n", " ")
                    .strip()
                )
                clean_channel = (
                    UntrustedContent.sanitize_text(meta.channel_title)
                    .replace("|", "/")
                    .replace("\r\n", " ")
                    .replace("\r", " ")
                    .replace("\n", " ")
                    .strip()
                )
                pub_date = (meta.published_at.split("T")[0] if meta.published_at else "Unknown").replace("|", "/")
                clean_dur = meta.duration_str.replace("|", "/")
                title_link = f"[{clean_title}]({meta.video_url})"
                score_str = f"{candidate.signals.composite_score:.2f}"

                lines.append(
                    f"| #{candidate.rank} | {title_link} | {clean_channel} | {clean_dur} | "
                    f"{meta.view_count:,} | {meta.like_count:,} | {pub_date} | {score_str} |"
                )
            lines.append("")

        # 4. Detailed Video Breakdowns (Stage 1 Map outputs)
        analyses_to_render = result.video_analyses
        if not analyses_to_render and result.top_analysis:
            analyses_to_render = [result.top_analysis]

        if analyses_to_render:
            lines.append("---")
            lines.append("")
            lines.append("## 🎬 Top 候選深度拆解 (Detailed Video Breakdowns)")
            lines.append("")

            for idx, analysis in enumerate(analyses_to_render):
                meta = analysis.metadata
                raw_b_title = UntrustedContent.sanitize_text(meta.title).rstrip("\\")
                clean_title = (
                    raw_b_title
                    .replace("[", "\\[")
                    .replace("]", "\\]")
                    .replace("\r\n", " ")
                    .replace("\r", " ")
                    .replace("\n", " ")
                    .strip()
                )
                clean_channel = (
                    UntrustedContent.sanitize_text(meta.channel_title)
                    .replace("|", "/")
                    .replace("\r\n", " ")
                    .replace("\r", " ")
                    .replace("\n", " ")
                    .strip()
                )
                published_date = meta.published_at.split("T")[0] if meta.published_at else "Unknown"

                lines.append(f"### #{idx + 1} [{clean_title}]({meta.video_url})")
                lines.append("")
                lines.append("| 頻道 | 發布日期 | 影片時長 | 觀看次數 | 按讚數 | 狀態 |")
                lines.append("|---|---|---|---|---|---|")
                status_tag = "✅ 成功解析逐字稿" if analysis.status == "SUCCESS" else "⚠️ NO_TRANSCRIPT"
                lines.append(
                    f"| {clean_channel} | {published_date} | {meta.duration_str} | "
                    f"{meta.view_count:,} | {meta.like_count:,} | {status_tag} |"
                )
                lines.append("")

                if analysis.status == "NO_TRANSCRIPT":
                    lines.append("> [!WARNING]")
                    lines.append("> **無可用逐字稿（NO_TRANSCRIPT）**：")
                    lines.append("> 此影片未提供字幕，或字幕功能已停用。已自動啟動價值底線機制，呈現基本元數據與描述欄章節導覽。")
                    lines.append("")
                    if analysis.overview:
                        lines.append(f"> ℹ️ **備註詳情**：{analysis.overview}")
                        lines.append("")
                    if meta.chapters:
                        lines.append("#### 📑 原片描述欄章節 (Video Chapters)")
                        lines.append("")
                        for ch in meta.chapters:
                            chapter_url = f"https://youtu.be/{meta.video_id}?t={ch.start_seconds}"
                            clean_ch_title = UntrustedContent.sanitize_text(ch.title)
                            lines.append(f"- [`{ch.timestamp_str}`]({chapter_url}) **{clean_ch_title}**")
                        lines.append("")
                else:
                    if analysis.overview:
                        lines.append(f"> 🎯 **核心摘要**：{analysis.overview}")
                        lines.append("")

                    if analysis.key_points:
                        lines.append("#### ⏱️ 時間戳記重點脈絡 (Timestamped Key Points)")
                        lines.append("")
                        lines.append("| 時間戳記 | 核心重點 | 原音佐證 (Short Quote) |")
                        lines.append("|---|---|---|")
                        for kp in analysis.key_points:
                            link_md = f"[`{kp.timestamp_str}`]({kp.deep_link})"
                            point_clean = (
                                UntrustedContent.sanitize_text(kp.point)
                                .replace("|", "/")
                                .replace("\r", " ")
                                .replace("\n", " ")
                                .strip()
                            )
                            if kp.quote:
                                quote_text = (
                                    UntrustedContent.sanitize_text(kp.quote)
                                    .replace("|", "/")
                                    .replace("\r", " ")
                                    .replace("\n", " ")
                                    .strip()
                                )
                                quote_clean = f"「{quote_text}」"
                            else:
                                quote_clean = "-"
                            lines.append(f"| {link_md} | {point_clean} | {quote_clean} |")
                        lines.append("")

                    if analysis.takeaways:
                        lines.append("#### 💡 關鍵啟示與行動洞察 (Key Takeaways)")
                        lines.append("")
                        for t in analysis.takeaways:
                            clean_takeaway = (
                                UntrustedContent.sanitize_text(t)
                                .replace("\r\n", " ")
                                .replace("\r", " ")
                                .replace("\n", " ")
                                .strip()
                            )
                            lines.append(f"- {clean_takeaway}")
                        lines.append("")

                    if meta.chapters:
                        lines.append("#### 📑 官方描述欄章節索引 (Original Chapters)")
                        lines.append("")
                        for ch in meta.chapters:
                            chapter_url = f"https://youtu.be/{meta.video_id}?t={ch.start_seconds}"
                            clean_ch_title = (
                                UntrustedContent.sanitize_text(ch.title)
                                .replace("\r\n", " ")
                                .replace("\r", " ")
                                .replace("\n", " ")
                                .strip()
                            )
                            lines.append(f"- [`{ch.timestamp_str}`]({chapter_url}) {clean_ch_title}")
                        lines.append("")

        return "\n".join(lines).strip()

    @classmethod
    def render_topic_json(cls, result: TopicSearchResult) -> str:
        """Render machine-readable JSON representation for topic search result."""
        return json.dumps(result.to_dict(), indent=2, ensure_ascii=False)

    @classmethod
    def render_history_markdown(cls, history_data: Dict[str, Any]) -> str:
        """Render markdown report for history and deduplication status."""
        total_records = history_data.get("total_records", 0)
        cooldown_count = history_data.get("cooldown_count", 0)
        cooldown_days = history_data.get("cooldown_days", 30)
        recent_entries = history_data.get("recent_entries", [])

        lines = []
        lines.append("# 📜 YouTube Scout 歷史紀錄與去重狀態")
        lines.append("")
        lines.append("| 總記錄數 | 冷卻中影片數 | 冷卻天數 |")
        lines.append("|---|---|---|")
        lines.append(f"| {total_records:,} | {cooldown_count:,} | {cooldown_days} 天 |")
        lines.append("")

        lines.append("## 🕒 最近推薦紀錄 (Recent Records)")
        lines.append("")

        if not recent_entries:
            lines.append("> *目前歷史庫中尚無紀錄。*")
        else:
            lines.append("| 影片 ID | 標題 | 頻道 | 狀態 | 最近推薦 | 累計次數 | 查詢 |")
            lines.append("|---|---|---|---|---|---|---|")
            for entry in recent_entries:
                v_id = (
                    UntrustedContent.sanitize_text(entry.get("video_id", ""))
                    .replace("|", "/")
                    .replace("\r\n", " ")
                    .replace("\r", " ")
                    .replace("\n", " ")
                    .strip()
                )
                raw_title = UntrustedContent.sanitize_text(entry.get("title", "")).rstrip("\\")
                clean_title = (
                    raw_title
                    .replace("[", "\\[")
                    .replace("]", "\\]")
                    .replace("|", "/")
                    .replace("\r\n", " ")
                    .replace("\r", " ")
                    .replace("\n", " ")
                    .strip()
                )
                clean_channel = (
                    UntrustedContent.sanitize_text(entry.get("channel_title", ""))
                    .replace("|", "/")
                    .replace("\r\n", " ")
                    .replace("\r", " ")
                    .replace("\n", " ")
                    .strip()
                )
                status = (
                    UntrustedContent.sanitize_text(entry.get("status", ""))
                    .replace("|", "/")
                    .replace("\r\n", " ")
                    .replace("\r", " ")
                    .replace("\n", " ")
                    .strip()
                )
                last_rec = (
                    UntrustedContent.sanitize_text(entry.get("last_recommended_date", ""))
                    .replace("|", "/")
                    .replace("\r\n", " ")
                    .replace("\r", " ")
                    .replace("\n", " ")
                    .strip()
                )
                count = entry.get("recommend_count", 1)
                raw_query = entry.get("query")
                clean_query = (
                    UntrustedContent.sanitize_text(raw_query)
                    .replace("|", "/")
                    .replace("\r\n", " ")
                    .replace("\r", " ")
                    .replace("\n", " ")
                    .strip()
                    if raw_query
                    else "-"
                )
                video_link = f"[`{v_id}`](https://youtu.be/{v_id})" if v_id else "-"
                lines.append(
                    f"| {video_link} | {clean_title} | {clean_channel} | {status} | {last_rec} | {count} | {clean_query} |"
                )

        return "\n".join(lines).strip()

    @classmethod
    def render_history_json(cls, history_data: Dict[str, Any]) -> str:
        """Render machine-readable JSON representation for history data."""
        return json.dumps(history_data, indent=2, ensure_ascii=False)



