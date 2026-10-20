"""Data models and content security boundaries for YouTube Scout."""

import re
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any


class UntrustedContent:
    """Sanitizer and security boundary for external untrusted YouTube content."""

    # Disallowed control chars (keep \t, \n, \r)
    _CONTROL_CHAR_RE = re.compile(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]')
    # Common HTML tags to strip while preserving programming syntax like std::vector<int>
    _COMMON_HTML_TAGS = (
        r'script|style|iframe|object|embed|form|input|button|svg|math|a|abbr|address|area|'
        r'article|aside|audio|b|base|bdi|bdo|blockquote|body|br|canvas|caption|cite|code|'
        r'col|colgroup|data|datalist|dd|del|details|dfn|dialog|div|dl|dt|em|fieldset|'
        r'figcaption|figure|footer|head|header|hgroup|h[1-6]|hr|html|i|img|ins|kbd|label|'
        r'legend|li|link|main|map|mark|meta|meter|nav|noscript|ol|optgroup|option|output|'
        r'p|param|picture|pre|progress|q|rp|rt|ruby|s|samp|section|select|small|source|'
        r'span|strong|sub|summary|sup|table|tbody|td|template|textarea|tfoot|th|thead|'
        r'time|title|tr|track|u|ul|var|video|wbr'
    )
    _HTML_TAG_RE = re.compile(rf'</?(?:{_COMMON_HTML_TAGS})\b[^>]*>', re.DOTALL | re.IGNORECASE)
    # Custom sandbox boundary tags to prevent delimiter collision and prompt escape
    _SANDBOX_TAG_RE = re.compile(r'</?untrusted_[a-zA-Z0-9_-]*\b[^>]*>', re.IGNORECASE)
    # Dangerous HTML elements with content
    _DANGEROUS_ELEMENTS_RE = re.compile(
        r'<(script|style|iframe|object|embed|form|input|button|svg|math)[^>]*>.*?</\1>',
        re.DOTALL | re.IGNORECASE
    )

    @classmethod
    def strip_control_chars(cls, text: str) -> str:
        """Strip non-printable and dangerous control characters."""
        if not text:
            return ""
        return cls._CONTROL_CHAR_RE.sub('', text)

    @classmethod
    def strip_html(cls, text: str) -> str:
        """Strip HTML tags and sandbox delimiters while preserving generic brackets and code comparisons."""
        if not text:
            return ""
        cleaned = text
        prev = None
        iterations = 0
        while prev != cleaned and iterations < 10:
            prev = cleaned
            cleaned = cls._DANGEROUS_ELEMENTS_RE.sub('', cleaned)
            cleaned = cls._SANDBOX_TAG_RE.sub('', cleaned)
            cleaned = cls._HTML_TAG_RE.sub('', cleaned)
            iterations += 1
        return cleaned

    @classmethod
    def sanitize_text(cls, text: Any, max_chars: Optional[int] = None) -> str:
        """Sanitize text by coercing to string, stripping control chars, removing HTML, and capping length."""
        if text is None:
            return ""
        text_str = str(text) if not isinstance(text, str) else text
        sanitized = cls.strip_control_chars(text_str)
        sanitized = cls.strip_html(sanitized)
        if max_chars and len(sanitized) > max_chars:
            sanitized = sanitized[:max_chars] + " ...[truncated]"
        return sanitized.strip()

    @classmethod
    def is_safe_url(cls, url: str) -> bool:
        """Verify URL uses safe http or https schemes."""
        if not url:
            return False
        clean_url = url.strip().lower()
        return clean_url.startswith("http://") or clean_url.startswith("https://")


@dataclass
class VideoChapter:
    """Structured chapter timestamp marker extracted from video description."""
    start_seconds: int
    timestamp_str: str
    title: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "start_seconds": self.start_seconds,
            "timestamp_str": self.timestamp_str,
            "title": self.title,
        }


@dataclass
class VideoMetadata:
    """Standardized metadata fetched from YouTube Data API."""
    video_id: str
    title: str
    channel_title: str
    published_at: str
    duration_seconds: int
    duration_str: str
    view_count: int
    like_count: int
    description: str
    chapters: List[VideoChapter] = field(default_factory=list)

    @property
    def video_url(self) -> str:
        return f"https://youtu.be/{self.video_id}"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "video_id": self.video_id,
            "video_url": self.video_url,
            "title": self.title,
            "channel_title": self.channel_title,
            "published_at": self.published_at,
            "duration_seconds": self.duration_seconds,
            "duration_str": self.duration_str,
            "view_count": self.view_count,
            "like_count": self.like_count,
            "description": self.description,
            "chapters": [c.to_dict() for c in self.chapters],
        }


@dataclass
class TranscriptSegment:
    """Single timed segment of an ASR or manual transcript."""
    text: str
    start: float
    duration: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "text": self.text,
            "start": round(self.start, 2),
            "duration": round(self.duration, 2),
        }


@dataclass
class KeyPoint:
    """Individual timestamped summary point with deep link."""
    point: str
    timestamp_seconds: int
    timestamp_str: str
    deep_link: str
    quote: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "point": self.point,
            "timestamp_seconds": self.timestamp_seconds,
            "timestamp_str": self.timestamp_str,
            "deep_link": self.deep_link,
            "quote": self.quote,
        }


@dataclass
class VideoAnalysis:
    """Structured architectural analysis and key points for a YouTube video."""
    metadata: VideoMetadata
    status: str  # "SUCCESS" or "NO_TRANSCRIPT"
    overview: str = ""
    key_points: List[KeyPoint] = field(default_factory=list)
    takeaways: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "metadata": self.metadata.to_dict(),
            "status": self.status,
            "overview": self.overview,
            "key_points": [kp.to_dict() for kp in self.key_points],
            "takeaways": self.takeaways,
        }


@dataclass
class SignalScores:
    """Quantitative signal scores for video ranking."""
    view_velocity: float
    engagement_ratio: float
    recency_decay: float
    composite_score: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "view_velocity": self.view_velocity,
            "engagement_ratio": self.engagement_ratio,
            "recency_decay": self.recency_decay,
            "composite_score": self.composite_score,
        }


@dataclass
class ScoredVideoCandidate:
    """Ranked and scored video candidate with metadata and signals."""
    metadata: VideoMetadata
    signals: SignalScores
    rank: int

    def to_dict(self) -> Dict[str, Any]:
        return {
            "metadata": self.metadata.to_dict(),
            "signals": self.signals.to_dict(),
            "rank": self.rank,
        }


@dataclass
class TechRadarItem:
    """Technology radar item categorized with adoption status."""
    name: str
    category: str
    recommendation: str  # adopt, trial, assess, hold
    description: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "category": self.category,
            "recommendation": self.recommendation,
            "description": self.description,
        }


@dataclass
class TopicRadarSynthesis:
    """Stage 2 Reduce synthesis across top-ranked candidate video analyses."""
    summary: str
    consensus_points: List[str] = field(default_factory=list)
    divergence_points: List[str] = field(default_factory=list)
    tech_radar: List[TechRadarItem] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "summary": self.summary,
            "consensus_points": self.consensus_points,
            "divergence_points": self.divergence_points,
            "tech_radar": [item.to_dict() if hasattr(item, "to_dict") else item for item in self.tech_radar],
        }


@dataclass
class TopicSearchResult:
    """Topic search and ranking radar outcome."""
    query: str
    total_candidates_found: int
    filtered_candidates_count: int
    candidates: List[ScoredVideoCandidate] = field(default_factory=list)
    top_analysis: Optional[VideoAnalysis] = None
    radar_synthesis: Optional[TopicRadarSynthesis] = None
    video_analyses: List[VideoAnalysis] = field(default_factory=list)
    exhaustion_notice: bool = False
    cooldown_filtered_count: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "query": self.query,
            "total_candidates_found": self.total_candidates_found,
            "filtered_candidates_count": self.filtered_candidates_count,
            "candidates": [c.to_dict() for c in self.candidates],
            "top_analysis": self.top_analysis.to_dict() if self.top_analysis else None,
            "radar_synthesis": self.radar_synthesis.to_dict() if self.radar_synthesis else None,
            "video_analyses": [va.to_dict() for va in self.video_analyses],
            "exhaustion_notice": self.exhaustion_notice,
            "cooldown_filtered_count": self.cooldown_filtered_count,
        }



