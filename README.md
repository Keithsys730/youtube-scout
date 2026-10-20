# YouTube Scout 🛰️🎬

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python: 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![CLI: youtube--scout](https://img.shields.io/badge/CLI-youtube--scout-orange.svg)](https://github.com/Keithsys730/youtube-scout)

Autonomous YouTube technical radar and timestamped architectural insight scout. Transforms long-form engineering talks, tutorials, and conference sessions into deep-linked, actionable summaries and cross-video radar reports.

```
                           YouTube Scout Pipeline
┌──────────────────┐      ┌─────────────────────────────┐      ┌───────────────────────────┐
│ YouTube Video /  │ ───> │ Swappable Transcript Chain  │ ───> │ Untrusted Content Defense │
│ Topic Search     │      │ • YouTube Transcript API    │      │ • Injection sanitization  │
└──────────────────┘      │ • yt-dlp Auto-Sub fallback  │      │ • Value-floor degradation │
                          │ • Value floor (Graceful)    │      └───────────────────────────┘
                          └─────────────────────────────┘                    │
                                                                             ▼
┌──────────────────┐      ┌─────────────────────────────┐      ┌───────────────────────────┐
│ SQLite WAL Ledger│ <─── │ Timestamped Deep Link Report│ <─── │ LLM Architectural Insight │
│ • 30-day cooldown│      │ (youtu.be/<id>?t=<seconds>) │      │ • Single video summary    │
│ • Dedup ledger   │      └─────────────────────────────┘      │ • Cross-topic consensus   │
└──────────────────┘                                           └───────────────────────────┘
```

---

## Key Features

1. **Robust Multi-Layer Transcript Fallback Chain**:
   - Primary: `youtube-transcript-api` (clean official/manual subtitles)
   - Secondary Fallback: `yt-dlp --write-auto-sub --skip-download` (auto-generated speech recognition)
   - Value Floor: Emits a structured `NO_TRANSCRIPT` card preserving video metadata, view counts, and author chapter marks. **Never crashes.**
2. **Untrusted Content Sanitizer & Injection Defense**:
   - Isolates all external user-generated titles, descriptions, and transcripts behind strict passive evidence delimiters (`UntrustedContent`), neutralizing prompt injection attacks.
3. **Timestamped Deep-Linked Summaries**:
   - Extracts key architectural points and technical arguments with clickable time-coded deep links (`https://youtu.be/<video_id>?t=<seconds>`).
4. **Cross-Video Topic Radar & Consensus Synthesis**:
   - Searches YouTube topics, ranks candidates by composite engagement/velocity signals, and synthesizes multi-video consensus points, opposing perspectives, and architectural takeaways.
5. **Persistent SQLite WAL Deduplication Ledger**:
   - Tracks analyzed videos in high-concurrency WAL mode with a 30-day cooldown to prevent duplicate token expenditure.

---

## Installation

```bash
git clone https://github.com/Keithsys730/youtube-scout.git
cd youtube-scout
python -m pip install -e ".[dev]"
```

---

## Environment Variables

| Variable | Description | Required |
|---|---|---|
| `YOUTUBE_API_KEY` | Google YouTube Data API v3 key for search and video metadata | Yes (for live network runs) |
| `OPENAI_API_KEY` | OpenAI API key (or OpenAI-compatible endpoint) for LLM analysis | Yes (for live LLM analysis) |

---

## Configuration (config.yaml)

By default, config.yaml sets default_language: zh-TW (Traditional Chinese) for summary output. To generate reports in English or another language, set default_language: en in config.yaml.

---

## CLI Usage

### 1. Single Video Deep-Dive
Analyze an individual video by URL or 11-character video ID:

```bash
# Generate markdown report with timestamped deep links
python -m youtube_scout.cli --url "https://www.youtube.com/watch?v=dQw4w9WgXcQ"

# Export machine-readable JSON output
python -m youtube_scout.cli --url "dQw4w9WgXcQ" --json

# Save report directly to file
python -m youtube_scout.cli --url "dQw4w9WgXcQ" --output reports/summary.md
```

### 2. Multi-Video Topic Radar & Cross Synthesis
Search and synthesize top engineering videos for a topic:

```bash
# Run radar scan for a topic
python -m youtube_scout.cli --query "Rust async runtime" --limit 5

# Export radar synthesis
python -m youtube_scout.cli --query "Local LLM inference" --output reports/radar_llm.md
```

---

## Testing

Run the 100% offline pytest suite (mocked fixtures, zero API keys or network connection required):

```bash
pytest tests -v
```

---

## License

This project is licensed under the [MIT License](LICENSE).

---

## AI Agent Skill Integration (Claude Code / Codex / Gemini)

This repository ships with a portable [SKILL.md](SKILL.md). Copy or symlink it into ~/.claude/skills/youtube-scout/, ~/.codex/skills/youtube-scout/, or ~/.gemini/config/skills/youtube-scout/ to give your AI agents on-demand YouTube architectural intelligence.
