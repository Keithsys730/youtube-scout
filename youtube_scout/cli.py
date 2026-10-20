"""Command Line Interface for YouTube Scout."""

import os
import re
import sys
import logging
import argparse
from typing import Optional, List
from .runner import ScoutRunner
from .reporter import VideoReporter
from .fetcher import MissingApiKeyError, InvalidUrlError, MetadataFetchError


def configure_logging(verbose: bool = False) -> None:
    """Configure console logging level and format."""
    level = logging.DEBUG if verbose else logging.WARNING
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S"
    )


def main(argv: Optional[List[str]] = None) -> int:
    """CLI entry point for YouTube Scout single-video and topic-search radar.
    
    Returns 0 on success (including NO_TRANSCRIPT graceful completion),
    and non-zero on actionable errors (e.g. missing API key).
    """
    if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except Exception:
            pass

    parser = argparse.ArgumentParser(
        prog="youtube-scout",
        description="YouTube Scout - Autonomous YouTube radar and timestamped architectural insight scout"
    )
    parser.add_argument(
        "--url", "-u",
        type=str,
        required=False,
        default=None,
        help="YouTube video URL or 11-character video ID"
    )
    parser.add_argument(
        "--query", "-q",
        type=str,
        default=None,
        help="Search query or topic for discovery radar"
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=10,
        help="Maximum ranked candidates to display (default: 10)"
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=None,
        help="Number of top candidates to analyze with LLM (default: from config, usually 3)"
    )
    parser.add_argument(
        "--rank-only",
        action="store_true",
        default=False,
        help="Only rank candidates without running LLM transcript summarization"
    )
    parser.add_argument(
        "--print",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Print report to console (default: True, use --no-print to silence)"
    )
    parser.add_argument(
        "--json", "-j",
        action="store_true",
        default=False,
        help="Output in structured JSON format instead of Markdown"
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default=None,
        help="Optional file path to save output report"
    )
    parser.add_argument(
        "--config", "-c",
        type=str,
        default="config.yaml",
        help="Path to YAML configuration file (default: config.yaml)"
    )
    parser.add_argument(
        "--db-path",
        type=str,
        default=None,
        help="Path to SQLite history database file"
    )
    parser.add_argument(
        "--cooldown-days",
        type=int,
        default=None,
        help="Cooldown period in days to filter out previously seen videos"
    )
    parser.add_argument(
        "--no-dedup",
        action="store_true",
        default=False,
        help="Disable history deduplication filtering"
    )
    parser.add_argument(
        "--force",
        action="store_true",
        default=False,
        help="Bypass history cooldown check and force re-analysis"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        default=False,
        help="Execute pipeline without recording videos into history database"
    )
    parser.add_argument(
        "--inspect-history",
        action="store_true",
        default=False,
        help="Display summary statistics and recent entries from the history database and exit"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        default=False,
        help="Enable verbose debug logging"
    )

    args = parser.parse_args(argv)
    configure_logging(args.verbose)

    if args.cooldown_days is not None and args.cooldown_days < 0:
        sys.stderr.write("Error: --cooldown-days must be a non-negative integer.\n")
        return 1

    if args.inspect_history:
        try:
            runner = ScoutRunner(
                api_key=os.environ.get("YOUTUBE_API_KEY") or "history_inspect_dummy_key",
                config_path=args.config,
                db_path=args.db_path,
                cooldown_days=args.cooldown_days,
            )
            cd_days = args.cooldown_days if args.cooldown_days is not None else runner.cooldown_days
            history_data = runner.ledger.inspect_history(cooldown_days=cd_days)
            if args.json:
                output_content = VideoReporter.render_history_json(history_data)
            else:
                output_content = VideoReporter.render_history_markdown(history_data)

            if args.print:
                print(output_content)

            if args.output:
                parent_dir = os.path.dirname(os.path.abspath(args.output))
                if parent_dir:
                    os.makedirs(parent_dir, exist_ok=True)
                with open(args.output, "w", encoding="utf-8") as f:
                    f.write(output_content)

            return 0
        except Exception as e:
            logging.getLogger("youtube_scout.cli").debug("Unhandled CLI exception:", exc_info=True)
            safe_msg = re.sub(r'([?&]key=)[^&\s]+', r'\1[REDACTED]', str(e))
            for env_key in ("YOUTUBE_API_KEY", "OPENAI_API_KEY"):
                val = os.environ.get(env_key)
                if val:
                    safe_msg = safe_msg.replace(val, "[REDACTED]")
            sys.stderr.write(f"Error: {safe_msg}\n")
            return 1

    # Mutual exclusivity enforcement
    if not args.url and not args.query:
        sys.stderr.write("Error: Please specify either --url/-u <video_url> or --query/-q <search_query>.\n")
        return 1

    if args.url and args.query:
        sys.stderr.write("Error: Cannot specify both --url and --query at the same time.\n")
        return 1

    if args.query is not None and not args.query.strip():
        sys.stderr.write("Error: Search query cannot be empty or whitespace only.\n")
        return 1

    if args.limit is not None and args.limit <= 0:
        sys.stderr.write("Error: --limit must be a positive integer.\n")
        return 1

    if args.top_n is not None and args.top_n <= 0:
        sys.stderr.write("Error: --top-n must be a positive integer.\n")
        return 1

    if args.url and args.rank_only:
        sys.stderr.write("Error: --rank-only can only be used with --query/-q.\n")
        return 1

    if args.url and args.top_n is not None:
        sys.stderr.write("Error: --top-n can only be used with --query/-q.\n")
        return 1

    if args.top_n is not None and args.limit is not None and args.top_n > args.limit:
        sys.stderr.write("Error: --top-n cannot exceed --limit.\n")
        return 1

    try:
        runner_kwargs = {"config_path": args.config}
        if args.db_path is not None:
            runner_kwargs["db_path"] = args.db_path
        if args.cooldown_days is not None:
            runner_kwargs["cooldown_days"] = args.cooldown_days
        runner = ScoutRunner(**runner_kwargs)

        if args.query:
            topic_kwargs = {
                "query": args.query,
                "limit": args.limit,
                "top_n": args.top_n,
                "rank_only": args.rank_only,
            }
            if args.no_dedup:
                topic_kwargs["no_dedup"] = True
            if args.force:
                topic_kwargs["force"] = True
            if args.dry_run:
                topic_kwargs["dry_run"] = True

            result = runner.run_topic(**topic_kwargs)
            if args.json:
                output_content = VideoReporter.render_topic_json(result)
            else:
                output_content = VideoReporter.render_topic_markdown(result, limit=args.limit)
        else:
            if args.dry_run:
                analysis = runner.run_video(args.url, dry_run=True)
            else:
                analysis = runner.run_video(args.url)
            if args.json:
                output_content = VideoReporter.render_json(analysis)
            else:
                output_content = VideoReporter.render_markdown(analysis)

        if args.print:
            print(output_content)

        if args.output:
            parent_dir = os.path.dirname(os.path.abspath(args.output))
            if parent_dir:
                os.makedirs(parent_dir, exist_ok=True)
            with open(args.output, "w", encoding="utf-8") as f:
                f.write(output_content)

        return 0

    except MissingApiKeyError as e:
        sys.stderr.write(f"Error: {e}\n")
        return 1
    except (InvalidUrlError, MetadataFetchError) as e:
        sys.stderr.write(f"Error: {e}\n")
        return 1
    except Exception as e:
        logging.getLogger("youtube_scout.cli").debug("Unhandled CLI exception:", exc_info=True)
        safe_msg = re.sub(r'([?&]key=)[^&\s]+', r'\1[REDACTED]', str(e))
        for env_key in ("YOUTUBE_API_KEY", "OPENAI_API_KEY"):
            val = os.environ.get(env_key)
            if val:
                safe_msg = safe_msg.replace(val, "[REDACTED]")
        sys.stderr.write(f"Error: {safe_msg}\n")
        return 1



if __name__ == "__main__":
    sys.exit(main())
