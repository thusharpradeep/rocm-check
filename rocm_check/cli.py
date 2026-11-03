"""Command-line entry point for rocm_check."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from rich.console import Console

from . import __version__
from .models import ScanResult, Severity
from .report import render_json, render_markdown, render_terminal
from .rules import RuleError, load_rules
from .scanner import ScanError, scan_target
from .source import SourceError

FORMAT_CHOICES = ("terminal", "json", "html", "markdown")
FAIL_ON_CHOICES = ("none", "blocker", "warning", "info", "high", "medium", "low")
DEFAULT_HTML_REPORT = "rocm-report.html"

_FAIL_ON_ALIASES = {
    "high": "blocker",
    "medium": "warning",
    "low": "info",
}
_SEVERITY_RANK = {
    Severity.BLOCKER: 0,
    Severity.WARNING: 1,
    Severity.INFO: 2,
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="rocm-check",
        description=(
            "Scan a local directory or GitHub repository for CUDA-specific code "
            "that would break on AMD ROCm."
        ),
    )
    parser.add_argument(
        "target",
        nargs="?",
        default=".",
        help="local directory or GitHub URL / owner/repo (default: current directory)",
    )
    parser.add_argument(
        "--format",
        choices=FORMAT_CHOICES,
        default=None,
        help="output format (default: terminal)",
    )
    parser.add_argument(
        "--output",
        metavar="PATH",
        help=(
            "write the selected format to PATH; use '-' for stdout "
            "(not valid with --format terminal)"
        ),
    )
    parser.add_argument(
        "--json",
        metavar="PATH",
        help="legacy shorthand for writing a JSON report to PATH ('-' for stdout)",
    )
    parser.add_argument(
        "--markdown",
        metavar="PATH",
        help="legacy shorthand for writing a Markdown report to PATH ('-' for stdout)",
    )
    parser.add_argument(
        "--rules",
        metavar="PATH",
        action="append",
        default=[],
        help="add a custom YAML rules file (repeatable; ids override built-ins)",
    )
    parser.add_argument(
        "--rules-only",
        action="store_true",
        help="use only the files passed with --rules, not the built-in rules",
    )
    parser.add_argument(
        "--exclude",
        metavar="GLOB",
        action="append",
        default=[],
        help="exclude matching paths (repeatable)",
    )
    parser.add_argument(
        "--fail-on",
        choices=FAIL_ON_CHOICES,
        default="blocker",
        help=(
            "exit 1 when a finding at or above this severity exists "
            "(default: blocker; aliases: high, medium, low)"
        ),
    )
    parser.add_argument(
        "--no-color",
        action="store_true",
        help="disable ANSI colors in terminal output",
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {__version__}",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the command-line interface and return a process exit code."""

    parser = build_parser()
    args = parser.parse_args(argv)
    output_format = args.format or "terminal"

    if args.rules_only and not args.rules:
        return _error("--rules-only requires at least one --rules PATH")
    if output_format == "terminal" and args.output:
        return _error(
            "--output requires --format json, html, or markdown; "
            "terminal output goes to stdout"
        )
    stdout_reports = _stdout_reports(args, output_format)
    if len(stdout_reports) > 1:
        return _error(
            "only one report can be written to stdout; use a file path for the others"
        )
    if output_format == "json" and args.json:
        return _error("use either --format json or --json, not both")
    if output_format == "markdown" and args.markdown:
        return _error("use either --format markdown or --markdown, not both")

    try:
        rules = load_rules(args.rules, include_builtin=not args.rules_only)
        result = scan_target(
            args.target,
            rules=rules,
            extra_exclude=tuple(args.exclude),
        )
    except (RuleError, SourceError, ScanError) as exc:
        return _error(str(exc))

    try:
        if output_format == "terminal":
            if not stdout_reports:
                render_terminal(result, console=_console(args.no_color))
        elif output_format == "json":
            _deliver(render_json(result), args.output or "-", "JSON")
        elif output_format == "markdown":
            _deliver(render_markdown(result), args.output or "-", "Markdown")
        elif output_format == "html":
            from .html_report import render_html

            _deliver(
                render_html(result),
                args.output or DEFAULT_HTML_REPORT,
                "HTML",
            )

        if args.json:
            _deliver(render_json(result), args.json, "JSON")
        if args.markdown:
            _deliver(render_markdown(result), args.markdown, "Markdown")
    except (OSError, RuntimeError) as exc:
        return _error(f"could not write report: {exc}")

    return _exit_code(result, args.fail_on)


def _stdout_reports(args: argparse.Namespace, output_format: str) -> list[str]:
    reports: list[str] = []
    if args.json == "-":
        reports.append("--json")
    if args.markdown == "-":
        reports.append("--markdown")
    if output_format == "json" and not args.output:
        reports.append("--format json")
    if output_format == "markdown" and not args.output:
        reports.append("--format markdown")
    if output_format == "html" and args.output == "-":
        reports.append("--format html")
    return reports


def _console(no_color: bool) -> Console:
    return Console(no_color=no_color or "NO_COLOR" in os.environ)


def _deliver(content: str, destination: str, label: str) -> None:
    if destination == "-":
        print(content)
        return

    path = Path(destination).expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    print(f"rocm-check: wrote {label} report to {path}", file=sys.stderr)


def _exit_code(result: ScanResult, fail_on: str) -> int:
    if fail_on == "none":
        return 0
    threshold = _SEVERITY_RANK[Severity(_FAIL_ON_ALIASES.get(fail_on, fail_on))]
    if any(_SEVERITY_RANK[finding.severity] <= threshold for finding in result.findings):
        return 1
    return 0


def _error(message: str) -> int:
    print(f"rocm-check: error: {message}", file=sys.stderr)
    return 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
