"""Render scan results as a terminal report, JSON, or Markdown."""

from __future__ import annotations

import html
import json
from typing import Any

from rich import box
from rich.console import Console
from rich.table import Table
from rich.text import Text

from . import __version__
from .models import Finding, ScanResult, Severity

SEVERITY_LABEL = {
    Severity.BLOCKER: "BLOCKER",
    Severity.WARNING: "WARNING",
    Severity.INFO: "INFO",
}
SEVERITY_STYLE = {
    Severity.BLOCKER: "bold red",
    Severity.WARNING: "yellow",
    Severity.INFO: "cyan",
}


def summary_counts(result: ScanResult) -> dict[str, int]:
    """Return finding counts keyed by severity value."""

    counts = {severity.value: 0 for severity in Severity}
    for finding in result.findings:
        counts[finding.severity.value] += 1
    return counts


def render_terminal(
    result: ScanResult,
    console: Console | None = None,
) -> None:
    """Print a compact terminal report."""

    console = console or Console()
    counts = summary_counts(result)
    style = _score_style(result.score)

    title = Text("rocm-check", style="bold")
    title.append(f"  v{__version__}", style="dim")
    console.print(title)

    summary = Table.grid(padding=(0, 2))
    summary.add_column(style="bold", no_wrap=True)
    summary.add_column(overflow="fold")
    summary.add_row("Target", Text(result.target))
    files = f"{len(result.files_scanned)} scanned"
    if result.files_skipped:
        files += f", {result.files_skipped} skipped"
    summary.add_row("Files", files)

    issues = Text()
    issues.append(f"{counts['blocker']} blockers", style=SEVERITY_STYLE[Severity.BLOCKER])
    issues.append("   ")
    issues.append(f"{counts['warning']} warnings", style=SEVERITY_STYLE[Severity.WARNING])
    issues.append("   ")
    issues.append(f"{counts['info']} info", style=SEVERITY_STYLE[Severity.INFO])
    summary.add_row("Issues", issues)

    score = Text()
    score.append(f"{result.score}/100", style=f"bold {style}")
    score.append(f"   {result.grade}", style=style)
    summary.add_row("Score", score)
    console.print(summary)

    if not result.findings:
        console.print()
        console.print("[green]No CUDA-specific code detected.[/green]")
        return

    console.print()
    table = Table(
        title=f"Findings ({len(result.findings)})",
        box=box.SIMPLE,
        show_lines=False,
        show_edge=False,
        expand=True,
        padding=(0, 1),
    )
    table.add_column("Severity", no_wrap=True, width=9)
    table.add_column("Location", max_width=34, overflow="fold")
    table.add_column("Finding and Suggested ROCm fix", ratio=1, overflow="fold")

    for finding in result.findings:
        location = Text(f"{finding.path}:{finding.line}")
        location.append(f"\n{finding.rule_id}", style="dim")
        if finding.tags:
            location.append(f"\n[{', '.join(finding.tags)}]", style="dim")

        cell = Text(finding.explanation)
        cell.append("\n")
        cell.append(f"  {finding.excerpt}", style="dim italic")
        cell.append("\n  Suggested ROCm fix: ", style="bold")
        cell.append(finding.suggestion)

        table.add_row(
            Text(
                SEVERITY_LABEL[finding.severity],
                style=SEVERITY_STYLE[finding.severity],
            ),
            location,
            cell,
        )

    console.print(table)
    console.print(
        "[dim]blocker = will not build or run | "
        "warning = likely porting work | "
        "info = review on ROCm[/dim]"
    )


def render_json(result: ScanResult) -> str:
    """Return a machine-readable report."""

    payload = {
        "tool": "rocm-check",
        "version": __version__,
        "target": result.target,
        "score": result.score,
        "grade": result.grade,
        "summary": summary_counts(result),
        "files_scanned": result.files_scanned,
        "files_skipped": result.files_skipped,
        "rules_loaded": result.rules_loaded,
        "findings": [_finding_dict(finding) for finding in result.findings],
    }
    return json.dumps(payload, indent=2, ensure_ascii=False)


def render_markdown(result: ScanResult) -> str:
    """Return a pull-request-friendly Markdown report."""

    counts = summary_counts(result)
    lines = [
        "# ROCm Compatibility Report",
        "",
        f"**Target:** `{result.target}`",
        "",
        "## Summary",
        "",
        "| Metric | Value |",
        "| --- | --- |",
        f"| Compatibility score | **{result.score}/100** - {result.grade} |",
        f"| Files scanned | {len(result.files_scanned)} |",
        f"| Files skipped | {result.files_skipped} |",
        f"| Blockers | {counts['blocker']} |",
        f"| Warnings | {counts['warning']} |",
        f"| Info | {counts['info']} |",
        "",
        "## Findings",
        "",
    ]

    if not result.findings:
        lines.append("_No CUDA-specific code detected._")
    else:
        lines.append(
            "| # | Location | Severity | Rule | Explanation | Suggested fix |"
        )
        lines.append("| --- | --- | --- | --- | --- | --- |")
        for index, finding in enumerate(result.findings, start=1):
            location = f"`{_md_cell(f'{finding.path}:{finding.line}')}`"
            location += f"<br><code>{html.escape(finding.excerpt)}</code>"
            rule = f"`{finding.rule_id}`"
            if finding.tags:
                rule += "<br>" + " ".join(f"`{tag}`" for tag in finding.tags)
            lines.append(
                f"| {index} | {location} | {SEVERITY_LABEL[finding.severity]} | "
                f"{rule} | {_md_cell(finding.explanation)} | "
                f"{_md_cell(finding.suggestion)} |"
            )

    lines.extend(
        [
            "",
            "---",
            "",
            f"_Generated by rocm-check {__version__}._",
            "",
        ]
    )
    return "\n".join(lines)


def _finding_dict(finding: Finding) -> dict[str, Any]:
    return {
        "file": finding.path,
        "line": finding.line,
        "column": finding.column,
        "severity": finding.severity.value,
        "rule_id": finding.rule_id,
        "rule": finding.rule_name,
        "excerpt": finding.excerpt,
        "explanation": finding.explanation,
        "suggestion": finding.suggestion,
        "tags": list(finding.tags),
    }


def _score_style(score: int) -> str:
    if score >= 90:
        return "green"
    if score >= 70:
        return "yellow"
    if score >= 40:
        return "dark_orange"
    return "red"


def _md_cell(text: str) -> str:
    return text.replace("|", "\\|").replace("\r", "").replace("\n", "<br>")
