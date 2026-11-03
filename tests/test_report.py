from __future__ import annotations

import json
from dataclasses import replace
from io import StringIO
from pathlib import Path

from rich.console import Console

from rocm_check.models import ScanResult
from rocm_check.report import render_json, render_markdown, render_terminal
from rocm_check.rules import load_builtin_rules
from rocm_check.scanner import scan_path

SAMPLE = Path(__file__).parent / "samples" / "cuda_project"


def sample_result() -> ScanResult:
    return scan_path(SAMPLE, rules=load_builtin_rules())


def test_json_report_contains_summary_and_findings() -> None:
    result = sample_result()

    payload = json.loads(render_json(result))

    assert payload["tool"] == "rocm-check"
    assert payload["target"] == str(SAMPLE.resolve())
    assert payload["score"] == result.score
    assert payload["grade"] == result.grade
    assert payload["summary"] == {
        "blocker": 12,
        "warning": 3,
        "info": 3,
    }
    assert payload["files_scanned"] == result.files_scanned
    assert payload["files_skipped"] == 0
    assert len(payload["findings"]) == 18

    finding = payload["findings"][0]
    assert set(finding) == {
        "file",
        "line",
        "column",
        "severity",
        "rule_id",
        "rule",
        "excerpt",
        "explanation",
        "suggestion",
        "tags",
    }
    assert finding["severity"] == "blocker"
    assert finding["file"] == "Dockerfile"
    assert finding["line"] == 1

    tagged = {
        item["rule_id"]: item["tags"]
        for item in payload["findings"]
        if item["tags"]
    }
    assert tagged == {
        "cuda-source-file": ["hipify-able"],
        "cuda-runtime-api": ["hipify-able"],
    }


def test_json_report_is_valid_for_empty_result() -> None:
    payload = json.loads(render_json(ScanResult(target="empty", score=100)))

    assert payload["summary"] == {"blocker": 0, "warning": 0, "info": 0}
    assert payload["findings"] == []


def test_markdown_report_has_score_and_finding_table() -> None:
    result = sample_result()

    markdown = render_markdown(result)

    assert markdown.startswith("# ROCm Compatibility Report")
    assert f"**{result.score}/100**" in markdown
    assert result.grade in markdown
    assert "## Findings" in markdown
    assert "| # | Location | Severity | Rule | Explanation | Suggested fix |" in markdown
    assert "`kernels/ops.cu:1`" in markdown
    assert "cuda-docker-base" in markdown
    assert "rocm/pytorch" in markdown
    assert markdown.count("hipify-able") == 3
    assert markdown.count("\n|") >= 10


def test_markdown_report_handles_clean_result() -> None:
    markdown = render_markdown(ScanResult(target="clean", score=100, grade="ROCm-ready"))

    assert "No CUDA-specific code detected." in markdown


def test_markdown_escapes_table_separators() -> None:
    result = sample_result()
    result.findings[0] = replace(
        result.findings[0],
        explanation="a | b",
        suggestion="use x | y",
    )

    markdown = render_markdown(result)

    assert "a \\| b" in markdown
    assert "use x \\| y" in markdown


def test_terminal_report_renders_summary_and_findings() -> None:
    result = sample_result()
    buffer = StringIO()
    console = Console(
        file=buffer, force_terminal=False, width=200, color_system=None
    )

    render_terminal(result, console=console)
    output = buffer.getvalue()

    assert "rocm-check" in output
    assert str(result.score) in output
    assert result.grade in output
    assert "cuda-docker-base" in output
    assert "CUDA-based Docker image" not in output  # table uses rule ids
    assert "Suggested ROCm fix" in output
    assert "hipify-able" in output
    assert "╭" not in output
    assert "╮" not in output
    assert "╰" not in output
    assert "╯" not in output
    assert "█" not in output


def test_terminal_report_handles_clean_result() -> None:
    buffer = StringIO()
    console = Console(
        file=buffer, force_terminal=False, width=120, color_system=None
    )

    render_terminal(ScanResult(target="clean", score=100), console=console)

    assert "No CUDA-specific code detected." in buffer.getvalue()


def test_terminal_report_uses_colors_in_a_terminal() -> None:
    buffer = StringIO()
    console = Console(
        file=buffer, force_terminal=True, width=160, color_system="standard"
    )

    render_terminal(sample_result(), console=console)

    assert "\x1b[" in buffer.getvalue()
