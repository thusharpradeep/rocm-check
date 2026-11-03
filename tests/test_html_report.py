from __future__ import annotations

from dataclasses import replace
from pathlib import Path

from rocm_check.html_report import render_html
from rocm_check.models import ScanResult
from rocm_check.rules import load_builtin_rules
from rocm_check.scanner import scan_path

SAMPLE = Path(__file__).parent / "samples" / "cuda_project"


def sample_result() -> ScanResult:
    return scan_path(SAMPLE, rules=load_builtin_rules())


def test_html_report_contains_summary_and_analysis_sections() -> None:
    html = render_html(sample_result())

    assert html.startswith("<!doctype html>")
    assert "ROCm compatibility report" in html
    assert "0/100" in html
    assert "CUDA-locked" in html
    assert "Files scanned" in html
    assert "Blockers" in html
    assert "Warnings" in html
    assert "Info / compatible usage" in html
    assert "Severity distribution" in html
    assert "Issue breakdown" in html
    assert "Detected CUDA APIs" in html
    assert "cudaMalloc" in html
    assert "cudaDeviceSynchronize" in html
    assert "Detected dependencies" in html
    assert "flash-attn" in html
    assert "bitsandbytes" in html
    assert "kernels/ops.cu" in html
    assert "Suggested ROCm fix" in html
    assert "hipify-able" in html


def test_html_report_has_search_filters_and_expandable_findings() -> None:
    html = render_html(sample_result())

    assert 'id="finding-search"' in html
    assert 'data-severity-filter="all"' in html
    assert 'data-severity-filter="blocker"' in html
    assert 'data-severity-filter="warning"' in html
    assert 'data-severity-filter="info"' in html
    assert 'id="file-filter"' in html
    assert "<details" in html
    assert 'class="finding"' in html
    assert 'data-file="Dockerfile"' in html
    assert 'data-severity="blocker"' in html


def test_html_report_groups_findings_by_category_and_file() -> None:
    html = render_html(sample_result())

    assert "PyTorch API" in html
    assert "CUDA and C++" in html
    assert "Dependencies" in html
    assert "Build and packaging" in html
    assert "Containers" in html
    assert "Environment" in html
    assert "Dockerfile" in html
    assert "requirements.txt" in html
    assert "train.py" in html


def test_html_report_escapes_finding_content() -> None:
    result = sample_result()
    result.findings[0] = replace(
        result.findings[0],
        excerpt="<img src=x onerror=alert(1)>",
        explanation="<script>alert('x')</script>",
        suggestion="<b>use rocm</b>",
        tags=("<unsafe>",),
    )

    html = render_html(result)

    assert "<img src=x" not in html
    assert "&lt;img src=x onerror=alert(1)&gt;" in html
    assert "<script>alert('x')</script>" not in html
    assert "&lt;script&gt;alert" in html
    assert "<b>use rocm</b>" not in html
    assert "&lt;b&gt;use rocm&lt;/b&gt;" in html
    assert "<unsafe>" not in html
    assert "&lt;unsafe&gt;" in html


def test_html_report_does_not_expand_placeholders_from_findings() -> None:
    result = sample_result()
    result.findings[0] = replace(
        result.findings[0],
        excerpt="__VISIBLE_COUNT__",
        explanation="__SCORE__",
        suggestion="__TITLE__",
    )

    html = render_html(result)

    assert "__VISIBLE_COUNT__" in html
    assert "__SCORE__" in html
    assert "__TITLE__" in html


def test_html_report_handles_clean_result() -> None:
    html = render_html(ScanResult(target="clean", score=100, grade="ROCm-ready"))

    assert "100/100" in html
    assert "No CUDA-specific code detected" in html
    assert "0 findings" in html


def test_html_report_is_self_contained_and_responsive() -> None:
    html = render_html(sample_result())

    assert '<meta name="viewport"' in html
    assert "<style>" in html
    assert "<script>" in html
    assert 'src="http' not in html
    assert 'href="http' not in html
    assert "prefers-color-scheme" in html
    assert 'id="theme-toggle"' in html
