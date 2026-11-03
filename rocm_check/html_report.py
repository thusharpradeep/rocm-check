"""Build a self-contained HTML report from a scan result."""

from __future__ import annotations

import html
import re
from collections import Counter, defaultdict
from importlib import resources
from typing import Iterable

from . import __version__
from .models import Finding, ScanResult, Severity

_PLACEHOLDER_RE = re.compile(r"__[A-Z][A-Z0-9_]*__")
_SEVERITY_ORDER = (Severity.BLOCKER, Severity.WARNING, Severity.INFO)
_SEVERITY_LABEL = {
    Severity.BLOCKER: "Blocker",
    Severity.WARNING: "Warning",
    Severity.INFO: "Info",
}
_SEVERITY_RANK = {
    Severity.BLOCKER: 0,
    Severity.WARNING: 1,
    Severity.INFO: 2,
}
_BUILD_RULES = {
    "cuda-nvcc",
    "cuda-arch-flags",
    "torch-cuda-extension",
    "cuda-wheel-index",
    "cuda-version-pin",
}
_CONTAINER_RULES = {"cuda-docker-base", "cuda-only-package-install"}
_ENVIRONMENT_RULES = {"cuda-visible-devices", "cuda-home-env", "nvidia-smi"}
_CUDA_CPP_RULES = {
    "cuda-runtime-api",
    "cuda-include",
    "cuda-kernel-launch",
    "cuda-source-file",
}
_DEPENDENCY_RULES = {
    "dep-flash-attn": "flash-attn",
    "dep-bitsandbytes": "bitsandbytes",
    "dep-xformers": "xformers",
    "dep-apex": "apex",
    "dep-cupy": "cupy",
    "dep-pycuda": "pycuda",
    "dep-cuda-python": "cuda-python",
    "dep-tensorrt": "tensorrt",
    "dep-cudnn": "cudnn",
    "dep-onnxruntime-gpu": "onnxruntime-gpu",
    "dep-jax-cuda": "jax[cuda]",
}
_IMPORT_PACKAGES = {
    "flash_attn": "flash-attn",
    "flash-attn": "flash-attn",
    "bitsandbytes": "bitsandbytes",
    "xformers": "xformers",
    "apex": "apex",
    "cupy": "cupy",
    "pycuda": "pycuda",
}


def render_html(result: ScanResult) -> str:
    """Return a standalone HTML report with no external assets."""

    template = _read_asset("templates/report.html")
    style = _read_asset("static/report.css")
    script = _read_asset("static/report.js")
    counts = _severity_counts(result.findings)
    categories = _category_counts(result.findings)
    apis, dependencies = _detected_items(result.findings)
    files = _file_groups(result.findings)

    replacements = {
        "__TITLE__": _esc(f"ROCm compatibility report: {result.target}"),
        "__STYLE__": style,
        "__SCRIPT__": script,
        "__TARGET__": _esc(result.target),
        "__VERSION__": _esc(__version__),
        "__SCORE__": str(result.score),
        "__GRADE__": _esc(result.grade),
        "__SCORE_CLASS__": _score_class(result.score),
        "__SUMMARY_CARDS__": _summary_cards(result, counts),
        "__SEVERITY_CHART__": _severity_chart(counts),
        "__CATEGORY_BREAKDOWN__": _category_breakdown(categories),
        "__DETECTED_SECTION__": _detected_section(apis, dependencies),
        "__FILTER_BUTTONS__": _filter_buttons(counts, len(result.findings)),
        "__FILE_OPTIONS__": _file_options(files),
        "__FILE_RESULTS__": _file_results(files),
        "__VISIBLE_COUNT__": (
            f"{len(result.findings)} findings"
            if result.findings
            else "0 findings"
        ),
    }
    return _PLACEHOLDER_RE.sub(
        lambda match: replacements.get(match.group(0), match.group(0)),
        template,
    )


def _read_asset(relative: str) -> str:
    try:
        return (
            resources.files("rocm_check")
            .joinpath(relative)
            .read_text(encoding="utf-8")
        )
    except (OSError, ModuleNotFoundError) as exc:
        raise RuntimeError(f"could not read HTML report asset {relative}: {exc}") from exc


def _esc(value: object) -> str:
    return html.escape(str(value), quote=True)


def _severity_counts(findings: Iterable[Finding]) -> dict[str, int]:
    counts = {severity.value: 0 for severity in Severity}
    for finding in findings:
        counts[finding.severity.value] += 1
    return counts


def _score_class(score: int) -> str:
    if score >= 90:
        return "good"
    if score >= 70:
        return "warn"
    if score >= 40:
        return "attention"
    return "bad"


def _summary_cards(result: ScanResult, counts: dict[str, int]) -> str:
    needs_attention = counts["blocker"] + counts["warning"]
    cards = [
        ("Compatibility score", f"{result.score}/100", result.grade, _score_class(result.score)),
        ("Files scanned", str(len(result.files_scanned)), f"{result.files_skipped} skipped", "neutral"),
        ("Blockers", str(counts["blocker"]), "Will not build or run", "blocker"),
        ("Warnings", str(counts["warning"]), "Likely porting work", "warning"),
        ("Info / compatible usage", str(counts["info"]), "Review on ROCm", "info"),
        ("Needs attention", str(needs_attention), "Blockers and warnings", "attention"),
    ]
    return "\n".join(
        f"""
        <article class="stat-card stat-{css_class}">
          <span class="stat-label">{_esc(label)}</span>
          <strong class="stat-value">{_esc(value)}</strong>
          <span class="stat-note">{_esc(note)}</span>
        </article>"""
        for label, value, note, css_class in cards
    )


def _severity_chart(counts: dict[str, int]) -> str:
    total = sum(counts.values())
    rows: list[str] = []
    for severity in _SEVERITY_ORDER:
        count = counts[severity.value]
        percentage = (count / total * 100) if total else 0.0
        rows.append(
            f"""
            <div class="severity-row">
              <div class="severity-name">
                <span class="severity-dot {severity.value}"></span>
                {_esc(_SEVERITY_LABEL[severity])}
              </div>
              <div class="severity-track" aria-hidden="true">
                <span class="severity-fill {severity.value}" style="width: {percentage:.1f}%"></span>
              </div>
              <div class="severity-count">{count}</div>
            </div>"""
        )
    return "".join(rows)


def _category_counts(findings: Iterable[Finding]) -> dict[str, Counter[str]]:
    categories: dict[str, Counter[str]] = defaultdict(Counter)
    for finding in findings:
        categories[_category_for(finding.rule_id)][finding.severity.value] += 1
    return dict(categories)


def _category_for(rule_id: str) -> str:
    if rule_id.startswith("dep-") or rule_id == "cuda-only-package-import":
        return "Dependencies"
    if rule_id in _CONTAINER_RULES:
        return "Containers"
    if rule_id in _BUILD_RULES:
        return "Build and packaging"
    if rule_id in _ENVIRONMENT_RULES:
        return "Environment"
    if rule_id in _CUDA_CPP_RULES:
        return "CUDA and C++"
    if rule_id == "torch-cuda-nvtx":
        return "Profiling"
    if rule_id == "torch-cuda-graph":
        return "CUDA graphs"
    if rule_id == "torch-cuda-amp":
        return "Mixed precision"
    if rule_id == "nccl-reference":
        return "Distributed"
    if rule_id == "cuda-only-import":
        return "Python API"
    if rule_id.startswith("torch-") or rule_id in {
        "tensor-dot-cuda",
        "hardcoded-cuda-device",
    }:
        return "PyTorch API"
    return "Other"


def _category_breakdown(categories: dict[str, Counter[str]]) -> str:
    if not categories:
        return '<p class="muted">No findings to break down.</p>'

    ordered = sorted(
        categories.items(),
        key=lambda item: (-sum(item[1].values()), item[0]),
    )
    rows = []
    for name, counts in ordered:
        total = sum(counts.values())
        rows.append(
            f"""
            <tr>
              <td>{_esc(name)}</td>
              <td>{counts.get("blocker", 0)}</td>
              <td>{counts.get("warning", 0)}</td>
              <td>{counts.get("info", 0)}</td>
              <td><strong>{total}</strong></td>
            </tr>"""
        )
    return f"""
    <table class="data-table">
      <thead>
        <tr><th>Category</th><th>Blockers</th><th>Warnings</th><th>Info</th><th>Total</th></tr>
      </thead>
      <tbody>{''.join(rows)}</tbody>
    </table>"""


def _detected_items(findings: Iterable[Finding]) -> tuple[list[str], list[str]]:
    apis: set[str] = set()
    dependencies: set[str] = set()
    for finding in findings:
        if finding.rule_id == "cuda-runtime-api":
            apis.update(re.findall(r"\bcuda[A-Z][A-Za-z0-9_]*\b", finding.excerpt))
        elif finding.rule_id == "cuda-include":
            apis.update(re.findall(r"[<\"]([^>\"]+\.h)[>\"]", finding.excerpt))
        elif finding.rule_id in _DEPENDENCY_RULES:
            dependencies.add(_DEPENDENCY_RULES[finding.rule_id])
        elif finding.rule_id in {"cuda-only-package-import", "cuda-only-package-install"}:
            for package, label in _IMPORT_PACKAGES.items():
                if package in finding.excerpt:
                    dependencies.add(label)
        elif finding.rule_id == "cuda-only-import":
            for package in ("pycuda", "cupy", "numba.cuda"):
                if package in finding.excerpt:
                    dependencies.add(package)
    return sorted(apis), sorted(dependencies)


def _detected_section(apis: list[str], dependencies: list[str]) -> str:
    api_items = _chips(apis, "No CUDA APIs or headers detected.")
    dependency_items = _chips(dependencies, "No CUDA-only dependencies detected.")
    return f"""
    <div class="detected-grid">
      <div>
        <h3>Detected CUDA APIs</h3>
        {api_items}
      </div>
      <div>
        <h3>Detected dependencies</h3>
        {dependency_items}
      </div>
    </div>"""


def _chips(items: list[str], empty_message: str) -> str:
    if not items:
        return f'<p class="muted">{_esc(empty_message)}</p>'
    return '<div class="chip-group">' + "".join(
        f'<span class="chip">{_esc(item)}</span>' for item in items
    ) + "</div>"


def _file_groups(findings: Iterable[Finding]) -> dict[str, list[Finding]]:
    grouped: dict[str, list[Finding]] = defaultdict(list)
    for finding in findings:
        grouped[finding.path].append(finding)
    for items in grouped.values():
        items.sort(
            key=lambda finding: (
                _SEVERITY_RANK[finding.severity],
                finding.line,
                finding.column,
            )
        )
    return dict(sorted(grouped.items()))


def _filter_buttons(counts: dict[str, int], total: int) -> str:
    buttons = [("all", "All", total)]
    for severity in _SEVERITY_ORDER:
        buttons.append((severity.value, _SEVERITY_LABEL[severity], counts[severity.value]))
    return "".join(
        f'<button type="button" data-severity-filter="{value}"'
        f' class="filter-button{" active" if value == "all" else ""}">'
        f'{_esc(label)} <span>{count}</span></button>'
        for value, label, count in buttons
    )


def _file_options(groups: dict[str, list[Finding]]) -> str:
    options = ['<option value="all">All files</option>']
    options.extend(
        f'<option value="{_esc(path)}">{_esc(path)}</option>' for path in groups
    )
    return "".join(options)


def _file_results(groups: dict[str, list[Finding]]) -> str:
    if not groups:
        return (
            '<div class="empty-state">'
            "<strong>No CUDA-specific code detected.</strong>"
            "<span>0 findings</span>"
            "</div>"
        )

    sections = []
    for path, findings in groups.items():
        counts = _severity_counts(findings)
        summary = _file_summary(counts)
        items = "".join(_finding_html(finding) for finding in findings)
        sections.append(
            f"""
            <section class="file-group" data-file="{_esc(path)}">
              <div class="file-heading">
                <h3>{_esc(path)}</h3>
                <span>{summary}</span>
              </div>
              {items}
            </section>"""
        )
    return "".join(sections)


def _file_summary(counts: dict[str, int]) -> str:
    parts = []
    for severity in _SEVERITY_ORDER:
        count = counts[severity.value]
        if count:
            label = _SEVERITY_LABEL[severity].lower()
            parts.append(f"{count} {label}{'' if count == 1 else 's'}")
    parts.append(f"{sum(counts.values())} total")
    return " | ".join(parts)


def _finding_html(finding: Finding) -> str:
    search_text = " ".join(
        [
            finding.path,
            finding.rule_id,
            finding.rule_name,
            finding.excerpt,
            finding.explanation,
            finding.suggestion,
            " ".join(finding.tags),
        ]
    ).lower()
    tags = ""
    if finding.tags:
        tags = '<div class="tag-group">' + "".join(
            f'<span class="tag">{_esc(tag)}</span>' for tag in finding.tags
        ) + "</div>"
    return f"""
    <details class="finding" data-severity="{finding.severity.value}" data-file="{_esc(finding.path)}" data-search="{_esc(search_text)}">
      <summary>
        <span class="severity-badge {finding.severity.value}">{_esc(_SEVERITY_LABEL[finding.severity])}</span>
        <span class="finding-title">{_esc(finding.rule_name)}</span>
        <span class="finding-rule">{_esc(finding.rule_id)}</span>
        <span class="finding-line">line {finding.line}</span>
      </summary>
      <div class="finding-body">
        <p class="finding-excerpt"><code>{_esc(finding.excerpt)}</code></p>
        <p>{_esc(finding.explanation)}</p>
        <p class="finding-fix"><strong>Suggested ROCm fix:</strong> {_esc(finding.suggestion)}</p>
        {tags}
      </div>
    </details>"""
