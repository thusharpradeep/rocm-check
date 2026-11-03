"""Walk a source tree and apply the loaded rules."""

from __future__ import annotations

import fnmatch
import os
import re
from pathlib import Path, PurePosixPath
from typing import Iterable, Optional, Sequence

from .models import Finding, Rule, ScanResult, Severity
from .rules import compile_flags, load_builtin_rules
from .scoring import compatibility_score, grade_for

DEFAULT_INCLUDE = (
    "*.py",
    "*.pyi",
    "*.cu",
    "*.cuh",
    "*.cpp",
    "*.cc",
    "*.cxx",
    "*.h",
    "*.hpp",
    "requirements*.txt",
    "setup.py",
    "setup.cfg",
    "pyproject.toml",
    "Dockerfile",
    "Dockerfile.*",
    "*.Dockerfile",
)

DEFAULT_EXCLUDE = (
    ".git",
    ".hg",
    ".svn",
    ".venv",
    "venv",
    "env",
    "node_modules",
    "__pycache__",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    ".tox",
    ".nox",
    ".eggs",
    "*.egg-info",
    "dist",
    "build",
    ".idea",
    ".vscode",
    "site-packages",
)

MAX_FILE_SIZE = 2_000_000
MAX_MATCHES_PER_RULE_FILE = 500
EXCERPT_LIMIT = 200

_SEVERITY_RANK = {
    Severity.BLOCKER: 0,
    Severity.WARNING: 1,
    Severity.INFO: 2,
}


class ScanError(ValueError):
    """Raised when the scan target or configuration is unusable."""


CompiledRules = Sequence[tuple[Rule, Optional[re.Pattern[str]]]]


def scan_path(
    root: str | Path,
    *,
    rules: Iterable[Rule] | None = None,
    include: Sequence[str] | None = None,
    exclude: Sequence[str] | None = None,
    extra_exclude: Sequence[str] = (),
    max_file_size: int = MAX_FILE_SIZE,
    target: str | None = None,
) -> ScanResult:
    """Scan a local directory and return findings plus a compatibility score."""

    root_path = Path(root).expanduser()
    if not root_path.exists():
        raise ScanError(f"scan path not found: {root_path}")
    if not root_path.is_dir():
        raise ScanError(f"scan path is not a directory: {root_path}")
    root_path = root_path.resolve()

    rule_list = list(rules) if rules is not None else load_builtin_rules()
    compiled = _compile_rules(rule_list)
    include_globs = tuple(include) if include is not None else DEFAULT_INCLUDE
    exclude_patterns = tuple(exclude or ()) + tuple(DEFAULT_EXCLUDE) + tuple(extra_exclude)

    findings: list[Finding] = []
    files_scanned: list[str] = []
    files_skipped = 0

    for path, relative in _iter_files(root_path, include_globs, exclude_patterns):
        try:
            data = path.read_bytes()
        except OSError:
            files_skipped += 1
            continue

        if len(data) > max_file_size or b"\x00" in data[:4096]:
            files_skipped += 1
            continue

        text = data.decode("utf-8-sig", errors="replace")
        files_scanned.append(relative)
        findings.extend(_scan_file(relative, text, compiled))

    findings = _dedupe(findings)
    findings.sort(key=lambda finding: (
        _SEVERITY_RANK[finding.severity], finding.path, finding.line, finding.column
    ))
    files_scanned.sort()
    score = compatibility_score(findings)

    return ScanResult(
        target=target or str(root_path),
        root=root_path,
        findings=findings,
        files_scanned=files_scanned,
        files_skipped=files_skipped,
        rules_loaded=len(rule_list),
        score=score,
        grade=grade_for(score),
    )


def scan_target(
    target: str,
    *,
    rules: Iterable[Rule] | None = None,
    include: Sequence[str] | None = None,
    exclude: Sequence[str] | None = None,
    extra_exclude: Sequence[str] = (),
    max_file_size: int = MAX_FILE_SIZE,
) -> ScanResult:
    """Resolve a local path or GitHub URL and scan it."""

    from .source import resolve_source

    with resolve_source(target) as source:
        return scan_path(
            source.root,
            rules=rules,
            include=include,
            exclude=exclude,
            extra_exclude=extra_exclude,
            max_file_size=max_file_size,
            target=source.label,
        )


def _compile_rules(rules: Iterable[Rule]) -> list[tuple[Rule, re.Pattern[str] | None]]:
    compiled: list[tuple[Rule, re.Pattern[str] | None]] = []
    for rule in rules:
        if rule.type in {"regex", "dependency"}:
            compiled.append((rule, re.compile(rule.pattern, compile_flags(rule.flags))))
        else:
            compiled.append((rule, None))
    return compiled


def _iter_files(
    root: Path,
    include: Sequence[str],
    exclude: Sequence[str],
) -> Iterable[tuple[Path, str]]:
    for dirpath, dirnames, filenames in os.walk(root):
        current = Path(dirpath)
        relative_dir = current.relative_to(root)
        dirnames[:] = sorted(
            name
            for name in dirnames
            if not _is_excluded(name, relative_dir / name, exclude)
        )

        for name in sorted(filenames):
            relative = relative_dir / name
            relative_posix = relative.as_posix()
            if _is_excluded(name, relative, exclude):
                continue
            if not _is_included(name, relative_posix, include):
                continue
            yield current / name, relative_posix


def _is_excluded(name: str, relative: Path, patterns: Sequence[str]) -> bool:
    relative_posix = relative.as_posix()
    parts = relative.parts
    for pattern in patterns:
        if fnmatch.fnmatch(name, pattern) or fnmatch.fnmatch(relative_posix, pattern):
            return True
        if pattern in parts:
            return True
    return False


def _is_included(name: str, relative_posix: str, globs: Sequence[str]) -> bool:
    return any(
        fnmatch.fnmatch(name, glob) or fnmatch.fnmatch(relative_posix, glob)
        for glob in globs
    )


def _scan_file(
    relative: str,
    text: str,
    compiled: CompiledRules,
) -> list[Finding]:
    name = PurePosixPath(relative).name
    findings: list[Finding] = []

    for rule, regex in compiled:
        if not _is_included(name, relative, rule.file_globs):
            continue

        if rule.type == "file":
            findings.append(_finding(rule, relative, 1, 1, relative))
            continue

        if regex is None:  # pragma: no cover - defensive
            continue

        rule_matches = 0
        if rule.type == "regex":
            for match in regex.finditer(text):
                if match.start() == match.end():
                    continue
                findings.append(_finding_for_index(rule, relative, text, match.start()))
                rule_matches += 1
                if rule_matches >= MAX_MATCHES_PER_RULE_FILE:
                    break
            continue

        # dependency: match the code part of each line so comments are ignored
        for line_number, line in enumerate(text.splitlines(), start=1):
            code = line.split("#", 1)[0]
            match = regex.search(code)
            if match is None:
                continue
            findings.append(
                _finding(rule, relative, line_number, match.start() + 1, line.strip())
            )
            rule_matches += 1
            if rule_matches >= MAX_MATCHES_PER_RULE_FILE:
                break

    return findings


def _finding_for_index(rule: Rule, relative: str, text: str, index: int) -> Finding:
    line_number = text.count("\n", 0, index) + 1
    line_start = text.rfind("\n", 0, index) + 1
    column = index - line_start + 1
    line_end = text.find("\n", index)
    if line_end == -1:
        line_end = len(text)
    excerpt = text[line_start:line_end].strip()
    return _finding(rule, relative, line_number, column, excerpt)


def _finding(
    rule: Rule,
    relative: str,
    line: int,
    column: int,
    excerpt: str,
) -> Finding:
    excerpt = excerpt.strip() or rule.name
    if len(excerpt) > EXCERPT_LIMIT:
        excerpt = excerpt[: EXCERPT_LIMIT - 3] + "..."
    return Finding(
        rule_id=rule.id,
        rule_name=rule.name,
        severity=rule.severity,
        path=relative,
        line=line,
        column=column,
        excerpt=excerpt,
        explanation=rule.explanation,
        suggestion=rule.suggestion,
        tags=rule.tags,
    )


def _dedupe(findings: Iterable[Finding]) -> list[Finding]:
    seen: set[tuple[str, str, int]] = set()
    unique: list[Finding] = []
    for finding in findings:
        key = (finding.rule_id, finding.path, finding.line)
        if key in seen:
            continue
        seen.add(key)
        unique.append(finding)
    return unique
