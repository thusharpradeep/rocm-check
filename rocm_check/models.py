"""Data models shared by the rules engine, scanner, and reporters."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path


class Severity(str, Enum):
    """How badly a finding affects a ROCm port."""

    BLOCKER = "blocker"
    WARNING = "warning"
    INFO = "info"

    def __str__(self) -> str:  # pragma: no cover - trivial
        return self.value


@dataclass(frozen=True)
class Rule:
    """A single detection rule loaded from YAML."""

    id: str
    name: str
    severity: Severity
    type: str  # "regex" | "dependency" | "file"
    explanation: str
    suggestion: str
    pattern: str = ""
    file_globs: tuple[str, ...] = ("*",)
    flags: tuple[str, ...] = ()
    tags: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "file_globs", tuple(self.file_globs))
        object.__setattr__(self, "flags", tuple(self.flags))
        object.__setattr__(self, "tags", tuple(self.tags))


@dataclass(frozen=True)
class Finding:
    """One rule match, anchored to a file and line."""

    rule_id: str
    rule_name: str
    severity: Severity
    path: str
    line: int
    column: int
    excerpt: str
    explanation: str
    suggestion: str
    tags: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "tags", tuple(self.tags))


@dataclass
class ScanResult:
    """Everything a reporter needs to render a scan."""

    target: str
    root: Path | None = None
    findings: list[Finding] = field(default_factory=list)
    files_scanned: list[str] = field(default_factory=list)
    files_skipped: int = 0
    rules_loaded: int = 0
    score: int = 100
    grade: str = "ROCm-ready"
