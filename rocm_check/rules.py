"""Load and validate detection rules from YAML."""

from __future__ import annotations

import re
from importlib import resources
from pathlib import Path
from typing import Iterable, Sequence

import yaml

from .models import Rule, Severity

RULE_VERSION = 1
RULE_TYPES = {"regex", "dependency", "file"}
REQUIRED_FIELDS = ("id", "name", "severity", "type", "explanation", "suggestion")
ALLOWED_FIELDS = set(REQUIRED_FIELDS) | {"pattern", "file_globs", "flags", "tags"}
FLAG_MAP = {
    "ignorecase": re.IGNORECASE,
    "multiline": re.MULTILINE,
    "dotall": re.DOTALL,
}


class RuleError(ValueError):
    """Raised when a rules file is missing, malformed, or invalid."""


def compile_flags(flags: Sequence[str]) -> int:
    """Translate YAML flag names into :mod:`re` flags."""

    value = 0
    for flag in flags:
        value |= FLAG_MAP[flag]
    return value


def load_builtin_rules() -> list[Rule]:
    """Load the rules shipped inside the package."""

    try:
        text = (
            resources.files("rocm_check")
            .joinpath("rules.yaml")
            .read_text(encoding="utf-8")
        )
    except (OSError, ModuleNotFoundError) as exc:  # pragma: no cover - packaging error
        raise RuleError(f"could not read built-in rules.yaml: {exc}") from exc
    return load_rules_from_text(text, source="built-in rules")


def load_rules_file(path: str | Path) -> list[Rule]:
    """Load one external YAML rules file."""

    path = Path(path).expanduser()
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise RuleError(f"{path}: could not read rules file: {exc}") from exc
    return load_rules_from_text(text, source=str(path))


def load_rules_from_text(text: str, source: str = "<rules>") -> list[Rule]:
    """Parse and validate a YAML rules document."""

    try:
        document = yaml.safe_load(text)
    except yaml.YAMLError as exc:
        raise RuleError(f"{source}: invalid YAML: {exc}") from exc

    if not isinstance(document, dict):
        raise RuleError(f"{source}: expected a mapping with a 'rules' list")

    version = document.get("version", RULE_VERSION)
    if version != RULE_VERSION:
        raise RuleError(
            f"{source}: unsupported rules version {version!r}; expected {RULE_VERSION}"
        )

    raw_rules = document.get("rules")
    if not isinstance(raw_rules, list):
        raise RuleError(f"{source}: expected a 'rules' list")

    rules: list[Rule] = []
    seen: set[str] = set()
    for index, raw in enumerate(raw_rules):
        rule = _parse_rule(raw, source, index)
        if rule.id in seen:
            raise RuleError(f"{source}: duplicate rule id '{rule.id}'")
        seen.add(rule.id)
        rules.append(rule)
    return rules


def load_rules(
    custom_paths: Iterable[str | Path] = (),
    *,
    include_builtin: bool = True,
) -> list[Rule]:
    """Load built-in rules plus any custom files; later rules override by id."""

    rules: list[Rule] = []
    if include_builtin:
        rules.extend(load_builtin_rules())
    for path in custom_paths:
        rules.extend(load_rules_file(path))

    merged: dict[str, Rule] = {}
    for rule in rules:
        merged[rule.id] = rule
    return list(merged.values())


def _parse_rule(raw: object, source: str, index: int) -> Rule:
    where = f"{source}: rule #{index + 1}"
    if not isinstance(raw, dict):
        raise RuleError(f"{where}: expected a mapping")

    unknown = sorted(set(raw) - ALLOWED_FIELDS)
    if unknown:
        raise RuleError(f"{where}: unknown field(s): {', '.join(unknown)}")

    for name in REQUIRED_FIELDS:
        value = raw.get(name)
        if not isinstance(value, str) or not value.strip():
            raise RuleError(f"{where}: missing or empty '{name}'")

    rule_type = raw["type"].strip().lower()
    if rule_type not in RULE_TYPES:
        raise RuleError(
            f"{where}: invalid type {raw['type']!r}; expected one of "
            f"{', '.join(sorted(RULE_TYPES))}"
        )

    try:
        severity = Severity(raw["severity"].strip().lower())
    except ValueError as exc:
        raise RuleError(
            f"{where}: invalid severity {raw['severity']!r}; expected "
            "blocker, warning, or info"
        ) from exc

    pattern = raw.get("pattern")
    if pattern is None:
        pattern = ""
    if not isinstance(pattern, str):
        raise RuleError(f"{where}: 'pattern' must be a string")
    if rule_type in {"regex", "dependency"} and not pattern.strip():
        raise RuleError(f"{where}: 'pattern' is required for type '{rule_type}'")

    raw_flags = raw.get("flags", [])
    if not isinstance(raw_flags, list) or not all(
        isinstance(flag, str) for flag in raw_flags
    ):
        raise RuleError(f"{where}: 'flags' must be a list of strings")
    flags: list[str] = []
    for flag in raw_flags:
        key = flag.strip().lower()
        if key not in FLAG_MAP:
            raise RuleError(
                f"{where}: unknown flag {flag!r}; expected "
                f"{', '.join(sorted(FLAG_MAP))}"
            )
        flags.append(key)

    if rule_type in {"regex", "dependency"}:
        try:
            re.compile(pattern, compile_flags(flags))
        except re.error as exc:
            raise RuleError(f"{where}: invalid pattern: {exc}") from exc

    raw_globs = raw.get("file_globs", ["*"])
    if (
        not isinstance(raw_globs, list)
        or not raw_globs
        or not all(isinstance(glob, str) and glob.strip() for glob in raw_globs)
    ):
        raise RuleError(f"{where}: 'file_globs' must be a non-empty list of strings")

    raw_tags = raw.get("tags", [])
    if not isinstance(raw_tags, list) or not all(
        isinstance(tag, str) and tag.strip() for tag in raw_tags
    ):
        raise RuleError(f"{where}: 'tags' must be a list of strings")
    tags: list[str] = []
    for tag in raw_tags:
        value = tag.strip()
        if value not in tags:
            tags.append(value)

    return Rule(
        id=raw["id"].strip(),
        name=raw["name"].strip(),
        severity=severity,
        type=rule_type,
        explanation=raw["explanation"].strip(),
        suggestion=raw["suggestion"].strip(),
        pattern=pattern,
        file_globs=tuple(glob.strip() for glob in raw_globs),
        flags=tuple(flags),
        tags=tuple(tags),
    )
