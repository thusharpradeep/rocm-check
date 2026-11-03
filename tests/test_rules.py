from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from rocm_check.models import Rule, Severity
from rocm_check.rules import (
    RuleError,
    load_builtin_rules,
    load_rules,
    load_rules_file,
    load_rules_from_text,
)


def make_rule(**overrides: object) -> dict:
    rule = {
        "id": "custom-rule",
        "name": "Custom rule",
        "severity": "warning",
        "type": "regex",
        "pattern": r"cuda:[0-9]",
        "explanation": "A hard-coded device index.",
        "suggestion": "Use a device variable.",
    }
    rule.update(overrides)
    return rule


def write_rules(tmp_path: Path, *rules: dict, version: int = 1) -> Path:
    path = tmp_path / "rules.yaml"
    path.write_text(
        yaml.safe_dump({"version": version, "rules": list(rules)}), encoding="utf-8"
    )
    return path


def test_builtin_rules_cover_required_detections() -> None:
    rules = load_builtin_rules()
    ids = {rule.id for rule in rules}

    assert {
        "torch-cuda-call",
        "tensor-dot-cuda",
        "hardcoded-cuda-device",
        "cuda-runtime-api",
        "cuda-include",
        "cuda-source-file",
        "cuda-nvcc",
        "cuda-docker-base",
        "dep-flash-attn",
        "dep-bitsandbytes",
        "dep-xformers",
        "dep-apex",
    } <= ids
    assert {Severity.BLOCKER, Severity.WARNING, Severity.INFO} == {
        rule.severity for rule in rules
    }
    assert all(isinstance(rule, Rule) for rule in rules)


def test_builtin_rule_ids_are_unique() -> None:
    rules = load_builtin_rules()

    assert len({rule.id for rule in rules}) == len(rules)


def test_plain_cuda_usage_is_info_and_specific_features_stay_warnings() -> None:
    by_id = {rule.id: rule for rule in load_builtin_rules()}

    for rule_id in ("torch-cuda-call", "tensor-dot-cuda", "hardcoded-cuda-device"):
        assert by_id[rule_id].severity is Severity.INFO, rule_id
    for rule_id in ("torch-cuda-nvtx", "torch-cuda-graph", "torch-cuda-amp"):
        assert by_id[rule_id].severity is Severity.WARNING, rule_id

    assert by_id["cuda-source-file"].tags == ("hipify-able",)
    assert by_id["cuda-runtime-api"].tags == ("hipify-able",)


def test_custom_rule_tags_load(tmp_path: Path) -> None:
    path = write_rules(tmp_path, make_rule(tags=["hipify-able"]))

    rule = load_rules([path], include_builtin=False)[0]

    assert rule.tags == ("hipify-able",)


def test_invalid_tags_are_rejected(tmp_path: Path) -> None:
    path = write_rules(tmp_path, make_rule(tags="hipify-able"))

    with pytest.raises(RuleError, match="'tags' must be a list of strings"):
        load_rules_file(path)


def test_custom_regex_rule_loads(tmp_path: Path) -> None:
    path = write_rules(
        tmp_path,
        make_rule(
            id="custom-cuda-string",
            pattern="cuda:[0-9]",
            file_globs=["*.py"],
            flags=["IGNORECASE"],
        ),
    )

    rules = load_rules([path], include_builtin=False)

    assert len(rules) == 1
    rule = rules[0]
    assert rule.id == "custom-cuda-string"
    assert rule.severity is Severity.WARNING
    assert rule.type == "regex"
    assert rule.file_globs == ("*.py",)
    assert rule.flags == ("ignorecase",)
    assert rule.pattern == "cuda:[0-9]"


def test_yaml_can_define_file_rule_without_pattern(tmp_path: Path) -> None:
    path = write_rules(
        tmp_path,
        make_rule(type="file", pattern=None, file_globs=["*.cu"]),
    )

    rule = load_rules([path], include_builtin=False)[0]

    assert rule.type == "file"
    assert rule.pattern == ""


def test_custom_rules_extend_and_override_builtins(tmp_path: Path) -> None:
    path = write_rules(
        tmp_path,
        make_rule(
            id="torch-cuda-call",
            name="Overridden name",
            severity="info",
        ),
    )

    rules = load_rules([path], include_builtin=True)
    by_id = {rule.id: rule for rule in rules}

    assert by_id["torch-cuda-call"].severity is Severity.INFO
    assert by_id["torch-cuda-call"].name == "Overridden name"
    assert "cuda-docker-base" in by_id


def test_load_rules_file_uses_path_in_error(tmp_path: Path) -> None:
    path = tmp_path / "rules.yaml"
    path.write_text("rules: [", encoding="utf-8")

    with pytest.raises(RuleError, match="rules.yaml"):
        load_rules_file(path)


def test_missing_rules_key_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "rules.yaml"
    path.write_text("version: 1\n", encoding="utf-8")

    with pytest.raises(RuleError, match="rules"):
        load_rules_file(path)


def test_unsupported_version_is_rejected(tmp_path: Path) -> None:
    path = write_rules(tmp_path, make_rule(), version=99)

    with pytest.raises(RuleError, match="version"):
        load_rules_file(path)


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"severity": "fatal"}, "severity"),
        ({"type": "telepathy"}, "type"),
        ({"pattern": "("}, "pattern"),
        ({"id": ""}, "id"),
        ({"explanation": ""}, "explanation"),
        ({"unexpected": True}, "unexpected"),
    ],
)
def test_invalid_rules_are_rejected(
    tmp_path: Path, overrides: dict, message: str
) -> None:
    path = write_rules(tmp_path, make_rule(**overrides))

    with pytest.raises(RuleError, match=message):
        load_rules_file(path)


def test_duplicate_ids_in_one_file_are_rejected(tmp_path: Path) -> None:
    path = write_rules(tmp_path, make_rule(), make_rule())

    with pytest.raises(RuleError, match="duplicate"):
        load_rules_file(path)


def test_rules_from_text_reports_source_name() -> None:
    text = yaml.safe_dump({"version": 1, "rules": [make_rule(severity="fatal")]})

    with pytest.raises(RuleError, match="inline-rules"):
        load_rules_from_text(text, source="inline-rules")


def test_unknown_flags_are_rejected(tmp_path: Path) -> None:
    path = write_rules(tmp_path, make_rule(flags=["unicode"]))

    with pytest.raises(RuleError, match="flag"):
        load_rules_file(path)


def test_empty_file_globs_are_rejected(tmp_path: Path) -> None:
    path = write_rules(tmp_path, make_rule(file_globs=[]))

    with pytest.raises(RuleError, match="file_globs"):
        load_rules_file(path)
