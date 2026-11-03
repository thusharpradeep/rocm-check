from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from rocm_check import __version__
from rocm_check.cli import main

SAMPLE = Path(__file__).parent / "samples" / "cuda_project"


def test_sample_scan_exits_1_on_blockers(capsys: pytest.CaptureFixture[str]) -> None:
    code = main([str(SAMPLE)])

    captured = capsys.readouterr()
    assert code == 1
    assert "rocm-check" in captured.out
    assert "CUDA-locked" in captured.out


def test_json_stdout_is_machine_readable(
    capsys: pytest.CaptureFixture[str],
) -> None:
    code = main([str(SAMPLE), "--json", "-", "--fail-on", "none"])

    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert code == 0
    assert payload["tool"] == "rocm-check"
    assert payload["findings"]
    assert captured.err == ""


def test_exports_json_file(tmp_path: Path) -> None:
    output = tmp_path / "reports" / "report.json"

    code = main([str(SAMPLE), "--json", str(output), "--fail-on", "none"])

    assert code == 0
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["score"] == 0
    assert len(payload["findings"]) == 18


def test_exports_markdown_file(tmp_path: Path) -> None:
    output = tmp_path / "reports" / "report.md"

    code = main([str(SAMPLE), "--markdown", str(output), "--fail-on", "none"])

    assert code == 0
    assert output.read_text(encoding="utf-8").startswith("# ROCm Compatibility Report")


def test_markdown_stdout(capsys: pytest.CaptureFixture[str]) -> None:
    code = main([str(SAMPLE), "--markdown", "-", "--fail-on", "none"])

    captured = capsys.readouterr()
    assert code == 0
    assert captured.out.startswith("# ROCm Compatibility Report")
    assert "rocm-check" not in captured.out.splitlines()[0]


def test_clean_repository_exits_0(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    target = tmp_path / "repo"
    target.mkdir()
    (target / "app.py").write_text("print('hello')\n", encoding="utf-8")
    output = tmp_path / "report.json"

    code = main([str(target), "--json", str(output)])

    payload = json.loads(output.read_text(encoding="utf-8"))
    assert code == 0
    assert payload["score"] == 100
    assert payload["findings"] == []


def test_fail_on_warning_threshold(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    target = tmp_path / "repo"
    target.mkdir()
    (target / "app.py").write_text(
        "torch.backends.cuda.matmul.allow_tf32 = True\n", encoding="utf-8"
    )

    assert main([str(target), "--json", "-", "--fail-on", "blocker"]) == 0
    capsys.readouterr()
    assert main([str(target), "--json", "-", "--fail-on", "warning"]) == 1
    capsys.readouterr()


def test_info_findings_do_not_fail_warning_threshold(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    target = tmp_path / "repo"
    target.mkdir()
    (target / "app.py").write_text('device = "cuda:0"\n', encoding="utf-8")

    code = main([str(target), "--json", "-", "--fail-on", "warning"])
    payload = json.loads(capsys.readouterr().out)

    assert code == 0
    assert payload["summary"] == {"blocker": 0, "warning": 0, "info": 1}
    assert main([str(target), "--json", "-", "--fail-on", "info"]) == 1
    capsys.readouterr()


def test_fail_on_none_always_exits_0(capsys: pytest.CaptureFixture[str]) -> None:
    code = main([str(SAMPLE), "--json", "-", "--fail-on", "none"])

    capsys.readouterr()
    assert code == 0


def test_exclude_flag_reaches_scanner(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    target = tmp_path / "repo"
    (target / "legacy").mkdir(parents=True)
    (target / "legacy" / "old.py").write_text(
        "torch.cuda.is_available()\n", encoding="utf-8"
    )
    (target / "app.py").write_text("print('ok')\n", encoding="utf-8")

    code = main(
        [
            str(target),
            "--exclude",
            "legacy",
            "--json",
            "-",
            "--fail-on",
            "none",
        ]
    )

    payload = json.loads(capsys.readouterr().out)
    assert code == 0
    assert payload["files_scanned"] == ["app.py"]
    assert payload["findings"] == []


def test_missing_target_returns_2(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    code = main([str(tmp_path / "missing")])

    captured = capsys.readouterr()
    assert code == 2
    assert "error" in captured.err
    assert "not found" in captured.err


def test_rules_only_without_rules_returns_2(
    capsys: pytest.CaptureFixture[str],
) -> None:
    code = main([".", "--rules-only"])

    captured = capsys.readouterr()
    assert code == 2
    assert "--rules" in captured.err


def test_custom_rules_file(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    target = tmp_path / "repo"
    target.mkdir()
    (target / "app.py").write_text("CUSTOM_MARKER = 1\n", encoding="utf-8")
    rules_path = tmp_path / "custom.yaml"
    rules_path.write_text(
        yaml.safe_dump(
            {
                "version": 1,
                "rules": [
                    {
                        "id": "custom-marker",
                        "name": "Custom marker",
                        "severity": "info",
                        "type": "regex",
                        "pattern": "CUSTOM_MARKER",
                        "file_globs": ["*.py"],
                        "explanation": "Test marker.",
                        "suggestion": "Remove it.",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    code = main(
        [
            str(target),
            "--rules",
            str(rules_path),
            "--rules-only",
            "--json",
            "-",
            "--fail-on",
            "none",
        ]
    )

    payload = json.loads(capsys.readouterr().out)
    assert code == 0
    assert [finding["rule_id"] for finding in payload["findings"]] == [
        "custom-marker"
    ]


def test_invalid_rules_file_returns_2(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    target = tmp_path / "repo"
    target.mkdir()
    (target / "app.py").write_text("print('x')\n", encoding="utf-8")
    rules_path = tmp_path / "broken.yaml"
    rules_path.write_text("rules: [", encoding="utf-8")

    code = main([str(target), "--rules", str(rules_path), "--json", "-"])

    captured = capsys.readouterr()
    assert code == 2
    assert "invalid YAML" in captured.err


def test_two_stdout_exports_are_rejected(
    capsys: pytest.CaptureFixture[str],
) -> None:
    code = main([str(SAMPLE), "--json", "-", "--markdown", "-"])

    captured = capsys.readouterr()
    assert code == 2
    assert "stdout" in captured.err


def test_version_flag(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as excinfo:
        main(["--version"])

    assert excinfo.value.code == 0
    assert __version__ in capsys.readouterr().out


def test_help_lists_output_options(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as excinfo:
        main(["--help"])

    output = capsys.readouterr().out
    assert excinfo.value.code == 0
    assert "--format" in output
    assert "--output" in output
    assert "--no-color" in output


def test_format_json_writes_to_stdout(
    capsys: pytest.CaptureFixture[str],
) -> None:
    code = main([str(SAMPLE), "--format", "json", "--fail-on", "none"])

    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert code == 0
    assert payload["tool"] == "rocm-check"
    assert payload["summary"]["blocker"] == 12
    assert "Findings" not in captured.out
    assert captured.err == ""


def test_format_json_writes_to_output_file(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    output = tmp_path / "reports" / "report.json"

    code = main(
        [
            str(SAMPLE),
            "--format",
            "json",
            "--output",
            str(output),
            "--fail-on",
            "none",
        ]
    )

    captured = capsys.readouterr()
    assert code == 0
    assert captured.out == ""
    assert "wrote JSON report" in captured.err
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["summary"]["blocker"] == 12


def test_format_markdown_writes_to_stdout(
    capsys: pytest.CaptureFixture[str],
) -> None:
    code = main([str(SAMPLE), "--format", "markdown", "--fail-on", "none"])

    captured = capsys.readouterr()
    assert code == 0
    assert captured.out.startswith("# ROCm Compatibility Report")
    assert captured.err == ""


def test_format_html_writes_to_stdout(
    capsys: pytest.CaptureFixture[str],
) -> None:
    code = main(
        [
            str(SAMPLE),
            "--format",
            "html",
            "--output",
            "-",
            "--fail-on",
            "none",
        ]
    )

    captured = capsys.readouterr()
    assert code == 0
    assert captured.out.startswith("<!doctype html>")
    assert captured.err == ""


def test_format_html_writes_to_output_file(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    output = tmp_path / "report.html"

    code = main(
        [
            str(SAMPLE),
            "--format",
            "html",
            "--output",
            str(output),
            "--fail-on",
            "none",
        ]
    )

    captured = capsys.readouterr()
    assert code == 0
    assert output.read_text(encoding="utf-8").startswith("<!doctype html>")
    assert "wrote HTML report" in captured.err


def test_format_html_uses_default_output(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.chdir(tmp_path)

    code = main([str(SAMPLE), "--format", "html", "--fail-on", "none"])

    captured = capsys.readouterr()
    output = tmp_path / "rocm-report.html"
    assert code == 0
    assert output.is_file()
    assert "wrote HTML report" in captured.err


def test_output_is_rejected_for_terminal_format(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    code = main(
        [
            str(SAMPLE),
            "--format",
            "terminal",
            "--output",
            str(tmp_path / "report.txt"),
            "--fail-on",
            "none",
        ]
    )

    captured = capsys.readouterr()
    assert code == 2
    assert "--output" in captured.err
    assert "terminal" in captured.err


def test_format_json_stdout_conflicts_with_legacy_stdout(
    capsys: pytest.CaptureFixture[str],
) -> None:
    code = main(
        [
            str(SAMPLE),
            "--format",
            "json",
            "--json",
            "-",
            "--fail-on",
            "none",
        ]
    )

    captured = capsys.readouterr()
    assert code == 2
    assert "stdout" in captured.err


def test_fail_on_high_alias(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    target = tmp_path / "clean"
    target.mkdir()
    (target / "app.py").write_text("print('ok')\n", encoding="utf-8")

    assert main([str(target), "--format", "json", "--fail-on", "high"]) == 0
    capsys.readouterr()
    assert main([str(SAMPLE), "--format", "json", "--fail-on", "high"]) == 1
    capsys.readouterr()


def test_output_path_with_spaces_and_windows_separators(
    tmp_path: Path,
) -> None:
    target = tmp_path / "project with spaces"
    target.mkdir()
    (target / "app.py").write_text(
        "torch.cuda.is_available()\n", encoding="utf-8"
    )
    output = tmp_path / "reports" / "report with spaces.json"

    code = main(
        [
            str(target),
            "--format",
            "json",
            "--output",
            str(output),
            "--fail-on",
            "none",
        ]
    )

    payload = json.loads(output.read_text(encoding="utf-8"))
    assert code == 0
    assert payload["summary"] == {"blocker": 0, "warning": 0, "info": 1}
