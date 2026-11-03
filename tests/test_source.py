from __future__ import annotations

from pathlib import Path

import pytest

from rocm_check.source import (
    GithubTarget,
    SourceError,
    parse_github_target,
    resolve_source,
)
from rocm_check.scanner import scan_target


@pytest.mark.parametrize(
    ("target", "owner", "repo", "branch"),
    [
        ("https://github.com/ROCm/rocBLAS", "ROCm", "rocBLAS", None),
        ("https://github.com/ROCm/rocBLAS.git", "ROCm", "rocBLAS", None),
        ("https://github.com/ROCm/rocBLAS/", "ROCm", "rocBLAS", None),
        ("https://github.com/ROCm/rocBLAS/tree/develop", "ROCm", "rocBLAS", "develop"),
        ("git@github.com:ROCm/rocBLAS.git", "ROCm", "rocBLAS", None),
        ("ROCm/rocBLAS", "ROCm", "rocBLAS", None),
        ("ROCm/rocBLAS.git", "ROCm", "rocBLAS", None),
    ],
)
def test_parse_github_targets(
    target: str, owner: str, repo: str, branch: str | None
) -> None:
    parsed = parse_github_target(target)

    assert parsed == GithubTarget(owner=owner, repo=repo, branch=branch)
    assert parsed.display == f"https://github.com/{owner}/{repo}"
    assert parsed.clone_url == f"https://github.com/{owner}/{repo}.git"


def test_parse_github_target_returns_none_for_plain_paths() -> None:
    assert parse_github_target("some/local/project") is None
    assert parse_github_target(".") is None


@pytest.mark.parametrize(
    "target",
    [
        "https://gitlab.com/owner/repo",
        "https://example.com/archive.zip",
        "git@gitlab.com:owner/repo.git",
    ],
)
def test_parse_github_target_rejects_other_hosts(target: str) -> None:
    with pytest.raises(SourceError, match="GitHub"):
        parse_github_target(target)


def test_resolve_local_directory(tmp_path: Path) -> None:
    with resolve_source(str(tmp_path)) as source:
        assert source.root == tmp_path.resolve()
        assert source.label == str(tmp_path.resolve())
        assert source.is_remote is False


def test_resolve_rejects_files(tmp_path: Path) -> None:
    path = tmp_path / "file.py"
    path.write_text("x = 1\n", encoding="utf-8")

    with pytest.raises(SourceError, match="not a directory"):
        with resolve_source(str(path)):
            pass


def test_resolve_missing_target(tmp_path: Path) -> None:
    with pytest.raises(SourceError, match="not found"):
        with resolve_source(str(tmp_path / "missing")):
            pass


def test_remote_target_clones_into_a_temp_dir(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}

    def fake_clone(target: GithubTarget, destination: Path) -> None:
        captured["target"] = target
        destination.mkdir(parents=True)
        (destination / "train.py").write_text("print('hi')\n", encoding="utf-8")

    monkeypatch.setattr("rocm_check.source._clone_repo", fake_clone)

    with resolve_source("https://github.com/acme/widget") as source:
        assert source.is_remote is True
        assert source.label == "https://github.com/acme/widget"
        assert (source.root / "train.py").exists()
        cloned_root = source.root
        assert captured["target"] == GithubTarget("acme", "widget")

    assert not cloned_root.exists()


def test_clone_reports_missing_git(monkeypatch: pytest.MonkeyPatch) -> None:
    def raise_missing(*args: object, **kwargs: object) -> None:
        raise FileNotFoundError("git")

    monkeypatch.setattr("rocm_check.source.subprocess.run", raise_missing)

    with pytest.raises(SourceError, match="git"):
        with resolve_source("https://github.com/acme/widget"):
            pass


def test_clone_reports_git_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    class Result:
        returncode = 1
        stdout = ""
        stderr = "repository not found"

    monkeypatch.setattr(
        "rocm_check.source.subprocess.run", lambda *args, **kwargs: Result()
    )

    with pytest.raises(SourceError, match="repository not found"):
        with resolve_source("https://github.com/acme/widget"):
            pass


def test_clone_uses_requested_branch(monkeypatch: pytest.MonkeyPatch) -> None:
    commands: list[list[str]] = []

    def fake_run(command: list[str], **kwargs: object) -> object:
        commands.append(command)

        class Result:
            returncode = 0
            stdout = ""
            stderr = ""

        return Result()

    monkeypatch.setattr("rocm_check.source.subprocess.run", fake_run)

    with resolve_source("https://github.com/acme/widget/tree/dev"):
        pass

    assert commands[0][:4] == ["git", "clone", "--depth", "1"]
    assert commands[0][4:6] == ["--branch", "dev"]
    assert commands[0][-2].endswith("widget.git")


def test_scan_target_scans_local_directory(tmp_path: Path) -> None:
    (tmp_path / "train.py").write_text("torch.cuda.is_available()\n", encoding="utf-8")

    result = scan_target(str(tmp_path))

    assert result.target == str(tmp_path.resolve())
    assert [finding.rule_id for finding in result.findings] == ["torch-cuda-call"]
