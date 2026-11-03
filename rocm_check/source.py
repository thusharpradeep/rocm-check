"""Resolve local directories and GitHub URLs into a scannable root."""

from __future__ import annotations

import re
import subprocess
import tempfile
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

CLONE_TIMEOUT_SECONDS = 300

_HTTPS_RE = re.compile(
    r"^https?://(?:www\.)?github\.com/"
    r"(?P<owner>[A-Za-z0-9_.-]+)/"
    r"(?P<repo>[A-Za-z0-9_.-]+?)"
    r"(?:\.git)?"
    r"(?:/tree/(?P<branch>[^/?#]+))?"
    r"/?(?:[?#].*)?$"
)
_SSH_RE = re.compile(
    r"^git@github\.com:"
    r"(?P<owner>[A-Za-z0-9_.-]+)/"
    r"(?P<repo>[A-Za-z0-9_.-]+?)"
    r"(?:\.git)?$"
)
_SHORTHAND_RE = re.compile(
    r"^(?P<owner>[A-Za-z0-9_.-]+)/(?P<repo>[A-Za-z0-9_.-]+)$"
)


class SourceError(ValueError):
    """Raised when a scan target cannot be resolved."""


@dataclass(frozen=True)
class GithubTarget:
    owner: str
    repo: str
    branch: str | None = None

    @property
    def display(self) -> str:
        return f"https://github.com/{self.owner}/{self.repo}"

    @property
    def clone_url(self) -> str:
        return f"{self.display}.git"


@dataclass(frozen=True)
class Source:
    root: Path
    label: str
    is_remote: bool


def parse_github_target(target: str) -> GithubTarget | None:
    """Parse GitHub URL / SSH / ``owner/repo`` forms; return None for paths."""

    text = target.strip()
    if not text:
        return None

    match = _HTTPS_RE.match(text) or _SSH_RE.match(text)
    if match is not None:
        repo = _strip_git_suffix(match.group("repo"))
        return GithubTarget(
            owner=match.group("owner"),
            repo=repo,
            branch=match.groupdict().get("branch"),
        )

    if "://" in text or text.startswith("git@"):
        raise SourceError(
            f"only GitHub repository URLs are supported, got: {target}"
        )

    match = _SHORTHAND_RE.match(text)
    if match is not None:
        return GithubTarget(
            owner=match.group("owner"),
            repo=_strip_git_suffix(match.group("repo")),
        )
    return None


@contextmanager
def resolve_source(target: str) -> Iterator[Source]:
    """Yield a source directory; remote repositories are cloned and cleaned up."""

    local_path = Path(target).expanduser()
    if local_path.exists():
        if not local_path.is_dir():
            raise SourceError(f"target is not a directory: {local_path}")
        resolved = local_path.resolve()
        yield Source(root=resolved, label=str(resolved), is_remote=False)
        return

    github = parse_github_target(target)
    if github is None:
        raise SourceError(f"target not found: {target}")

    with tempfile.TemporaryDirectory(prefix="rocm-check-") as temporary:
        destination = Path(temporary) / "repo"
        _clone_repo(github, destination)
        yield Source(root=destination, label=github.display, is_remote=True)


def _clone_repo(target: GithubTarget, destination: Path) -> None:
    command = ["git", "clone", "--depth", "1"]
    if target.branch:
        command.extend(["--branch", target.branch])
    command.extend([target.clone_url, str(destination)])

    try:
        completed = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=CLONE_TIMEOUT_SECONDS,
        )
    except FileNotFoundError as exc:
        raise SourceError(
            "git is required to scan GitHub URLs but was not found on PATH"
        ) from exc
    except subprocess.TimeoutExpired as exc:
        raise SourceError(
            f"git clone timed out after {CLONE_TIMEOUT_SECONDS} seconds"
        ) from exc

    if completed.returncode != 0:
        detail = (completed.stderr or completed.stdout or "").strip()
        raise SourceError(
            f"git clone failed for {target.display}: {detail or 'unknown error'}"
        )


def _strip_git_suffix(repo: str) -> str:
    return repo[:-4] if repo.endswith(".git") else repo
