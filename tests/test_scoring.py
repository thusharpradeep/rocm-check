from __future__ import annotations

import pytest

from rocm_check.models import Finding, Severity
from rocm_check.scoring import compatibility_score, grade_for


def finding(severity: Severity, path: str = "train.py", line: int = 1) -> Finding:
    return Finding(
        rule_id=f"rule-{severity.value}",
        rule_name="Test rule",
        severity=severity,
        path=path,
        line=line,
        column=1,
        excerpt="x",
        explanation="x",
        suggestion="x",
    )


def test_no_findings_scores_100() -> None:
    assert compatibility_score([]) == 100


@pytest.mark.parametrize(
    ("severities", "expected"),
    [
        ([Severity.BLOCKER], 90),
        ([Severity.WARNING], 97),
        ([Severity.INFO], 100),
        ([Severity.BLOCKER, Severity.WARNING, Severity.INFO], 87),
        ([Severity.BLOCKER] * 20, 0),
    ],
)
def test_score_is_weighted_and_never_negative(
    severities: list[Severity], expected: int
) -> None:
    assert compatibility_score([finding(severity) for severity in severities]) == expected


def test_info_findings_do_not_reduce_the_score() -> None:
    assert compatibility_score([finding(Severity.INFO) for _ in range(50)]) == 100


@pytest.mark.parametrize(
    ("score", "expected"),
    [
        (100, "ROCm-ready"),
        (90, "ROCm-ready"),
        (89, "Mostly portable"),
        (70, "Mostly portable"),
        (69, "Needs porting work"),
        (40, "Needs porting work"),
        (39, "Heavily CUDA-bound"),
        (1, "Heavily CUDA-bound"),
        (0, "CUDA-locked"),
    ],
)
def test_grade_boundaries(score: int, expected: str) -> None:
    assert grade_for(score) == expected
