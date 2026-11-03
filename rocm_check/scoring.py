"""Compatibility scoring.

The score is a simple, explainable penalty model. Every finding subtracts
points based on severity; the total is capped so a very CUDA-heavy repository
lands at 0 rather than going negative.
"""

from __future__ import annotations

from typing import Iterable

from .models import Finding, Severity

PENALTIES = {
    Severity.BLOCKER: 10,
    Severity.WARNING: 3,
    Severity.INFO: 0,
}


def compatibility_score(findings: Iterable[Finding]) -> int:
    """Return a 0-100 ROCm compatibility score."""

    penalty = sum(PENALTIES[finding.severity] for finding in findings)
    return max(0, 100 - min(100, penalty))


def grade_for(score: int) -> str:
    """Map a numeric score to a short human label."""

    if score >= 90:
        return "ROCm-ready"
    if score >= 70:
        return "Mostly portable"
    if score >= 40:
        return "Needs porting work"
    if score > 0:
        return "Heavily CUDA-bound"
    return "CUDA-locked"
