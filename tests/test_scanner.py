from __future__ import annotations

from pathlib import Path

import pytest

from rocm_check.models import Finding, Rule, Severity
from rocm_check.rules import load_builtin_rules
from rocm_check.scanner import ScanError, scan_path
from rocm_check.scoring import compatibility_score, grade_for

SAMPLE = Path(__file__).parent / "samples" / "cuda_project"


def write(root: Path, relative: str, text: str) -> Path:
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def findings_for(result, rule_id: str) -> list[Finding]:
    return [finding for finding in result.findings if finding.rule_id == rule_id]


def test_sample_repo_detects_required_cuda_constructs() -> None:
    result = scan_path(SAMPLE, rules=load_builtin_rules())

    rule_ids = {finding.rule_id for finding in result.findings}
    assert {
        "torch-cuda-call",
        "tensor-dot-cuda",
        "hardcoded-cuda-device",
        "cuda-only-package-import",
        "cuda-source-file",
        "cuda-include",
        "cuda-runtime-api",
        "cuda-docker-base",
        "cuda-nvcc",
        "cuda-home-env",
        "torch-cuda-extension",
        "dep-flash-attn",
        "dep-bitsandbytes",
        "cuda-wheel-index",
    } <= rule_ids
    assert "portable.py" not in {finding.path for finding in result.findings}
    assert result.score < 50
    assert result.grade in {"Heavily CUDA-bound", "CUDA-locked"}
    assert result.files_scanned == [
        "Dockerfile",
        "kernels/ops.cu",
        "portable.py",
        "requirements.txt",
        "setup.py",
        "train.py",
    ]


def test_findings_carry_locations_and_metadata() -> None:
    result = scan_path(SAMPLE, rules=load_builtin_rules())

    assert [finding.line for finding in findings_for(result, "tensor-dot-cuda")] == [6]
    assert [finding.line for finding in findings_for(result, "hardcoded-cuda-device")] == [7]
    assert [finding.line for finding in findings_for(result, "torch-cuda-call")] == [9]
    assert [finding.line for finding in findings_for(result, "cuda-runtime-api")] == [10, 12]
    assert [finding.line for finding in findings_for(result, "cuda-source-file")] == [1]

    for finding in result.findings:
        assert finding.severity in Severity
        assert finding.line >= 1
        assert finding.column >= 1
        assert finding.excerpt
        assert finding.explanation
        assert finding.suggestion
        assert "\\" not in finding.path


def test_findings_are_sorted_by_severity_then_location() -> None:
    result = scan_path(SAMPLE, rules=load_builtin_rules())
    rank = {Severity.BLOCKER: 0, Severity.WARNING: 1, Severity.INFO: 2}

    keys = [(rank[finding.severity], finding.path, finding.line) for finding in result.findings]
    assert keys == sorted(keys)


def test_repeated_matches_on_one_line_are_deduped(tmp_path: Path) -> None:
    write(
        tmp_path,
        "dup.py",
        "x = torch.cuda.is_available() or torch.cuda.device_count()\n",
    )

    result = scan_path(tmp_path)

    assert len(findings_for(result, "torch-cuda-call")) == 1


def test_default_excluded_directories_are_not_scanned(tmp_path: Path) -> None:
    write(tmp_path, ".git/bad.py", "torch.cuda.is_available()\n")
    write(tmp_path, "node_modules/bad.py", "torch.cuda.is_available()\n")
    write(tmp_path, "__pycache__/bad.py", "torch.cuda.is_available()\n")
    write(tmp_path, "venv/bad.py", "torch.cuda.is_available()\n")
    write(tmp_path, "ok.py", "print('hello')\n")

    result = scan_path(tmp_path)

    assert result.files_scanned == ["ok.py"]
    assert result.findings == []


def test_extra_exclude_hides_directories(tmp_path: Path) -> None:
    write(tmp_path, "legacy/old.py", "torch.cuda.is_available()\n")
    write(tmp_path, "current.py", "print('ok')\n")

    included = scan_path(tmp_path)
    excluded = scan_path(tmp_path, extra_exclude=("legacy",))

    assert findings_for(included, "torch-cuda-call")
    assert findings_for(excluded, "torch-cuda-call") == []
    assert excluded.files_scanned == ["current.py"]


def test_unsupported_files_are_ignored(tmp_path: Path) -> None:
    write(tmp_path, "notes.md", "Use torch.cuda.is_available() here.\n")
    write(tmp_path, "data.json", '{"device": "cuda"}\n')
    write(tmp_path, "script.sh", "nvidia-smi\n")
    write(tmp_path, "model.cu", "__global__ void k() {}\n")

    result = scan_path(tmp_path)

    assert result.files_scanned == ["model.cu"]
    assert {finding.path for finding in result.findings} == {"model.cu"}


def test_binary_files_are_skipped(tmp_path: Path) -> None:
    (tmp_path / "binary.py").write_bytes(b"\x00\x01torch.cuda\n")

    result = scan_path(tmp_path)

    assert result.files_scanned == []
    assert result.files_skipped == 1


def test_rule_file_globs_restrict_matches(tmp_path: Path) -> None:
    write(tmp_path, "notes.cpp", "// torch.cuda.is_available() is Python-only\n")
    write(tmp_path, "clean.py", "print('ok')\n")

    result = scan_path(tmp_path)

    assert findings_for(result, "torch-cuda-call") == []


def test_custom_rule_objects_are_applied(tmp_path: Path) -> None:
    write(tmp_path, "app.py", "USE_CUDA_ONLY_MARKER = True\n")
    rule = Rule(
        id="custom-marker",
        name="Custom marker",
        severity=Severity.INFO,
        type="regex",
        pattern="CUDA_ONLY_MARKER",
        explanation="Test marker.",
        suggestion="Remove the marker.",
        file_globs=("*.py",),
    )

    result = scan_path(tmp_path, rules=[rule])

    assert len(result.findings) == 1
    finding = result.findings[0]
    assert finding.rule_id == "custom-marker"
    assert finding.severity is Severity.INFO
    assert finding.line == 1


def test_requirements_variants_are_scanned(tmp_path: Path) -> None:
    write(tmp_path, "requirements-dev.txt", "flash-attn>=2\n")

    result = scan_path(tmp_path)

    assert [finding.rule_id for finding in result.findings] == ["dep-flash-attn"]


def test_numba_cuda_import_is_detected(tmp_path: Path) -> None:
    write(tmp_path, "kernels.py", "from numba import cuda\n")

    result = scan_path(tmp_path)

    assert [finding.rule_id for finding in result.findings] == ["cuda-only-import"]


def test_cuda_package_installed_in_dockerfile_is_detected(tmp_path: Path) -> None:
    write(
        tmp_path,
        "Dockerfile",
        "FROM ubuntu:22.04\nRUN pip install flash-attn\n",
    )

    result = scan_path(tmp_path)

    assert [finding.rule_id for finding in result.findings] == [
        "cuda-only-package-install"
    ]


def test_rocm_base_image_is_not_flagged(tmp_path: Path) -> None:
    write(tmp_path, "Dockerfile", "FROM rocm/pytorch:latest\n")

    result = scan_path(tmp_path)

    assert findings_for(result, "cuda-docker-base") == []


def test_dependency_comments_are_ignored(tmp_path: Path) -> None:
    write(tmp_path, "requirements.txt", "# flash-attn is intentionally not installed\n")

    result = scan_path(tmp_path)

    assert result.findings == []


def test_minimal_pytorch_cuda_project_is_rocm_ready(tmp_path: Path) -> None:
    write(
        tmp_path,
        "model.py",
        "import torch\n"
        "\n"
        'device = "cuda"\n'
        "if torch.cuda.is_available():\n"
        "    model = build_model().cuda()\n"
        "    batch = batch.to(device)\n",
    )

    result = scan_path(tmp_path)

    assert result.findings
    assert all(finding.severity is Severity.INFO for finding in result.findings)
    assert not any(finding.severity is Severity.BLOCKER for finding in result.findings)
    assert result.score >= 90
    assert result.grade == "ROCm-ready"


def test_cuda_specific_pytorch_features_stay_warnings(tmp_path: Path) -> None:
    write(
        tmp_path,
        "profile.py",
        "import torch\n"
        "torch.cuda.nvtx.range_push('step')\n"
        "graph = torch.cuda.CUDAGraph()\n"
        "with torch.cuda.amp.autocast():\n"
        "    pass\n",
    )

    result = scan_path(tmp_path)

    warning_ids = {
        finding.rule_id
        for finding in result.findings
        if finding.severity is Severity.WARNING
    }
    assert warning_ids == {"torch-cuda-nvtx", "torch-cuda-graph", "torch-cuda-amp"}
    assert findings_for(result, "torch-cuda-call") == []


def test_cuda_sources_and_runtime_calls_are_tagged_hipify_able(tmp_path: Path) -> None:
    write(
        tmp_path,
        "kernel.cu",
        "#include <cuda_runtime.h>\n"
        "void allocate() {\n"
        "    cudaMalloc(nullptr, 0);\n"
        "}\n",
    )

    result = scan_path(tmp_path)

    source = findings_for(result, "cuda-source-file")[0]
    runtime = findings_for(result, "cuda-runtime-api")[0]
    for finding in (source, runtime):
        assert finding.tags == ("hipify-able",)
        assert "hipify-perl <file>" in finding.suggestion
        assert "hipify-clang <file>" in finding.suggestion


def test_score_and_grade_match_findings() -> None:
    result = scan_path(SAMPLE, rules=load_builtin_rules())

    assert result.score == compatibility_score(result.findings)
    assert result.grade == grade_for(result.score)


def test_scan_path_rejects_files(tmp_path: Path) -> None:
    path = write(tmp_path, "single.py", "print('x')\n")

    with pytest.raises(ScanError, match="directory"):
        scan_path(path)


def test_scan_path_rejects_missing_directories(tmp_path: Path) -> None:
    with pytest.raises(ScanError, match="not found"):
        scan_path(tmp_path / "does-not-exist")
