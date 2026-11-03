# rocm-check

Static scanner for CUDA-specific code that would break on AMD ROCm.

`rocm-check` walks a local directory or a GitHub repository, applies a YAML rule
set, and reports every place that assumes NVIDIA's stack: `torch.cuda` calls,
hard-coded `"cuda"` device strings, `.cu` files, `nvcc`, CUDA base images, and
CUDA-only Python packages. Each finding includes a severity, an explanation, and
a ROCm-compatible fix. Results come with a 0-100 compatibility score and can be
printed in the terminal, written as JSON for automation, or saved as a
self-contained HTML report. Plain `torch.cuda` usage is reported as info and
does not lower the score; CUDA-specific features such as NVTX, graph capture, and
`torch.cuda.amp` stay warnings.

## What it scans

- Python: `.py`, `.pyi`
- CUDA/C++: `.cu`, `.cuh`, `.cpp`, `.cc`, `.cxx`, `.h`, `.hpp`
- Dependencies and builds: `requirements*.txt`, `setup.py`, `setup.cfg`,
  `pyproject.toml`
- Containers: `Dockerfile`, `Dockerfile.*`, `*.Dockerfile`

It skips `.git`, virtualenvs, `node_modules`, caches, `build`, `dist`, and other
generated directories by default. Files over 2 MB and binary files are skipped.

## Install

Python 3.9 or newer is required.

Install from PyPI with pip or pipx:

```bash
pip install rocm-check
pipx install rocm-check
```

To install from a checkout instead:

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
python -m pip install .
# or install the command in an isolated environment:
pipx install .
```

Both installation methods provide two equivalent commands, `rocm_check` and
`rocm-check`. When installed with pipx or into a user or global Python
environment, either command works from any directory without activating a
virtual environment. On Windows the launchers live in the Python `Scripts`
directory; add that directory to `PATH` if the command is not found.

For development, install the editable package with the test and release tools:

```bash
python -m pip install -e ".[dev]"
```

Running `python -m rocm_check` also works, but the installed commands are the
supported interface.

## Usage

`rocm_check` and `rocm-check` are the same command. Examples use `rocm-check`.

```bash
# Current directory (the target defaults to ".")
rocm-check .

# A project directory
rocm-check path/to/project

# GitHub URL, GitHub SSH URL, or owner/repo shorthand
rocm-check https://github.com/ROCm/rocBLAS
rocm-check git@github.com:ROCm/rocBLAS.git
rocm-check ROCm/rocBLAS

# Scan a branch
rocm-check https://github.com/ROCm/rocBLAS/tree/develop

# Skip vendored code
rocm-check . --exclude third_party --exclude benchmarks

# Terminal report (default)
rocm-check ./project

# JSON to stdout
rocm-check ./project --format json

# JSON to a file
rocm-check ./project --format json --output report.json

# HTML report, written to rocm-report.html by default
rocm-check ./project --format html

# HTML report with an explicit path
rocm-check ./project --format html --output report.html

# Fail CI on warnings as well as blockers
rocm-check ./project --fail-on warning

# high, medium, and low are aliases for blocker, warning, and info
rocm-check ./project --fail-on high

# Add project-specific rules, or replace the built-in set entirely
rocm-check . --rules team-rules.yaml
rocm-check . --rules team-rules.yaml --rules-only

# Legacy report flags remain available
rocm-check . --json report.json --markdown report.md
```

| Option | Meaning |
| --- | --- |
| `target` | Local directory, GitHub URL, or `owner/repo` shorthand. Defaults to `.` |
| `--format terminal\|json\|html\|markdown` | Output format. Defaults to `terminal` |
| `--output PATH` | Write the selected format to `PATH`. Use `-` for stdout |
| `--json PATH` | Legacy shorthand for a JSON report |
| `--markdown PATH` | Legacy shorthand for a Markdown report |
| `--rules PATH` | Add a custom YAML rule file. Repeatable |
| `--rules-only` | Use only files passed with `--rules` |
| `--exclude GLOB` | Exclude matching paths. Repeatable |
| `--fail-on none\|blocker\|warning\|info` | Exit code threshold. Defaults to `blocker` |
| `--no-color` | Disable ANSI colors |
| `--version`, `--help` | Print version or usage information |

`--fail-on` also accepts `high`, `medium`, and `low` as aliases for `blocker`,
`warning`, and `info`. `--output` is not valid with `--format terminal`, since
the terminal report already writes to stdout. `--format json` and
`--format markdown` write to stdout when `--output` is omitted. `--format html`
writes to `rocm-report.html` when `--output` is omitted. Use `--output -` to
write HTML to stdout instead. Parent directories are created automatically, and
paths may contain spaces.

GitHub targets are shallow-cloned with `git` into a temporary directory, scanned,
and deleted. Git must be on `PATH`; private repositories work when your normal
git credentials are configured. An existing local path always wins over the
`owner/repo` shorthand.

### Terminal report

Severity and score are the only colored elements. Set `NO_COLOR=1` or pass
`--no-color` when the terminal does not support ANSI colors. This is the report
for `tests/samples/cuda_project`, with most rows omitted:

```text
rocm-check  v0.2.1
Target  tests/samples/cuda_project
Files   6 scanned
Issues  12 blockers   3 warnings   3 info
Score   0/100   CUDA-locked

                                 Findings (18)
 Severity  | Location                 | Finding and Suggested ROCm fix
-----------+--------------------------+--------------------------------
 BLOCKER   | Dockerfile:1             | Starts from an NVIDIA CUDA image. The
           | cuda-docker-base         | image ships the CUDA runtime and
           |                          | drivers, neither of which works with
           |                          | AMD GPUs.
           |                          |   FROM
           |                          | nvidia/cuda:12.1.1-devel-ubuntu22.04
           |                          |   Suggested ROCm fix: Use a ROCm base
           |                          | image (`rocm/pytorch:latest`, ...
...
blocker = will not build or run | warning = likely porting work | info = review on ROCm
```

Findings tagged `hipify-able` show the tag under their rule id.

### HTML report

`--format html` produces one file with the stylesheet and script embedded, so it
opens directly in a browser without a server or network access. The report
contains:

- Overall score, files scanned, blocker, warning, and info counts
- Severity distribution and issue counts by category
- Detected CUDA APIs and CUDA-only dependencies
- File-by-file findings with expandable details and suggested ROCm fixes
- Search, severity filters, file filtering, expand and collapse controls
- Light and dark themes, with the initial mode following the operating system

The default output path is `rocm-report.html` in the current directory. Pass
`--output -` to send the HTML to stdout or `--output path/report.html` to choose
a path. Generated reports are not part of the package.

### Exit codes

| Code | Meaning |
| --- | --- |
| `0` | No findings at or above the `--fail-on` threshold |
| `1` | At least one finding met the threshold |
| `2` | Bad arguments, unresolvable target, invalid rules file, or write error |

`--fail-on` accepts `none`, `blocker` (default), `warning`, or `info`.
`high`, `medium`, and `low` are accepted as aliases for `blocker`, `warning`,
and `info`.

## Compatibility score

Every occurrence is weighted by severity:

```text
penalty = blockers * 10 + warnings * 3 + info * 0
score   = max(0, 100 - min(100, penalty))
```

| Score | Grade |
| --- | --- |
| 90-100 | ROCm-ready |
| 70-89 | Mostly portable |
| 40-69 | Needs porting work |
| 1-39 | Heavily CUDA-bound |
| 0 | CUDA-locked |

The score is a triage signal, not a guarantee. It counts occurrences rather than
unique rules, so a large CUDA codebase can reach 0 quickly. A single blocker
scores 90, a single warning 97, and info findings do not reduce the score.

## Rules

The built-in rule set lives in [`rocm_check/rules.yaml`](rocm_check/rules.yaml)
and contains 35 rules: 22 blockers, 9 warnings, and 4 info.

### Python

| Rule id | Severity | Catches |
| --- | --- | --- |
| `torch-cuda-call` | info | plain `torch.cuda` namespace usage |
| `torch-cuda-nvtx` | warning | `torch.cuda.nvtx` profiling |
| `torch-cuda-graph` | warning | CUDA graph capture (`torch.cuda.CUDAGraph`, `torch.cuda.graph`) |
| `torch-cuda-amp` | warning | `torch.cuda.amp` mixed precision |
| `tensor-dot-cuda` | info | `.cuda()` tensor/module moves |
| `hardcoded-cuda-device` | info | `"cuda"` / `"cuda:0"` device strings |
| `torch-backend-cuda` | warning | `torch.backends.cuda` tuning |
| `cuda-only-import` | blocker | `pycuda`, `cupy`, `numba.cuda`, `cuda` imports |
| `cuda-only-package-import` | blocker | `flash_attn`, `bitsandbytes`, `xformers`, `apex` imports |
| `cuda-visible-devices` | warning | `CUDA_VISIBLE_DEVICES` |
| `cuda-home-env` | warning | `CUDA_HOME`, `CUDA_PATH`, `CUDA_ROOT` |
| `nvidia-smi` | warning | `nvidia-smi` invocations |
| `nccl-reference` | info | direct NCCL references |

### CUDA / C++

| Rule id | Severity | Catches |
| --- | --- | --- |
| `cuda-runtime-api` | blocker | `cudaMalloc`, `cudaMemcpy`, and friends; tagged `hipify-able` |
| `cuda-include` | blocker | CUDA toolkit headers (`cuda_runtime.h`, cuBLAS, cuDNN, ...) |
| `cuda-kernel-launch` | warning | `<<< ... >>>` launches |
| `cuda-source-file` | blocker | `.cu` and `.cuh` files; tagged `hipify-able` |

`hipify-able` findings include suggested `hipify-perl` and `hipify-clang`
commands in their fix text.

### Build and containers

| Rule id | Severity | Catches |
| --- | --- | --- |
| `cuda-nvcc` | blocker | `nvcc` in builds or Dockerfiles |
| `cuda-arch-flags` | warning | `sm_XX`, `compute_XX`, `-gencode`, `-arch` |
| `torch-cuda-extension` | blocker | `CUDAExtension` |
| `cuda-only-package-install` | blocker | CUDA-only packages in Dockerfile `pip install` commands |
| `cuda-docker-base` | blocker | `nvidia/cuda`, `nvcr.io/nvidia`, CUDA-tagged images |
| `cuda-wheel-index` | blocker | PyTorch `/whl/cuXXX` indexes |
| `cuda-version-pin` | blocker | `+cu121`-style wheel pins |

### Dependencies

All dependency rules are blockers and apply to `requirements*.txt`, `setup.py`,
`setup.cfg`, and `pyproject.toml`:

`dep-flash-attn`, `dep-bitsandbytes`, `dep-xformers`, `dep-apex`, `dep-cupy`,
`dep-pycuda`, `dep-cuda-python`, `dep-tensorrt`, `dep-cudnn`,
`dep-onnxruntime-gpu`, `dep-jax-cuda`.

### Custom rules

`--rules path.yaml` loads extra rules after the built-ins. A custom rule with the
same `id` overrides the built-in one. Use `--rules-only` with one or more
`--rules` files to ignore the built-in set completely.

```yaml
version: 1
rules:
  - id: our-training-helper
    name: Internal CUDA-only helper
    severity: blocker
    type: regex
    pattern: >-
      \brequire_cuda_backend\s*\(
    file_globs: ["*.py"]
    flags: ["ignorecase"]
    explanation: >-
      Our helper raises unless an NVIDIA GPU and the CUDA toolkit are present.
    suggestion: >-
      Replace it with a device check that accepts ROCm
      (`torch.version.hip is not None`) and fall back to CPU.
```

Rule fields:

| Field | Required | Meaning |
| --- | --- | --- |
| `id` | yes | Stable unique identifier |
| `name` | yes | Short human-readable label |
| `severity` | yes | `blocker`, `warning`, or `info` |
| `type` | yes | `regex`, `dependency`, or `file` |
| `pattern` | for regex/dependency | Python regular expression |
| `file_globs` | no | Globs matched against the basename or relative path (default `["*"]`) |
| `flags` | no | `ignorecase`, `multiline`, `dotall` |
| `tags` | no | Optional labels, for example `["hipify-able"]` |
| `explanation` | yes | Why it breaks or needs review on ROCm |
| `suggestion` | yes | Concrete ROCm-compatible fix |

Rule types:

- `regex` searches the whole file text.
- `dependency` checks the code part of each line (comments after `#` are
  ignored), which is useful for requirements, TOML, and `setup.py`.
- `file` reports every file whose name or relative path matches `file_globs`,
  anchored at line 1.

A missing field, unknown field or flag, invalid severity/type, duplicate id in
one file, or uncompilable regex produces a clear error and exit code 2.

## JSON and Markdown reports

`--format json` writes a deterministic report to stdout. Add `--output PATH` to
write it to a file. The legacy `--json PATH` flag does the same thing:

```bash
rocm-check tests/samples/cuda_project --format json
rocm-check tests/samples/cuda_project --format json --output report.json
rocm-check tests/samples/cuda_project --json report.json
```

```json
{
  "tool": "rocm-check",
  "version": "0.2.1",
  "target": "tests/samples/cuda_project",
  "score": 0,
  "grade": "CUDA-locked",
  "summary": { "blocker": 12, "warning": 3, "info": 3 },
  "files_scanned": ["Dockerfile", "kernels/ops.cu", "portable.py", "requirements.txt", "setup.py", "train.py"],
  "files_skipped": 0,
  "rules_loaded": 35,
  "findings": [
    {
      "file": "Dockerfile",
      "line": 1,
      "column": 1,
      "severity": "blocker",
      "rule_id": "cuda-docker-base",
      "rule": "CUDA-based Docker image",
      "excerpt": "FROM nvidia/cuda:12.1.1-devel-ubuntu22.04",
      "explanation": "Starts from an NVIDIA CUDA image. ...",
      "suggestion": "Use a ROCm base image (`rocm/pytorch:latest`, ...).",
      "tags": []
    }
  ]
}
```

Findings from `.cu`/`.cuh` files and CUDA Runtime API calls include
`"tags": ["hipify-able"]`.

`--format markdown` or `--markdown PATH` writes a table-based report suitable for
a pull request or CI artifact. Use a file path for all but one report when
combining formats in a single run, because only one report can use stdout.

## CI example

```yaml
name: rocm-check
on: [push, pull_request]

jobs:
  rocm:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.12"
      - run: pip install .
      - run: >-
          rocm-check . --fail-on warning
          --format html --output rocm-report.html
          --json rocm-report.json --markdown rocm-report.md
      - uses: actions/upload-artifact@v4
        if: always()
        with:
          name: rocm-report
          path: |
            rocm-report.html
            rocm-report.json
            rocm-report.md
```

## Limitations

- This is static, regex-based analysis. A CUDA mention inside a comment or an
  unrelated string can be reported; the location and excerpt make it quick to
  judge.
- Plain `torch.cuda` calls, `.cuda()` moves, and `"cuda"` device strings are
  info: ROCm PyTorch implements the CUDA API on top of HIP, so much of it works.
  Info findings do not reduce the score; they surface NVIDIA assumptions for
  review and testing. CUDA-specific features such as NVTX, graph capture, and
  `torch.cuda.amp` stay warnings.
- Dependency rules reflect common upstream packaging. A ROCm fork or a recent
  multi-backend release may still be usable; verify the version.
- It does not resolve transitive dependencies, execute builds, inspect compiled
  binaries, or probe hardware.
- GitHub scans require `git`, clone one branch shallowly, and do not fall back to
  tarball downloads.
- The default include list is intentionally narrow. New file types require a
  small code change; new detections for the supported file types only require
  YAML.

## Build a release

The version is defined once in `rocm_check/__init__.py`; `pyproject.toml` reads it
from there. Build a wheel and a source distribution with:

```bash
python -m pip install -e ".[dev]"
python -m build
python -m twine check dist/*
```

The wheel includes `rules.yaml`, the HTML template, and the CSS and JavaScript
assets. Releases are published at https://pypi.org/project/rocm-check/. To upload
a checked build by hand, configure a PyPI API token and run:

```bash
python -m twine upload dist/*
```

The workflow in `.github/workflows/publish.yml` builds and uploads a release with
PyPI trusted publishing. The trusted publisher is configured for
`thusharpradeep/rocm-check`, workflow `publish.yml`, and environment `pypi`. The
workflow runs when a GitHub release is published or when it is started manually.

## npm wrapper

There is no npm package. A JavaScript wrapper would still require a working
Python installation and would add a second release process without exposing the
scanner to new users. PyPI and pipx are the supported distribution paths. If a
wrapper is added later, it should only locate and invoke the installed Python
command, and it should fail with a clear message when Python is unavailable.

## Development

```bash
python -m pip install -e ".[dev]"
python -m pytest
```

The test suite covers the rules engine, scanner, scoring, source resolution
(GitHub cloning is mocked), all report formats, the CLI, packaging metadata, and
the sample project under `tests/samples/`. To see the scanner in action:

```bash
rocm-check tests/samples/cuda_project
```

Project layout:

```text
rocm_check/
  cli.py             argument parsing, output routing, exit codes
  scanner.py         file discovery, rule application, finding deduplication
  rules.py           YAML loading, validation, merging
  rules.yaml         built-in detections
  scoring.py         0-100 compatibility score
  source.py          local paths, GitHub parsing, shallow clone
  report.py          terminal, JSON, and Markdown rendering
  html_report.py     HTML report data preparation
  templates/         self-contained report shell
  static/            report stylesheet and interactions
tests/
  samples/cuda_project/
```

## Uninstall

```bash
pip uninstall rocm-check
pipx uninstall rocm-check
```

## License

MIT. See [LICENSE](LICENSE).
