"""Real SWE-bench Verified scoring (runs locally, where the network is open).

This is the piece the Cloud/Web sandbox couldn't run: HuggingFace and arbitrary
GitHub clones are blocked there. On a normal machine it works.

Per instance it: clones the repo @ base_commit, builds a venv, `pip install -e .`,
lets the agent edit the source in place, then scores the SWE-bench way — apply the
dataset's gold *test* patch and require the specific FAIL_TO_PASS tests to pass and
PASS_TO_PASS tests to stay green. Instances whose env won't build are skipped.

Best-effort: without the official per-version Docker specs, some installs will fail;
those instances are skipped and reported rather than faked.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

from fusion.workspace import Workspace

# Repos that install cleanly from source with plain pip (no C/build toolchain).
# Avoid matplotlib / scikit-learn / seaborn / astropy (compiled deps).
ALLOWLIST = [
    "psf/requests", "pallets/flask", "pytest-dev/pytest", "pydata/xarray",
    "pylint-dev/pylint", "sphinx-doc/sphinx", "sympy/sympy", "django/django",
]

CACHE = Path.home() / ".cache" / "fusion_swebench"


def _run(cmd, cwd=None, timeout=900):
    return subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, timeout=timeout)


def _prepare(row: dict) -> dict | None:
    """Clone + venv + install one instance. Returns a task dict, or None if it won't build."""
    iid = row["instance_id"]
    d = CACHE / iid
    venv_py = d / ".venv" / "bin" / "python"
    marker = d / ".ready"
    if marker.exists() and venv_py.exists():
        return _task(row, d, venv_py)

    try:
        CACHE.mkdir(parents=True, exist_ok=True)
        if not (d / ".git").exists():
            _run(["git", "clone", "--quiet",
                  f"https://github.com/{row['repo']}.git", str(d)])
        _run(["git", "checkout", "--quiet", "--force", row["base_commit"]], cwd=d)
        _run(["git", "clean", "-fdq"], cwd=d)
        if not venv_py.exists():
            _run(["python3", "-m", "venv", str(d / ".venv")])
        _run([str(venv_py), "-m", "pip", "install", "--quiet", "-U", "pip", "wheel"])
        # Try common extras, then fall back to a bare editable install.
        for spec in ("-e .[test]", "-e .[dev]", "-e .[testing]", "-e ."):
            r = _run([str(venv_py), "-m", "pip", "install", "--quiet", *spec.split()], cwd=d)
            if r.returncode == 0:
                break
        _run([str(venv_py), "-m", "pip", "install", "--quiet", "pytest"])
        # Sanity: pytest importable?
        r = _run([str(venv_py), "-c", "import pytest"], cwd=d)
        if r.returncode != 0:
            print(f"  [skip] {iid}: env build failed ({r.stderr.strip()[:120]})")
            return None
        marker.write_text("ok")
        return _task(row, d, venv_py)
    except Exception as exc:  # noqa: BLE001
        print(f"  [skip] {iid}: {type(exc).__name__}: {exc}")
        return None


def _task(row: dict, d: Path, venv_py: Path) -> dict:
    f2p = json.loads(row["FAIL_TO_PASS"]) if isinstance(row["FAIL_TO_PASS"], str) else row["FAIL_TO_PASS"]
    p2p = json.loads(row["PASS_TO_PASS"]) if isinstance(row["PASS_TO_PASS"], str) else row["PASS_TO_PASS"]
    return {
        "instance_id": row["instance_id"],
        "template_dir": str(d),
        "in_place": True,                       # edit the clone directly (egg-link)
        "test_cmd": f"{venv_py} -m pytest -q",  # agent's own check (existing suite)
        "problem_statement": row["problem_statement"],
        "_dir": str(d), "_venv": str(venv_py),
        "_test_patch": row["test_patch"], "_f2p": f2p, "_p2p": p2p,
        "reset": _reset, "scorer": _scorer,
    }


def _reset(task: dict) -> None:
    d = task["_dir"]
    _run(["git", "checkout", "--quiet", "--", "."], cwd=d)
    _run(["git", "clean", "-fdq"], cwd=d)


def _scorer(task: dict) -> bool:
    """Apply gold test patch; require FAIL_TO_PASS + PASS_TO_PASS to pass."""
    d = Path(task["_dir"])
    patch = d / ".gold_test.patch"
    patch.write_text(task["_test_patch"])
    applied = _run(["git", "apply", str(patch)], cwd=d).returncode == 0
    if not applied:
        _run(["git", "apply", "--3way", str(patch)], cwd=d)
    ids = list(task["_f2p"]) + list(task["_p2p"])
    try:
        r = _run([task["_venv"], "-m", "pytest", "-q", "--no-header", "-p", "no:cacheprovider", *ids],
                 cwd=d, timeout=600)
        resolved = r.returncode == 0
    except Exception:
        resolved = False
    _run(["git", "apply", "-R", str(patch)], cwd=d)  # revert gold tests
    return resolved


def load(limit: int = 6) -> list[dict]:
    from datasets import load_dataset  # needs HF network (works locally)
    ds = load_dataset("princeton-nlp/SWE-bench_Verified", split="test")
    rank = {r: i for i, r in enumerate(ALLOWLIST)}
    rows = [r for r in ds if r["repo"] in rank]
    rows.sort(key=lambda r: rank[r["repo"]])       # small/clean repos first
    tasks: list[dict] = []
    for row in rows:
        if len(tasks) >= limit:
            break
        print(f"Preparing {row['instance_id']} ({row['repo']}) ...")
        t = _prepare(row)
        if t:
            tasks.append(t)
    print(f"Prepared {len(tasks)}/{limit} instances.")
    return tasks
