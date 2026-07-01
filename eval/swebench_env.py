"""Real SWE-bench Verified scoring (runs locally, where the network is open).

This is the piece the Cloud/Web sandbox couldn't run: HuggingFace and arbitrary
GitHub clones are blocked there. On a normal machine it works.

Per instance: clone repo @ base_commit, build a venv, `pip install -e .`, then a
**validation gate** — apply the dataset's gold test patch and require the
FAIL_TO_PASS tests to collect and FAIL at base. Instances that don't validate
(broken env, non-pytest test ids, already-passing tests) are skipped with a
printed reason BEFORE any LLM spend, so "resolved" is trustworthy by construction.

Scoring after an agent run: apply the gold test patch, run FAIL_TO_PASS (must all
pass) and a deterministic sample of <=30 PASS_TO_PASS (must stay green) as separate
pytest invocations, then revert the patch.
"""

from __future__ import annotations

import json
import random
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

# Pure-python, pip-installable repos whose SWE-bench test ids are pytest-style.
# django/django is excluded: its FAIL_TO_PASS ids use the Django-runner format
# ("test_x (app.Class)"), not pytest node ids.
ALLOWLIST = [
    "psf/requests", "pallets/flask", "pytest-dev/pytest", "pydata/xarray",
    "pylint-dev/pylint", "sphinx-doc/sphinx", "sympy/sympy",
]

P2P_SAMPLE = 30          # cap PASS_TO_PASS ids per instance (argv + runtime)
TEST_TIMEOUT = 600       # seconds, agent-facing and scoring runs
CACHE = Path.home() / ".cache" / "fusion_swebench"

# Untracked files we must never `git clean` away inside an instance dir.
_KEEP = ["-e", ".venv", "-e", ".ready"]


def _run(cmd, cwd=None, timeout=900):
    return subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, timeout=timeout)


def _pick_python() -> str:
    """Old (2019-23) codebases often break on 3.12+; prefer 3.11/3.10."""
    for exe in ("python3.11", "python3.10", "python3"):
        if shutil.which(exe):
            return exe
    return sys.executable


def _ids(raw) -> list[str]:
    return json.loads(raw) if isinstance(raw, str) else list(raw)


def _apply_gold(d: Path, test_patch: str):
    """Apply the gold test patch from a temp file OUTSIDE the repo. Returns
    (ok, revert_fn)."""
    with tempfile.NamedTemporaryFile("w", suffix=".patch", delete=False) as fh:
        fh.write(test_patch)
        pfile = fh.name
    ok = _run(["git", "apply", pfile], cwd=d).returncode == 0
    if not ok:
        ok = _run(["git", "apply", "--3way", pfile], cwd=d).returncode == 0

    def revert():
        _run(["git", "checkout", "--quiet", "--", "."], cwd=d)
        _run(["git", "clean", "-fdq", *_KEEP], cwd=d)
        Path(pfile).unlink(missing_ok=True)

    return ok, revert


def _pytest(venv_py: str, d: Path, ids: list[str], timeout=TEST_TIMEOUT):
    return _run([venv_py, "-m", "pytest", "-q", "--no-header",
                 "-p", "no:cacheprovider", *ids], cwd=d, timeout=timeout)


# --- env build ---------------------------------------------------------------
def _build_env(row: dict) -> tuple[Path, str] | None:
    iid = row["instance_id"]
    d = CACHE / iid
    venv_py = d / ".venv" / "bin" / "python"
    if (d / ".ready").exists() and venv_py.exists():
        return d, str(venv_py)

    try:
        CACHE.mkdir(parents=True, exist_ok=True)
        if not (d / ".git").exists():
            r = _run(["git", "clone", "--quiet",
                      f"https://github.com/{row['repo']}.git", str(d)])
            if r.returncode != 0:
                print(f"  [skip] {iid}: clone failed")
                return None
        _run(["git", "checkout", "--quiet", "--force", row["base_commit"]], cwd=d)
        _run(["git", "clean", "-fdq", *_KEEP], cwd=d)
        if not venv_py.exists():
            _run([_pick_python(), "-m", "venv", str(d / ".venv")])
        _run([str(venv_py), "-m", "pip", "install", "--quiet", "-U", "pip", "wheel", "setuptools"])
        for spec in ("-e .[test]", "-e .[dev]", "-e .[testing]", "-e ."):
            if _run([str(venv_py), "-m", "pip", "install", "--quiet", *spec.split()],
                    cwd=d).returncode == 0:
                break
        _run([str(venv_py), "-m", "pip", "install", "--quiet", "pytest"])
        if _run([str(venv_py), "-c", "import pytest"], cwd=d).returncode != 0:
            print(f"  [skip] {iid}: env build failed (pytest not importable)")
            return None
        (d / ".ready").write_text("ok")
        return d, str(venv_py)
    except Exception as exc:  # noqa: BLE001
        print(f"  [skip] {iid}: {type(exc).__name__}: {exc}")
        return None


# --- validation gate ---------------------------------------------------------
def _validate(row: dict, d: Path, venv_py: str) -> str | None:
    """Return a skip-reason, or None if the instance is scoreable.

    Requirement: with the gold test patch applied, the FAIL_TO_PASS ids must
    collect and FAIL at base (pytest exit code 1)."""
    ok, revert = _apply_gold(d, row["test_patch"])
    if not ok:
        revert()
        return "gold test patch does not apply"
    try:
        rc = _pytest(venv_py, d, _ids(row["FAIL_TO_PASS"])).returncode
    except subprocess.TimeoutExpired:
        revert()
        return "F2P run timed out"
    revert()
    if rc == 0:
        return "F2P already passes at base (bad instance/env)"
    if rc in (4, 5):
        return "F2P ids not collectable (non-pytest format or missing)"
    if rc in (2, 3):
        return f"pytest errored (rc={rc}) — env likely broken"
    return None  # rc == 1: tests ran and failed, as they should


# --- scoring -----------------------------------------------------------------
def _scorer(task: dict) -> tuple[bool, str]:
    d = Path(task["_dir"])
    venv = task["_venv"]
    ok, revert = _apply_gold(d, task["_test_patch"])
    if not ok:
        revert()
        return False, "gold test patch failed to apply (agent edited test files?)"
    try:
        f2p_rc = _pytest(venv, d, task["_f2p"]).returncode
        p2p = task["_p2p"]
        if p2p:
            p2p_rc = _pytest(venv, d, p2p).returncode
        else:
            p2p_rc = 0
    except subprocess.TimeoutExpired:
        revert()
        return False, "scoring run timed out"
    revert()
    f2p_ok, p2p_ok = f2p_rc == 0, p2p_rc == 0
    detail = (f"f2p({len(task['_f2p'])}) {'PASS' if f2p_ok else 'FAIL'}; "
              f"p2p({len(p2p)} sampled) {'PASS' if p2p_ok else 'FAIL'}")
    return f2p_ok and p2p_ok, detail


def _reset(task: dict) -> None:
    d = task["_dir"]
    _run(["git", "checkout", "--quiet", "--force", task["_base_commit"]], cwd=d)
    _run(["git", "clean", "-fdq", *_KEEP], cwd=d)


def _task(row: dict, d: Path, venv_py: str) -> dict:
    p2p_all = sorted(_ids(row["PASS_TO_PASS"]))
    p2p = (random.Random(0).sample(p2p_all, P2P_SAMPLE)
           if len(p2p_all) > P2P_SAMPLE else p2p_all)
    return {
        "instance_id": row["instance_id"],
        "template_dir": str(d),
        "in_place": True,
        "test_cmd": f"{venv_py} -m pytest -q -p no:cacheprovider",
        "test_timeout": TEST_TIMEOUT,
        "problem_statement": row["problem_statement"],
        "_dir": str(d), "_venv": venv_py, "_base_commit": row["base_commit"],
        "_test_patch": row["test_patch"],
        "_f2p": _ids(row["FAIL_TO_PASS"]), "_p2p": p2p,
        "reset": _reset, "scorer": _scorer,
    }


# --- loader ------------------------------------------------------------------
def load(limit: int = 15) -> list[dict]:
    from datasets import load_dataset  # needs HF network (works locally)
    ds = load_dataset("princeton-nlp/SWE-bench_Verified", split="test")
    rank = {r: i for i, r in enumerate(ALLOWLIST)}
    rows = sorted((r for r in ds if r["repo"] in rank),
                  key=lambda r: (rank[r["repo"]], r["instance_id"]))
    # Oversample so validation-gate skips don't starve the run.
    candidates = rows[: limit * 3]
    tasks: list[dict] = []
    for row in candidates:
        if len(tasks) >= limit:
            break
        iid = row["instance_id"]
        print(f"Preparing {iid} ({row['repo']}) ...", flush=True)
        built = _build_env(row)
        if not built:
            continue
        d, venv_py = built
        reason = _validate(row, d, venv_py)
        if reason:
            print(f"  [skip] {iid}: {reason}")
            continue
        print(f"  [ok]   {iid}: env valid, F2P fails at base as expected")
        tasks.append(_task(row, d, venv_py))
    print(f"\nPrepared {len(tasks)}/{limit} scoreable instances "
          f"(from {len(candidates)} candidates).")
    return tasks
