"""Real SWE-bench Verified scoring (runs locally, where the network is open).

This is the piece the Cloud/Web sandbox couldn't run: HuggingFace and arbitrary
GitHub clones are blocked there. On a normal machine it works.

Per instance: clone repo @ base_commit, build a venv, `pip install -e .`, then a
**validation gate** — apply the dataset's gold test patch and require BOTH that
the FAIL_TO_PASS tests collect and FAIL at base AND that the sampled PASS_TO_PASS
tests PASS at base. The second check matters: these venvs use current dep versions
(not SWE-bench's pinned images), so an instance whose P2P sample is already red at
base can never score `resolved` no matter what the agent does — it must be skipped,
not run. Instances that don't validate are skipped with a printed reason BEFORE any
LLM spend, so "resolved" is trustworthy by construction.

Scoring after an agent run: restore the test files touched by the gold test patch
to their base state (official SWE-bench semantics — agent edits to tests never
count), apply the gold test patch, run FAIL_TO_PASS (must all pass) and the same
deterministic sample of <=30 PASS_TO_PASS (must stay green) as separate pytest
invocations, then revert the patch.
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

# Dataset coordinates for both the local loader and the official Docker harness.
DATASET_NAME = "princeton-nlp/SWE-bench_Verified"
SPLIT = "test"

# Untracked files we must never `git clean` away inside an instance dir.
# (*.egg-info guards editable-install metadata in repos that don't gitignore it.)
_KEEP = ["-e", ".venv", "-e", ".ready", "-e", "*.egg-info"]


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


def _patch_paths(patch: str) -> list[str]:
    """File paths touched by a unified diff (from its `diff --git a/x b/y` lines)."""
    paths: set[str] = set()
    for line in patch.splitlines():
        if line.startswith("diff --git "):
            for tok in line.split()[2:4]:
                p = tok[2:] if tok[:2] in ("a/", "b/") else tok
                if p and p != "dev/null":
                    paths.add(p)
    return sorted(paths)


def _apply_gold(d: Path, test_patch: str, base_commit: str):
    """Apply the gold test patch from a temp file OUTSIDE the repo. Returns
    (ok, revert_fn).

    The files the patch touches are first restored to their base state:
    official SWE-bench discards agent edits to test files, and applying onto
    edited tests would otherwise fail and score the run unresolvable."""
    for rel in _patch_paths(test_patch):
        r = _run(["git", "checkout", "--quiet", "--force", base_commit, "--", rel], cwd=d)
        if r.returncode != 0:  # file doesn't exist at base: the patch creates it
            (d / rel).unlink(missing_ok=True)
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


def _pytest_digest(proc, max_chars: int = 1200) -> str:
    """The informative part of a pytest run: any FAILED/ERROR lines plus the
    tail (which holds the summary line). Kept small enough to log per run."""
    out = (proc.stdout or "") + (("\n" + proc.stderr) if proc.stderr else "")
    flagged = [ln for ln in out.splitlines()
               if ("FAILED" in ln or "ERROR" in ln or ln.startswith("E   ")
                   or " passed" in ln or " failed" in ln or " error" in ln)]
    head = "\n".join(flagged[:20])
    tail = out.strip()[-max_chars:]
    combined = (head + "\n...\n" + tail) if head and head not in tail else tail
    return combined[-max_chars:].strip() or "(no pytest output)"


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
def _p2p_sample(row: dict) -> list[str]:
    """Deterministic <=P2P_SAMPLE-id sample; the SAME sample is used for the
    base-state validation and for scoring, so the gate actually covers scoring."""
    p2p_all = sorted(_ids(row["PASS_TO_PASS"]))
    return (random.Random(0).sample(p2p_all, P2P_SAMPLE)
            if len(p2p_all) > P2P_SAMPLE else p2p_all)


def _validate(row: dict, d: Path, venv_py: str, p2p: list[str]) -> str | None:
    """Return a skip-reason, or None if the instance is scoreable.

    Requirements, with the gold test patch applied at base:
      * FAIL_TO_PASS ids collect and FAIL (pytest exit code 1), and
      * the P2P sample PASSES (exit code 0) — otherwise `resolved` is
        unreachable in this env and the instance would only burn budget."""
    ok, revert = _apply_gold(d, row["test_patch"], row["base_commit"])
    if not ok:
        revert()
        return "gold test patch does not apply"
    try:
        rc = _pytest(venv_py, d, _ids(row["FAIL_TO_PASS"])).returncode
        p2p_rc = _pytest(venv_py, d, p2p).returncode if p2p else 0
    except subprocess.TimeoutExpired:
        revert()
        return "F2P/P2P validation run timed out"
    revert()
    if rc == 0:
        return "F2P already passes at base (bad instance/env)"
    if rc in (4, 5):
        return "F2P ids not collectable (non-pytest format or missing)"
    if rc in (2, 3):
        return f"pytest errored (rc={rc}) — env likely broken"
    if p2p_rc != 0:
        return (f"P2P sample fails at base (rc={p2p_rc}) — dep drift vs the "
                f"pinned SWE-bench env; instance unscoreable here")
    return None  # F2P fails, P2P passes: exactly the state scoring assumes


# --- scoring -----------------------------------------------------------------
def _rc_label(rc: int) -> str:
    """PASS / FAIL are test outcomes; anything else means pytest itself broke
    (collection error, internal error, bad env) and must not read as 'the
    agent's fix was wrong'."""
    if rc == 0:
        return "PASS"
    if rc == 1:
        return "FAIL"
    return f"ERROR(rc={rc})"


def _scorer(task: dict) -> tuple[bool, str]:
    d = Path(task["_dir"])
    venv = task["_venv"]
    ok, revert = _apply_gold(d, task["_test_patch"], task["_base_commit"])
    if not ok:
        revert()
        task["_score_artifacts"] = {"apply_ok": False}
        return False, "gold test patch failed to apply even after resetting test files"
    try:
        f2p_proc = _pytest(venv, d, task["_f2p"])
        p2p = task["_p2p"]
        p2p_proc = _pytest(venv, d, p2p) if p2p else None
    except subprocess.TimeoutExpired:
        revert()
        task["_score_artifacts"] = {"apply_ok": True, "timed_out": True}
        return False, "scoring run timed out"
    revert()
    f2p_rc = f2p_proc.returncode
    p2p_rc = p2p_proc.returncode if p2p_proc else 0
    # Stash the full scoring evidence so the audit log can show WHY a run
    # failed — a wrong fix, a crashed pytest, or an untouched file all differ.
    task["_score_artifacts"] = {
        "apply_ok": True,
        "f2p_rc": f2p_rc, "f2p_ids": task["_f2p"],
        "f2p_output": _pytest_digest(f2p_proc),
        "p2p_rc": p2p_rc, "p2p_n": len(p2p),
        "p2p_output": _pytest_digest(p2p_proc) if p2p_proc else "(no P2P ids)",
    }
    detail = (f"f2p({len(task['_f2p'])}) {_rc_label(f2p_rc)}; "
              f"p2p({len(p2p)} sampled) {_rc_label(p2p_rc)}")
    return f2p_rc == 0 and p2p_rc == 0, detail


def score_with_gold_patch(task: dict) -> tuple[bool, str]:
    """$0 pipeline self-test: reset the repo, apply the dataset's GOLD SOLUTION
    patch (the known-correct human fix), then run the exact scorer used for
    agent runs. A correct scorer MUST return resolved=True here. If it doesn't,
    the scoring pipeline — not the agent — is what's failing real runs."""
    _reset(task)
    d = Path(task["_dir"])
    patch = task.get("_gold_patch") or ""
    if not patch.strip():
        return False, "no gold solution patch in dataset row"
    with tempfile.NamedTemporaryFile("w", suffix=".patch", delete=False) as fh:
        fh.write(patch)
        pfile = fh.name
    applied = _run(["git", "apply", pfile], cwd=d).returncode == 0
    if not applied:
        applied = _run(["git", "apply", "--3way", pfile], cwd=d).returncode == 0
    Path(pfile).unlink(missing_ok=True)
    if not applied:
        _reset(task)
        return False, "gold SOLUTION patch failed to apply (repo/base mismatch)"
    resolved, detail = _scorer(task)
    _reset(task)
    return resolved, detail


def _reset(task: dict) -> None:
    d = task["_dir"]
    _run(["git", "checkout", "--quiet", "--force", task["_base_commit"]], cwd=d)
    _run(["git", "clean", "-fdq", *_KEEP], cwd=d)


def _task(row: dict, d: Path, venv_py: str, p2p: list[str],
          backend: str = "local") -> dict:
    task = {
        "instance_id": row["instance_id"],
        "template_dir": str(d),
        "in_place": True,
        "test_cmd": f"{venv_py} -m pytest -q -p no:cacheprovider",
        "test_timeout": TEST_TIMEOUT,
        "problem_statement": row["problem_statement"],
        "backend": backend,
        "_dir": str(d), "_venv": venv_py, "_base_commit": row["base_commit"],
        "_test_patch": row["test_patch"], "_gold_patch": row.get("patch", ""),
        "_f2p": _ids(row["FAIL_TO_PASS"]), "_p2p": p2p,
        "_dataset": DATASET_NAME, "_split": SPLIT,
        "reset": _reset,
    }
    # Local backend scores with the local pytest scorer; docker backend defers
    # scoring to the official harness (driver calls docker_score with the diff),
    # so it must NOT carry a `scorer` key.
    if backend == "local":
        task["scorer"] = _scorer
    return task


def _sorted_rows(ds):
    rank = {r: i for i, r in enumerate(ALLOWLIST)}
    return sorted((r for r in ds if r["repo"] in rank),
                  key=lambda r: (rank[r["repo"]], r["instance_id"]))


def list_instance_ids(limit: int = 5) -> list[str]:
    """Just the first `limit` allowlisted Verified instance_ids — no env build.
    Used by the Docker gold self-test, which needs only IDs (Docker does the rest)."""
    from datasets import load_dataset
    ds = load_dataset(DATASET_NAME, split=SPLIT)
    return [r["instance_id"] for r in _sorted_rows(ds)[:limit]]


# --- loader ------------------------------------------------------------------
def load(limit: int = 15, backend: str = "local") -> list[dict]:
    """Prepare scoreable instances.

    local backend: strict gate — the local venv must reproduce the bug AND keep
      PASS_TO_PASS green at base, because the local pytest scorer is authoritative.
    docker backend: relaxed gate — the official harness scores in a pinned image,
      so we only need a runnable local checkout for the agent's edit/iterate loop.
      We keep instances whose FAIL_TO_PASS ids at least *collect* (so the agent's
      own test runs are meaningful) but do NOT require PASS_TO_PASS to pass at
      base — that local drift is exactly what Docker scoring exists to bypass.
    """
    from datasets import load_dataset  # needs HF network (works locally)
    ds = load_dataset(DATASET_NAME, split=SPLIT)
    candidates = _sorted_rows(ds)[: limit * 5]
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
        p2p = _p2p_sample(row)
        if backend == "docker":
            reason = _validate_docker(row, d, venv_py)
            ok_msg = "local checkout runnable (Docker will score authoritatively)"
        else:
            reason = _validate(row, d, venv_py, p2p)
            ok_msg = "env valid — F2P fails and P2P passes at base"
        if reason:
            print(f"  [skip] {iid}: {reason}")
            continue
        print(f"  [ok]   {iid}: {ok_msg}")
        tasks.append(_task(row, d, venv_py, p2p, backend=backend))
    print(f"\nPrepared {len(tasks)}/{limit} instance(s) for backend='{backend}' "
          f"(from {len(candidates)} candidates).")
    return tasks


def _validate_docker(row: dict, d: Path, venv_py: str) -> str | None:
    """Relaxed gate for docker backend: only reject instances the agent could
    not meaningfully iterate on locally (env broken, or FAIL_TO_PASS ids that
    don't even collect). PASS_TO_PASS drift at base is tolerated — Docker scores."""
    ok, revert = _apply_gold(d, row["test_patch"], row["base_commit"])
    if not ok:
        revert()
        return "gold test patch does not apply to local checkout"
    try:
        rc = _pytest(venv_py, d, _ids(row["FAIL_TO_PASS"])).returncode
    except subprocess.TimeoutExpired:
        revert()
        return "F2P collection run timed out"
    revert()
    if rc in (4, 5):
        return "F2P ids not collectable (non-pytest format or missing)"
    if rc in (2, 3):
        return f"pytest errored (rc={rc}) — local env too broken to iterate"
    return None  # rc 0 (bug not reproduced locally) or 1 (reproduced) both fine
