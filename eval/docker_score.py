"""Authoritative SWE-bench scoring via the official Docker harness.

Why this exists: building a local venv with *current* dependency versions cannot
reproduce SWE-bench Verified's pinned per-instance environments, so PASS_TO_PASS
control tests go red at base and nothing is scoreable (see the skip reasons the
local gate prints). The official harness runs each instance in its own Docker
image with the correct pinned deps, so a correct patch actually scores resolved.

This module is a thin, version-pinned wrapper over `swebench.harness.run_evaluation`
(swebench >= 4.1). It shells out to the harness the exact same way the SWE-bench
leaderboard does — we do not re-implement grading — and reads back the per-instance
`report.json` the harness writes.

Two entry points:
  * `score_patch(instance_id, model_patch, ...)`  — score one agent patch.
  * `gold_selftest(instance_ids, ...)`            — score the GOLD patches
        (`-p gold`); every instance MUST resolve, proving the pipeline itself.

Everything here is Docker-side; nothing depends on the flaky local venv.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

# swebench writes per-instance reports here, relative to the harness CWD.
_LOG_DIR = Path("logs") / "run_evaluation"
KEY_INSTANCE_ID = "instance_id"
KEY_MODEL = "model_name_or_path"
KEY_PREDICTION = "model_patch"


class DockerUnavailable(RuntimeError):
    pass


def preflight() -> tuple[bool, str]:
    """Is the Docker scoring path usable? Returns (ok, human-readable reason)."""
    if shutil.which("docker") is None:
        return False, "docker CLI not found on PATH (install Docker Desktop)"
    info = subprocess.run(["docker", "info"], capture_output=True, text=True)
    if info.returncode != 0:
        return False, ("docker daemon not responding — is Docker Desktop running?\n"
                       + info.stderr.strip()[:300])
    try:
        import swebench  # noqa: F401
    except Exception:  # noqa: BLE001
        return False, "the `swebench` package is not installed (pip install swebench)"
    return True, "docker + swebench ready"


def _slug(model_name: str) -> str:
    # The harness derives the report subdir from the model name this way.
    return model_name.replace("/", "__")


def _report_path(run_id: str, model_name: str, instance_id: str) -> Path:
    return _LOG_DIR / run_id / _slug(model_name) / instance_id / "report.json"


def _run_harness(*, predictions_path: str, instance_ids: list[str], run_id: str,
                 dataset_name: str, split: str, namespace: str | None,
                 workers: int, timeout: int, force_rebuild: bool,
                 cwd: Path) -> subprocess.CompletedProcess:
    cmd = [
        sys.executable, "-m", "swebench.harness.run_evaluation",
        "--dataset_name", dataset_name,
        "--split", split,
        "--predictions_path", predictions_path,
        "--run_id", run_id,
        "--instance_ids", *instance_ids,
        "--max_workers", str(workers),
        "--timeout", str(timeout),
        "--cache_level", "env",           # keep env images, drop instance images after
        "--force_rebuild", "true" if force_rebuild else "false",
    ]
    # namespace "swebench" (the harness default) PULLS prebuilt images from
    # Docker Hub; "none" forces local builds. Pass through explicitly.
    if namespace:
        cmd += ["--namespace", namespace]
    return subprocess.run(cmd, cwd=str(cwd), capture_output=True, text=True)


def _detail_from_report(rep: dict, instance_id: str) -> tuple[bool, str]:
    """(resolved, human detail) from a per-instance report.json body."""
    inner = rep.get(instance_id, {})
    resolved = bool(inner.get("resolved", False))
    if not inner.get("patch_successfully_applied", True) and "patch_successfully_applied" in inner:
        return resolved, "patch did not apply cleanly in the pinned env"
    ts = inner.get("tests_status", {})

    def frac(kind: str) -> str:
        s = len(ts.get(kind, {}).get("success", []))
        f = len(ts.get(kind, {}).get("failure", []))
        tot = s + f
        fails = ts.get(kind, {}).get("failure", [])
        tail = f" (failed: {', '.join(fails[:3])}{'…' if len(fails) > 3 else ''})" if f else ""
        return f"{s}/{tot}{tail}"

    return resolved, f"F2P {frac('FAIL_TO_PASS')}; P2P {frac('PASS_TO_PASS')}"


def score_patch(instance_id: str, model_patch: str, *, dataset_name: str,
                split: str = "test", run_id: str, model_name: str,
                namespace: str | None = "swebench", workers: int = 1,
                timeout: int = 1800, force_rebuild: bool = False,
                cwd: Path | None = None) -> dict:
    """Score a single agent patch in the official pinned Docker env.

    Returns a dict: resolved(bool), detail(str), applied(bool), report(dict),
    harness_tail(str). An empty patch short-circuits to unresolved (the agent
    changed nothing) without invoking Docker."""
    cwd = cwd or Path.cwd()
    patch = (model_patch or "").strip()
    if not patch or patch == "(no changes)":
        return {"resolved": False, "detail": "empty patch (agent produced no diff)",
                "applied": False, "report": {}, "harness_tail": ""}

    with tempfile.NamedTemporaryFile("w", suffix=".jsonl", delete=False,
                                     dir=str(cwd)) as fh:
        json.dump({KEY_INSTANCE_ID: instance_id, KEY_MODEL: model_name,
                   KEY_PREDICTION: model_patch}, fh)
        fh.write("\n")
        preds = fh.name

    try:
        proc = _run_harness(predictions_path=preds, instance_ids=[instance_id],
                            run_id=run_id, dataset_name=dataset_name, split=split,
                            namespace=namespace, workers=workers, timeout=timeout,
                            force_rebuild=force_rebuild, cwd=cwd)
    finally:
        Path(preds).unlink(missing_ok=True)

    tail = (proc.stdout or "")[-1500:] + (("\n" + proc.stderr[-800:]) if proc.stderr else "")
    rep_file = cwd / _report_path(run_id, model_name, instance_id)
    if not rep_file.exists():
        return {"resolved": False,
                "detail": "harness produced no report (build/pull failed? see harness_tail)",
                "applied": False, "report": {}, "harness_tail": tail}
    try:
        rep = json.loads(rep_file.read_text())
    except json.JSONDecodeError:
        return {"resolved": False, "detail": "report.json unreadable",
                "applied": False, "report": {}, "harness_tail": tail}
    resolved, detail = _detail_from_report(rep, instance_id)
    applied = bool(rep.get(instance_id, {}).get("patch_successfully_applied", True))
    return {"resolved": resolved, "detail": detail, "applied": applied,
            "report": rep.get(instance_id, {}), "harness_tail": tail}


def gold_selftest(instance_ids: list[str], *, dataset_name: str, split: str = "test",
                  run_id: str, namespace: str | None = "swebench", workers: int = 2,
                  timeout: int = 1800, cwd: Path | None = None) -> dict:
    """Score the GOLD patches (`-p gold`). Every instance MUST resolve; this is
    the authoritative proof that the Docker scoring pipeline is correct.

    Returns: {"resolved_ids": [...], "unresolved_ids": [...], "harness_tail": str}.
    The gold model name is fixed by the harness to 'gold'."""
    cwd = cwd or Path.cwd()
    proc = _run_harness(predictions_path="gold", instance_ids=instance_ids,
                        run_id=run_id, dataset_name=dataset_name, split=split,
                        namespace=namespace, workers=workers, timeout=timeout,
                        force_rebuild=False, cwd=cwd)
    tail = (proc.stdout or "")[-2000:] + (("\n" + proc.stderr[-1000:]) if proc.stderr else "")
    resolved, unresolved = [], []
    for iid in instance_ids:
        rep_file = cwd / _report_path(run_id, "gold", iid)
        ok = False
        if rep_file.exists():
            try:
                ok = bool(json.loads(rep_file.read_text()).get(iid, {}).get("resolved"))
            except json.JSONDecodeError:
                ok = False
        (resolved if ok else unresolved).append(iid)
    return {"resolved_ids": resolved, "unresolved_ids": unresolved, "harness_tail": tail}
