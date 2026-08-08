"""Per-cell task materialization.

SWE-bench's loader caches one editable checkout per instance.  That checkout is
safe for serial runs but cannot be reset/edited by concurrent variants.  Each
cell therefore receives a cheap local clone with shared immutable git objects
and an independent worktree/index.
"""

from __future__ import annotations

import os
import shlex
import shutil
import subprocess
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator


class IsolationError(RuntimeError):
    pass


def _run(cmd: list[str], cwd: Path | None = None) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, timeout=300)


def _reset_cell(task: dict) -> None:
    root = Path(task["_dir"])
    base = task["_base_commit"]
    checkout = _run(["git", "checkout", "--quiet", "--force", base], cwd=root)
    clean = _run(["git", "clean", "-fdq"], cwd=root)
    if checkout.returncode or clean.returncode:
        raise IsolationError(
            f"could not reset isolated checkout: "
            f"{checkout.stderr.strip() or clean.stderr.strip()}")


def _prepend_pythonpath(test_cmd: str, root: Path) -> str:
    # Editable virtualenvs point at the loader's cache checkout. CWD normally
    # wins, but src-layout projects need an explicit prefix to ensure tests
    # import the cell's private clone rather than another cell's base tree.
    entries = f"{root}{os.pathsep}{root / 'src'}"
    return f"PYTHONPATH={shlex.quote(entries)}${{PYTHONPATH:+:$PYTHONPATH}} {test_cmd}"


@contextmanager
def isolated_task(base_task: dict, workspace_dir: Path, *, seed: int) -> Iterator[dict]:
    """Yield a shallow-cloned task whose mutable repository is cell-private."""
    task = dict(base_task)
    task["seed"] = seed
    if not task.get("in_place"):
        # Native tasks already use Workspace.from_template(), which creates a
        # unique copy per invocation. Keep that established fast path.
        yield task
        return

    source = Path(task["template_dir"]).resolve()
    repo = workspace_dir / "repo"
    if workspace_dir.exists():
        shutil.rmtree(workspace_dir)
    workspace_dir.mkdir(parents=True, exist_ok=True)
    clone = _run(["git", "clone", "--shared", "--quiet", str(source), str(repo)])
    if clone.returncode != 0:
        shutil.rmtree(workspace_dir, ignore_errors=True)
        raise IsolationError(f"local clone failed: {clone.stderr.strip()[:500]}")
    base = str(task.get("_base_commit") or "HEAD")
    checkout = _run(["git", "checkout", "--quiet", "--force", base], cwd=repo)
    if checkout.returncode != 0:
        shutil.rmtree(workspace_dir, ignore_errors=True)
        raise IsolationError(f"base checkout failed: {checkout.stderr.strip()[:500]}")

    task.update({
        "template_dir": str(repo),
        "_dir": str(repo),
        "reset": _reset_cell,
        "test_cmd": _prepend_pythonpath(task["test_cmd"], repo),
    })
    try:
        yield task
    finally:
        shutil.rmtree(workspace_dir, ignore_errors=True)
