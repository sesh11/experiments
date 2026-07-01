"""Task loaders behind one interface.

Native mini-set (default): self-contained bug-fix tasks bundled in the repo. No
network, no Docker — runnable end-to-end for the price of the LLM calls, ideal for
the Tier-0 smoke run and cheap iteration.

SWE-bench Verified (optional): a small slice of the public benchmark via the
`datasets` package. Each task is a real repo checked out at its base commit. This
path needs network + git and is heavier; use it to scale up once the harness works.

Every task dict has: instance_id, problem_statement, template_dir, test_cmd.
"""

from __future__ import annotations

import os
import subprocess
import tempfile
from pathlib import Path

_HERE = Path(__file__).parent
_DATA = _HERE / "tasks_data"

_NATIVE = [
    {
        "instance_id": "native__slugify",
        "template_dir": str(_DATA / "slugify"),
        "test_cmd": "python -m pytest -q",
        "problem_statement": (
            "slugify(' Hello, World! ') returns '-hello-world-' but should return "
            "'hello-world'. Leading/trailing separators must be trimmed, and runs of "
            "non-alphanumeric characters should collapse to a single hyphen."
        ),
    },
    {
        "instance_id": "native__median",
        "template_dir": str(_DATA / "median"),
        "test_cmd": "python -m pytest -q",
        "problem_statement": (
            "median() is wrong for even-length inputs: median([1,2,3,4]) returns 3 but "
            "should return 2.5 (the average of the two middle values). Odd-length inputs "
            "are already correct."
        ),
    },
]


def load_native(limit: int | None = None) -> list[dict]:
    tasks = list(_NATIVE)
    return tasks[:limit] if limit else tasks


def load_swebench_verified(limit: int = 10) -> list[dict]:
    """Small slice of SWE-bench Verified. Requires `datasets`, network, and git."""
    from datasets import load_dataset  # local import: optional dependency

    ds = load_dataset("princeton-nlp/SWE-bench_Verified", split="test")
    tasks = []
    for row in ds.select(range(min(limit, len(ds)))):
        template_dir = _checkout_repo(row["repo"], row["base_commit"])
        tasks.append({
            "instance_id": row["instance_id"],
            "template_dir": template_dir,
            # Fall back to a broad test run; refine per-instance with FAIL_TO_PASS if needed.
            "test_cmd": "python -m pytest -q",
            "problem_statement": row["problem_statement"],
            "fail_to_pass": row.get("FAIL_TO_PASS"),
        })
    return tasks


def _checkout_repo(repo: str, base_commit: str) -> str:
    """Clone `owner/name` at base_commit into a cached template dir."""
    cache = Path(tempfile.gettempdir()) / "fusion_swebench" / f"{repo.replace('/', '__')}__{base_commit[:8]}"
    if cache.exists():
        return str(cache)
    cache.parent.mkdir(parents=True, exist_ok=True)
    url = f"https://github.com/{repo}.git"
    subprocess.run(["git", "clone", "--quiet", url, str(cache)], check=True)
    subprocess.run(["git", "-C", str(cache), "checkout", "--quiet", base_commit], check=True)
    return str(cache)


def load(source: str, limit: int | None = None) -> list[dict]:
    if source == "native":
        return load_native(limit)
    if source == "swebench":
        # Real per-instance venv build + gold-patch scoring (runs locally).
        from . import swebench_env
        return swebench_env.load(limit or 6)
    raise ValueError(f"unknown task source: {source}")
