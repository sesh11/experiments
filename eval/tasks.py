"""Task loaders behind one interface.

Native mini-set (default): self-contained bug-fix tasks bundled in the repo. No
network, no Docker — runnable end-to-end for the price of the LLM calls, ideal for
the Tier-0 smoke run and cheap iteration.

SWE-bench Verified: real instances with real FAIL_TO_PASS / PASS_TO_PASS scoring —
see eval/swebench_env.py. Needs network (HuggingFace + GitHub), so it runs locally,
not in the Cloud/Web sandbox.

Every task dict has: instance_id, problem_statement, template_dir, test_cmd.
"""

from __future__ import annotations

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


def load(source: str, limit: int | None = None,
         backend: str = "local",
         instance_ids: list[str] | None = None) -> list[dict]:
    if source == "native":
        return load_native(limit)
    if source == "swebench":
        # Real per-instance envs. backend='local' scores with a local pytest
        # scorer; backend='docker' defers to the official SWE-bench harness.
        # instance_ids (if given) pins the exact set — e.g. the gold-verified set.
        from . import swebench_env
        return swebench_env.load(limit or 15, backend=backend,
                                 instance_ids=instance_ids)
    raise ValueError(f"unknown task source: {source}")
