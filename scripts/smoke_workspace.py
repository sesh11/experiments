"""Smoke-test the workspace/test loop WITHOUT any LLM spend.

Verifies, for each native task, that:
  1. the buggy template fails its tests,
  2. applying the known fix makes them pass,
  3. the diff is non-empty.

Run from the repo root:  python scripts/smoke_workspace.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from eval import tasks
from fusion.workspace import Workspace

# Known-good fixes applied via exact string replacement.
FIXES = {
    "native__slugify": ("slugify.py",
                         'return text',
                         'return text.strip("-")'),
    "native__median": ("stats.py",
                        "    return s[mid]",
                        "    if n % 2 == 0:\n        return (s[mid - 1] + s[mid]) / 2\n    return s[mid]"),
}


def main() -> int:
    ok = True
    for task in tasks.load_native():
        tid = task["instance_id"]
        ws = Workspace.from_template(task["template_dir"], task["test_cmd"])
        try:
            passed_before, _ = ws.run_tests()
            path, old, new = FIXES[tid]
            msg = ws.str_replace(path, old, new)
            passed_after, out = ws.run_tests()
            diff = ws.diff()
            good = (not passed_before) and passed_after and "no such" not in msg and diff != "(no changes)"
            status = "PASS" if good else "FAIL"
            print(f"[{status}] {tid}: before={passed_before} after={passed_after} "
                  f"edit={msg!r}")
            if not good:
                ok = False
                print(out)
        finally:
            ws.cleanup()
    print("\nSMOKE OK" if ok else "\nSMOKE FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
