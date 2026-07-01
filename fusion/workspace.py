"""A per-task sandboxed working copy: file ops, search, patch, and test execution.

A Workspace is a plain directory the agent edits in place. We snapshot the
original file contents on load so we can emit a unified diff of whatever the
agent changed (used by the judge and for reporting).
"""

from __future__ import annotations

import difflib
import os
import shutil
import subprocess
import tempfile
from pathlib import Path


class Workspace:
    def __init__(self, root: str | Path, test_cmd: str, *,
                 use_git: bool = False, test_timeout: int = 120):
        self.root = Path(root).resolve()
        self.test_cmd = test_cmd
        self.use_git = use_git and (self.root / ".git").exists()
        self.test_timeout = test_timeout
        self._original: dict[str, str] = {}
        if not self.use_git:
            # Snapshot for difflib. Skipped in git mode: reading every .py in a
            # sympy-sized repo is slow, and `git diff` is authoritative there.
            for p in self._py_files():
                self._original[str(p.relative_to(self.root))] = p.read_text()

    # --- lifecycle ---------------------------------------------------------
    @classmethod
    def from_template(cls, template_dir: str | Path, test_cmd: str,
                      **kwargs) -> "Workspace":
        """Copy a task template into a fresh temp dir so runs never collide."""
        tmp = tempfile.mkdtemp(prefix="fusion_ws_")
        dst = Path(tmp) / "repo"
        shutil.copytree(template_dir, dst)
        return cls(dst, test_cmd, **kwargs)

    def cleanup(self) -> None:
        parent = self.root.parent
        if parent.name.startswith("fusion_ws_") or "fusion_ws_" in str(parent):
            shutil.rmtree(parent, ignore_errors=True)

    # --- helpers -----------------------------------------------------------
    def _py_files(self) -> list[Path]:
        return sorted(p for p in self.root.rglob("*.py") if p.is_file())

    def _safe(self, rel: str) -> Path:
        p = (self.root / rel).resolve()
        if not str(p).startswith(str(self.root)):
            raise ValueError(f"path escapes workspace: {rel}")
        return p

    # --- read-side tools ---------------------------------------------------
    def read_file(self, path: str, start: int | None = None, end: int | None = None) -> str:
        p = self._safe(path)
        if not p.exists():
            return f"ERROR: no such file: {path}"
        lines = p.read_text().splitlines()
        s = (start - 1) if start else 0
        e = end if end else len(lines)
        chunk = lines[max(0, s):e]
        width = len(str(e))
        return "\n".join(f"{i + s + 1:>{width}}\t{ln}" for i, ln in enumerate(chunk))

    def list_dir(self, path: str = ".") -> str:
        p = self._safe(path)
        if not p.exists():
            return f"ERROR: no such directory: {path}"
        entries = []
        for child in sorted(p.iterdir()):
            tag = "/" if child.is_dir() else ""
            entries.append(child.name + tag)
        return "\n".join(entries) or "(empty)"

    def search(self, pattern: str, max_hits: int = 60) -> str:
        import re
        try:
            rx = re.compile(pattern)
        except re.error as exc:
            return f"ERROR: bad regex: {exc}"
        if self.use_git:
            # git grep searches tracked files only (excludes .venv etc.) and is
            # far faster than a Python walk over a large repo.
            proc = subprocess.run(
                ["git", "grep", "-nIE", pattern, "--", "*.py"],
                cwd=self.root, capture_output=True, text=True, timeout=60,
            )
            if proc.returncode > 1:
                return f"ERROR: git grep: {proc.stderr.strip()[:200]}"
            lines = proc.stdout.splitlines()
            if not lines:
                return "(no matches)"
            out = lines[:max_hits]
            if len(lines) > max_hits:
                out.append(f"... ({len(lines) - max_hits} more hits truncated)")
            return "\n".join(out)
        hits = []
        for p in self._py_files():
            rel = p.relative_to(self.root)
            for i, ln in enumerate(p.read_text().splitlines(), 1):
                if rx.search(ln):
                    hits.append(f"{rel}:{i}: {ln.strip()}")
                    if len(hits) >= max_hits:
                        return "\n".join(hits) + "\n... (truncated)"
        return "\n".join(hits) or "(no matches)"

    # --- write-side tools --------------------------------------------------
    def str_replace(self, path: str, old: str, new: str) -> str:
        p = self._safe(path)
        if not p.exists():
            return f"ERROR: no such file: {path}"
        text = p.read_text()
        count = text.count(old)
        if count == 0:
            return "ERROR: old_str not found (must match exactly, including whitespace)"
        if count > 1:
            return f"ERROR: old_str matches {count} times; make it unique"
        p.write_text(text.replace(old, new, 1))
        return f"OK: edited {path}"

    def create_file(self, path: str, content: str) -> str:
        p = self._safe(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content)
        return f"OK: wrote {path}"

    # --- verification ------------------------------------------------------
    def run_tests(self, target: str | None = None) -> tuple[bool, str]:
        cmd = self.test_cmd
        if target:
            # target is model-supplied; keep it to a plain path / -k expression.
            if any(c in target for c in ";|&`$><\n"):
                return False, "ERROR: invalid characters in target"
            cmd = f"{cmd} {target}"
        try:
            proc = subprocess.run(
                cmd, shell=True, cwd=self.root,
                capture_output=True, text=True, timeout=self.test_timeout,
            )
        except subprocess.TimeoutExpired:
            return False, "ERROR: test command timed out"
        out = (proc.stdout + "\n" + proc.stderr).strip()
        if len(out) > 4000:
            out = out[:2000] + "\n... (truncated) ...\n" + out[-2000:]
        return proc.returncode == 0, out

    # --- diff --------------------------------------------------------------
    def diff(self) -> str:
        if self.use_git:
            return self._git_diff()
        chunks = []
        current = {str(p.relative_to(self.root)): p.read_text() for p in self._py_files()}
        keys = sorted(set(self._original) | set(current))
        for rel in keys:
            before = self._original.get(rel, "")
            after = current.get(rel, "")
            if before == after:
                continue
            d = difflib.unified_diff(
                before.splitlines(keepends=True), after.splitlines(keepends=True),
                fromfile=f"a/{rel}", tofile=f"b/{rel}",
            )
            chunks.append("".join(d))
        return "\n".join(chunks) or "(no changes)"

    def _git_diff(self) -> str:
        proc = subprocess.run(["git", "diff"], cwd=self.root,
                              capture_output=True, text=True, timeout=120)
        parts = [proc.stdout] if proc.stdout.strip() else []
        # New files created by the agent are untracked; synthesize their diffs.
        ls = subprocess.run(["git", "ls-files", "--others", "--exclude-standard",
                             "--", "*.py"],
                            cwd=self.root, capture_output=True, text=True, timeout=60)
        for rel in ls.stdout.splitlines():
            if rel.startswith(".venv/") or not rel.strip():
                continue
            try:
                after = (self.root / rel).read_text()
            except Exception:
                continue
            d = difflib.unified_diff([], after.splitlines(keepends=True),
                                     fromfile="/dev/null", tofile=f"b/{rel}")
            parts.append("".join(d))
        return "\n".join(p for p in parts if p) or "(no changes)"
