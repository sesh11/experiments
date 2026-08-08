"""Persistent experiment cells and atomic, resumable run state.

The manifest is the authoritative journal for a run.  Human-facing audit logs
and aggregate CSV/JSON files are derived artifacts and can be rebuilt from the
per-cell result files after a crash.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
import fcntl
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


SCHEMA_VERSION = 1
_SAFE = re.compile(r"[^A-Za-z0-9_.-]+")


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def atomic_write_text(path: Path, text: str) -> None:
    """Durably replace *path* without exposing a partially written file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    tmp = Path(tmp_name)
    try:
        with os.fdopen(fd, "w") as fh:
            fh.write(text)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


def atomic_write_json(path: Path, value: Any) -> None:
    atomic_write_text(path, json.dumps(value, indent=2, sort_keys=True) + "\n")


def _slug(value: str, limit: int = 52) -> str:
    clean = _SAFE.sub("-", value).strip("-._") or "cell"
    return clean[:limit]


@dataclass(frozen=True)
class CellSpec:
    """One independently schedulable experiment execution."""

    index: int
    task_index: int
    task_id: str
    variant: str
    config_name: str
    config: dict[str, Any]
    repetition: int
    seed: int
    cell_id: str

    @classmethod
    def create(cls, *, index: int, task_index: int, task_id: str,
               variant: str, config_name: str, config: dict[str, Any],
               repetition: int, seed: int) -> "CellSpec":
        identity = {
            "task_id": task_id,
            "variant": variant,
            "config_name": config_name,
            "config": config,
            "repetition": repetition,
            "seed": seed,
        }
        digest = hashlib.sha256(canonical_json(identity).encode()).hexdigest()[:12]
        label = _slug(f"{task_id}__{variant}__{config_name}__r{repetition + 1}")
        return cls(index=index, task_index=task_index, task_id=task_id,
                   variant=variant, config_name=config_name, config=config,
                   repetition=repetition, seed=seed,
                   cell_id=f"{label}__{digest}")


def build_cells(tasks: list[dict], variants: list[str], configurations: list[dict],
                repetitions: int, base_seed: int) -> list[CellSpec]:
    cells: list[CellSpec] = []
    index = 0
    # Task-major/variant-major ordering preserves the old serial run order.
    for task_index, task in enumerate(tasks):
        for variant in variants:
            for cfg in configurations:
                for repetition in range(repetitions):
                    cells.append(CellSpec.create(
                        index=index, task_index=task_index,
                        task_id=task["instance_id"], variant=variant,
                        config_name=cfg["name"], config=cfg["run_config"],
                        repetition=repetition, seed=base_seed + repetition,
                    ))
                    index += 1
    return cells


def new_run_id() -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    nonce = hashlib.sha256(os.urandom(16)).hexdigest()[:6]
    return f"{stamp}_{nonce}"


class RunStore:
    """Atomic run manifest plus one atomic result file per completed cell."""

    def __init__(self, run_dir: Path, manifest: dict) -> None:
        self.run_dir = run_dir
        self.manifest_path = run_dir / "manifest.json"
        self.cells_dir = run_dir / "cells"
        self.workspaces_dir = run_dir / "workspaces"
        self.manifest = manifest

    @classmethod
    def create(cls, *, out_dir: Path, run_id: str, experiment: dict,
               cells: list[CellSpec], budget_usd: float,
               resources: dict) -> "RunStore":
        if (not run_id or run_id in {".", ".."} or len(run_id) > 80
                or _SAFE.search(run_id)):
            raise ValueError(
                "run id must be 1-80 characters using only letters, numbers, '.', '_' or '-'")
        run_dir = out_dir / "runs" / run_id
        if run_dir.exists():
            raise FileExistsError(
                f"run '{run_id}' already exists; use --resume {run_id} or choose another --run-id")
        now = datetime.now(timezone.utc).isoformat()
        manifest = {
            "schema_version": SCHEMA_VERSION,
            "run_id": run_id,
            "created_at": now,
            "updated_at": now,
            "experiment": experiment,
            "budget_usd": budget_usd,
            "resources": resources,
            "cells": {
                c.cell_id: {
                    "spec": asdict(c), "status": "pending", "attempts": 0,
                    "reserved_usd": 0.0, "actual_cost_usd": 0.0,
                    "result_file": None,
                } for c in cells
            },
        }
        store = cls(run_dir, manifest)
        store.cells_dir.mkdir(parents=True, exist_ok=True)
        store.workspaces_dir.mkdir(parents=True, exist_ok=True)
        store.save()
        return store

    @classmethod
    def load(cls, manifest_path: Path) -> "RunStore":
        manifest = json.loads(manifest_path.read_text())
        if manifest.get("schema_version") != SCHEMA_VERSION:
            raise ValueError(
                f"unsupported run manifest schema {manifest.get('schema_version')!r}; "
                f"expected {SCHEMA_VERSION}")
        store = cls(manifest_path.parent, manifest)
        # A process cannot still own these states if we are resuming its manifest.
        # Their private workspaces are disposable and will be rebuilt from base.
        for entry in store.manifest["cells"].values():
            if entry["status"] in {"running", "scoring"}:
                entry["status"] = "pending"
                entry["reserved_usd"] = 0.0
        store.save()
        return store

    def save(self) -> None:
        self.manifest["updated_at"] = datetime.now(timezone.utc).isoformat()
        atomic_write_json(self.manifest_path, self.manifest)

    def specs(self) -> list[CellSpec]:
        specs = [CellSpec(**entry["spec"])
                 for entry in self.manifest["cells"].values()]
        return sorted(specs, key=lambda c: c.index)

    def status(self, cell_id: str) -> str:
        return self.manifest["cells"][cell_id]["status"]

    @property
    def spent_usd(self) -> float:
        return sum(float(c.get("actual_cost_usd", 0.0) or 0.0)
                   for c in self.manifest["cells"].values()
                   if c["status"] == "completed")

    def mark_running(self, cell_id: str, reserved_usd: float) -> None:
        entry = self.manifest["cells"][cell_id]
        entry["status"] = "running"
        entry["attempts"] += 1
        entry["reserved_usd"] = round(reserved_usd, 6)
        self.save()

    def mark_scoring(self, cell_id: str) -> None:
        self.manifest["cells"][cell_id]["status"] = "scoring"
        self.save()

    def mark_pending(self, cell_id: str) -> None:
        entry = self.manifest["cells"][cell_id]
        entry["status"] = "pending"
        entry["reserved_usd"] = 0.0
        self.save()

    def complete(self, cell_id: str, payload: dict, actual_cost_usd: float) -> None:
        cell_dir = self.cells_dir / cell_id
        result_path = cell_dir / "result.json"
        atomic_write_json(result_path, payload)
        entry = self.manifest["cells"][cell_id]
        entry.update({
            "status": "completed",
            "reserved_usd": 0.0,
            "actual_cost_usd": actual_cost_usd,
            "result_file": str(result_path.relative_to(self.run_dir)),
        })
        self.save()

    def skip_budget(self, cell_id: str, reason: str) -> None:
        entry = self.manifest["cells"][cell_id]
        entry.update({"status": "skipped_budget", "reserved_usd": 0.0,
                      "skip_reason": reason})
        self.save()

    def reopen_budget_skips(self) -> None:
        changed = False
        for entry in self.manifest["cells"].values():
            if entry["status"] == "skipped_budget":
                entry["status"] = "pending"
                entry.pop("skip_reason", None)
                changed = True
        if changed:
            self.save()

    def completed_payloads(self) -> list[dict]:
        payloads: list[dict] = []
        for spec in self.specs():
            entry = self.manifest["cells"][spec.cell_id]
            if entry["status"] != "completed" or not entry.get("result_file"):
                continue
            path = self.run_dir / entry["result_file"]
            if not path.exists():
                # A manifest claiming completion without its atomic result is not
                # trustworthy; make it recoverable on the next scheduler pass.
                entry["status"] = "pending"
                entry["actual_cost_usd"] = 0.0
                entry["result_file"] = None
                self.save()
                continue
            payloads.append(json.loads(path.read_text()))
        return payloads

    def summary_counts(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for entry in self.manifest["cells"].values():
            counts[entry["status"]] = counts.get(entry["status"], 0) + 1
        return counts


def resolve_manifest(out_dir: Path, resume: str) -> Path:
    candidate = Path(resume)
    if candidate.is_dir():
        candidate = candidate / "manifest.json"
    elif candidate.name != "manifest.json" and not candidate.exists():
        candidate = out_dir / "runs" / resume / "manifest.json"
    if not candidate.exists():
        raise FileNotFoundError(f"resume manifest not found: {candidate}")
    return candidate.resolve()


@contextmanager
def run_lease(run_dir: Path):
    """Prevent two coordinators from executing/resuming the same run."""
    run_dir.mkdir(parents=True, exist_ok=True)
    lock_path = run_dir / ".coordinator.lock"
    handle = lock_path.open("a+")
    acquired = False
    try:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            handle.seek(0)
            owner = handle.read().strip() or "another process"
            raise RuntimeError(
                f"run is already owned by {owner}; refusing duplicate execution") from exc
        acquired = True
        handle.seek(0)
        handle.truncate()
        handle.write(f"pid {os.getpid()}\n")
        handle.flush()
        yield
    finally:
        try:
            if acquired:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        finally:
            handle.close()
