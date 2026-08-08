"""EC2-friendly worker sizing and lightweight dispatch backpressure."""

from __future__ import annotations

import os
import shutil
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path

GIB = 1024 ** 3


@dataclass(frozen=True)
class ResourceSnapshot:
    cpus: int
    memory_available_bytes: int
    disk_available_bytes: int
    load_1m: float = 0.0

    def summary(self) -> dict:
        out = asdict(self)
        out["memory_available_gib"] = round(self.memory_available_bytes / GIB, 2)
        out["disk_available_gib"] = round(self.disk_available_bytes / GIB, 2)
        return out


def _available_memory() -> int:
    # Linux/EC2: MemAvailable includes reclaimable cache and is the right signal
    # for deciding whether another checkout/test process can safely start.
    meminfo = Path("/proc/meminfo")
    if meminfo.exists():
        values = {}
        for line in meminfo.read_text().splitlines():
            key, _, raw = line.partition(":")
            if raw:
                values[key] = int(raw.strip().split()[0]) * 1024
        if values.get("MemAvailable"):
            return values["MemAvailable"]
    # macOS developer runs. EC2 never takes this path.
    try:
        proc = subprocess.run(
            ["sysctl", "-n", "hw.memsize"], capture_output=True,
            text=True, timeout=5,
        )
        if proc.returncode == 0:
            return int(proc.stdout.strip())
    except Exception:  # noqa: BLE001
        pass
    try:
        pages = os.sysconf("SC_AVPHYS_PAGES")
        page_size = os.sysconf("SC_PAGE_SIZE")
        return int(pages * page_size)
    except (ValueError, OSError, AttributeError):
        return 4 * GIB  # cautious fallback


def snapshot(path: Path) -> ResourceSnapshot:
    try:
        load_1m = float(os.getloadavg()[0])
    except (AttributeError, OSError):
        load_1m = 0.0
    return ResourceSnapshot(
        cpus=max(1, os.cpu_count() or 1),
        memory_available_bytes=max(1, _available_memory()),
        disk_available_bytes=max(1, shutil.disk_usage(path).free),
        load_1m=round(load_1m, 3),
    )


def _parse(value: str | int, *, label: str) -> int | None:
    if isinstance(value, int):
        if value < 1:
            raise ValueError(f"{label} must be 'auto' or a positive integer")
        return value
    if value == "auto":
        return None
    try:
        parsed = int(value)
    except ValueError as exc:
        raise ValueError(f"{label} must be 'auto' or a positive integer") from exc
    if parsed < 1:
        raise ValueError(f"{label} must be 'auto' or a positive integer")
    return parsed


def choose_workers(*, workers: str | int, docker_workers: str | int,
                   queued_cells: int, docker_enabled: bool,
                   path: Path) -> tuple[int, int, ResourceSnapshot]:
    """Resolve auto/N settings using conservative CPU, RAM, and disk headroom.

    Agent cells spend much of their time waiting on model calls but can run
    repository tests, so auto allows one per vCPU while budgeting ~2 GiB each.
    Docker scorers are heavier: ~2 vCPU, ~6 GiB, and ~8 GiB free disk each.
    Explicit values are honored exactly; runtime backpressure still stops new
    dispatch when the host is under acute memory/disk pressure.
    """
    snap = snapshot(path)
    queued = max(1, queued_cells)
    explicit_workers = _parse(workers, label="--workers")
    explicit_docker = _parse(docker_workers, label="--docker-workers")

    # Always leave at least 2 GiB / 15% RAM and 10 GiB disk to the OS, Docker
    # daemon, package caches, and the coordinator.
    memory_headroom = max(2 * GIB, int(snap.memory_available_bytes * 0.15))
    usable_memory = max(GIB, snap.memory_available_bytes - memory_headroom)
    usable_disk = max(GIB, snap.disk_available_bytes - 10 * GIB)

    if docker_enabled:
        auto_docker = max(1, min(
            queued,
            max(1, snap.cpus // 3),
            # Use 8 GiB for admission even though steady-state observation is
            # closer to 6 GiB; image pulls/builds have transient peaks.
            max(1, usable_memory // (8 * GIB)),
            max(1, usable_disk // (8 * GIB)),
        ))
        docker_count = explicit_docker or int(auto_docker)
    else:
        # This pool only performs cheap judge/finalization work for local runs;
        # do not reserve Docker-class memory for it.
        docker_count = explicit_docker or 1

    # Auto agent sizing accounts for the scoring pool running at the same time.
    # Explicit settings remain exact by design and are reported prominently.
    scoring_memory = docker_count * 6 * GIB if docker_enabled else 0
    scoring_cpus = docker_count * 2 if docker_enabled else 0
    memory_after_scoring = max(GIB, usable_memory - scoring_memory)
    cpu_after_scoring = max(1, snap.cpus - scoring_cpus)
    auto_agents = max(1, min(
        queued,
        cpu_after_scoring,
        max(1, memory_after_scoring // (2 * GIB)),
        max(1, usable_disk // (3 * GIB)),
    ))
    agent_count = explicit_workers or int(auto_agents)

    if not docker_enabled and explicit_docker is None:
        docker_count = min(queued, agent_count)
    return agent_count, docker_count, snap


def dispatch_headroom(path: Path) -> tuple[bool, str, ResourceSnapshot]:
    """Fast pressure check before starting another cell.

    Fixed pools bound steady-state use; this check handles a host whose memory
    or disk was consumed by another process after the run began.
    """
    snap = snapshot(path)
    if snap.memory_available_bytes < GIB:
        return (False,
                f"available memory is only {snap.memory_available_bytes / GIB:.1f} GiB",
                snap)
    if snap.disk_available_bytes < 5 * GIB:
        return (False,
                f"available disk is only {snap.disk_available_bytes / GIB:.1f} GiB",
                snap)
    return True, "ok", snap
