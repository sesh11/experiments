"""Timing aggregation and human-readable formatting for evaluation runs."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Iterable


PHASE_FIELDS = (
    "queue_wait_seconds",
    "workspace_setup_seconds",
    "runtime_wall_seconds",
    "workspace_cleanup_seconds",
    "agent_wall_seconds",
    "scoring_queue_wait_seconds",
    "docker_lock_wait_seconds",
    "docker_scoring_wall_seconds",
    "judge_wall_seconds",
    "scoring_wall_seconds",
    "cell_elapsed_seconds",
)


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def format_duration(seconds: float | int | None) -> str:
    """Format a duration compactly without losing multi-hour context."""
    if seconds is None:
        return "--:--"
    measured = max(0.0, float(seconds))
    if measured < 1:
        return f"{measured:.2f}s"
    if measured < 10:
        return f"{measured:.1f}s"
    total = round(measured)
    hours, remainder = divmod(total, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours:d}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"


def _percentile(values: list[float], fraction: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * fraction
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def _stats(values: Iterable[float]) -> dict:
    measured = [max(0.0, float(value)) for value in values]
    if not measured:
        return {"count": 0, "total": 0.0, "mean": 0.0,
                "p50": 0.0, "p95": 0.0, "max": 0.0}
    return {
        "count": len(measured),
        "total": round(sum(measured), 3),
        "mean": round(sum(measured) / len(measured), 3),
        "p50": round(_percentile(measured, 0.50), 3),
        "p95": round(_percentile(measured, 0.95), 3),
        "max": round(max(measured), 3),
    }


def _phase_stats(rows: list[dict]) -> dict[str, dict]:
    return {
        field: _stats(row[field] for row in rows if row.get(field) is not None)
        for field in PHASE_FIELDS
    }


def summarize_timing(rows: list[dict], *, execution_history: list[dict] | None = None,
                     created_at: str | None = None) -> dict:
    """Build machine-readable overall and per-configuration timing summaries."""
    history = execution_history or []
    active_seconds = sum(float(item.get("wall_seconds", 0) or 0) for item in history)
    calendar_seconds = None
    if created_at:
        try:
            created = datetime.fromisoformat(created_at)
            if created.tzinfo is None:
                created = created.replace(tzinfo=timezone.utc)
            calendar_seconds = max(
                0.0, (datetime.now(timezone.utc) - created).total_seconds())
        except (TypeError, ValueError):
            calendar_seconds = None

    groups: dict[tuple[str, str], list[dict]] = {}
    for row in rows:
        key = (str(row.get("variant", "unknown")),
               str(row.get("config", "default")))
        groups.setdefault(key, []).append(row)

    by_variant_config = []
    for (variant, config), group in sorted(groups.items()):
        by_variant_config.append({
            "variant": variant,
            "config": config,
            "cells": len(group),
            "phases": _phase_stats(group),
        })

    return {
        "generated_at": utc_now_iso(),
        "run": {
            "completed_cells": len(rows),
            "invocations": len(history),
            "active_wall_seconds": round(active_seconds, 3),
            "calendar_elapsed_seconds": (
                round(calendar_seconds, 3) if calendar_seconds is not None else None),
            "throughput_cells_per_hour": (
                round(len(rows) * 3600 / active_seconds, 3)
                if rows and active_seconds else None),
        },
        "overall": {"cells": len(rows), "phases": _phase_stats(rows)},
        "by_variant_config": by_variant_config,
        "notes": {
            "active_wall_seconds": (
                "Sum of scheduler invocation time; excludes downtime between resumes."),
            "calendar_elapsed_seconds": (
                "Wall-clock age from run creation; includes downtime between resumes."),
            "phase_totals": (
                "Work-seconds summed across cells; parallel phase totals can exceed run wall time."),
        },
    }
