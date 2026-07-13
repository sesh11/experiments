"""Variant registry + the orchestration patterns.

A *variant* names one experiment cell: an orchestration pattern bound to a
runtime and model(s). The eval driver only ever calls `run_variant(name, task,
cfg)` and consumes the returned PolicyResult — everything else is internal.

Legacy fusion variants (frontier_only / sidekick_only / scout) delegate to
fusion.policies unchanged until the parity milestone retires them.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

from fusion import config
from fusion.llm import BudgetExceeded, Ledger
from fusion.workspace import Workspace
from runtimes.base import AgentRuntime, RuntimeUnavailable


@dataclass
class PolicyResult:
    variant: str
    resolved: bool
    diff: str
    summary: str
    ledger: dict
    budget_hit: bool = False
    error: str = ""
    resolve_detail: str = ""
    steps: int = 0                       # tool-loop steps the main agent used
    finished: bool = False               # did the agent call finish()?
    trace: list = field(default_factory=list)         # main-agent tool-call trace
    scout_trace: list = field(default_factory=list)   # sub-agent (scout) trace
    score_artifacts: dict = field(default_factory=dict)  # pytest evidence from scoring


def make_ws(task: dict) -> Workspace:
    """Native tasks are copied to a temp dir; swebench tasks are edited in place
    (the venv's editable install points at the clone), reset via git first."""
    if task.get("in_place"):
        if task.get("reset"):
            task["reset"](task)
        return Workspace(task["template_dir"], task["test_cmd"],
                         use_git=True, test_timeout=task.get("test_timeout", 600))
    return Workspace.from_template(task["template_dir"], task["test_cmd"],
                                   test_timeout=task.get("test_timeout", 120))


def finalize(variant: str, task: dict, ws: Workspace, ledger: Ledger, *,
             summary: str, steps: int, finished: bool, trace: list,
             scout_trace: list | None = None, budget_hit: bool = False,
             error: str = "", extra: dict | None = None) -> PolicyResult:
    """Capture the diff, route scoring, package the result, clean up.

    Shared by every pattern (and the legacy fusion policies) so scoring
    behavior stays identical across runtimes.
    """
    err = error
    # Capture the agent's diff BEFORE scoring: the scorer applies/reverts the
    # gold test patch and must not pollute the recorded change.
    diff = ws.diff()
    # Scoring routes three ways:
    #   * docker backend -> deferred; the driver scores `diff` via the official
    #     SWE-bench harness (a pinned env the local checkout can't reproduce).
    #   * local swebench  -> the local pytest scorer (task["scorer"]).
    #   * native          -> the task's own test command.
    detail = ""
    if task.get("backend") == "docker":
        resolved, detail = False, "(pending docker scoring)"
    elif task.get("scorer"):
        try:
            scored = task["scorer"](task)
            if isinstance(scored, tuple):
                resolved, detail = bool(scored[0]), str(scored[1])
            else:
                resolved = bool(scored)
        except Exception as exc:  # noqa: BLE001
            resolved, err = False, err or f"scorer: {exc}"
    else:
        resolved, _ = ws.run_tests()
    artifacts = dict(task.get("_score_artifacts", {}))
    if extra:
        artifacts["runtime_extra"] = extra
    out = PolicyResult(
        variant=variant, resolved=resolved, diff=diff, summary=summary,
        ledger=ledger.summary(), budget_hit=budget_hit, error=err,
        resolve_detail=detail, steps=steps, finished=finished, trace=trace,
        scout_trace=scout_trace or [],
        score_artifacts=artifacts,
    )
    ws.cleanup()
    return out


def single_agent(task: dict, cfg: config.RunConfig, *, variant: str,
                 runtime: AgentRuntime, model: str) -> PolicyResult:
    """One agent, one runtime, full task — the parity-milestone pattern."""
    ledger = Ledger(cap_usd=cfg.budget_usd)
    ok, reason = runtime.preflight()
    if not ok:
        return PolicyResult(variant=variant, resolved=False, diff="", summary="",
                            ledger=ledger.summary(),
                            error=f"runtime '{runtime.name}' unavailable: {reason}")
    ws = make_ws(task)
    summary, steps, finished = "", 0, False
    trace: list = []
    extra: dict = {}
    budget_hit = False
    err = ""
    try:
        rr = runtime.run(task, ws, model=model, ledger=ledger, cfg=cfg)
        summary, steps, finished = rr.summary, rr.steps, rr.finished
        trace, extra = rr.trace, rr.extra
    except BudgetExceeded as exc:
        budget_hit = True
        summary = f"(budget hit) {exc}"
    except Exception as exc:  # keep the run alive; record the failure
        err = f"{type(exc).__name__}: {exc}"
    return finalize(variant, task, ws, ledger, summary=summary, steps=steps,
                    finished=finished, trace=trace, budget_hit=budget_hit,
                    error=err, extra=extra)


# --- registry ---------------------------------------------------------------
def _fusion_runtime() -> AgentRuntime:
    from runtimes.fusion_rt import FusionRuntime
    return FusionRuntime()


def _stirrup_runtime() -> AgentRuntime:
    from runtimes.stirrup_rt import StirrupRuntime
    return StirrupRuntime()


def _pi_runtime() -> AgentRuntime:
    from runtimes.pi_rt import PiRuntime
    return PiRuntime()


@dataclass(frozen=True)
class VariantSpec:
    """One experiment cell: pattern + runtime + model(s).

    `sidekick_model` and `pattern` are the seams where Scout delegation and
    confidence-gated routing plug in after the parity milestone.
    """
    runtime_factory: Callable[[], AgentRuntime]
    main_model: str
    sidekick_model: str | None = None
    pattern: str = "single_agent"


_REGISTRY: dict[str, VariantSpec] = {
    "baseline-fusion": VariantSpec(_fusion_runtime, config.MODEL_MAIN),
    "baseline-stirrup": VariantSpec(_stirrup_runtime, config.MODEL_MAIN),
    "baseline-pi": VariantSpec(_pi_runtime, config.MODEL_MAIN),
}

_LEGACY = ("frontier_only", "sidekick_only", "scout")

# Legacy defaults preserved so a bare `python -m eval.run_eval` is unchanged.
ALL_VARIANTS: list[str] = list(_LEGACY)


def run_variant(name: str, task: dict, cfg: config.RunConfig) -> PolicyResult:
    """Run one variant on one task; never raises for an unavailable runtime."""
    if name in _LEGACY:
        from fusion import policies
        return policies.run_variant(name, task, cfg)
    spec = _REGISTRY.get(name)
    if spec is None:
        raise ValueError(f"unknown variant: {name}")
    try:
        runtime = spec.runtime_factory()
    except (ImportError, RuntimeUnavailable) as exc:
        return PolicyResult(variant=name, resolved=False, diff="", summary="",
                            ledger=Ledger(cap_usd=cfg.budget_usd).summary(),
                            error=f"runtime unavailable: {exc}")
    return single_agent(task, cfg, variant=name, runtime=runtime,
                        model=spec.main_model)
