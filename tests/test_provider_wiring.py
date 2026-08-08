from __future__ import annotations

import io
from types import SimpleNamespace

import pytest

from fusion import config, policies
from fusion.llm import Ledger
from orchestrator import variants
from runtimes import pi_rt, stirrup_rt


def test_legacy_policy_uses_run_config_models(monkeypatch) -> None:
    seen = {}

    def fake(task, cfg, **kwargs):
        seen.update(kwargs)
        return "ok"

    monkeypatch.setattr(policies, "_run_single_agent", fake)
    cfg = config.RunConfig(
        provider="openrouter",
        main_model="openai/gpt-5",
        sidekick_model="google/gemini-3-flash",
    )
    assert policies.run_variant("frontier_only", {}, cfg) == "ok"
    assert seen["role"] == "main"
    assert seen["model"] == "openai/gpt-5"


def test_registry_baseline_uses_configured_main_model(monkeypatch) -> None:
    seen = {}
    runtime = SimpleNamespace()
    monkeypatch.setitem(
        variants._REGISTRY, "test-runtime",
        variants.VariantSpec(lambda: runtime),
    )

    def fake_single(task, cfg, **kwargs):
        seen.update(kwargs)
        return "ok"

    monkeypatch.setattr(variants, "single_agent", fake_single)
    cfg = config.RunConfig(provider="openrouter", main_model="openai/gpt-5")
    assert variants.run_variant("test-runtime", {}, cfg) == "ok"
    assert seen["runtime"] is runtime
    assert seen["model"] == "openai/gpt-5"


def test_stirrup_qualifies_openrouter_model_and_forwards_gateway_settings(
        monkeypatch, tmp_path) -> None:
    captured = {}

    class Client:
        def __init__(self, **kwargs) -> None:
            captured.update(kwargs)

    async def fake_arun(self, task, ws, client, cfg):
        return "ok"

    monkeypatch.setattr(stirrup_rt, "LedgerLiteLLMClient", Client)
    monkeypatch.setattr(stirrup_rt.StirrupRuntime, "_arun", fake_arun)
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    cfg = config.RunConfig(
        provider="openrouter",
        main_model="anthropic/claude-sonnet-5",
        openrouter_base_url="https://gateway.example/v1",
        openrouter_site_url="https://app.example",
        openrouter_app_name="Fusion",
    )
    ws = SimpleNamespace(root=tmp_path)
    result = stirrup_rt.StirrupRuntime().run(
        {}, ws, model=cfg.main_model, ledger=Ledger(), cfg=cfg)

    assert result == "ok"
    assert captured["model"] == "openrouter/anthropic/claude-sonnet-5"
    assert captured["pricing_model"] == "anthropic/claude-sonnet-5"
    assert captured["api_key"] == "test-key"
    assert captured["kwargs"] == {
        "api_base": "https://gateway.example/v1",
        "extra_headers": {
            "HTTP-Referer": "https://app.example",
            "X-OpenRouter-Title": "Fusion",
        },
    }


def test_stirrup_refuses_unpriced_model_before_spending(monkeypatch, tmp_path) -> None:
    cfg = config.RunConfig(provider="openrouter", main_model="openai/gpt-5")
    with pytest.raises(config.PricingUnavailable, match="does not expose"):
        stirrup_rt.StirrupRuntime().run(
            {}, SimpleNamespace(root=tmp_path), model=cfg.main_model,
            ledger=Ledger(), cfg=cfg,
        )


def test_pi_passes_openrouter_provider_and_model(monkeypatch, tmp_path) -> None:
    captured = {}

    class FakeProcess:
        def __init__(self, cmd, **kwargs):
            captured["cmd"] = cmd
            self.stdout = io.StringIO("")
            self.returncode = 0

        def poll(self):
            return self.returncode

        def wait(self, timeout=None):
            return self.returncode

        def kill(self):
            self.returncode = -9

        def terminate(self):
            self.returncode = -15

    monkeypatch.setattr(pi_rt, "_find_pi", lambda: "/bin/pi")
    monkeypatch.setattr(pi_rt.subprocess, "Popen", FakeProcess)
    cfg = config.RunConfig(
        provider="openrouter", main_model="anthropic/claude-sonnet-5")
    pi_rt.PiRuntime().run(
        {"problem_statement": "fix"}, SimpleNamespace(root=tmp_path),
        model=cfg.main_model, ledger=Ledger(), cfg=cfg,
    )
    cmd = captured["cmd"]
    assert cmd[cmd.index("--provider") + 1] == "openrouter"
    assert cmd[cmd.index("--model") + 1] == "anthropic/claude-sonnet-5"
