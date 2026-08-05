from __future__ import annotations

import pytest

from fusion import config
from eval.run_eval import _config_from_args, build_parser
from eval import tasks


def test_api_model_preserves_anthropic_defaults_across_providers() -> None:
    assert config.api_model("anthropic", "claude-sonnet-5") == "claude-sonnet-5"
    assert config.api_model("anthropic", "anthropic/claude-sonnet-5") == "claude-sonnet-5"
    assert config.api_model("openrouter", "claude-sonnet-5") == "anthropic/claude-sonnet-5"
    assert config.api_model("openrouter", "openai/gpt-5") == "openai/gpt-5"


def test_provider_model_validation_is_actionable() -> None:
    with pytest.raises(config.ConfigurationError, match="use --provider openrouter"):
        config.api_model("anthropic", "openai/gpt-5")
    with pytest.raises(config.ConfigurationError, match="needs a provider prefix"):
        config.api_model("openrouter", "gpt-5")
    with pytest.raises(config.ConfigurationError, match="unsupported provider"):
        config.api_model("unknown", "model")


def test_litellm_model_uses_the_selected_gateway() -> None:
    assert config.litellm_model("anthropic", "claude-sonnet-5") == (
        "anthropic/claude-sonnet-5"
    )
    assert config.litellm_model("openrouter", "anthropic/claude-sonnet-5") == (
        "openrouter/anthropic/claude-sonnet-5"
    )


def test_pricing_normalization_does_not_strip_unrelated_providers() -> None:
    assert config.pricing_key("claude-sonnet-5-20260801") == (
        "anthropic/claude-sonnet-5"
    )
    assert config.pricing_key("anthropic/claude-haiku-4-5:floor") == (
        "anthropic/claude-haiku-4-5"
    )
    assert config.pricing_key("other/claude-sonnet-5") == "other/claude-sonnet-5"
    assert config.pricing_key("anthropic/claude-sonnet-50") == (
        "anthropic/claude-sonnet-50"
    )
    with pytest.raises(config.PricingUnavailable, match="no pinned pricing"):
        config.cost_for("openai/gpt-5", input_tokens=1, output_tokens=1)


def test_run_config_validates_main_sidekick_and_judge_models() -> None:
    cfg = config.RunConfig(
        provider="openrouter",
        main_model="anthropic/claude-sonnet-5",
        sidekick_model="anthropic/claude-haiku-4-5",
        judge_provider="openrouter",
        judge_model="openai/gpt-5",
    )
    assert cfg.provider == "openrouter"
    assert cfg.judge_model == "openai/gpt-5"


def test_cli_provider_selection_flows_to_models_and_judge(monkeypatch) -> None:
    monkeypatch.delenv("JUDGE_PROVIDER", raising=False)
    monkeypatch.delenv("JUDGE_MODEL", raising=False)
    args = build_parser().parse_args([
        "--provider", "openrouter",
        "--main-model", "openai/gpt-5",
        "--sidekick-model", "google/gemini-3-flash",
    ])
    cfg = _config_from_args(args)
    assert cfg.provider == "openrouter"
    assert cfg.main_model == "openai/gpt-5"
    assert cfg.sidekick_model == "google/gemini-3-flash"
    assert cfg.judge_provider == "openrouter"
    assert cfg.judge_model == "openai/gpt-5"


def test_native_tasks_use_the_active_python_interpreter() -> None:
    assert all(" -m pytest -q" in task["test_cmd"]
               for task in tasks.load_native())
