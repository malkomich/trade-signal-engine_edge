from __future__ import annotations

from typing import cast

import pytest

from trade_signal_edge.config import RuntimeConfig
from trade_signal_edge.providers import (
    ProviderConfig,
    ProviderName,
    build_provider,
    load_provider_config,
    provider_policies,
    resolve_provider_name,
    selected_provider_policy,
)


def test_provider_policies_cover_supported_providers() -> None:
    policies = provider_policies()

    assert [policy.name for policy in policies] == ["synthetic", "alpaca"]
    assert policies[0].cost_tier == "free"
    assert policies[1].requires_credentials is True
    assert "raw bars" in policies[1].redistribution


def test_selected_provider_policy_tracks_config() -> None:
    policy = selected_provider_policy(ProviderConfig(name="alpaca"))

    assert policy.name == "alpaca"
    assert policy.access_model == "external market-data API"


def test_provider_config_repr_masks_secrets() -> None:
    rendered = repr(ProviderConfig(name="alpaca", alpaca_api_key_id="key", alpaca_api_secret_key="secret"))

    assert "secret" not in rendered
    assert "alpaca_api_key_id" not in rendered


def test_resolve_provider_name_accepts_case_insensitive_values() -> None:
    assert resolve_provider_name("ALPACA") == "alpaca"
    assert resolve_provider_name(" synthetic ") == "synthetic"
    assert resolve_provider_name("") == "synthetic"


def test_resolve_provider_name_rejects_invalid_values() -> None:
    with pytest.raises(ValueError, match="unsupported provider"):
        resolve_provider_name("invalid")


def test_load_provider_config_uses_runtime_values() -> None:
    runtime = RuntimeConfig(
        provider="alpaca",
        alpaca_feed="sip",
        alpaca_api_key_id="key",
        alpaca_api_secret_key="secret",
    )

    config = load_provider_config(runtime)

    assert config.name == "alpaca"
    assert config.alpaca_feed == "sip"
    assert config.alpaca_api_key_id == "key"
    assert config.alpaca_api_secret_key == "secret"


def test_build_provider_defaults_to_synthetic() -> None:
    provider = build_provider(ProviderConfig())

    assert provider.__class__.__name__ == "SyntheticProvider"


def test_build_provider_requires_alpaca_credentials() -> None:
    with pytest.raises(ValueError, match="alpaca provider requires"):
        build_provider(ProviderConfig(name="alpaca"))


def test_build_provider_rejects_unimplemented_provider_name() -> None:
    provider = ProviderConfig(name=cast(ProviderName, "polygon"))

    with pytest.raises(NotImplementedError, match="not implemented"):
        build_provider(provider)
