from __future__ import annotations

import pytest
from typing import cast

from trade_signal_edge.providers import (
    ProviderSelection,
    ProviderName,
    build_provider,
    load_provider_selection,
    resolve_provider_name,
)


def test_build_provider_defaults_to_synthetic() -> None:
    provider = build_provider(ProviderSelection())

    assert provider.__class__.__name__ == "SyntheticProvider"


def test_build_provider_requires_alpaca_credentials() -> None:
    with pytest.raises(ValueError, match="alpaca provider requires"):
        build_provider(ProviderSelection(name="alpaca"))


def test_build_provider_selects_alpaca_when_configured() -> None:
    provider = build_provider(
        ProviderSelection(
            name="alpaca",
            alpaca_api_key_id="key",
            alpaca_api_secret_key="secret",
        )
    )

    assert provider.__class__.__name__ == "AlpacaProvider"


def test_build_provider_rejects_unimplemented_provider_name() -> None:
    provider = ProviderSelection(name=cast(ProviderName, "polygon"))

    with pytest.raises(NotImplementedError, match="not implemented"):
        build_provider(provider)


def test_resolve_provider_name_accepts_case_insensitive_values() -> None:
    assert resolve_provider_name("ALPACA") == "alpaca"
    assert resolve_provider_name(" synthetic ") == "synthetic"
    assert resolve_provider_name("") == "synthetic"


def test_resolve_provider_name_rejects_invalid_values() -> None:
    with pytest.raises(ValueError, match="unsupported provider"):
        resolve_provider_name("invalid")


def test_load_provider_selection_reads_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("EDGE_PROVIDER", "alpaca")
    monkeypatch.setenv("ALPACA_DATA_FEED", "sip")
    monkeypatch.setenv("ALPACA_API_KEY_ID", "key")
    monkeypatch.setenv("ALPACA_API_SECRET_KEY", "secret")

    selection = load_provider_selection()

    assert selection.name == "alpaca"
    assert selection.alpaca_feed == "sip"
    assert selection.alpaca_api_key_id == "key"
    assert selection.alpaca_api_secret_key == "secret"


def test_provider_selection_repr_masks_secrets() -> None:
    selection = ProviderSelection(name="alpaca", alpaca_api_key_id="key", alpaca_api_secret_key="secret")

    rendered = repr(selection)

    assert "secret" not in rendered
    assert "alpaca_api_key_id" not in rendered
