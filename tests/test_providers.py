from __future__ import annotations

import os
from unittest.mock import patch

import pytest

from trade_signal_edge.providers import (
    ProviderSelection,
    SyntheticProvider,
    build_provider,
    load_provider_selection,
    resolve_provider_name,
)


def test_resolve_provider_name_accepts_valid_values() -> None:
    assert resolve_provider_name("synthetic") == "synthetic"
    assert resolve_provider_name("ALPACA") == "alpaca"
    assert resolve_provider_name("  ") == "synthetic"


def test_resolve_provider_name_rejects_invalid_values() -> None:
    with pytest.raises(ValueError, match="unsupported provider"):
        resolve_provider_name("invalid")


def test_load_provider_selection_reads_environment() -> None:
    with patch.dict(
        os.environ,
        {
            "EDGE_PROVIDER": "alpaca",
            "ALPACA_API_KEY_ID": "key-id",
            "ALPACA_API_SECRET_KEY": "secret",
            "ALPACA_DATA_FEED": "sip",
        },
        clear=True,
    ):
        selection = load_provider_selection()

    assert selection == ProviderSelection(
        name="alpaca",
        alpaca_api_key_id="key-id",
        alpaca_api_secret_key="secret",
        alpaca_feed="sip",
    )


def test_build_provider_returns_synthetic_by_default() -> None:
    provider = build_provider(ProviderSelection())

    assert isinstance(provider, SyntheticProvider)
