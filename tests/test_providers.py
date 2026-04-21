import pytest

from trade_signal_edge.providers import ProviderSelection, build_provider


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
