from __future__ import annotations

from trade_signal_edge.config import load_runtime_config


def test_load_runtime_config_defaults(monkeypatch) -> None:
    monkeypatch.delenv("EDGE_SYMBOL", raising=False)
    monkeypatch.delenv("EDGE_BARS", raising=False)
    monkeypatch.delenv("EDGE_PROVIDER", raising=False)
    monkeypatch.delenv("API_BASE_URL", raising=False)
    monkeypatch.delenv("ALPACA_DATA_FEED", raising=False)
    monkeypatch.delenv("ALPACA_API_KEY_ID", raising=False)
    monkeypatch.delenv("ALPACA_API_SECRET_KEY", raising=False)

    runtime = load_runtime_config()

    assert runtime.symbol == "AAPL"
    assert runtime.bars == 60
    assert runtime.provider == "synthetic"
    assert runtime.api_base_url is None
    assert runtime.alpaca_feed == "iex"
    assert runtime.alpaca_api_key_id is None
    assert runtime.alpaca_api_secret_key is None


def test_load_runtime_config_reads_environment(monkeypatch) -> None:
    monkeypatch.setenv("EDGE_SYMBOL", "MSFT")
    monkeypatch.setenv("EDGE_BARS", "120")
    monkeypatch.setenv("EDGE_PROVIDER", "alpaca")
    monkeypatch.setenv("API_BASE_URL", "https://api.example.com")
    monkeypatch.setenv("ALPACA_DATA_FEED", "sip")
    monkeypatch.setenv("ALPACA_API_KEY_ID", "key")
    monkeypatch.setenv("ALPACA_API_SECRET_KEY", "secret")

    runtime = load_runtime_config()

    assert runtime.symbol == "MSFT"
    assert runtime.bars == 120
    assert runtime.provider == "alpaca"
    assert runtime.api_base_url == "https://api.example.com"
    assert runtime.alpaca_feed == "sip"
    assert runtime.alpaca_api_key_id == "key"
    assert runtime.alpaca_api_secret_key == "secret"
