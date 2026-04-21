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
    monkeypatch.delenv("EDGE_DEPLOYMENT_PROFILE", raising=False)
    monkeypatch.delenv("EDGE_LOG_LEVEL", raising=False)
    monkeypatch.delenv("EDGE_METRICS_ENABLED", raising=False)
    monkeypatch.delenv("EDGE_SECRET_SOURCE", raising=False)

    runtime = load_runtime_config()

    assert runtime.symbol == "AAPL"
    assert runtime.bars == 60
    assert runtime.provider == "synthetic"
    assert runtime.api_base_url is None
    assert runtime.alpaca_feed == "iex"
    assert runtime.alpaca_api_key_id is None
    assert runtime.alpaca_api_secret_key is None
    assert runtime.deployment_profile == "pi"
    assert runtime.log_level == "INFO"
    assert runtime.metrics_enabled is False
    assert runtime.secret_source == "environment"


def test_load_runtime_config_reads_environment(monkeypatch) -> None:
    monkeypatch.setenv("EDGE_SYMBOL", "MSFT")
    monkeypatch.setenv("EDGE_BARS", "120")
    monkeypatch.setenv("EDGE_PROVIDER", "alpaca")
    monkeypatch.setenv("API_BASE_URL", "https://api.example.com")
    monkeypatch.setenv("ALPACA_DATA_FEED", "sip")
    monkeypatch.setenv("ALPACA_API_KEY_ID", "key")
    monkeypatch.setenv("ALPACA_API_SECRET_KEY", "secret")
    monkeypatch.setenv("EDGE_DEPLOYMENT_PROFILE", "pi-zero")
    monkeypatch.setenv("EDGE_LOG_LEVEL", "debug")
    monkeypatch.setenv("EDGE_METRICS_ENABLED", "yes")
    monkeypatch.setenv("EDGE_SECRET_SOURCE", "sealed-secret")

    runtime = load_runtime_config()

    assert runtime.symbol == "MSFT"
    assert runtime.bars == 120
    assert runtime.provider == "alpaca"
    assert runtime.api_base_url == "https://api.example.com"
    assert runtime.alpaca_feed == "sip"
    assert runtime.alpaca_api_key_id == "key"
    assert runtime.alpaca_api_secret_key == "secret"
    assert runtime.deployment_profile == "pi-zero"
    assert runtime.log_level == "DEBUG"
    assert runtime.metrics_enabled is True
    assert runtime.secret_source == "sealed-secret"


def test_load_runtime_config_invalid_log_level_falls_back(monkeypatch) -> None:
    monkeypatch.setenv("EDGE_LOG_LEVEL", "verbose")

    runtime = load_runtime_config()

    assert runtime.log_level == "INFO"
