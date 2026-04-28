from __future__ import annotations

from trade_signal_edge.config import load_runtime_config


def test_load_runtime_config_defaults(monkeypatch) -> None:
    monkeypatch.delenv("EDGE_SYMBOL", raising=False)
    monkeypatch.delenv("EDGE_SYMBOLS", raising=False)
    monkeypatch.delenv("EDGE_BENCHMARK_SYMBOL", raising=False)
    monkeypatch.delenv("EDGE_SESSION_ID", raising=False)
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
    assert runtime.symbols == ("AAPL", "AMZN", "GOOGL", "META", "MSFT", "NVDA", "PLTR", "TSLA")
    assert runtime.benchmark_symbol == "QQQ"
    assert runtime.session_id == "nasdaq-live"
    assert runtime.bars == 60
    assert runtime.provider == "synthetic"
    assert runtime.api_base_url is None
    assert runtime.session_timezone == "America/New_York"
    assert runtime.entry_threshold == 0.7
    assert runtime.exit_threshold == 0.6
    assert runtime.buy_signal_weights["sma"] == 0.8
    assert runtime.sell_signal_weights["sma"] == 0.8
    assert runtime.buy_timeframe_weights == {"1m": 1.0, "5m": 0.75, "15m": 0.5}
    assert runtime.sell_timeframe_weights == {"1m": 1.0, "5m": 0.75, "15m": 0.5}
    assert runtime.alpaca_feed == "iex"
    assert runtime.alpaca_api_key_id is None
    assert runtime.alpaca_api_secret_key is None
    assert runtime.deployment_profile == "pi"
    assert runtime.log_level == "INFO"
    assert runtime.metrics_enabled is False
    assert runtime.secret_source == "environment"


def test_load_runtime_config_reads_environment(monkeypatch) -> None:
    monkeypatch.setenv("EDGE_SYMBOL", "MSFT")
    monkeypatch.setenv("EDGE_SYMBOLS", "MSFT,NVDA")
    monkeypatch.setenv("EDGE_BARS", "120")
    monkeypatch.setenv("EDGE_PROVIDER", "alpaca")
    monkeypatch.setenv("API_BASE_URL", "https://api.example.com")
    monkeypatch.setenv("EDGE_BENCHMARK_SYMBOL", "QQQ")
    monkeypatch.setenv("EDGE_SESSION_ID", "session-42")
    monkeypatch.setenv("ALPACA_DATA_FEED", "sip")
    monkeypatch.setenv("ALPACA_API_KEY_ID", "key")
    monkeypatch.setenv("ALPACA_API_SECRET_KEY", "secret")
    monkeypatch.setenv("EDGE_DEPLOYMENT_PROFILE", "pi-zero")
    monkeypatch.setenv("EDGE_LOG_LEVEL", "debug")
    monkeypatch.setenv("EDGE_METRICS_ENABLED", "yes")
    monkeypatch.setenv("EDGE_SECRET_SOURCE", "sealed-secret")

    runtime = load_runtime_config()

    assert runtime.symbol == "MSFT"
    assert runtime.symbols == ("MSFT", "NVDA")
    assert runtime.benchmark_symbol == "QQQ"
    assert runtime.session_id == "session-42"
    assert runtime.bars == 120
    assert runtime.provider == "alpaca"
    assert runtime.api_base_url == "https://api.example.com"
    assert runtime.session_timezone == "America/New_York"
    assert runtime.entry_threshold == 0.7
    assert runtime.exit_threshold == 0.6
    assert runtime.alpaca_feed == "sip"
    assert runtime.alpaca_api_key_id == "key"
    assert runtime.alpaca_api_secret_key == "secret"
    assert runtime.deployment_profile == "pi-zero"
    assert runtime.log_level == "DEBUG"
    assert runtime.metrics_enabled is True
    assert runtime.secret_source == "sealed-secret"


def test_load_runtime_config_reads_secret_files(monkeypatch, tmp_path) -> None:
    key_id = tmp_path / "alpaca_key_id"
    key_secret = tmp_path / "alpaca_key_secret"
    key_id.write_text("key-file", encoding="utf-8")
    key_secret.write_text("secret-file", encoding="utf-8")

    monkeypatch.delenv("ALPACA_API_KEY_ID", raising=False)
    monkeypatch.delenv("ALPACA_API_SECRET_KEY", raising=False)
    monkeypatch.setenv("ALPACA_API_KEY_ID_FILE", str(key_id))
    monkeypatch.setenv("ALPACA_API_SECRET_KEY_FILE", str(key_secret))

    runtime = load_runtime_config()

    assert runtime.alpaca_api_key_id == "key-file"
    assert runtime.alpaca_api_secret_key == "secret-file"


def test_load_runtime_config_invalid_log_level_falls_back(monkeypatch) -> None:
    monkeypatch.setenv("EDGE_LOG_LEVEL", "verbose")

    runtime = load_runtime_config()

    assert runtime.log_level == "INFO"


def test_load_runtime_config_blank_benchmark_and_session_fallback_to_defaults(monkeypatch) -> None:
    monkeypatch.setenv("EDGE_BENCHMARK_SYMBOL", "   ")
    monkeypatch.setenv("EDGE_SESSION_ID", "   ")

    runtime = load_runtime_config()

    assert runtime.benchmark_symbol == "QQQ"
    assert runtime.session_id == "nasdaq-live"


def test_load_runtime_config_legacy_symbol_env_accepts_comma_separated_values(monkeypatch) -> None:
    monkeypatch.delenv("EDGE_SYMBOLS", raising=False)
    monkeypatch.setenv("EDGE_SYMBOL", "MSFT, NVDA, MSFT")

    runtime = load_runtime_config()

    assert runtime.symbols == ("MSFT", "NVDA")
    assert runtime.symbol == "MSFT"
