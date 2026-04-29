from __future__ import annotations

from dataclasses import dataclass, field
import os
from pathlib import Path

from .models import default_signal_weights, default_timeframe_weights


@dataclass(slots=True)
class RuntimeConfig:
    symbol: str = "AAPL"
    symbols: tuple[str, ...] = ("AAPL", "AMZN", "GOOGL", "META", "MSFT", "NVDA", "PLTR", "TSLA")
    benchmark_symbol: str = "QQQ"
    session_id: str = "nasdaq-live"
    bars: int = 60
    provider: str = "synthetic"
    api_base_url: str | None = None
    session_timezone: str = "America/New_York"
    entry_threshold: float = 0.7
    exit_threshold: float = 0.6
    entry_exit_margin: float = 0.1
    entry_gate_cap: float = 0.56
    buy_signal_weights: dict[str, float] = field(default_factory=default_signal_weights)
    sell_signal_weights: dict[str, float] = field(default_factory=default_signal_weights)
    buy_timeframe_weights: dict[str, float] = field(default_factory=default_timeframe_weights)
    sell_timeframe_weights: dict[str, float] = field(default_factory=default_timeframe_weights)
    optimizer_learning_rate: float = 0.12
    optimizer_bias_cap: float = 0.08
    entry_profile: dict[str, float] = field(default_factory=dict)
    exit_profile: dict[str, float] = field(default_factory=dict)
    alpaca_feed: str = "iex"
    alpaca_api_key_id: str | None = None
    alpaca_api_secret_key: str | None = None
    deployment_profile: str = "pi"
    log_level: str = "INFO"
    metrics_enabled: bool = False
    secret_source: str = "environment"


def load_runtime_config() -> RuntimeConfig:
    defaults = RuntimeConfig()
    symbols = _resolve_symbols(defaults)
    symbol = symbols[0] if symbols else defaults.symbol
    return RuntimeConfig(
        symbol=symbol,
        symbols=symbols,
        benchmark_symbol=_resolve_symbol_env("EDGE_BENCHMARK_SYMBOL", defaults.benchmark_symbol),
        session_id=_resolve_text_env("EDGE_SESSION_ID", defaults.session_id),
        bars=int(os.getenv("EDGE_BARS", str(defaults.bars))),
        api_base_url=os.getenv("API_BASE_URL", defaults.api_base_url),
        provider=(os.getenv("EDGE_PROVIDER", defaults.provider) or defaults.provider).strip().lower(),
        alpaca_feed=(os.getenv("ALPACA_DATA_FEED", defaults.alpaca_feed) or defaults.alpaca_feed).strip().lower(),
        alpaca_api_key_id=_read_optional_value("ALPACA_API_KEY_ID", "ALPACA_API_KEY_ID_FILE"),
        alpaca_api_secret_key=_read_optional_value("ALPACA_API_SECRET_KEY", "ALPACA_API_SECRET_KEY_FILE"),
        deployment_profile=(os.getenv("EDGE_DEPLOYMENT_PROFILE", defaults.deployment_profile) or defaults.deployment_profile).strip().lower(),
        log_level=_parse_log_level(os.getenv("EDGE_LOG_LEVEL"), defaults.log_level),
        metrics_enabled=_parse_bool(os.getenv("EDGE_METRICS_ENABLED")),
        secret_source=(os.getenv("EDGE_SECRET_SOURCE", defaults.secret_source) or defaults.secret_source).strip().lower(),
        entry_gate_cap=float(os.getenv("EDGE_ENTRY_GATE_CAP", str(defaults.entry_gate_cap))),
        buy_signal_weights=dict(defaults.buy_signal_weights),
        sell_signal_weights=dict(defaults.sell_signal_weights),
        buy_timeframe_weights=dict(defaults.buy_timeframe_weights),
        sell_timeframe_weights=dict(defaults.sell_timeframe_weights),
    )


def _resolve_symbol_env(name: str, fallback: str) -> str:
    candidate = (os.getenv(name) or "").strip().upper()
    return candidate or fallback


def _resolve_text_env(name: str, fallback: str) -> str:
    candidate = (os.getenv(name) or "").strip()
    return candidate or fallback


def _resolve_symbols(defaults: RuntimeConfig) -> tuple[str, ...]:
    symbols = _parse_symbols(os.getenv("EDGE_SYMBOLS"))
    if symbols:
        return symbols
    legacy_symbols = _parse_symbols(os.getenv("EDGE_SYMBOL"))
    if legacy_symbols:
        return legacy_symbols
    return defaults.symbols


def _parse_log_level(value: str | None, fallback: str) -> str:
    valid_levels = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
    candidate = (value or fallback).strip().upper()
    return candidate if candidate in valid_levels else fallback


def _parse_bool(value: str | None) -> bool:
    """Parse a truthy environment value for lightweight feature flags."""
    if value is None:
        return False
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _read_optional_value(value_env: str, file_env: str) -> str | None:
    file_path = os.getenv(file_env)
    if file_path:
        return Path(file_path).read_text(encoding="utf-8").strip() or None
    value = os.getenv(value_env)
    if value is None:
        return None
    candidate = value.strip()
    return candidate or None


def _parse_symbols(value: str | None) -> tuple[str, ...]:
    if value is None:
        return ()
    items = [item.strip().upper() for item in value.split(",")]
    symbols = tuple(item for item in items if item)
    if len(symbols) != len(set(symbols)):
        deduped: list[str] = []
        for symbol in symbols:
            if symbol not in deduped:
                deduped.append(symbol)
        return tuple(deduped)
    return symbols
