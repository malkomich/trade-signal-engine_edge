from __future__ import annotations

from dataclasses import dataclass
import os


@dataclass(slots=True)
class RuntimeConfig:
    symbol: str = "AAPL"
    bars: int = 60
    provider: str = "synthetic"
    api_base_url: str | None = None
    alpaca_feed: str = "iex"
    alpaca_api_key_id: str | None = None
    alpaca_api_secret_key: str | None = None


def load_runtime_config() -> RuntimeConfig:
    defaults = RuntimeConfig()
    return RuntimeConfig(
        symbol=os.getenv("EDGE_SYMBOL", defaults.symbol),
        bars=int(os.getenv("EDGE_BARS", str(defaults.bars))),
        provider=os.getenv("EDGE_PROVIDER", defaults.provider).strip().lower(),
        api_base_url=os.getenv("API_BASE_URL"),
        alpaca_feed=os.getenv("ALPACA_DATA_FEED", defaults.alpaca_feed).strip().lower(),
        alpaca_api_key_id=os.getenv("ALPACA_API_KEY_ID"),
        alpaca_api_secret_key=os.getenv("ALPACA_API_SECRET_KEY"),
    )
