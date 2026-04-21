from __future__ import annotations

from dataclasses import dataclass
import os


@dataclass(slots=True)
class RuntimeConfig:
    symbol: str = "AAPL"
    bars: int = 60
    api_base_url: str | None = None


def load_runtime_config() -> RuntimeConfig:
    return RuntimeConfig(
        symbol=os.getenv("EDGE_SYMBOL", "AAPL"),
        bars=int(os.getenv("EDGE_BARS", "60")),
        api_base_url=os.getenv("API_BASE_URL"),
    )
