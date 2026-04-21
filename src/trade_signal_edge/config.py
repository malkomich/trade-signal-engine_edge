from __future__ import annotations

from dataclasses import dataclass
import os


@dataclass(slots=True)
class RuntimeConfig:
    symbol: str = "AAPL"
    bars: int = 60
    api_base_url: str | None = None
    deployment_profile: str = "pi"
    log_level: str = "INFO"
    metrics_enabled: bool = False
    secret_source: str = "environment"


def load_runtime_config() -> RuntimeConfig:
    defaults = RuntimeConfig()
    return RuntimeConfig(
        symbol=os.getenv("EDGE_SYMBOL", defaults.symbol),
        bars=int(os.getenv("EDGE_BARS", str(defaults.bars))),
        api_base_url=os.getenv("API_BASE_URL", defaults.api_base_url),
        deployment_profile=os.getenv("EDGE_DEPLOYMENT_PROFILE", defaults.deployment_profile).strip().lower(),
        log_level=os.getenv("EDGE_LOG_LEVEL", defaults.log_level).strip().upper(),
        metrics_enabled=_parse_bool(os.getenv("EDGE_METRICS_ENABLED")),
        secret_source=os.getenv("EDGE_SECRET_SOURCE", defaults.secret_source).strip().lower(),
    )


def _parse_bool(value: str | None) -> bool:
    if value is None:
        return False
    return value.strip().lower() in {"1", "true", "yes", "on"}
