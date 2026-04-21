from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Literal, Protocol, Sequence, cast, get_args
from urllib import error, parse, request
import json

from .config import RuntimeConfig
from .models import Bar

ProviderName = Literal["synthetic", "alpaca"]


class MarketDataProvider(Protocol):
    def history(self, symbol: str, bars: int) -> Sequence[Bar]:
        raise NotImplementedError


@dataclass(frozen=True, slots=True)
class ProviderPolicy:
    name: ProviderName
    access_model: str
    cost_tier: str
    redistribution: str
    commercial_use: str
    notes: str
    requires_credentials: bool = False


PROVIDER_POLICIES: tuple[ProviderPolicy, ...] = (
    ProviderPolicy(
        name="synthetic",
        access_model="local-only generated data",
        cost_tier="free",
        redistribution="safe for local development and tests because the bars are generated in-process",
        commercial_use="development and test only",
        notes="No external market-data license is involved.",
    ),
    ProviderPolicy(
        name="alpaca",
        access_model="external market-data API",
        cost_tier="account-plan dependent",
        redistribution="review Alpaca terms before sharing raw bars or derived market data outside the app boundary",
        commercial_use="subject to the Alpaca account plan and market-data terms",
        notes="Use this provider for live ingestion; keep API keys out of source control.",
        requires_credentials=True,
    ),
)


@dataclass(slots=True)
class SyntheticProvider:
    drift: float = 0.22
    base_price: float = 180.0

    def history(self, symbol: str, bars: int) -> list[Bar]:
        start = datetime.now(tz=timezone.utc) - timedelta(minutes=bars)
        current = self.base_price
        output: list[Bar] = []
        for index in range(bars):
            current += self.drift if index > 10 else 0.08
            output.append(
                Bar(
                    symbol=symbol,
                    timestamp=start + timedelta(minutes=index),
                    open=current - 0.12,
                    high=current + 0.35,
                    low=current - 0.25,
                    close=current,
                    volume=1_000 + index * 15,
                )
            )
        return output


@dataclass(slots=True)
class AlpacaProvider:
    api_key_id: str
    api_secret_key: str
    feed: str = "iex"
    base_url: str = "https://data.alpaca.markets"
    timeout_seconds: int = 15

    def history(self, symbol: str, bars: int) -> list[Bar]:
        start = (datetime.now(tz=timezone.utc) - timedelta(minutes=max(bars * 3, 45))).isoformat()
        params = {
            "symbols": symbol,
            "timeframe": "1Min",
            "start": start,
            "limit": str(min(max(bars, 1), 1000)),
            "adjustment": "raw",
            "feed": self.feed,
            "sort": "asc",
        }
        payload = self._get_json("/v2/stocks/bars", params)
        items = self._extract_bars(payload, symbol)
        if len(items) < bars:
            latest_payload = self._get_json("/v2/stocks/bars/latest", {"symbols": symbol, "feed": self.feed})
            items.extend(self._extract_bars(latest_payload, symbol))
        return items[-bars:]

    def _get_json(self, path: str, params: dict[str, str]) -> dict[str, object]:
        url = f"{self.base_url}{path}?{parse.urlencode(params)}"
        req = request.Request(
            url,
            headers={
                "APCA-API-KEY-ID": self.api_key_id,
                "APCA-API-SECRET-KEY": self.api_secret_key,
            },
            method="GET",
        )
        try:
            with request.urlopen(req, timeout=self.timeout_seconds) as response:
                return json.loads(response.read().decode("utf-8"))
        except error.HTTPError as exc:
            raise RuntimeError(f"alpaca request failed with status {exc.code}") from exc
        except error.URLError as exc:
            raise RuntimeError("alpaca request failed") from exc

    def _extract_bars(self, payload: dict[str, object], symbol: str) -> list[Bar]:
        bars_payload = payload.get("bars")
        if isinstance(bars_payload, dict):
            raw_bars = bars_payload.get(symbol)
            if not isinstance(raw_bars, list):
                return []
            return [self._normalize_bar(symbol, item) for item in raw_bars if isinstance(item, dict)]
        if isinstance(bars_payload, list):
            return [
                self._normalize_bar(str(item.get("S", symbol)), item)
                for item in bars_payload
                if isinstance(item, dict)
            ]
        return []

    def _normalize_bar(self, symbol: str, item: dict[str, object]) -> Bar:
        timestamp = item.get("t")
        if not isinstance(timestamp, str):
            raise ValueError("alpaca bar missing timestamp")
        return Bar(
            symbol=symbol,
            timestamp=datetime.fromisoformat(timestamp.replace("Z", "+00:00")),
            open=float(item.get("o", 0.0)),
            high=float(item.get("h", 0.0)),
            low=float(item.get("l", 0.0)),
            close=float(item.get("c", 0.0)),
            volume=float(item.get("v", 0.0)),
        )


@dataclass(slots=True)
class ProviderConfig:
    name: ProviderName = "synthetic"
    alpaca_api_key_id: str | None = None
    alpaca_api_secret_key: str | None = None
    alpaca_feed: str = "iex"


def resolve_provider_name(value: str | None) -> ProviderName:
    if value is None:
        return "synthetic"
    normalized = value.strip().lower()
    if not normalized:
        return "synthetic"
    allowed = get_args(ProviderName)
    if normalized not in allowed:
        raise ValueError(f"unsupported provider {value!r}. Supported: {', '.join(allowed)}")
    return cast(ProviderName, normalized)


def provider_policies() -> tuple[ProviderPolicy, ...]:
    return PROVIDER_POLICIES


def selected_provider_policy(config: ProviderConfig) -> ProviderPolicy:
    for policy in PROVIDER_POLICIES:
        if policy.name == config.name:
            return policy
    raise NotImplementedError(f"provider {config.name!r} is not implemented")


def load_provider_config(runtime: RuntimeConfig) -> ProviderConfig:
    return ProviderConfig(
        name=resolve_provider_name(runtime.provider),
        alpaca_api_key_id=runtime.alpaca_api_key_id,
        alpaca_api_secret_key=runtime.alpaca_api_secret_key,
        alpaca_feed=runtime.alpaca_feed,
    )


def build_provider(config: ProviderConfig) -> MarketDataProvider:
    if config.name == "synthetic":
        return SyntheticProvider()
    if config.name == "alpaca":
        if not config.alpaca_api_key_id or not config.alpaca_api_secret_key:
            raise ValueError("alpaca provider requires ALPACA_API_KEY_ID and ALPACA_API_SECRET_KEY")
        # Raw bars are consumed for local computation only; redistribution policy is documented in the catalog.
        return AlpacaProvider(
            api_key_id=config.alpaca_api_key_id,
            api_secret_key=config.alpaca_api_secret_key,
            feed=config.alpaca_feed,
        )
    raise NotImplementedError(f"provider {config.name!r} is not implemented")
