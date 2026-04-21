from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Literal, Protocol, Sequence, cast, get_args
from urllib import error, parse, request
import json
import os

from .models import Bar

ProviderName = Literal["synthetic", "alpaca"]


class MarketDataProvider(Protocol):
    def history(self, symbol: str, bars: int) -> Sequence[Bar]:
        raise NotImplementedError


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
class ProviderSelection:
    name: ProviderName = "synthetic"
    alpaca_api_key_id: str | None = None
    alpaca_api_secret_key: str | None = None
    alpaca_feed: str = "iex"


ProviderConfig = ProviderSelection


def resolve_provider_name(value: str | None) -> ProviderName:
    if value is None:
        return "synthetic"
    normalized = value.strip().lower()
    allowed = get_args(ProviderName)
    if normalized not in allowed:
        raise ValueError(f"unsupported provider {value!r}. Supported: {', '.join(allowed)}")
    return cast(ProviderName, normalized)


def load_provider_selection() -> ProviderSelection:
    return ProviderSelection(
        name=resolve_provider_name(os.getenv("EDGE_PROVIDER", "synthetic")),
        alpaca_api_key_id=os.getenv("ALPACA_API_KEY_ID"),
        alpaca_api_secret_key=os.getenv("ALPACA_API_SECRET_KEY"),
        alpaca_feed=os.getenv("ALPACA_DATA_FEED", "iex").strip().lower(),
    )


def build_provider(selection: ProviderSelection) -> MarketDataProvider:
    if selection.name == "alpaca":
        if not selection.alpaca_api_key_id or not selection.alpaca_api_secret_key:
            raise ValueError("alpaca provider requires ALPACA_API_KEY_ID and ALPACA_API_SECRET_KEY")
        return AlpacaProvider(
            api_key_id=selection.alpaca_api_key_id,
            api_secret_key=selection.alpaca_api_secret_key,
            feed=selection.alpaca_feed,
        )
    return SyntheticProvider()
