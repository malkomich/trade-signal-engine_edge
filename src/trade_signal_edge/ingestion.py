from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timedelta
from typing import Sequence

from .models import Bar

BAR_INTERVAL = timedelta(minutes=1)
MAX_SYNTHETIC_GAP_INTERVALS = 30


def ingest_bars(bars: Sequence[Bar]) -> list[Bar]:
    normalized = _normalize_bars(bars)
    return _fill_missing_bars(normalized)


def _normalize_bars(bars: Sequence[Bar]) -> list[Bar]:
    if not bars:
        raise ValueError("bars cannot be empty")

    symbols = {bar.symbol for bar in bars}
    if len(symbols) != 1:
        raise ValueError("bars must belong to a single symbol")

    indexed_bars = sorted(enumerate(bars), key=lambda item: (item[1].timestamp, item[0]))
    deduped: dict[datetime, Bar] = {}
    for _, bar in indexed_bars:
        deduped[bar.timestamp] = bar
    return [deduped[timestamp] for timestamp in sorted(deduped)]


def _fill_missing_bars(bars: Sequence[Bar]) -> list[Bar]:
    if not bars:
        return []

    filled = [bars[0]]
    for bar in bars[1:]:
        previous = filled[-1]
        missing_intervals = int((bar.timestamp - previous.timestamp) // BAR_INTERVAL) - 1
        if missing_intervals > MAX_SYNTHETIC_GAP_INTERVALS:
            filled.append(bar)
            continue

        expected = previous.timestamp + BAR_INTERVAL
        while expected < bar.timestamp:
            filled.append(_synthetic_bar(previous, expected))
            previous = filled[-1]
            expected = previous.timestamp + BAR_INTERVAL
        filled.append(bar)
    return filled


def _synthetic_bar(previous: Bar, timestamp: datetime) -> Bar:
    return replace(
        previous,
        timestamp=timestamp,
        open=previous.close,
        high=previous.close,
        low=previous.close,
        close=previous.close,
        volume=0.0,
    )
