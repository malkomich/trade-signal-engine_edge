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


def resample_bars(bars: Sequence[Bar], interval_minutes: int) -> list[Bar]:
    if interval_minutes <= 1:
        return list(bars)

    normalized = _normalize_bars(bars)
    if not normalized:
        return []

    bucketed: list[Bar] = []
    current_bucket: list[Bar] = []
    current_key = _bucket_key(normalized[0].timestamp, interval_minutes)
    for bar in normalized:
        bucket_key = _bucket_key(bar.timestamp, interval_minutes)
        if bucket_key != current_key and current_bucket:
            bucketed.append(_aggregate_bucket(current_bucket, current_key))
            current_bucket = []
            current_key = bucket_key
        current_bucket.append(bar)

    if current_bucket:
        bucketed.append(_aggregate_bucket(current_bucket, current_key))
    return bucketed


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


def _bucket_key(timestamp: datetime, interval_minutes: int) -> datetime:
    total_minutes = timestamp.hour * 60 + timestamp.minute
    bucket_minutes = (total_minutes // interval_minutes) * interval_minutes
    hour = bucket_minutes // 60
    minute = bucket_minutes % 60
    return timestamp.replace(hour=hour, minute=minute, second=0, microsecond=0)


def _aggregate_bucket(bucket: Sequence[Bar], bucket_key: datetime) -> Bar:
    first = bucket[0]
    last = bucket[-1]
    return Bar(
        symbol=first.symbol,
        timestamp=bucket_key,
        open=first.open,
        high=max(bar.high for bar in bucket),
        low=min(bar.low for bar in bucket),
        close=last.close,
        volume=sum(bar.volume for bar in bucket),
    )
