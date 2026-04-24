from __future__ import annotations

from datetime import datetime, timedelta, timezone

from trade_signal_edge.ingestion import MAX_SYNTHETIC_GAP_INTERVALS, ingest_bars, resample_bars
from trade_signal_edge.models import Bar


def test_ingest_bars_skips_unbounded_gaps() -> None:
    bars = [
        Bar(
            symbol="AAPL",
            timestamp=datetime(2026, 4, 21, 9, 30, tzinfo=timezone.utc),
            open=100.0,
            high=101.0,
            low=99.5,
            close=100.5,
            volume=1_000.0,
        ),
        Bar(
            symbol="AAPL",
            timestamp=datetime(2026, 4, 21, 9, 30, tzinfo=timezone.utc) + timedelta(minutes=MAX_SYNTHETIC_GAP_INTERVALS + 2),
            open=110.0,
            high=111.0,
            low=109.5,
            close=110.5,
            volume=1_500.0,
        ),
    ]

    normalized = ingest_bars(bars)

    assert len(normalized) == 2
    assert normalized[0].timestamp == datetime(2026, 4, 21, 9, 30, tzinfo=timezone.utc)
    assert normalized[1].timestamp == datetime(2026, 4, 21, 9, 30, tzinfo=timezone.utc) + timedelta(minutes=MAX_SYNTHETIC_GAP_INTERVALS + 2)


def test_resample_bars_aggregates_open_high_low_close_volume() -> None:
    bars = [
        Bar(symbol="AAPL", timestamp=datetime(2026, 4, 21, 9, 30, tzinfo=timezone.utc), open=100.0, high=101.0, low=99.5, close=100.5, volume=1_000.0),
        Bar(symbol="AAPL", timestamp=datetime(2026, 4, 21, 9, 31, tzinfo=timezone.utc), open=100.5, high=102.0, low=100.25, close=101.5, volume=1_500.0),
        Bar(symbol="AAPL", timestamp=datetime(2026, 4, 21, 9, 32, tzinfo=timezone.utc), open=101.5, high=103.0, low=101.0, close=102.5, volume=2_000.0),
    ]

    resampled = resample_bars(bars, 5)

    assert len(resampled) == 1
    assert resampled[0].open == 100.0
    assert resampled[0].high == 103.0
    assert resampled[0].low == 99.5
    assert resampled[0].close == 102.5
    assert resampled[0].volume == 4_500.0


def test_resample_bars_returns_empty_for_empty_input() -> None:
    assert resample_bars([], 5) == []


def test_resample_bars_keeps_day_boundaries_separate() -> None:
    bars = [
        Bar(symbol="AAPL", timestamp=datetime(2026, 4, 21, 23, 55, tzinfo=timezone.utc), open=100.0, high=101.0, low=99.5, close=100.5, volume=1_000.0),
        Bar(symbol="AAPL", timestamp=datetime(2026, 4, 22, 0, 5, tzinfo=timezone.utc), open=101.0, high=102.0, low=100.5, close=101.5, volume=1_500.0),
    ]

    resampled = resample_bars(bars, 15)

    assert len(resampled) == 2
    assert resampled[0].timestamp == datetime(2026, 4, 21, 23, 45, tzinfo=timezone.utc)
    assert resampled[1].timestamp == datetime(2026, 4, 22, 0, 0, tzinfo=timezone.utc)
