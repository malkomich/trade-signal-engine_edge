from __future__ import annotations

from datetime import datetime, timezone

import pytest

from trade_signal_edge.ingestion import ingest_bars
from trade_signal_edge.models import Bar


def test_ingest_bars_sorts_deduplicates_and_fills_gaps() -> None:
    bars = [
        Bar(
            symbol="AAPL",
            timestamp=datetime(2026, 4, 21, 9, 32, tzinfo=timezone.utc),
            open=103.0,
            high=104.0,
            low=102.5,
            close=103.5,
            volume=1_500.0,
        ),
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
            timestamp=datetime(2026, 4, 21, 9, 32, tzinfo=timezone.utc),
            open=104.0,
            high=105.0,
            low=103.5,
            close=104.5,
            volume=1_800.0,
        ),
    ]

    normalized = ingest_bars(bars)

    assert [bar.timestamp for bar in normalized] == [
        datetime(2026, 4, 21, 9, 30, tzinfo=timezone.utc),
        datetime(2026, 4, 21, 9, 31, tzinfo=timezone.utc),
        datetime(2026, 4, 21, 9, 32, tzinfo=timezone.utc),
    ]
    assert normalized[1].open == pytest.approx(100.5)
    assert normalized[1].high == pytest.approx(100.5)
    assert normalized[1].low == pytest.approx(100.5)
    assert normalized[1].close == pytest.approx(100.5)
    assert normalized[1].volume == pytest.approx(0.0)
    assert normalized[2].close == pytest.approx(104.5)


def test_ingest_bars_rejects_mixed_symbols() -> None:
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
            symbol="MSFT",
            timestamp=datetime(2026, 4, 21, 9, 31, tzinfo=timezone.utc),
            open=200.0,
            high=201.0,
            low=199.5,
            close=200.5,
            volume=1_000.0,
        ),
    ]

    with pytest.raises(ValueError, match="single symbol"):
        ingest_bars(bars)
