from __future__ import annotations

from datetime import datetime, timedelta, timezone

from trade_signal_edge.ingestion import MAX_SYNTHETIC_GAP_INTERVALS, ingest_bars
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
