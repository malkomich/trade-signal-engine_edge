from datetime import datetime, timedelta, timezone

from trade_signal_edge.indicators import IndicatorCalculator
from trade_signal_edge.models import Bar


def build_bars(count: int = 60) -> list[Bar]:
    start = datetime(2026, 4, 20, 13, 30, tzinfo=timezone.utc)
    bars: list[Bar] = []
    price = 100.0
    for index in range(count):
        price += 0.5
        bars.append(
            Bar(
                symbol="MSFT",
                timestamp=start + timedelta(minutes=index),
                open=price - 0.2,
                high=price + 0.4,
                low=price - 0.4,
                close=price,
                volume=1_000 + index * 10,
            )
        )
    return bars


def test_indicator_calculator_produces_snapshot() -> None:
    snapshot = IndicatorCalculator().compute(build_bars())

    assert snapshot.symbol == "MSFT"
    assert snapshot.close > 0
    assert snapshot.sma_fast is not None
    assert snapshot.ema_fast is not None
    assert snapshot.macd_histogram is not None


def test_indicator_calculator_uses_independent_volume_profile_window() -> None:
    calculator = IndicatorCalculator(bollinger_period=50, volume_profile_period=5)
    snapshot = calculator.compute(build_bars(12))

    assert snapshot.volume_profile is not None
    assert 0.0 <= snapshot.volume_profile <= 1.0
