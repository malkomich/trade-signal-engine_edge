from datetime import datetime, timezone

from trade_signal_edge.models import IndicatorSnapshot, SignalAction, SignalConfig, TradeState
from trade_signal_edge.signal_engine import SignalEngine


def test_signal_engine_raises_buy_alert_when_trend_is_aligned() -> None:
    snapshot = IndicatorSnapshot(
        symbol="NVDA",
        timestamp=datetime(2026, 4, 20, 13, 30, tzinfo=timezone.utc),
        close=102.0,
        sma_fast=101.5,
        sma_slow=100.2,
        ema_fast=101.8,
        ema_slow=100.6,
        vwap=100.8,
        rsi=62.0,
        atr=1.25,
        plus_di=28.0,
        minus_di=14.0,
        adx=26.0,
        macd=1.1,
        macd_signal=0.85,
        macd_histogram=0.25,
        stochastic_k=34.0,
        stochastic_d=28.0,
    )

    decision = SignalEngine().evaluate(snapshot, TradeState.FLAT)

    assert decision.action is SignalAction.BUY_ALERT
    assert decision.entry_score > decision.exit_score


def test_signal_engine_vetoes_entries_when_exit_pressure_is_high() -> None:
    snapshot = IndicatorSnapshot(
        symbol="TSLA",
        timestamp=datetime(2026, 4, 20, 13, 30, tzinfo=timezone.utc),
        close=198.0,
        sma_fast=200.0,
        sma_slow=198.8,
        ema_fast=199.5,
        ema_slow=198.6,
        vwap=197.5,
        rsi=72.0,
        atr=2.9,
        plus_di=22.0,
        minus_di=17.0,
        adx=23.0,
        macd=0.15,
        macd_signal=0.1,
        macd_histogram=0.05,
        stochastic_k=88.0,
        stochastic_d=91.0,
    )

    decision = SignalEngine().evaluate(snapshot, TradeState.FLAT)

    assert decision.action is SignalAction.HOLD
    assert decision.reasons == ()


def test_signal_engine_uses_configured_entry_exit_margin() -> None:
    snapshot = IndicatorSnapshot(
        symbol="AAPL",
        timestamp=datetime(2026, 4, 20, 13, 30, tzinfo=timezone.utc),
        close=190.0,
        sma_fast=189.5,
        sma_slow=188.8,
        ema_fast=189.7,
        ema_slow=189.1,
        vwap=189.0,
        rsi=57.0,
        atr=1.1,
        plus_di=23.0,
        minus_di=16.0,
        adx=22.0,
        macd=0.45,
        macd_signal=0.3,
        macd_histogram=0.15,
        stochastic_k=49.0,
        stochastic_d=45.0,
    )

    loose = SignalEngine(SignalConfig(entry_threshold=0.65, exit_threshold=0.55, entry_exit_margin=0.0)).evaluate(snapshot, TradeState.FLAT)
    strict = SignalEngine(SignalConfig(entry_threshold=0.65, exit_threshold=0.55, entry_exit_margin=0.8)).evaluate(snapshot, TradeState.FLAT)

    assert loose.action is SignalAction.BUY_ALERT
    assert strict.action is SignalAction.HOLD


def test_signal_engine_requires_long_entry_quality_even_with_a_good_score() -> None:
    snapshot = IndicatorSnapshot(
        symbol="AMZN",
        timestamp=datetime(2026, 4, 20, 13, 30, tzinfo=timezone.utc),
        close=100.4,
        sma_fast=99.8,
        sma_slow=100.2,
        ema_fast=99.7,
        ema_slow=100.3,
        vwap=100.8,
        rsi=61.0,
        atr=1.05,
        plus_di=19.0,
        minus_di=22.0,
        adx=24.0,
        macd=0.35,
        macd_signal=0.18,
        macd_histogram=0.17,
        stochastic_k=39.0,
        stochastic_d=42.0,
    )

    decision = SignalEngine(SignalConfig(entry_threshold=0.2, exit_threshold=0.55)).evaluate(snapshot, TradeState.FLAT)

    assert decision.action is SignalAction.HOLD


def test_signal_engine_uses_exit_pressure_for_open_positions() -> None:
    snapshot = IndicatorSnapshot(
        symbol="TSLA",
        timestamp=datetime(2026, 4, 20, 13, 30, tzinfo=timezone.utc),
        close=198.0,
        sma_fast=200.0,
        sma_slow=198.8,
        ema_fast=199.5,
        ema_slow=198.6,
        vwap=197.5,
        rsi=72.0,
        atr=2.9,
        plus_di=22.0,
        minus_di=17.0,
        adx=23.0,
        macd=0.15,
        macd_signal=0.1,
        macd_histogram=0.05,
        stochastic_k=88.0,
        stochastic_d=91.0,
    )

    decision = SignalEngine().evaluate(snapshot, TradeState.ACCEPTED_OPEN)

    assert decision.action is SignalAction.SELL_ALERT
    assert "exit-pressure" in decision.reasons


def test_signal_engine_accounts_for_benchmark_alignment() -> None:
    benchmark = IndicatorSnapshot(
        symbol="QQQ",
        timestamp=datetime(2026, 4, 20, 13, 30, tzinfo=timezone.utc),
        close=99.8,
        ema_fast=100.4,
        ema_slow=99.6,
    )
    snapshot = IndicatorSnapshot(
        symbol="AAPL",
        timestamp=datetime(2026, 4, 20, 13, 30, tzinfo=timezone.utc),
        close=103.5,
        sma_fast=103.0,
        sma_slow=102.0,
        ema_fast=103.1,
        ema_slow=102.2,
        vwap=102.4,
        rsi=58.0,
        atr=1.15,
        plus_di=25.0,
        minus_di=16.0,
        adx=24.0,
        macd=0.8,
        macd_signal=0.5,
        macd_histogram=0.3,
        stochastic_k=44.0,
        stochastic_d=39.0,
    )

    baseline = SignalEngine().evaluate(snapshot, TradeState.FLAT)
    decision = SignalEngine().evaluate(snapshot, TradeState.FLAT, benchmark)

    assert decision.action is SignalAction.BUY_ALERT
    assert decision.entry_score > baseline.entry_score
    assert decision.exit_score != baseline.exit_score
    assert any(reason.startswith("QQQ ") for reason in decision.reasons)
    assert 0.0 <= decision.entry_score <= 1.0
    assert 0.0 <= decision.exit_score <= 1.0
