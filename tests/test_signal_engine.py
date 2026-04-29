from datetime import datetime, timezone

import pytest

from trade_signal_edge.models import IndicatorSnapshot, SignalAction, SignalConfig, SignalTier, TradeState
from trade_signal_edge.signal_engine import SignalEngine


def test_signal_engine_raises_buy_alert_when_trend_is_aligned() -> None:
    snapshot = IndicatorSnapshot(
        symbol="NVDA",
        timestamp=datetime(2026, 4, 20, 18, 30, tzinfo=timezone.utc),
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
    assert decision.signal_tier is SignalTier.BALANCED_BUY
    assert decision.entry_score > decision.exit_score
    assert decision.signal_tier is not None


def test_signal_engine_classifies_buy_tiers_by_strength() -> None:
    engine = SignalEngine()

    assert engine._buy_signal_tier(0.84, 0.32, 0.80, 0.40, False) is SignalTier.CONVICTION_BUY
    assert engine._buy_signal_tier(0.74, 0.44, 0.64, 0.40, False) is SignalTier.BALANCED_BUY
    assert engine._buy_signal_tier(0.64, 0.62, 0.60, 0.55, True) is SignalTier.OPPORTUNISTIC_BUY
    assert engine._buy_signal_tier(0.57, 0.81, 0.55, 0.55, True) is SignalTier.SPECULATIVE_BUY


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
        timestamp=datetime(2026, 4, 20, 18, 30, tzinfo=timezone.utc),
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


@pytest.mark.parametrize(
    ("session_risk", "entry_score", "expected_action"),
    [
        (0.95, 0.71, SignalAction.HOLD),
        (0.95, 0.73, SignalAction.BUY_ALERT),
        (0.85, 0.63, SignalAction.HOLD),
        (0.85, 0.65, SignalAction.BUY_ALERT),
    ],
)
def test_signal_engine_applies_session_risk_to_entry_gate_boundaries(
    session_risk: float,
    entry_score: float,
    expected_action: SignalAction,
) -> None:
    engine = SignalEngine()
    snapshot = IndicatorSnapshot(
        symbol="NVDA",
        timestamp=datetime(2026, 4, 20, 13, 35, tzinfo=timezone.utc),
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

    decision = engine.decide_action(entry_score, 0.1, TradeState.FLAT, snapshot=snapshot, session_risk=session_risk)

    assert decision[0] is expected_action
    if expected_action is SignalAction.BUY_ALERT:
        assert decision[2] is not None


def test_signal_engine_derives_session_risk_from_snapshot_timestamp() -> None:
    engine = SignalEngine()
    opening_snapshot = IndicatorSnapshot(
        symbol="NVDA",
        timestamp=datetime(2026, 4, 20, 13, 35, tzinfo=timezone.utc),
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
    later_snapshot = IndicatorSnapshot(
        symbol="NVDA",
        timestamp=datetime(2026, 4, 20, 17, 35, tzinfo=timezone.utc),
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

    opening_decision = engine.decide_action(0.71, 0.1, TradeState.FLAT, snapshot=opening_snapshot)
    later_decision = engine.decide_action(0.71, 0.1, TradeState.FLAT, snapshot=later_snapshot)

    assert opening_decision[0] is SignalAction.HOLD
    assert later_decision[0] is SignalAction.BUY_ALERT


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


def test_signal_engine_requires_long_entry_quality_without_exit_pressure() -> None:
    snapshot = IndicatorSnapshot(
        symbol="AMZN",
        timestamp=datetime(2026, 4, 20, 13, 30, tzinfo=timezone.utc),
        close=100.1,
        sma_fast=99.4,
        sma_slow=100.2,
        ema_fast=100.6,
        ema_slow=99.8,
        vwap=100.8,
        rsi=44.0,
        atr=1.05,
        plus_di=21.0,
        minus_di=18.0,
        adx=18.0,
        macd=0.25,
        macd_signal=0.18,
        macd_histogram=0.07,
        stochastic_k=85.0,
        stochastic_d=82.0,
    )

    decision = SignalEngine(SignalConfig(entry_threshold=0.2, exit_threshold=0.55)).evaluate(snapshot, TradeState.FLAT)

    assert decision.action is SignalAction.HOLD


def test_signal_engine_rejects_missing_snapshot_for_entry_action() -> None:
    decision = SignalEngine().decide_action(0.9, 0.1, TradeState.FLAT, snapshot=None)

    assert decision[0] is SignalAction.HOLD
    assert decision[1] == ()
    assert decision[2] is None


def test_signal_engine_biases_are_covered_for_rsi_and_stochastic() -> None:
    engine = SignalEngine()

    assert engine._rsi_bias(72.0) == (-1.0, 1.0)
    assert engine._rsi_bias(60.0) == (0.9, -0.3)
    assert engine._rsi_bias(40.0) == (-0.4, 0.5)
    assert engine._stochastic_bias(15.0, 10.0) == (0.9, -0.3)
    assert engine._stochastic_bias(88.0, 85.0) == (-1.0, 1.0)
    assert engine._stochastic_bias(40.0, 35.0) == (0.5, -0.1)


def test_signal_engine_sell_pressure_bias_normalizes_and_gates_missing_obv() -> None:
    engine = SignalEngine()
    bearish_snapshot = IndicatorSnapshot(
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
        macd_histogram=-0.05,
        stochastic_k=88.0,
        stochastic_d=91.0,
        obv=1_000.0,
        relative_volume=1.2,
        volume_profile=0.22,
    )
    neutral_snapshot = IndicatorSnapshot(
        symbol="TSLA",
        timestamp=datetime(2026, 4, 20, 13, 30, tzinfo=timezone.utc),
        close=200.4,
        sma_fast=200.0,
        sma_slow=198.8,
        ema_fast=199.8,
        ema_slow=198.6,
        vwap=197.5,
        rsi=58.0,
        atr=2.9,
        plus_di=24.0,
        minus_di=16.0,
        adx=23.0,
        macd=0.15,
        macd_signal=0.1,
        macd_histogram=0.05,
        stochastic_k=52.0,
        stochastic_d=49.0,
        obv=None,
        relative_volume=1.2,
        volume_profile=0.22,
    )

    assert 0.0 <= engine._sell_pressure_bias(bearish_snapshot) <= 1.0
    assert engine._sell_pressure_bias(neutral_snapshot) < engine._sell_pressure_bias(bearish_snapshot)
    assert engine.is_strong_exit_pressure(neutral_snapshot) is False


def test_signal_engine_obv_bias_uses_relative_flow_context() -> None:
    engine = SignalEngine()

    assert engine._obv_bias(1_000.0, 1.2, 0.25) == (0.6, -0.1)
    assert engine._obv_bias(-1_000.0, 0.75, 0.1) == (-0.5, 0.7)
    assert engine._obv_bias(1_000.0, 0.75, 0.1) == (0.1, 0.1)
    assert engine._obv_bias(0.0, 1.0, 0.18) == (0.1, 0.1)


def test_signal_engine_uses_exit_pressure_for_open_positions() -> None:
    engine = SignalEngine()
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

    decision = engine.evaluate(snapshot, TradeState.ACCEPTED_OPEN)

    assert decision.action is SignalAction.SELL_ALERT
    assert decision.exit_score >= engine.config.exit_threshold
    assert "exit-pressure" in decision.reasons
    assert decision.signal_tier is None


def test_signal_engine_penalizes_opening_session_risk_for_entries() -> None:
    engine = SignalEngine()
    opening_snapshot = IndicatorSnapshot(
        symbol="NVDA",
        timestamp=datetime(2026, 4, 20, 13, 35, tzinfo=timezone.utc),
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
    late_snapshot = IndicatorSnapshot(
        symbol="NVDA",
        timestamp=datetime(2026, 4, 20, 17, 35, tzinfo=timezone.utc),
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

    opening_decision = engine.evaluate(opening_snapshot, TradeState.FLAT)
    late_decision = engine.evaluate(late_snapshot, TradeState.FLAT)

    assert opening_decision.action is SignalAction.HOLD
    assert opening_decision.entry_score < late_decision.entry_score
    assert late_decision.action is SignalAction.BUY_ALERT


def test_signal_engine_accounts_for_benchmark_alignment() -> None:
    benchmark = IndicatorSnapshot(
        symbol="QQQ",
        timestamp=datetime(2026, 4, 20, 18, 30, tzinfo=timezone.utc),
        close=99.8,
        ema_fast=100.4,
        ema_slow=99.6,
    )
    snapshot = IndicatorSnapshot(
        symbol="AAPL",
        timestamp=datetime(2026, 4, 20, 18, 30, tzinfo=timezone.utc),
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
