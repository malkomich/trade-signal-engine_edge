from datetime import datetime, timezone

import pytest

from trade_signal_edge.config import RuntimeConfig
from trade_signal_edge.models import (
    DEFAULT_ENTRY_GATE_CAP,
    IndicatorSnapshot,
    SignalAction,
    SignalConfig,
    SignalTier,
    TradeState,
)
from trade_signal_edge.signal_engine import (
    BUY_TIER_BALANCED_ENTRY,
    BUY_TIER_BALANCED_QUALITY,
    BUY_TIER_CONVICTION_ENTRY,
    BUY_TIER_CONVICTION_QUALITY,
    BUY_TIER_OPPORTUNISTIC_ENTRY,
    BUY_TIER_OPPORTUNISTIC_QUALITY,
    BUY_TIER_SPECULATIVE_ENTRY,
    BUY_TIER_SPECULATIVE_QUALITY,
    OPENING_SESSION_PENALTY_HIGH,
    OPENING_SESSION_PENALTY_LOW,
    OPENING_SESSION_PENALTY_MEDIUM,
    OPENING_SESSION_RISK_HIGH,
    OPENING_SESSION_RISK_LOW,
    OPENING_SESSION_RISK_MEDIUM,
    SignalEngine,
)


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
        rsi=32.0,
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
    assert decision.signal_tier is SignalTier.CONVICTION_BUY
    assert decision.entry_score > decision.exit_score
    assert decision.signal_tier is not None


def test_signal_engine_allows_buy_with_a_few_strong_categories() -> None:
    snapshot = IndicatorSnapshot(
        symbol="AAPL",
        timestamp=datetime(2026, 4, 20, 18, 30, tzinfo=timezone.utc),
        close=190.5,
        sma_fast=189.8,
        sma_slow=190.2,
        ema_fast=190.7,
        ema_slow=189.9,
        vwap=190.1,
        rsi=34.0,
        atr=1.15,
        plus_di=24.0,
        minus_di=19.0,
        adx=21.0,
        macd=0.42,
        macd_signal=0.31,
        macd_histogram=0.11,
        stochastic_k=48.0,
        stochastic_d=43.0,
        relative_volume=1.12,
        volume_profile=0.18,
    )

    decision = SignalEngine().evaluate(snapshot, TradeState.FLAT)

    assert decision.action is SignalAction.BUY_ALERT
    assert decision.signal_tier is not None
    assert any(reason.startswith("trend:") for reason in decision.reasons)
    assert any(reason.startswith("flow:") for reason in decision.reasons)
    assert any(reason.startswith("momentum:") for reason in decision.reasons)


def test_signal_engine_classifies_buy_tiers_by_strength() -> None:
    engine = SignalEngine()

    assert engine._buy_signal_tier(0.84, 0.32, 0.80, 4, 0.40, False) is SignalTier.CONVICTION_BUY
    assert engine._buy_signal_tier(0.74, 0.44, 0.64, 3, 0.40, False) is SignalTier.BALANCED_BUY
    assert engine._buy_signal_tier(0.64, 0.62, 0.60, 2, 0.55, True) is SignalTier.OPPORTUNISTIC_BUY
    assert engine._buy_signal_tier(0.57, 0.81, 0.55, 2, 0.55, True) is SignalTier.SPECULATIVE_BUY


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


def test_signal_engine_rejects_buy_when_rsi_is_overbought() -> None:
    snapshot = IndicatorSnapshot(
        symbol="TSLA",
        timestamp=datetime(2026, 4, 20, 18, 30, tzinfo=timezone.utc),
        close=198.0,
        sma_fast=197.2,
        sma_slow=196.9,
        ema_fast=197.4,
        ema_slow=197.0,
        vwap=196.8,
        rsi=72.0,
        atr=2.1,
        plus_di=28.0,
        minus_di=14.0,
        adx=27.0,
        macd=0.85,
        macd_signal=0.72,
        macd_histogram=0.13,
        stochastic_k=42.0,
        stochastic_d=38.0,
    )

    decision = SignalEngine().evaluate(snapshot, TradeState.FLAT)

    assert decision.action is SignalAction.HOLD


def test_signal_engine_rejects_buy_when_stochastic_is_overbought() -> None:
    snapshot = IndicatorSnapshot(
        symbol="NVDA",
        timestamp=datetime(2026, 4, 20, 18, 30, tzinfo=timezone.utc),
        close=102.0,
        sma_fast=101.5,
        sma_slow=100.2,
        ema_fast=101.8,
        ema_slow=100.6,
        vwap=100.8,
        rsi=38.0,
        atr=1.25,
        plus_di=28.0,
        minus_di=14.0,
        adx=26.0,
        macd=1.1,
        macd_signal=0.85,
        macd_histogram=0.25,
        stochastic_k=84.0,
        stochastic_d=82.0,
    )

    decision = SignalEngine().evaluate(snapshot, TradeState.FLAT)

    assert decision.action is SignalAction.HOLD


def test_signal_engine_rejects_buy_when_macd_is_bearish() -> None:
    snapshot = IndicatorSnapshot(
        symbol="META",
        timestamp=datetime(2026, 4, 20, 18, 30, tzinfo=timezone.utc),
        close=292.0,
        sma_fast=291.5,
        sma_slow=290.9,
        ema_fast=291.7,
        ema_slow=291.3,
        vwap=291.1,
        rsi=34.0,
        atr=1.8,
        plus_di=22.0,
        minus_di=20.0,
        adx=22.0,
        macd=-0.12,
        macd_signal=-0.05,
        macd_histogram=-0.07,
        stochastic_k=51.0,
        stochastic_d=47.0,
    )

    decision = SignalEngine().evaluate(snapshot, TradeState.FLAT)

    assert decision.action is SignalAction.HOLD


def test_signal_engine_ignores_bearish_exit_pressure_in_bullish_reversal_context() -> None:
    engine = SignalEngine()
    snapshot = IndicatorSnapshot(
        symbol="TSLA",
        timestamp=datetime(2026, 5, 4, 17, 36, tzinfo=timezone.utc),
        close=414.1,
        sma_fast=414.3,
        sma_slow=415.0,
        ema_fast=414.2,
        ema_slow=414.8,
        vwap=414.7,
        rsi=12.0,
        atr=3.5,
        plus_di=18.0,
        minus_di=22.0,
        adx=24.0,
        macd=-0.6,
        macd_signal=-0.4,
        macd_histogram=-0.2,
        stochastic_k=9.0,
        stochastic_d=10.0,
        relative_volume=1.15,
        volume_profile=0.18,
    )

    assert engine.is_strong_exit_pressure(snapshot) is True
    assert engine.is_strong_exit_pressure(snapshot, True) is False


def test_signal_engine_allows_buy_on_oversold_reversal_context() -> None:
    benchmark = IndicatorSnapshot(
        symbol="QQQ",
        timestamp=datetime(2026, 4, 20, 15, 22, tzinfo=timezone.utc),
        close=506.2,
        sma_fast=507.4,
        sma_slow=508.0,
        ema_fast=506.8,
        ema_slow=507.9,
        vwap=507.0,
        rsi=10.72,
        atr=4.2,
        plus_di=14.0,
        minus_di=24.0,
        adx=27.0,
        macd=-1.52,
        macd_signal=-1.08,
        macd_histogram=-0.44,
        stochastic_k=7.89,
        stochastic_d=9.6,
        relative_volume=1.32,
        volume_profile=0.24,
    )
    snapshot = IndicatorSnapshot(
        symbol="TSLA",
        timestamp=datetime(2026, 4, 20, 15, 22, tzinfo=timezone.utc),
        close=282.4,
        sma_fast=283.1,
        sma_slow=284.0,
        ema_fast=282.9,
        ema_slow=283.6,
        vwap=283.3,
        rsi=11.4,
        atr=3.1,
        plus_di=19.0,
        minus_di=21.0,
        adx=24.0,
        macd=-0.82,
        macd_signal=-0.51,
        macd_histogram=-0.31,
        stochastic_k=8.4,
        stochastic_d=10.1,
        relative_volume=1.24,
        volume_profile=0.21,
    )

    decision = SignalEngine().evaluate(snapshot, TradeState.FLAT, benchmark=benchmark)

    assert decision.action is SignalAction.BUY_ALERT
    assert decision.signal_tier is not None
    assert decision.entry_score > decision.exit_score
    assert "oversold-reversal-context" in decision.reasons
<<<<<<< HEAD


def test_signal_engine_allows_buy_on_benchmark_only_oversold_reversal_context() -> None:
    benchmark = IndicatorSnapshot(
        symbol="QQQ",
        timestamp=datetime(2026, 4, 20, 15, 22, tzinfo=timezone.utc),
        close=506.2,
        sma_fast=507.4,
        sma_slow=508.0,
        ema_fast=506.8,
        ema_slow=507.9,
        vwap=507.0,
        rsi=10.72,
        atr=4.2,
        plus_di=14.0,
        minus_di=24.0,
        adx=27.0,
        macd=-1.52,
        macd_signal=-1.08,
        macd_histogram=-0.44,
        stochastic_k=7.89,
        stochastic_d=9.6,
        relative_volume=1.32,
        volume_profile=0.24,
    )
    snapshot = IndicatorSnapshot(
        symbol="TSLA",
        timestamp=datetime(2026, 4, 20, 15, 22, tzinfo=timezone.utc),
        close=282.4,
        sma_fast=286.1,
        sma_slow=284.0,
        ema_fast=285.5,
        ema_slow=284.8,
        vwap=284.9,
        rsi=41.0,
        atr=3.1,
        plus_di=19.0,
        minus_di=21.0,
        adx=24.0,
        macd=-0.22,
        macd_signal=-0.19,
        macd_histogram=-0.03,
        stochastic_k=38.0,
        stochastic_d=35.0,
        relative_volume=1.18,
        volume_profile=0.19,
    )

    decision = SignalEngine().evaluate(snapshot, TradeState.FLAT, benchmark=benchmark)

    assert decision.action is SignalAction.BUY_ALERT
    assert "oversold-reversal-context" in decision.reasons
    assert any(reason.startswith("QQQ ") for reason in decision.reasons)


def test_signal_engine_scores_oversold_reversal_quality_context() -> None:
    snapshot = IndicatorSnapshot(
        symbol="TSLA",
        timestamp=datetime(2026, 4, 20, 15, 22, tzinfo=timezone.utc),
        close=282.4,
        sma_fast=283.1,
        sma_slow=284.0,
        ema_fast=282.9,
        ema_slow=283.6,
        vwap=283.3,
        rsi=11.4,
        atr=3.1,
        plus_di=19.0,
        minus_di=21.0,
        adx=24.0,
        macd=-0.82,
        macd_signal=-0.51,
        macd_histogram=-0.31,
        stochastic_k=8.4,
        stochastic_d=10.1,
        relative_volume=1.24,
        volume_profile=0.21,
    )

    quality = SignalEngine()._long_entry_quality_assessment(snapshot, None, True)

    assert quality.score > 0.0
    assert "trend:oversold-reversal" in quality.reasons
    assert any(reason.startswith("momentum:") for reason in quality.reasons)


def test_signal_engine_relaxes_buy_tier_support_in_oversold_reversal_context() -> None:
    engine = SignalEngine()

    tier = engine._buy_signal_tier(
        entry_score=0.52,
        risk_score=0.41,
        quality_score=0.42,
        supportive_signals=1,
        session_risk=0.45,
        strong_exit_pressure=False,
        bullish_reversal_context=True,
    )

    assert tier is SignalTier.SPECULATIVE_BUY


def test_signal_engine_allows_buy_on_benchmark_only_oversold_reversal_context() -> None:
    benchmark = IndicatorSnapshot(
        symbol="QQQ",
        timestamp=datetime(2026, 4, 20, 15, 22, tzinfo=timezone.utc),
        close=506.2,
        sma_fast=507.4,
        sma_slow=508.0,
        ema_fast=506.8,
        ema_slow=507.9,
        vwap=507.0,
        rsi=10.72,
        atr=4.2,
        plus_di=14.0,
        minus_di=24.0,
        adx=27.0,
        macd=-1.52,
        macd_signal=-1.08,
        macd_histogram=-0.44,
        stochastic_k=7.89,
        stochastic_d=9.6,
        relative_volume=1.32,
        volume_profile=0.24,
    )
    snapshot = IndicatorSnapshot(
        symbol="TSLA",
        timestamp=datetime(2026, 4, 20, 15, 22, tzinfo=timezone.utc),
        close=282.4,
        sma_fast=286.1,
        sma_slow=284.0,
        ema_fast=285.5,
        ema_slow=284.8,
        vwap=284.9,
        rsi=41.0,
        atr=3.1,
        plus_di=19.0,
        minus_di=21.0,
        adx=24.0,
        macd=-0.22,
        macd_signal=-0.19,
        macd_histogram=-0.03,
        stochastic_k=38.0,
        stochastic_d=35.0,
        relative_volume=1.18,
        volume_profile=0.19,
    )

    decision = SignalEngine().evaluate(snapshot, TradeState.FLAT, benchmark=benchmark)

    assert decision.action is SignalAction.BUY_ALERT
    assert "oversold-reversal-context" in decision.reasons
    assert any(reason.startswith("QQQ ") for reason in decision.reasons)


def test_signal_engine_scores_oversold_reversal_quality_context() -> None:
    snapshot = IndicatorSnapshot(
        symbol="TSLA",
        timestamp=datetime(2026, 4, 20, 15, 22, tzinfo=timezone.utc),
        close=282.4,
        sma_fast=283.1,
        sma_slow=284.0,
        ema_fast=282.9,
        ema_slow=283.6,
        vwap=283.3,
        rsi=11.4,
        atr=3.1,
        plus_di=19.0,
        minus_di=21.0,
        adx=24.0,
        macd=-0.82,
        macd_signal=-0.51,
        macd_histogram=-0.31,
        stochastic_k=8.4,
        stochastic_d=10.1,
        relative_volume=1.24,
        volume_profile=0.21,
    )

    quality = SignalEngine()._long_entry_quality_assessment(snapshot, None, True)

    assert quality.score > 0.0
    assert "trend:oversold-reversal" in quality.reasons
    assert any(reason.startswith("momentum:") for reason in quality.reasons)


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
        rsi=34.0,
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


def test_signal_config_and_runtime_share_entry_gate_cap_default() -> None:
    assert SignalConfig().entry_gate_cap == DEFAULT_ENTRY_GATE_CAP
    assert RuntimeConfig().entry_gate_cap == DEFAULT_ENTRY_GATE_CAP


def test_signal_engine_uses_session_entry_exit_margin_from_config() -> None:
    snapshot = IndicatorSnapshot(
        symbol="TSLA",
        timestamp=datetime(2026, 4, 20, 18, 30, tzinfo=timezone.utc),
        close=198.0,
        sma_fast=197.5,
        sma_slow=196.9,
        ema_fast=197.6,
        ema_slow=197.0,
        vwap=197.2,
        rsi=34.0,
        atr=1.9,
        plus_di=25.0,
        minus_di=16.0,
        adx=24.0,
        macd=0.58,
        macd_signal=0.45,
        macd_histogram=0.13,
        stochastic_k=49.0,
        stochastic_d=43.0,
    )

    loose = SignalEngine(
        SignalConfig(entry_threshold=0.65, exit_threshold=0.55, entry_exit_margin=0.0)
    ).evaluate(snapshot, TradeState.FLAT)
    strict = SignalEngine(
        SignalConfig(entry_threshold=0.65, exit_threshold=0.55, entry_exit_margin=0.8)
    ).evaluate(snapshot, TradeState.FLAT)

    assert loose.action is SignalAction.BUY_ALERT
    assert strict.action is SignalAction.HOLD


@pytest.mark.parametrize(
    ("session_risk", "entry_score", "expected_action"),
    [
        (0.95, 0.63, SignalAction.HOLD),
        (0.95, 0.69, SignalAction.HOLD),
        (0.85, 0.57, SignalAction.HOLD),
        (0.85, 0.63, SignalAction.HOLD),
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
        rsi=33.0,
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


@pytest.mark.parametrize(
    ("session_risk", "expected_penalty"),
    [
        (OPENING_SESSION_RISK_LOW - 0.01, 0.0),
        (OPENING_SESSION_RISK_LOW, OPENING_SESSION_PENALTY_LOW),
        (OPENING_SESSION_RISK_MEDIUM, OPENING_SESSION_PENALTY_MEDIUM),
        (OPENING_SESSION_RISK_HIGH, OPENING_SESSION_PENALTY_HIGH),
    ],
)
def test_signal_engine_opening_session_penalty_boundaries(session_risk: float, expected_penalty: float) -> None:
    engine = SignalEngine()
    assert engine._opening_session_penalty(session_risk) == expected_penalty


@pytest.mark.parametrize(
    ("entry_score", "risk_score", "quality_score", "supportive_signals", "expected_tier"),
    [
        (BUY_TIER_CONVICTION_ENTRY, 0.3, BUY_TIER_CONVICTION_QUALITY, 4, SignalTier.CONVICTION_BUY),
        (BUY_TIER_CONVICTION_ENTRY, 0.3, BUY_TIER_CONVICTION_QUALITY - 0.01, 3, SignalTier.BALANCED_BUY),
        (BUY_TIER_BALANCED_ENTRY, 0.4, BUY_TIER_BALANCED_QUALITY, 3, SignalTier.BALANCED_BUY),
        (BUY_TIER_BALANCED_ENTRY, 0.4, BUY_TIER_BALANCED_QUALITY - 0.01, 2, SignalTier.OPPORTUNISTIC_BUY),
        (BUY_TIER_OPPORTUNISTIC_ENTRY, 0.6, BUY_TIER_OPPORTUNISTIC_QUALITY, 2, SignalTier.OPPORTUNISTIC_BUY),
        (BUY_TIER_OPPORTUNISTIC_ENTRY, 0.6, BUY_TIER_OPPORTUNISTIC_QUALITY - 0.01, 2, SignalTier.SPECULATIVE_BUY),
        (BUY_TIER_SPECULATIVE_ENTRY, 0.7, BUY_TIER_SPECULATIVE_QUALITY, 2, SignalTier.SPECULATIVE_BUY),
        (BUY_TIER_SPECULATIVE_ENTRY - 0.01, 0.8, BUY_TIER_SPECULATIVE_QUALITY, 2, None),
    ],
)
def test_signal_engine_buy_tier_threshold_constants(
    entry_score: float,
    risk_score: float,
    quality_score: float,
    supportive_signals: int,
    expected_tier: SignalTier | None,
) -> None:
    engine = SignalEngine()
    assert engine._buy_signal_tier(entry_score, risk_score, quality_score, supportive_signals, 0.0, False) is expected_tier


def test_signal_engine_preserves_strong_setup_coverage_across_session_penalty() -> None:
    engine = SignalEngine()
    opening_snapshot = IndicatorSnapshot(
        symbol="NVDA",
        timestamp=datetime(2026, 4, 20, 13, 35, tzinfo=timezone.utc),
        close=100.4,
        sma_fast=100.5,
        sma_slow=100.2,
        ema_fast=100.7,
        ema_slow=100.4,
        vwap=100.7,
        rsi=33.0,
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
        close=100.4,
        sma_fast=100.5,
        sma_slow=100.2,
        ema_fast=100.7,
        ema_slow=100.4,
        vwap=100.7,
        rsi=33.0,
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
    later_decision = engine.evaluate(later_snapshot, TradeState.FLAT)

    assert opening_decision.action is SignalAction.HOLD
    assert opening_decision.signal_tier is None
    assert later_decision.action is SignalAction.BUY_ALERT
    assert later_decision.signal_tier is SignalTier.OPPORTUNISTIC_BUY
    assert opening_decision.entry_score < later_decision.entry_score


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
        rsi=33.0,
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
        rsi=33.0,
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

    opening_decision = engine.decide_action(0.63, 0.1, TradeState.FLAT, snapshot=opening_snapshot)
    later_decision = engine.decide_action(0.63, 0.1, TradeState.FLAT, snapshot=later_snapshot)

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
    assert engine._rsi_bias(60.0) == (-0.9, 0.9)
    assert engine._rsi_bias(40.0) == (0.6, -0.1)
    assert engine._rsi_bias(28.0) == (1.0, -0.4)
    assert engine._rsi_bias(10.0) == (1.0, -0.6)
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
        rsi=34.0,
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

    assert engine._obv_bias(1_000.0, 120.0, 1.2, 0.25) == (0.6, -0.1)
    assert engine._obv_bias(-1_000.0, -90.0, 0.75, 0.1) == (-0.5, 0.7)
    assert engine._obv_bias(1_000.0, 0.0, 0.75, 0.1) == (0.1, 0.1)
    assert engine._obv_bias(0.0, None, 1.0, 0.18) == (0.0, 0.0)


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
        close=100.4,
        sma_fast=100.5,
        sma_slow=100.2,
        ema_fast=100.7,
        ema_slow=100.4,
        vwap=100.7,
        rsi=49.0,
        atr=1.25,
        plus_di=24.0,
        minus_di=18.0,
        adx=22.0,
        macd=0.4,
        macd_signal=0.35,
        macd_histogram=0.05,
        stochastic_k=42.0,
        stochastic_d=39.0,
        obv=None,
        relative_volume=1.0,
        volume_profile=0.16,
    )
    late_snapshot = IndicatorSnapshot(
        symbol="NVDA",
        timestamp=datetime(2026, 4, 20, 17, 35, tzinfo=timezone.utc),
        close=100.4,
        sma_fast=100.5,
        sma_slow=100.2,
        ema_fast=100.7,
        ema_slow=100.4,
        vwap=100.7,
        rsi=49.0,
        atr=1.25,
        plus_di=24.0,
        minus_di=18.0,
        adx=22.0,
        macd=0.4,
        macd_signal=0.35,
        macd_histogram=0.05,
        stochastic_k=42.0,
        stochastic_d=39.0,
        obv=None,
        relative_volume=1.0,
        volume_profile=0.16,
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
        rsi=34.0,
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


def test_signal_engine_does_not_reward_bearish_benchmark_context() -> None:
    bullish_benchmark = IndicatorSnapshot(
        symbol="QQQ",
        timestamp=datetime(2026, 4, 20, 18, 30, tzinfo=timezone.utc),
        close=100.2,
        ema_fast=100.4,
        ema_slow=99.6,
    )
    bearish_benchmark = IndicatorSnapshot(
        symbol="QQQ",
        timestamp=datetime(2026, 4, 20, 18, 30, tzinfo=timezone.utc),
        close=99.8,
        ema_fast=99.2,
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
    bullish = SignalEngine().evaluate(snapshot, TradeState.FLAT, bullish_benchmark)
    bearish = SignalEngine().evaluate(snapshot, TradeState.FLAT, bearish_benchmark)

    assert bullish.entry_score > baseline.entry_score
    assert bearish.entry_score <= baseline.entry_score


def test_signal_engine_counts_category_support_not_raw_indicator_support() -> None:
    snapshot = IndicatorSnapshot(
        symbol="NVDA",
        timestamp=datetime(2026, 4, 20, 18, 30, tzinfo=timezone.utc),
        close=102.4,
        sma_fast=101.8,
        sma_slow=100.9,
        ema_fast=102.0,
        ema_slow=101.1,
        vwap=101.5,
        rsi=60.0,
        atr=1.2,
        plus_di=26.0,
        minus_di=16.0,
        adx=24.0,
        macd=0.75,
        macd_signal=0.5,
        macd_histogram=0.25,
        stochastic_k=46.0,
        stochastic_d=40.0,
        obv=2_000.0,
        obv_delta=120.0,
        relative_volume=1.25,
        volume_profile=0.22,
    )

    assessment = SignalEngine()._long_entry_quality_assessment(snapshot, None)

    assert assessment.supportive_signals == 5
    assert assessment.component_count > assessment.supportive_signals
