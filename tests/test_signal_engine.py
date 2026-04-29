from datetime import datetime, timezone

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
    assert decision.entry_score > decision.exit_score
    assert decision.signal_tier is not None


def test_signal_engine_classifies_buy_tiers_by_strength() -> None:
    conviction_snapshot = IndicatorSnapshot(
        symbol="NVDA",
        timestamp=datetime(2026, 4, 20, 18, 30, tzinfo=timezone.utc),
        close=102.0,
        sma_fast=101.9,
        sma_slow=100.2,
        ema_fast=101.8,
        ema_slow=100.6,
        vwap=100.8,
        rsi=63.0,
        atr=1.05,
        plus_di=30.0,
        minus_di=13.0,
        adx=28.0,
        macd=1.2,
        macd_signal=0.9,
        macd_histogram=0.3,
        stochastic_k=36.0,
        stochastic_d=30.0,
        obv=1_200.0,
        relative_volume=1.7,
        volume_profile=0.28,
        bollinger_middle=101.0,
        bollinger_upper=103.0,
        bollinger_lower=99.0,
    )
    balanced_snapshot = IndicatorSnapshot(
        symbol="NVDA",
        timestamp=datetime(2026, 4, 20, 18, 30, tzinfo=timezone.utc),
        close=101.0,
        sma_fast=100.8,
        sma_slow=100.5,
        ema_fast=100.9,
        ema_slow=100.6,
        vwap=100.7,
        rsi=55.5,
        atr=1.0,
        plus_di=24.0,
        minus_di=16.0,
        adx=22.0,
        macd=0.55,
        macd_signal=0.5,
        macd_histogram=0.05,
        stochastic_k=45.0,
        stochastic_d=42.0,
        relative_volume=0.9,
        volume_profile=0.12,
        bollinger_middle=100.5,
        bollinger_upper=101.7,
        bollinger_lower=99.2,
    )
    opportunistic_snapshot = IndicatorSnapshot(
        symbol="NVDA",
        timestamp=datetime(2026, 4, 20, 18, 30, tzinfo=timezone.utc),
        close=102.92089025354042,
        sma_fast=102.80073681563505,
        sma_slow=102.72893705465884,
        ema_fast=103.23476478229237,
        ema_slow=102.8593079359351,
        vwap=102.8985438721535,
        rsi=55.222126964790775,
        atr=1.1825331761480251,
        plus_di=22.328824305628803,
        minus_di=17.855653523529085,
        adx=23.426883679452448,
        macd=0.6505844309913789,
        macd_signal=0.15846527971934565,
        macd_histogram=0.1458967901289332,
        stochastic_k=41.04793293167423,
        stochastic_d=43.738786907931434,
        bollinger_middle=103.03945224759707,
        bollinger_upper=103.71663408782216,
        bollinger_lower=101.69862218280882,
        obv=None,
        relative_volume=0.9146317239760938,
        volume_profile=0.10569216672409912,
    )
    speculative_snapshot = IndicatorSnapshot(
        symbol="NVDA",
        timestamp=datetime(2026, 4, 20, 13, 35, tzinfo=timezone.utc),
        close=100.6,
        sma_fast=100.0,
        sma_slow=99.9,
        ema_fast=100.2,
        ema_slow=100.0,
        vwap=100.05,
        rsi=54.0,
        atr=1.7,
        plus_di=22.0,
        minus_di=19.0,
        adx=20.0,
        macd=0.35,
        macd_signal=0.32,
        macd_histogram=0.03,
        stochastic_k=53.0,
        stochastic_d=50.0,
        relative_volume=0.95,
        volume_profile=0.14,
        bollinger_middle=100.4,
        bollinger_upper=101.0,
        bollinger_lower=99.3,
    )

    assert SignalEngine().evaluate(conviction_snapshot, TradeState.FLAT).signal_tier is SignalTier.CONVICTION_BUY
    assert SignalEngine().evaluate(balanced_snapshot, TradeState.FLAT).signal_tier is SignalTier.BALANCED_BUY
    assert SignalEngine().evaluate(opportunistic_snapshot, TradeState.FLAT).signal_tier is SignalTier.OPPORTUNISTIC_BUY
    assert SignalEngine().evaluate(speculative_snapshot, TradeState.FLAT).signal_tier is SignalTier.SPECULATIVE_BUY


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

    decision = SignalEngine().evaluate(snapshot, TradeState.FLAT)

    assert decision.action is SignalAction.HOLD
    assert decision.entry_score < 0.7


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
