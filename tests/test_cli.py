from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import Mock

from trade_signal_edge.cli import (
    _build_market_snapshot_payload,
    _combine_timeframe_decisions,
    _publish_market_snapshots,
    _resolve_window_id_after_publish,
    _runtime_from_session_config,
)
from trade_signal_edge.config import load_runtime_config
from trade_signal_edge.models import Bar, IndicatorSnapshot, SignalAction, SignalDecision, SignalConfig, TradeState
from trade_signal_edge.signal_engine import SignalEngine


def make_decision(symbol: str, timestamp: datetime, entry_score: float, exit_score: float, reasons: tuple[str, ...] = ()) -> SignalDecision:
    return SignalDecision(
        symbol=symbol,
        timestamp=timestamp,
        entry_score=entry_score,
        exit_score=exit_score,
        action=SignalAction.HOLD,
        signal_tier=None,
        reasons=reasons,
    )


def test_combine_timeframe_decisions_prefers_weighted_entry_and_missing_timeframes() -> None:
    timestamp = datetime(2026, 4, 24, 13, 30, tzinfo=timezone.utc)
    snapshot = IndicatorSnapshot(
        symbol="NVDA",
        timestamp=timestamp,
        close=102.0,
        sma_fast=101.5,
        sma_slow=100.2,
        ema_fast=101.8,
        ema_slow=100.6,
        vwap=100.8,
        rsi=34.0,
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
    benchmark = IndicatorSnapshot(
        symbol="QQQ",
        timestamp=timestamp,
        close=101.0,
        ema_fast=100.8,
        ema_slow=100.1,
        sma_fast=100.7,
        sma_slow=100.0,
    )
    engine = SignalEngine(SignalConfig(entry_threshold=0.65, exit_threshold=0.55))

    decision = _combine_timeframe_decisions(
        "NVDA",
        {"1m": snapshot},
        {
            "1m": make_decision("NVDA", timestamp, 0.82, 0.21, ("trend-aligned",)),
        },
        {"1m": 1.0, "5m": 0.0, "15m": 0.0},
        {"1m": 1.0, "5m": 0.0, "15m": 0.0},
        engine,
        TradeState.FLAT,
        benchmark,
    )

    assert decision.action is SignalAction.BUY_ALERT
    assert decision.entry_score == 0.82
    assert decision.exit_score == 0.21
    assert "1m:trend-aligned" in decision.reasons
    assert "entry-qualified" in decision.reasons


def test_combine_timeframe_decisions_applies_benchmark_filter() -> None:
    timestamp = datetime(2026, 4, 24, 13, 30, tzinfo=timezone.utc)
    snapshot = IndicatorSnapshot(
        symbol="NVDA",
        timestamp=timestamp,
        close=102.0,
        sma_fast=101.5,
        sma_slow=100.2,
        ema_fast=101.8,
        ema_slow=100.6,
        vwap=100.8,
        rsi=34.0,
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
    bearish_benchmark = IndicatorSnapshot(
        symbol="QQQ",
        timestamp=timestamp,
        close=99.0,
        ema_fast=98.8,
        ema_slow=100.1,
        sma_fast=99.2,
        sma_slow=100.0,
    )
    class RecordingSignalEngine(SignalEngine):
        def __init__(self) -> None:
            super().__init__(SignalConfig(entry_threshold=0.65, exit_threshold=0.55))
            self.seen_benchmark = None

        def decide_action(self, *args, **kwargs):  # type: ignore[override]
            self.seen_benchmark = kwargs.get("benchmark")
            return super().decide_action(*args, **kwargs)

    engine = RecordingSignalEngine()

    decision = _combine_timeframe_decisions(
        "NVDA",
        {"1m": snapshot},
        {
            "1m": make_decision("NVDA", timestamp, 0.82, 0.21, ("trend-aligned",)),
        },
        {"1m": 1.0, "5m": 0.0, "15m": 0.0},
        {"1m": 1.0, "5m": 0.0, "15m": 0.0},
        engine,
        TradeState.FLAT,
        bearish_benchmark,
    )

    assert decision.action is SignalAction.BUY_ALERT
    assert engine.seen_benchmark is bearish_benchmark


def test_combine_timeframe_decisions_normalizes_buy_and_sell_independently() -> None:
    timestamp = datetime(2026, 4, 24, 13, 30, tzinfo=timezone.utc)
    snapshot = IndicatorSnapshot(
        symbol="AAPL",
        timestamp=timestamp,
        close=190.0,
        sma_fast=189.0,
        sma_slow=188.5,
        ema_fast=189.4,
        ema_slow=188.7,
        vwap=188.9,
        rsi=34.0,
        atr=1.2,
        plus_di=24.0,
        minus_di=17.0,
        adx=22.0,
        macd=0.7,
        macd_signal=0.5,
        macd_histogram=0.2,
        stochastic_k=42.0,
        stochastic_d=38.0,
    )
    engine = SignalEngine(SignalConfig(entry_threshold=0.65, exit_threshold=0.55))

    decision = _combine_timeframe_decisions(
        "AAPL",
        {"1m": snapshot},
        {
            "1m": make_decision("AAPL", timestamp, 0.82, 0.21, ("1m:trend-aligned",)),
        },
        {"1m": 0.5, "5m": 0.0, "15m": 0.0},
        {"1m": 1.0, "5m": 0.0, "15m": 0.0},
        engine,
        TradeState.FLAT,
    )

    assert decision.entry_score == 0.82
    assert decision.exit_score == 0.21
    assert decision.action is SignalAction.BUY_ALERT


def test_combine_timeframe_decisions_uses_exit_pressure_for_open_positions() -> None:
    timestamp = datetime(2026, 4, 24, 13, 30, tzinfo=timezone.utc)
    primary_snapshot = IndicatorSnapshot(
        symbol="TSLA",
        timestamp=timestamp,
        close=198.0,
        sma_fast=199.0,
        sma_slow=198.8,
        ema_fast=198.9,
        ema_slow=198.7,
        vwap=198.2,
        rsi=54.0,
        atr=2.2,
        plus_di=22.0,
        minus_di=17.0,
        adx=23.0,
        macd=0.03,
        macd_signal=0.02,
        macd_histogram=0.01,
        stochastic_k=55.0,
        stochastic_d=54.0,
    )
    exit_pressure_snapshot = IndicatorSnapshot(
        symbol="TSLA",
        timestamp=timestamp,
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
    engine = SignalEngine(SignalConfig(entry_threshold=0.65, exit_threshold=0.55))

    decision = _combine_timeframe_decisions(
        "TSLA",
        {"1m": primary_snapshot, "5m": exit_pressure_snapshot},
        {
            "1m": make_decision("TSLA", timestamp, 0.31, 0.72, ("1m:trend-pressure",)),
            "5m": make_decision("TSLA", timestamp, 0.28, 0.74, ("5m:trend-pressure",)),
        },
        {"1m": 1.0, "5m": 0.8, "15m": 0.0},
        {"1m": 1.0, "5m": 0.8, "15m": 0.0},
        engine,
        TradeState.ACCEPTED_OPEN,
    )

    assert decision.action is SignalAction.SELL_ALERT
    assert "exit-pressure" in decision.reasons
    assert "exit-qualified" in decision.reasons


def test_combine_timeframe_decisions_handles_zero_weights_without_crashing() -> None:
    timestamp = datetime(2026, 4, 24, 13, 30, tzinfo=timezone.utc)
    snapshot = IndicatorSnapshot(symbol="AAPL", timestamp=timestamp, close=190.0)
    engine = SignalEngine(SignalConfig(entry_threshold=0.65, exit_threshold=0.55))

    decision = _combine_timeframe_decisions(
        "AAPL",
        {"1m": snapshot},
        {"1m": make_decision("AAPL", timestamp, 0.34, 0.29, ("neutral",))},
        {"1m": 0.0, "5m": 0.0, "15m": 0.0},
        {"1m": 0.0, "5m": 0.0, "15m": 0.0},
        engine,
        TradeState.FLAT,
    )

    assert decision.action is SignalAction.HOLD


def test_combine_timeframe_decisions_prioritizes_fast_reversal_timeframes() -> None:
    timestamp = datetime(2026, 5, 4, 19, 22, tzinfo=timezone.utc)
    fast_snapshot = IndicatorSnapshot(
        symbol="NVDA",
        timestamp=timestamp,
        close=414.1,
        sma_fast=414.3,
        sma_slow=415.0,
        ema_fast=414.2,
        ema_slow=414.8,
        vwap=414.7,
        rsi=18.0,
        atr=3.0,
        plus_di=18.0,
        minus_di=22.0,
        adx=24.0,
        macd=-0.4,
        macd_signal=-0.2,
        macd_histogram=-0.2,
        stochastic_k=12.0,
        stochastic_d=14.0,
        relative_volume=1.2,
        volume_profile=0.2,
    )
    slow_snapshot = IndicatorSnapshot(
        symbol="NVDA",
        timestamp=timestamp,
        close=414.1,
        sma_fast=418.0,
        sma_slow=421.0,
        ema_fast=417.4,
        ema_slow=420.5,
        vwap=418.8,
        rsi=58.0,
        atr=3.0,
        plus_di=17.0,
        minus_di=24.0,
        adx=27.0,
        macd=-1.1,
        macd_signal=-0.8,
        macd_histogram=-0.3,
        stochastic_k=61.0,
        stochastic_d=58.0,
        relative_volume=0.92,
        volume_profile=0.14,
    )
    benchmark = IndicatorSnapshot(
        symbol="QQQ",
        timestamp=timestamp,
        close=500.0,
        sma_fast=501.0,
        sma_slow=503.0,
        ema_fast=500.5,
        ema_slow=502.0,
        vwap=501.0,
        rsi=12.0,
        atr=4.0,
        plus_di=14.0,
        minus_di=24.0,
        adx=27.0,
        macd=-1.0,
        macd_signal=-0.7,
        macd_histogram=-0.3,
        stochastic_k=9.0,
        stochastic_d=10.0,
        relative_volume=1.25,
        volume_profile=0.22,
    )
    engine = SignalEngine(SignalConfig(entry_threshold=0.62, exit_threshold=0.6))

    decision = _combine_timeframe_decisions(
        "NVDA",
        {
            "1m": fast_snapshot,
            "5m": fast_snapshot,
            "10m": fast_snapshot,
            "15m": fast_snapshot,
            "30m": slow_snapshot,
            "60m": slow_snapshot,
        },
        {
            "1m": make_decision("NVDA", timestamp, 0.92, 0.18, ("1m:reversal",)),
            "5m": make_decision("NVDA", timestamp, 0.81, 0.26, ("5m:reversal",)),
            "10m": make_decision("NVDA", timestamp, 0.56, 0.44, ("10m:mixed",)),
            "15m": make_decision("NVDA", timestamp, 0.49, 0.48, ("15m:mixed",)),
            "30m": make_decision("NVDA", timestamp, 0.24, 0.71, ("30m:bearish",)),
            "60m": make_decision("NVDA", timestamp, 0.18, 0.79, ("60m:bearish",)),
        },
        {"1m": 1.0, "5m": 0.85, "10m": 0.75, "15m": 0.6, "30m": 0.45, "60m": 0.3},
        {"1m": 1.0, "5m": 0.85, "10m": 0.75, "15m": 0.6, "30m": 0.45, "60m": 0.3},
        engine,
        TradeState.FLAT,
        benchmark,
    )

    assert decision.action is SignalAction.BUY_ALERT
    assert decision.entry_score > 0.6
    assert decision.exit_score < decision.entry_score


def test_runtime_from_session_config_filters_symbols_and_applies_margin() -> None:
    runtime = load_runtime_config()
    payload = {
        "selected_version": {
            "fields": [
                {"key": "monitored_symbols", "value": "AAPL,TSLA,NVDA,MSFT"},
                {"key": "entry_exit_margin", "value": "0.25"},
            ]
        }
    }

    updated = _runtime_from_session_config(runtime, payload)

    assert updated.symbols == ("TSLA", "NVDA")
    assert updated.symbol == "TSLA"
    assert updated.entry_exit_margin == 0.25


def test_runtime_from_session_config_falls_back_to_allowed_symbols_when_payload_is_unsupported() -> None:
    runtime = load_runtime_config()
    payload = {
        "selected_version": {
            "fields": [
                {"key": "monitored_symbols", "value": "AAPL,MSFT,GOOGL"},
            ]
        }
    }

    updated = _runtime_from_session_config(runtime, payload)

    assert updated.symbols == runtime.symbols
    assert updated.symbol == runtime.symbol


def test_resolve_window_id_after_publish_falls_back_to_open_windows_when_response_is_empty() -> None:
    session_client = Mock()
    session_client.load_open_windows.return_value = {"NVDA": "session:NVDA:decision-1"}

    resolved = _resolve_window_id_after_publish(
        publish_result=None,
        session_client=session_client,
        session_id="session-1",
        symbol="NVDA",
        current_window_id="",
        action=SignalAction.BUY_ALERT,
    )

    assert resolved == "session:NVDA:decision-1"
    session_client.load_open_windows.assert_called_once_with("session-1")


def test_build_market_snapshot_payload_omits_hold_signal_action() -> None:
    timestamp = datetime(2026, 4, 24, 13, 30, tzinfo=timezone.utc)
    snapshot = IndicatorSnapshot(
        symbol="AAPL",
        timestamp=timestamp,
        close=190.0,
    )
    bar = Bar(symbol="AAPL", timestamp=timestamp, open=189.5, high=190.5, low=189.2, close=190.0, volume=1_000.0)
    payload = _build_market_snapshot_payload(
        session_id="session-1",
        symbol="AAPL",
        timeframe="1m",
        bars_series=[bar],
        snapshot=snapshot,
        entry_score=0.42,
        exit_score=0.31,
        decision_action="HOLD",
        next_state="FLAT",
        benchmark_symbol="QQQ",
        regime="live market session",
        window_id="",
    )

    assert payload["signal_action"] is None
    assert payload["signal_tier"] is None
    assert payload["timeframe"] == "1m"


def test_publish_market_snapshots_publishes_each_timeframe_once() -> None:
    session_client = Mock()
    payloads = [
        ("1m", {"timeframe": "1m"}),
        ("5m", {"timeframe": "5m"}),
        ("15m", {"timeframe": "15m"}),
    ]
    errors: list[str] = []

    _publish_market_snapshots(session_client, "session-1", "AAPL", payloads, errors)

    assert errors == []
    assert session_client.publish_market_snapshot.call_count == 3
    assert session_client.publish_market_snapshot.call_args_list[0].args[0] == "session-1"
