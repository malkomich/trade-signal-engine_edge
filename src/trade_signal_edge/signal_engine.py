from __future__ import annotations

from dataclasses import dataclass, field
from math import isfinite

from .models import IndicatorSnapshot, SignalAction, SignalConfig, SignalDecision, TradeState


def _clamp(value: float, lower: float = 0.0, upper: float = 1.0) -> float:
    return max(lower, min(upper, value))


def _score_from_signal(raw_score: float, max_weight: float) -> float:
    if max_weight <= 0:
        return 0.5
    normalized = raw_score / max_weight
    return _clamp((normalized + 1.0) / 2.0)


@dataclass(slots=True)
class SignalEngine:
    config: SignalConfig = field(default_factory=SignalConfig)

    def evaluate(
        self,
        snapshot: IndicatorSnapshot,
        state: TradeState = TradeState.FLAT,
        benchmark: IndicatorSnapshot | None = None,
    ) -> SignalDecision:
        entry_raw = 0.0
        exit_raw = 0.0
        reasons: list[str] = []

        def add(signal_name: str, entry_value: float, exit_value: float) -> None:
            nonlocal entry_raw, exit_raw
            weight = float(self.config.weights[signal_name])
            entry_raw += weight * entry_value
            exit_raw += weight * exit_value

        sma_bias = self._trend_bias(snapshot.sma_fast, snapshot.sma_slow)
        ema_bias = self._trend_bias(snapshot.ema_fast, snapshot.ema_slow)
        vwap_bias = self._binary_bias(snapshot.close, snapshot.vwap)
        rsi_entry, rsi_exit = self._rsi_bias(snapshot.rsi)
        atr_entry, atr_exit = self._atr_bias(snapshot.atr, snapshot.close)
        dm_entry, dm_exit = self._dm_bias(snapshot.plus_di, snapshot.minus_di, snapshot.adx)
        macd_entry, macd_exit = self._macd_bias(snapshot.macd, snapshot.macd_signal, snapshot.macd_histogram)
        stochastic_entry, stochastic_exit = self._stochastic_bias(snapshot.stochastic_k, snapshot.stochastic_d)
        benchmark_entry, benchmark_exit, benchmark_reason = self._benchmark_bias(snapshot, benchmark)

        add("sma", sma_bias, -sma_bias)
        add("ema", ema_bias, -ema_bias)
        add("vwap", vwap_bias, -vwap_bias)
        add("rsi", rsi_entry, rsi_exit)
        add("atr", atr_entry, atr_exit)
        add("dm", dm_entry, dm_exit)
        add("macd", macd_entry, macd_exit)
        add("stochastic", stochastic_entry, stochastic_exit)
        entry_raw += benchmark_entry
        exit_raw += benchmark_exit

        max_weight = sum(float(weight) for weight in self.config.weights.values())
        # The benchmark term is capped separately in _benchmark_bias, so normalize each side with its own bound.
        benchmark_entry_weight = 0.575
        benchmark_exit_weight = 0.425
        entry_score = _score_from_signal(entry_raw, max_weight + benchmark_entry_weight)
        exit_score = _score_from_signal(exit_raw, max_weight + benchmark_exit_weight)
        strong_exit_pressure = self._strong_exit_pressure(snapshot)

        if state is TradeState.ACCEPTED_OPEN and (exit_score >= self.config.exit_threshold or strong_exit_pressure):
            if strong_exit_pressure:
                reasons.append("exit-pressure")
            action = SignalAction.SELL_ALERT
            if "exit-qualified" not in reasons:
                reasons.append("exit-qualified")
        elif state in {TradeState.FLAT, TradeState.REJECTED, TradeState.EXPIRED} and entry_score >= self.config.entry_threshold:
            action = SignalAction.BUY_ALERT
            reasons.append("entry-qualified")
        else:
            action = SignalAction.HOLD

        if benchmark_reason is not None:
            reasons.append(benchmark_reason)

        return SignalDecision(
            symbol=snapshot.symbol,
            timestamp=snapshot.timestamp,
            entry_score=round(entry_score, 4),
            exit_score=round(exit_score, 4),
            action=action,
            reasons=tuple(reasons),
        )

    def _strong_exit_pressure(self, snapshot: IndicatorSnapshot) -> bool:
        if snapshot.rsi is not None and snapshot.stochastic_k is not None:
            if snapshot.rsi >= 70 and snapshot.stochastic_k >= 80:
                return True
        if snapshot.vwap is not None and snapshot.macd_histogram is not None:
            if snapshot.close < snapshot.vwap and snapshot.macd_histogram < 0:
                return True
        return False

    def _trend_bias(self, fast: float | None, slow: float | None) -> float:
        if fast is None or slow is None or not isfinite(fast) or not isfinite(slow):
            return 0.0
        gap = fast - slow
        if gap > 0:
            return 1.0
        if gap < 0:
            return -1.0
        return 0.0

    def _binary_bias(self, close: float, reference: float | None) -> float:
        if reference is None or not isfinite(reference):
            return 0.0
        return 1.0 if close >= reference else -1.0

    def _rsi_bias(self, rsi: float | None) -> tuple[float, float]:
        if rsi is None or not isfinite(rsi):
            return 0.0, 0.0
        if rsi >= 70:
            return -0.8, 1.0
        if rsi >= 55:
            return 1.0, -0.4
        if rsi >= 45:
            return 0.2, 0.2
        if rsi >= 30:
            return -0.5, 0.6
        return -1.0, 0.9

    def _atr_bias(self, atr: float | None, close: float) -> tuple[float, float]:
        if atr is None or close <= 0 or not isfinite(atr):
            return 0.0, 0.0
        atr_ratio = atr / close
        if atr_ratio > 0.04:
            return -0.5, 0.8
        if atr_ratio < 0.008:
            return 0.2, -0.1
        return 0.6, 0.2

    def _dm_bias(self, plus_di: float | None, minus_di: float | None, adx: float | None) -> tuple[float, float]:
        if plus_di is None or minus_di is None:
            return 0.0, 0.0
        if plus_di > minus_di and (adx or 0) >= 20:
            return 1.0, -0.3
        if minus_di > plus_di and (adx or 0) >= 20:
            return -0.8, 1.0
        return 0.1, 0.1

    def _macd_bias(
        self,
        macd: float | None,
        macd_signal: float | None,
        macd_histogram: float | None,
    ) -> tuple[float, float]:
        if macd is None or macd_signal is None or macd_histogram is None:
            return 0.0, 0.0
        if macd > macd_signal and macd_histogram > 0:
            return 1.0, -0.5
        if macd < macd_signal and macd_histogram < 0:
            return -1.0, 1.0
        return 0.2, 0.2

    def _stochastic_bias(self, stochastic_k: float | None, stochastic_d: float | None) -> tuple[float, float]:
        if stochastic_k is None or stochastic_d is None:
            return 0.0, 0.0
        if stochastic_k < 20 and stochastic_k > stochastic_d:
            return 0.9, -0.3
        if stochastic_k > 80 and stochastic_k < stochastic_d:
            return -0.9, 0.9
        if stochastic_k > stochastic_d:
            return 0.3, 0.0
        return 0.0, 0.3

    def _benchmark_bias(
        self,
        snapshot: IndicatorSnapshot,
        benchmark: IndicatorSnapshot | None,
    ) -> tuple[float, float, str | None]:
        if benchmark is None:
            return 0.0, 0.0, None

        benchmark_trend = self._trend_bias(benchmark.ema_fast, benchmark.ema_slow)
        symbol_momentum = self._relative_momentum(snapshot.close, snapshot.ema_slow, snapshot.sma_slow)
        benchmark_momentum = self._relative_momentum(benchmark.close, benchmark.ema_slow, benchmark.sma_slow)
        relative_strength = symbol_momentum - benchmark_momentum
        relative_strength = max(-0.5, min(0.5, relative_strength))

        entry_bias = 0.35 * benchmark_trend + 0.45 * relative_strength
        exit_bias = -0.25 * benchmark_trend - 0.35 * relative_strength

        benchmark_label_raw = benchmark.symbol.strip()
        benchmark_label = benchmark_label_raw.lower() if benchmark_label_raw else "benchmark"
        if benchmark_trend > 0 and relative_strength > 0:
            return entry_bias, exit_bias, f"{benchmark_label}-aligned"
        if benchmark_trend < 0 and relative_strength < 0:
            return entry_bias, exit_bias, f"{benchmark_label}-pressure"
        return entry_bias, exit_bias, f"{benchmark_label}-mixed"

    def _relative_momentum(self, close: float, ema_slow: float | None, sma_slow: float | None) -> float:
        if close <= 0:
            return 0.0
        if ema_slow is not None and ema_slow > 0:
            return (close / ema_slow) - 1.0
        if sma_slow is not None and sma_slow > 0:
            return (close / sma_slow) - 1.0
        return 0.0
