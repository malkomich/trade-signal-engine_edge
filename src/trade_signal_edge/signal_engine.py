from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from math import isfinite
from zoneinfo import ZoneInfo

from .models import (
    IndicatorSnapshot,
    SignalAction,
    SignalConfig,
    SignalDecision,
    SignalTier,
    TradeState,
)

try:
    NEW_YORK_TIMEZONE = ZoneInfo("America/New_York")
except Exception:  # pragma: no cover - fallback for stripped tzdata environments
    NEW_YORK_TIMEZONE = timezone.utc

SELL_PRESSURE_MAX_SCORE = 1.3
BUY_ENTRY_FLOOR = 0.5
ENTRY_SCORE_RISK_ADJUSTMENT = 0.14
ENTRY_SCORE_SELL_PRESSURE_ADJUSTMENT = 0.05
OPENING_SESSION_RISK_HIGH = 0.95
OPENING_SESSION_RISK_MEDIUM = 0.85
OPENING_SESSION_RISK_LOW = 0.7
OPENING_SESSION_PENALTY_HIGH = 0.12
OPENING_SESSION_PENALTY_MEDIUM = 0.06
OPENING_SESSION_PENALTY_LOW = 0.03
ENTRY_MARGIN_BASE = 0.03
ENTRY_MARGIN_STRONG_EXIT_BONUS = 0.04
BUY_QUALITY_RSI_MIN = 48.0
BUY_QUALITY_RSI_MAX = 70.0
BUY_QUALITY_STOCHASTIC_MAX = 85.0
BUY_QUALITY_RELATIVE_VOLUME_MIN = 0.95
BUY_QUALITY_VOLUME_PROFILE_MIN = 0.15
BUY_QUALITY_SUPPORT_THRESHOLD = 0.62
BUY_QUALITY_MIN_SUPPORTING_SIGNALS = 2
BUY_QUALITY_TREND_WEIGHT = 0.4
BUY_QUALITY_FLOW_WEIGHT = 0.25
BUY_QUALITY_MOMENTUM_WEIGHT = 0.2
BUY_QUALITY_VOLATILITY_WEIGHT = 0.1
BUY_QUALITY_STRENGTH_WEIGHT = 0.05
BUY_RSI_OVERSOLD_THRESHOLD = 30.0
BUY_STOCHASTIC_OVERSOLD_THRESHOLD = 20.0
BUY_STOCHASTIC_LOW_THRESHOLD = 30.0
BUY_OVERSOLD_REVERSAL_ENTRY_BONUS = 0.35
BUY_OVERSOLD_REVERSAL_EXIT_PENALTY = 0.18
BUY_OVERSOLD_REVERSAL_MACD_ENTRY_BONUS = 0.5
BUY_OVERSOLD_REVERSAL_MACD_EXIT_PENALTY = 0.15
BUY_OVERSOLD_REVERSAL_ENTRY_BOOST = 1.2
BUY_OVERSOLD_REVERSAL_EXIT_BOOST = 0.18
BUY_OVERSOLD_REVERSAL_EXIT_SCORE_REDUCTION = 0.35
BUY_TIER_STRONG_EXIT_PENALTY = 0.08
BUY_TIER_HIGH_RISK_PENALTY = 0.05
BUY_TIER_CONVICTION_ENTRY = 0.73
BUY_TIER_CONVICTION_QUALITY = 0.68
BUY_TIER_CONVICTION_MAX_RISK = 0.52
BUY_TIER_BALANCED_ENTRY = 0.66
BUY_TIER_BALANCED_QUALITY = 0.58
BUY_TIER_BALANCED_MAX_RISK = 0.7
BUY_TIER_OPPORTUNISTIC_ENTRY = 0.56
BUY_TIER_OPPORTUNISTIC_QUALITY = 0.46
BUY_TIER_OPPORTUNISTIC_MAX_RISK = 0.84
BUY_TIER_SPECULATIVE_ENTRY = 0.5
BUY_TIER_SPECULATIVE_QUALITY = 0.36
BUY_TIER_SPECULATIVE_MAX_RISK = 0.94
BUY_TIER_CONVICTION_MIN_SUPPORTING_SIGNALS = 4
BUY_TIER_BALANCED_MIN_SUPPORTING_SIGNALS = 3
BUY_TIER_OPPORTUNISTIC_MIN_SUPPORTING_SIGNALS = 2
BUY_TIER_SPECULATIVE_MIN_SUPPORTING_SIGNALS = 2
BUY_RSI_BEARISH_THRESHOLD = 55.0
BUY_RSI_STRONG_ZONE_MAX = 35.0
BUY_STOCHASTIC_OVERBOUGHT_THRESHOLD = 78.0
BUY_MACD_BEARISH_CROSS_GUARD = 0.0


@dataclass(slots=True)
class QualitySlice:
    score: float
    supportive_signals: int
    component_count: int
    reasons: tuple[str, ...] = ()


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
            buy_weight = float(self.config.buy_weights[signal_name])
            sell_weight = float(self.config.sell_weights[signal_name])
            entry_raw += buy_weight * entry_value
            exit_raw += sell_weight * exit_value

        bullish_reversal_context = self._bullish_reversal_context(snapshot)
        sma_bias = self._trend_bias(snapshot.sma_fast, snapshot.sma_slow, bullish_reversal_context)
        ema_bias = self._trend_bias(snapshot.ema_fast, snapshot.ema_slow, bullish_reversal_context)
        vwap_bias = self._binary_bias(snapshot.close, snapshot.vwap, bullish_reversal_context)
        bollinger_entry, bollinger_exit = self._bollinger_bias(
            snapshot.close,
            snapshot.bollinger_middle,
            snapshot.bollinger_upper,
            snapshot.bollinger_lower,
        )
        rsi_entry, rsi_exit = self._rsi_bias(snapshot.rsi)
        atr_entry, atr_exit = self._atr_bias(snapshot.atr, snapshot.close)
        dm_entry, dm_exit = self._dm_bias(snapshot.plus_di, snapshot.minus_di, snapshot.adx)
        macd_entry, macd_exit = self._macd_bias(
            snapshot.macd,
            snapshot.macd_signal,
            snapshot.macd_histogram,
            bullish_reversal_context,
        )
        stochastic_entry, stochastic_exit = self._stochastic_bias(snapshot.stochastic_k, snapshot.stochastic_d)
        obv_entry, obv_exit = self._obv_bias(snapshot.obv, snapshot.obv_delta, snapshot.relative_volume, snapshot.volume_profile)
        relative_volume_entry, relative_volume_exit = self._relative_volume_bias(snapshot.relative_volume)
        volume_profile_entry, volume_profile_exit = self._volume_profile_bias(snapshot.volume_profile)
        benchmark_entry, benchmark_exit, benchmark_reason = self._benchmark_bias(snapshot, benchmark)
        profile_entry, profile_exit, profile_reason = self._optimization_bias(snapshot)
        risk_score = self._risk_score(snapshot)
        session_risk = self._session_risk(snapshot.timestamp)
        strong_exit_pressure = self.is_strong_exit_pressure(snapshot)
        sell_pressure = self._sell_pressure_bias(snapshot)
        quality_assessment = self._long_entry_quality_assessment(snapshot, benchmark, bullish_reversal_context)

        add("sma", sma_bias, -sma_bias)
        add("ema", ema_bias, -ema_bias)
        add("vwap", vwap_bias, -vwap_bias)
        add("bollinger", bollinger_entry, bollinger_exit)
        add("rsi", rsi_entry, rsi_exit)
        add("atr", atr_entry, atr_exit)
        add("dm", dm_entry, dm_exit)
        add("macd", macd_entry, macd_exit)
        add("stochastic", stochastic_entry, stochastic_exit)
        add("obv", obv_entry, obv_exit)
        add("relative_volume", relative_volume_entry, relative_volume_exit)
        add("volume_profile", volume_profile_entry, volume_profile_exit)
        entry_raw += benchmark_entry
        exit_raw += benchmark_exit
        entry_raw += profile_entry
        exit_raw += profile_exit
        if bullish_reversal_context:
            entry_raw += BUY_OVERSOLD_REVERSAL_ENTRY_BOOST
            exit_raw -= BUY_OVERSOLD_REVERSAL_EXIT_BOOST

        buy_max_weight = sum(float(weight) for weight in self.config.buy_weights.values())
        sell_max_weight = sum(float(weight) for weight in self.config.sell_weights.values())
        # The benchmark term is capped separately in _benchmark_bias, so normalize each side with its own bound.
        benchmark_entry_weight = 0.575
        benchmark_exit_weight = 0.425
        entry_score = _clamp(
            _score_from_signal(entry_raw, buy_max_weight + benchmark_entry_weight)
            * (1.0 - (risk_score * ENTRY_SCORE_RISK_ADJUSTMENT))
            * (1.0 - (sell_pressure * ENTRY_SCORE_SELL_PRESSURE_ADJUSTMENT))
        )
        exit_score = _clamp(
            _score_from_signal(exit_raw, sell_max_weight + benchmark_exit_weight)
            + (risk_score * 0.3)
            + (sell_pressure * 0.5)
            + (0.08 if strong_exit_pressure else 0.0)
        )
        if bullish_reversal_context:
            exit_score = _clamp(exit_score - BUY_OVERSOLD_REVERSAL_EXIT_SCORE_REDUCTION)
        action, action_reasons, signal_tier = self.decide_action(
            entry_score,
            exit_score,
            state,
            snapshot,
            benchmark,
            risk_score,
            session_risk,
            strong_exit_pressure,
            quality_assessment,
            bullish_reversal_context,
        )
        reasons.extend(action_reasons)
        if bullish_reversal_context:
            reasons.append("oversold-reversal-context")

        if benchmark_reason is not None:
            reasons.append(benchmark_reason)
        if profile_reason is not None:
            reasons.append(profile_reason)

        return SignalDecision(
            symbol=snapshot.symbol,
            timestamp=snapshot.timestamp,
            entry_score=round(entry_score, 4),
            exit_score=round(exit_score, 4),
            action=action,
            signal_tier=signal_tier,
            reasons=tuple(reasons),
        )

    def decide_action(
        self,
        entry_score: float,
        exit_score: float,
        state: TradeState,
        snapshot: IndicatorSnapshot | None = None,
        benchmark: IndicatorSnapshot | None = None,
        risk_score: float | None = None,
        session_risk: float | None = None,
        strong_exit_pressure: bool | None = None,
        quality_assessment: QualitySlice | None = None,
        bullish_reversal_context: bool | None = None,
    ) -> tuple[SignalAction, tuple[str, ...], SignalTier | None]:
        if strong_exit_pressure is None:
            strong_exit_pressure = self.is_strong_exit_pressure(snapshot)
        if session_risk is None and snapshot is not None:
            session_risk = self._session_risk(snapshot.timestamp)
        if bullish_reversal_context is None:
            bullish_reversal_context = self._bullish_reversal_context(snapshot) if snapshot is not None else self._bullish_reversal_context(benchmark)

        if state is TradeState.ACCEPTED_OPEN and exit_score >= self.config.exit_threshold:
            reasons: list[str] = []
            if strong_exit_pressure:
                reasons.append("exit-pressure")
            reasons.append("exit-qualified")
            return SignalAction.SELL_ALERT, tuple(reasons), None

        entry_gate = min(self.config.entry_threshold, self.config.entry_gate_cap)
        if session_risk is not None:
            opening_penalty = self._opening_session_penalty(session_risk)
            entry_gate += opening_penalty
            if bullish_reversal_context:
                entry_gate -= opening_penalty * 0.5
        entry_gate = max(BUY_ENTRY_FLOOR, entry_gate)
        if state in {TradeState.FLAT, TradeState.REJECTED, TradeState.EXPIRED} and entry_score >= entry_gate:
            if snapshot is None:
                return SignalAction.HOLD, (), None
            if not self._buy_momentum_gate(snapshot, benchmark, bullish_reversal_context):
                return SignalAction.HOLD, (), None
            if not self._buy_trend_gate(snapshot, benchmark, bullish_reversal_context):
                return SignalAction.HOLD, (), None
            if quality_assessment is None:
                quality_assessment = self._long_entry_quality_assessment(snapshot, benchmark, bullish_reversal_context)
            quality_score = quality_assessment.score
            min_supporting_signals = BUY_QUALITY_MIN_SUPPORTING_SIGNALS
            if bullish_reversal_context:
                min_supporting_signals = 1
            if quality_score <= 0 or quality_assessment.supportive_signals < min_supporting_signals:
                return SignalAction.HOLD, (), None
            effective_margin = max(ENTRY_MARGIN_BASE, self.config.entry_exit_margin * 0.6)
            if strong_exit_pressure:
                effective_margin += ENTRY_MARGIN_STRONG_EXIT_BONUS
            if bullish_reversal_context:
                reversal_margin = max(ENTRY_MARGIN_BASE * 0.5, self.config.entry_exit_margin * 0.25)
                effective_margin = min(effective_margin, reversal_margin)
            if exit_score >= entry_score - effective_margin:
                return SignalAction.HOLD, (), None
            if risk_score is None:
                risk_score = self._risk_score(snapshot)
            if session_risk is None:
                session_risk = self._session_risk(snapshot.timestamp)
            buy_tier = self._buy_signal_tier(
                entry_score=entry_score,
                risk_score=risk_score,
                quality_score=quality_score,
                supportive_signals=quality_assessment.supportive_signals,
                session_risk=session_risk,
                strong_exit_pressure=strong_exit_pressure,
            )
            if buy_tier is None:
                return SignalAction.HOLD, (), None
            reasons = ["entry-qualified", f"buy-tier:{buy_tier.value}"]
            reasons.extend(quality_assessment.reasons)
            return SignalAction.BUY_ALERT, tuple(dict.fromkeys(reasons)), buy_tier

        return SignalAction.HOLD, (), None

    def is_strong_exit_pressure(self, snapshot: IndicatorSnapshot | None) -> bool:
        if snapshot is None:
            return False
        if snapshot.rsi is not None and snapshot.rsi >= 70:
            return True
        if snapshot.stochastic_k is not None and snapshot.stochastic_d is not None:
            if snapshot.stochastic_k >= 80 and snapshot.stochastic_k >= snapshot.stochastic_d:
                return True
            if snapshot.stochastic_k >= 75 and snapshot.stochastic_k > snapshot.stochastic_d:
                return True
        if snapshot.vwap is not None and snapshot.macd_histogram is not None:
            if snapshot.close < snapshot.vwap and snapshot.macd_histogram < 0:
                return True
        if (
            snapshot.bollinger_middle is not None
            and snapshot.macd_histogram is not None
            and snapshot.close < snapshot.bollinger_middle
            and snapshot.macd_histogram < 0
        ):
            return True
        if snapshot.ema_fast is not None and snapshot.macd_histogram is not None:
            if snapshot.close < snapshot.ema_fast and snapshot.macd_histogram < 0:
                return True
        if snapshot.plus_di is not None and snapshot.minus_di is not None and snapshot.adx is not None:
            if snapshot.adx >= 20 and snapshot.minus_di > snapshot.plus_di:
                return True
        if snapshot.relative_volume is not None and snapshot.relative_volume < 0.7:
            return True
        if snapshot.volume_profile is not None and snapshot.volume_profile < 0.12:
            return True
        return False

    def _bullish_reversal_context(
        self,
        snapshot: IndicatorSnapshot | None,
        benchmark: IndicatorSnapshot | None = None,
    ) -> bool:
        for candidate in (snapshot, benchmark):
            if candidate is None:
                continue
            if candidate.rsi is None or candidate.stochastic_k is None or candidate.stochastic_d is None:
                continue
            if not isfinite(candidate.rsi) or not isfinite(candidate.stochastic_k) or not isfinite(candidate.stochastic_d):
                continue
            if candidate.rsi <= BUY_RSI_OVERSOLD_THRESHOLD and candidate.stochastic_k <= BUY_STOCHASTIC_OVERSOLD_THRESHOLD:
                return True
        return False

    def _buy_momentum_gate(
        self,
        snapshot: IndicatorSnapshot,
        benchmark: IndicatorSnapshot | None = None,
        bullish_reversal_context: bool | None = None,
    ) -> bool:
        if bullish_reversal_context is None:
            bullish_reversal_context = self._bullish_reversal_context(snapshot, benchmark)
        if snapshot.rsi is not None and isfinite(snapshot.rsi):
            if snapshot.rsi >= BUY_RSI_BEARISH_THRESHOLD:
                return False
        if snapshot.stochastic_k is not None and snapshot.stochastic_d is not None:
            if snapshot.stochastic_k >= BUY_STOCHASTIC_OVERBOUGHT_THRESHOLD and snapshot.stochastic_k >= snapshot.stochastic_d:
                return False
        if (
            snapshot.macd is not None
            and snapshot.macd_signal is not None
            and snapshot.macd_histogram is not None
            and snapshot.macd_histogram <= BUY_MACD_BEARISH_CROSS_GUARD
            and not bullish_reversal_context
        ):
            return False
        return True

    def _buy_trend_gate(
        self,
        snapshot: IndicatorSnapshot,
        benchmark: IndicatorSnapshot | None = None,
        bullish_reversal_context: bool | None = None,
    ) -> bool:
        if bullish_reversal_context is None:
            bullish_reversal_context = self._bullish_reversal_context(snapshot, benchmark)
        trend_votes = 0
        aligned_votes = 0
        if snapshot.vwap is not None and isfinite(snapshot.vwap):
            trend_votes += 1
            if snapshot.close >= snapshot.vwap:
                aligned_votes += 1
        if snapshot.ema_fast is not None and snapshot.ema_slow is not None:
            trend_votes += 1
            if snapshot.ema_fast >= snapshot.ema_slow:
                aligned_votes += 1
        if snapshot.sma_fast is not None and snapshot.sma_slow is not None:
            trend_votes += 1
            if snapshot.sma_fast >= snapshot.sma_slow:
                aligned_votes += 1

        if trend_votes == 0:
            # Reversal entries can proceed on momentum + flow alone when trend data is absent.
            return bullish_reversal_context
        if aligned_votes == 0:
            # Oversold reversals are allowed to override weak trend alignment.
            return bullish_reversal_context
        return True

    def _sell_pressure_bias(self, snapshot: IndicatorSnapshot) -> float:
        score = 0.0
        macd_bearish = snapshot.macd_histogram is not None and snapshot.macd_histogram < 0
        if snapshot.rsi is not None:
            if snapshot.rsi >= 70:
                score += 0.35
            elif snapshot.rsi >= 65:
                score += 0.2
        if snapshot.stochastic_k is not None and snapshot.stochastic_d is not None:
            if snapshot.stochastic_k >= 80 and snapshot.stochastic_k >= snapshot.stochastic_d:
                score += 0.35
            elif snapshot.stochastic_k >= 75 and snapshot.stochastic_k > snapshot.stochastic_d:
                score += 0.2
        if snapshot.vwap is not None and macd_bearish and snapshot.close < snapshot.vwap:
            score += 0.2
        if snapshot.bollinger_middle is not None and macd_bearish and snapshot.close < snapshot.bollinger_middle:
            score += 0.15
        if snapshot.ema_fast is not None and macd_bearish and snapshot.close < snapshot.ema_fast:
            score += 0.15
        if snapshot.plus_di is not None and snapshot.minus_di is not None and snapshot.adx is not None:
            if snapshot.adx >= 20 and snapshot.minus_di > snapshot.plus_di:
                score += 0.15
        if snapshot.relative_volume is not None and snapshot.relative_volume < 0.9:
            score += 0.1
        if snapshot.volume_profile is not None and snapshot.volume_profile < 0.15:
            score += 0.1
        if macd_bearish:
            score += 0.15
        return _clamp(score / SELL_PRESSURE_MAX_SCORE)

    def _opening_session_penalty(self, session_risk: float) -> float:
        if session_risk >= OPENING_SESSION_RISK_HIGH:
            return OPENING_SESSION_PENALTY_HIGH
        if session_risk >= OPENING_SESSION_RISK_MEDIUM:
            return OPENING_SESSION_PENALTY_MEDIUM
        if session_risk >= OPENING_SESSION_RISK_LOW:
            return OPENING_SESSION_PENALTY_LOW
        return 0.0

    def _trend_bias(self, fast: float | None, slow: float | None, bullish_reversal_context: bool = False) -> float:
        if fast is None or slow is None or not isfinite(fast) or not isfinite(slow):
            return 0.0
        gap = fast - slow
        if gap > 0:
            return 1.0
        if gap < 0:
            if bullish_reversal_context:
                return 0.5
            return -1.0
        return 0.0

    def _binary_bias(self, close: float, reference: float | None, bullish_reversal_context: bool = False) -> float:
        if reference is None or not isfinite(reference):
            return 0.0
        if close >= reference:
            return 1.0
        if bullish_reversal_context:
            return 0.45
        return -1.0

    def _rsi_bias(self, rsi: float | None) -> tuple[float, float]:
        if rsi is None or not isfinite(rsi):
            return 0.0, 0.0
        if rsi >= 70:
            return -1.0, 1.0
        if rsi >= BUY_RSI_BEARISH_THRESHOLD:
            return -0.9, 0.9
        if rsi >= 45:
            return -0.1, 0.2
        if rsi >= 35:
            return 0.6, -0.1
        if rsi >= BUY_RSI_OVERSOLD_THRESHOLD:
            return 0.9, -0.3
        if rsi >= 25:
            return 1.0, -0.4
        if rsi >= 20:
            return 1.0, -0.5
        return 1.0, -0.6

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
        bullish_reversal_context: bool = False,
    ) -> tuple[float, float]:
        if macd is None or macd_signal is None or macd_histogram is None:
            return 0.0, 0.0
        if macd > macd_signal and macd_histogram > 0:
            return 1.0, -0.5
        if macd > macd_signal and macd_histogram <= 0:
            return 0.5, 0.1
        if bullish_reversal_context and macd <= macd_signal and macd_histogram <= 0:
            return BUY_OVERSOLD_REVERSAL_MACD_ENTRY_BONUS, -BUY_OVERSOLD_REVERSAL_MACD_EXIT_PENALTY
        if macd < macd_signal and macd_histogram < 0:
            return -1.0, 1.0
        if macd < macd_signal and macd_histogram >= 0:
            return -0.4, 0.6
        return 0.0, 0.1

    def _bollinger_bias(
        self,
        close: float,
        middle: float | None,
        upper: float | None,
        lower: float | None,
    ) -> tuple[float, float]:
        if middle is None or upper is None or lower is None:
            return 0.0, 0.0
        if close >= upper:
            return 0.4, 1.0
        if close > middle:
            return 0.8, -0.2
        if close < lower:
            return -0.9, 1.0
        return -0.3, 0.3

    def _obv_bias(
        self,
        _obv: float | None,
        obv_delta: float | None,
        relative_volume: float | None,
        volume_profile: float | None,
    ) -> tuple[float, float]:
        if _obv is None or not isfinite(_obv):
            return 0.0, 0.0
        if obv_delta is not None and isfinite(obv_delta) and obv_delta > 0 and relative_volume is not None and relative_volume >= 1.1 and volume_profile is not None and volume_profile >= 0.18:
            return 0.6, -0.1
        if obv_delta is not None and isfinite(obv_delta) and obv_delta < 0 and relative_volume is not None and relative_volume < 0.9 and (volume_profile is None or volume_profile < 0.15):
            return -0.5, 0.7
        if _obv > 0:
            return 0.1, 0.1
        if _obv < 0:
            return -0.1, 0.2
        return 0.0, 0.0

    def _relative_volume_bias(self, relative_volume: float | None) -> tuple[float, float]:
        if relative_volume is None or not isfinite(relative_volume):
            return 0.0, 0.0
        if relative_volume >= 1.5:
            return 0.8, 0.2
        if relative_volume >= 1.0:
            return 0.5, 0.0
        if relative_volume < 0.8:
            return -0.4, 0.8
        return 0.2, 0.1

    def _volume_profile_bias(self, volume_profile: float | None) -> tuple[float, float]:
        if volume_profile is None or not isfinite(volume_profile):
            return 0.0, 0.0
        if volume_profile >= 0.25:
            return 0.5, 0.2
        if volume_profile < 0.1:
            return -0.3, 0.5
        return 0.2, 0.1

    def _stochastic_bias(self, stochastic_k: float | None, stochastic_d: float | None) -> tuple[float, float]:
        if stochastic_k is None or stochastic_d is None:
            return 0.0, 0.0
        if stochastic_k < 20 and stochastic_k > stochastic_d:
            return 0.9, -0.3
        if stochastic_k >= BUY_STOCHASTIC_OVERBOUGHT_THRESHOLD and stochastic_k >= stochastic_d:
            return -1.0, 1.0
        if stochastic_k > stochastic_d and stochastic_k < BUY_STOCHASTIC_OVERBOUGHT_THRESHOLD:
            return 0.5, -0.1
        if stochastic_k < stochastic_d and stochastic_k <= 30:
            return -0.4, 0.6
        return 0.0, 0.1

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

        benchmark_label = benchmark.symbol.strip().upper() if benchmark.symbol.strip() else "BENCHMARK"
        if self._bullish_reversal_context(benchmark):
            entry_bias += BUY_OVERSOLD_REVERSAL_ENTRY_BONUS
            exit_bias -= BUY_OVERSOLD_REVERSAL_EXIT_PENALTY
            return entry_bias, exit_bias, f"{benchmark_label} oversold reversal context"
        if benchmark_trend > 0 and relative_strength > 0:
            return entry_bias, exit_bias, f"{benchmark_label} market context aligned"
        if benchmark_trend < 0 and relative_strength < 0:
            return entry_bias, exit_bias, f"{benchmark_label} market context under pressure"
        return entry_bias, exit_bias, f"{benchmark_label} mixed market context"

    def _relative_momentum(self, close: float, ema_slow: float | None, sma_slow: float | None) -> float:
        if close <= 0:
            return 0.0
        if ema_slow is not None and ema_slow > 0:
            return (close / ema_slow) - 1.0
        if sma_slow is not None and sma_slow > 0:
            return (close / sma_slow) - 1.0
        return 0.0

    def _risk_score(self, snapshot: IndicatorSnapshot) -> float:
        volatility_risk = self._volatility_risk(snapshot.atr, snapshot.close)
        volatility_spike_risk = self._volatility_spike_risk(snapshot.atr, snapshot.close, snapshot.relative_volume)
        session_risk = self._session_risk(snapshot.timestamp)
        microstructure_risk = self._microstructure_risk(snapshot)
        return _clamp(
            (0.35 * volatility_risk)
            + (0.25 * volatility_spike_risk)
            + (0.20 * session_risk)
            + (0.20 * microstructure_risk)
        )

    def _volatility_risk(self, atr: float | None, close: float) -> float:
        if atr is None or close <= 0 or not isfinite(atr):
            return 0.5
        atr_ratio = atr / close
        if atr_ratio >= 0.04:
            return 1.0
        if atr_ratio >= 0.03:
            return 0.85
        if atr_ratio >= 0.02:
            return 0.65
        if atr_ratio >= 0.012:
            return 0.45
        return 0.25

    def _volatility_spike_risk(self, atr: float | None, close: float, relative_volume: float | None) -> float:
        if atr is None or close <= 0 or not isfinite(atr):
            return 0.5
        atr_ratio = atr / close
        score = 0.0
        if atr_ratio >= 0.04:
            score += 0.65
        elif atr_ratio >= 0.03:
            score += 0.5
        elif atr_ratio >= 0.02:
            score += 0.35
        elif atr_ratio >= 0.012:
            score += 0.2
        else:
            score += 0.1
        if relative_volume is not None and isfinite(relative_volume):
            if relative_volume >= 2.5:
                score += 0.35
            elif relative_volume >= 1.8:
                score += 0.25
            elif relative_volume <= 0.7:
                score += 0.3
            elif relative_volume <= 0.9:
                score += 0.15
        return _clamp(score)

    def _session_risk(self, timestamp: datetime) -> float:
        local_time = timestamp.astimezone(NEW_YORK_TIMEZONE)
        minute_of_day = local_time.hour * 60 + local_time.minute
        market_open = 9 * 60 + 30
        if minute_of_day < market_open:
            return 0.0
        if minute_of_day < 10 * 60:
            return 1.0
        if minute_of_day < 10 * 60 + 30:
            return 0.9
        if minute_of_day < 11 * 60 + 30:
            return 0.7
        if minute_of_day < 14 * 60 + 30:
            return 0.45
        if minute_of_day < 15 * 60 + 30:
            return 0.6
        return 0.8

    def _microstructure_risk(self, snapshot: IndicatorSnapshot) -> float:
        score = 0.0
        if snapshot.relative_volume is not None and isfinite(snapshot.relative_volume):
            if snapshot.relative_volume <= 0.8:
                score += 0.25
            elif snapshot.relative_volume >= 2.0:
                score += 0.2
        if snapshot.volume_profile is not None and isfinite(snapshot.volume_profile):
            if snapshot.volume_profile < 0.15:
                score += 0.35
            elif snapshot.volume_profile < 0.25:
                score += 0.15
        if snapshot.vwap is not None and snapshot.macd_histogram is not None:
            if snapshot.close < snapshot.vwap and snapshot.macd_histogram < 0:
                score += 0.2
        if snapshot.plus_di is not None and snapshot.minus_di is not None and snapshot.adx is not None:
            if snapshot.adx >= 20 and snapshot.minus_di > snapshot.plus_di:
                score += 0.2
        return _clamp(score)

    def _optimization_bias(self, snapshot: IndicatorSnapshot) -> tuple[float, float, str | None]:
        entry_profile = dict(self.config.entry_profile)
        exit_profile = dict(self.config.exit_profile)
        if not entry_profile and not exit_profile:
            return 0.0, 0.0, None

        current = self._snapshot_profile(snapshot)
        entry_distance = self._profile_distance(current, entry_profile)
        exit_distance = self._profile_distance(current, exit_profile)
        if entry_distance is None and exit_distance is None:
            return 0.0, 0.0, None
        if entry_distance is None:
            exit_similarity = 1.0 / (1.0 + exit_distance)
            bias = -min(self.config.optimizer_bias_cap, self.config.optimizer_learning_rate * exit_similarity)
            return bias, -bias, "optimizer profile favors exit"
        if exit_distance is None:
            entry_similarity = 1.0 / (1.0 + entry_distance)
            bias = min(self.config.optimizer_bias_cap, self.config.optimizer_learning_rate * entry_similarity)
            return bias, -bias, "optimizer profile favors entry"

        bias = (exit_distance - entry_distance) * self.config.optimizer_learning_rate
        bias = max(-self.config.optimizer_bias_cap, min(self.config.optimizer_bias_cap, bias))
        if abs(bias) < 1e-6:
            return 0.0, 0.0, "optimizer profile balanced"
        if bias > 0:
            return bias, -bias, "optimizer profile favors entry"
        return bias, -bias, "optimizer profile favors exit"

    def _profile_distance(self, current: dict[str, float], profile: dict[str, float]) -> float | None:
        total = 0.0
        count = 0
        for key, target in profile.items():
            if not isfinite(target):
                continue
            value = current.get(key)
            if value is None or not isfinite(value):
                continue
            scale = max(abs(target), 1.0)
            total += abs(value - target) / scale
            count += 1
        if count == 0:
            return None
        return total / count

    def _snapshot_profile(self, snapshot: IndicatorSnapshot) -> dict[str, float | None]:
        return {
            "close": snapshot.close,
            "sma_fast": snapshot.sma_fast,
            "sma_slow": snapshot.sma_slow,
            "ema_fast": snapshot.ema_fast,
            "ema_slow": snapshot.ema_slow,
            "vwap": snapshot.vwap,
            "rsi": snapshot.rsi,
            "atr": snapshot.atr,
            "plus_di": snapshot.plus_di,
            "minus_di": snapshot.minus_di,
            "adx": snapshot.adx,
            "macd": snapshot.macd,
            "macd_signal": snapshot.macd_signal,
            "macd_histogram": snapshot.macd_histogram,
            "stochastic_k": snapshot.stochastic_k,
            "stochastic_d": snapshot.stochastic_d,
            "bollinger_middle": snapshot.bollinger_middle,
            "bollinger_upper": snapshot.bollinger_upper,
            "bollinger_lower": snapshot.bollinger_lower,
            "obv": snapshot.obv,
            "obv_delta": snapshot.obv_delta,
            "relative_volume": snapshot.relative_volume,
            "volume_profile": snapshot.volume_profile,
        }

    def _long_entry_quality_score(self, snapshot: IndicatorSnapshot | None, benchmark: IndicatorSnapshot | None) -> float:
        if snapshot is None:
            return 0.0
        return self._long_entry_quality_assessment(snapshot, benchmark).score

    def _long_entry_quality_assessment(
        self,
        snapshot: IndicatorSnapshot,
        benchmark: IndicatorSnapshot | None,
        bullish_reversal_context: bool | None = None,
    ) -> QualitySlice:
        if bullish_reversal_context is None:
            bullish_reversal_context = self._bullish_reversal_context(snapshot, benchmark)
        trend = self._trend_quality_slice(snapshot, bullish_reversal_context)
        flow = self._flow_quality_slice(snapshot)
        momentum = self._momentum_quality_slice(snapshot, bullish_reversal_context)
        volatility = self._volatility_quality_slice(snapshot)
        strength = self._strength_quality_slice(snapshot)

        weighted_score = 0.0
        weighted_denominator = 0.0
        for category_weight, category in (
            (BUY_QUALITY_TREND_WEIGHT, trend),
            (BUY_QUALITY_FLOW_WEIGHT, flow),
            (BUY_QUALITY_MOMENTUM_WEIGHT, momentum),
            (BUY_QUALITY_VOLATILITY_WEIGHT, volatility),
            (BUY_QUALITY_STRENGTH_WEIGHT, strength),
        ):
            if category.component_count <= 0:
                continue
            weighted_score += category_weight * category.score
            weighted_denominator += category_weight
        if weighted_denominator > 0:
            weighted_score /= weighted_denominator

        if benchmark is not None and benchmark.ema_fast is not None and benchmark.ema_slow is not None:
            if benchmark.ema_fast >= benchmark.ema_slow or bullish_reversal_context:
                weighted_score = min(1.0, weighted_score + 0.05)

        weighted_score = _clamp(weighted_score)
        reasons = trend.reasons + flow.reasons + momentum.reasons + volatility.reasons + strength.reasons
        supportive_signals = sum(
            1 for category in (trend, flow, momentum, volatility, strength) if category.supportive_signals > 0
        )
        component_count = (
            trend.component_count
            + flow.component_count
            + momentum.component_count
            + volatility.component_count
            + strength.component_count
        )
        return QualitySlice(
            score=weighted_score,
            supportive_signals=supportive_signals,
            component_count=component_count,
            reasons=reasons,
        )

    def _trend_quality_slice(self, snapshot: IndicatorSnapshot, bullish_reversal_context: bool = False) -> QualitySlice:
        components: list[tuple[str, float, bool]] = []
        if snapshot.vwap is not None:
            score = 1.0 if snapshot.close >= snapshot.vwap else 0.25
            components.append(("trend:vwap-aligned", score, score >= BUY_QUALITY_SUPPORT_THRESHOLD))
        if snapshot.ema_fast is not None and snapshot.ema_slow is not None:
            score = 1.0 if snapshot.ema_fast > snapshot.ema_slow else 0.2
            components.append(("trend:ema-aligned", score, score >= BUY_QUALITY_SUPPORT_THRESHOLD))
        if snapshot.sma_fast is not None and snapshot.sma_slow is not None:
            score = 1.0 if snapshot.sma_fast >= snapshot.sma_slow else 0.2
            components.append(("trend:sma-aligned", score, score >= BUY_QUALITY_SUPPORT_THRESHOLD))
        if bullish_reversal_context and not any(score >= BUY_QUALITY_SUPPORT_THRESHOLD for _, score, _ in components):
            components.append(("trend:oversold-reversal", 0.8, True))
        if not components:
            return QualitySlice(score=0.0, supportive_signals=0, component_count=0, reasons=())
        score = sum(component_score for _, component_score, _ in components) / len(components)
        reasons = tuple(reason for reason, _, supportive in components if supportive)
        supportive = sum(1 for _, _, is_supportive in components if is_supportive)
        return QualitySlice(score=score, supportive_signals=supportive, component_count=len(components), reasons=reasons)

    def _flow_quality_slice(self, snapshot: IndicatorSnapshot) -> QualitySlice:
        components: list[tuple[str, float, bool]] = []
        if snapshot.relative_volume is not None:
            if snapshot.relative_volume >= 1.5:
                score = 1.0
            elif snapshot.relative_volume >= 1.1:
                score = 0.85
            elif snapshot.relative_volume >= BUY_QUALITY_RELATIVE_VOLUME_MIN:
                score = 0.7
            elif snapshot.relative_volume >= 0.8:
                score = 0.4
            else:
                score = 0.15
            components.append(("flow:relative-volume-confirmed", score, score >= BUY_QUALITY_SUPPORT_THRESHOLD))
        if snapshot.obv is not None and isfinite(snapshot.obv):
            if snapshot.obv_delta is not None and isfinite(snapshot.obv_delta):
                score = 1.0 if snapshot.obv_delta > 0 else 0.2
            else:
                score = 1.0 if snapshot.obv > 0 else 0.2
            components.append(("flow:obv-positive", score, score >= BUY_QUALITY_SUPPORT_THRESHOLD))
        if snapshot.volume_profile is not None:
            if snapshot.volume_profile >= 0.25:
                score = 1.0
            elif snapshot.volume_profile >= BUY_QUALITY_VOLUME_PROFILE_MIN:
                score = 0.8
            elif snapshot.volume_profile >= 0.1:
                score = 0.45
            else:
                score = 0.2
            components.append(("flow:volume-profile-supportive", score, score >= BUY_QUALITY_SUPPORT_THRESHOLD))
        if not components:
            return QualitySlice(score=0.0, supportive_signals=0, component_count=0, reasons=())
        score = sum(component_score for _, component_score, _ in components) / len(components)
        reasons = tuple(reason for reason, _, supportive in components if supportive)
        supportive = sum(1 for _, _, is_supportive in components if is_supportive)
        return QualitySlice(score=score, supportive_signals=supportive, component_count=len(components), reasons=reasons)

    def _momentum_quality_slice(self, snapshot: IndicatorSnapshot, bullish_reversal_context: bool = False) -> QualitySlice:
        components: list[tuple[str, float, bool]] = []
        if snapshot.rsi is not None:
            if snapshot.rsi <= BUY_RSI_OVERSOLD_THRESHOLD:
                score = 1.0
            elif snapshot.rsi <= BUY_RSI_STRONG_ZONE_MAX:
                score = 1.0
            elif snapshot.rsi <= 45:
                score = 0.75
            elif snapshot.rsi < BUY_RSI_BEARISH_THRESHOLD:
                score = 0.45
            else:
                score = 0.05
            components.append(("momentum:rsi-healthy", score, score >= BUY_QUALITY_SUPPORT_THRESHOLD))
        if snapshot.macd is not None and snapshot.macd_signal is not None and snapshot.macd_histogram is not None:
            if snapshot.macd > snapshot.macd_signal and snapshot.macd_histogram > 0:
                score = 1.0
            elif snapshot.macd > snapshot.macd_signal and snapshot.macd_histogram >= 0:
                score = 0.75
            elif snapshot.macd > snapshot.macd_signal:
                score = 0.45
            elif bullish_reversal_context and snapshot.macd <= snapshot.macd_signal and snapshot.macd_histogram <= 0:
                score = 0.7
            elif snapshot.macd < snapshot.macd_signal and snapshot.macd_histogram < 0:
                score = 0.0
            else:
                score = 0.25
            components.append(("momentum:macd-positive", score, score >= BUY_QUALITY_SUPPORT_THRESHOLD))
        if snapshot.stochastic_k is not None and snapshot.stochastic_d is not None:
            if snapshot.stochastic_k <= BUY_STOCHASTIC_OVERSOLD_THRESHOLD and snapshot.stochastic_k <= snapshot.stochastic_d:
                score = 1.0
            elif snapshot.stochastic_k > snapshot.stochastic_d and snapshot.stochastic_k < BUY_STOCHASTIC_OVERBOUGHT_THRESHOLD:
                score = 1.0
            elif snapshot.stochastic_k > snapshot.stochastic_d:
                score = 0.4
            elif snapshot.stochastic_k < BUY_STOCHASTIC_LOW_THRESHOLD and snapshot.stochastic_k <= snapshot.stochastic_d:
                score = 0.45
            else:
                score = 0.0
            components.append(("momentum:stochastic-rising", score, score >= BUY_QUALITY_SUPPORT_THRESHOLD))
        if not components:
            return QualitySlice(score=0.0, supportive_signals=0, component_count=0, reasons=())
        score = sum(component_score for _, component_score, _ in components) / len(components)
        reasons = tuple(reason for reason, _, supportive in components if supportive)
        supportive = sum(1 for _, _, is_supportive in components if is_supportive)
        return QualitySlice(score=score, supportive_signals=supportive, component_count=len(components), reasons=reasons)

    def _volatility_quality_slice(self, snapshot: IndicatorSnapshot) -> QualitySlice:
        components: list[tuple[str, float, bool]] = []
        if snapshot.atr is not None and snapshot.close > 0 and isfinite(snapshot.atr):
            atr_ratio = snapshot.atr / snapshot.close
            if 0.012 <= atr_ratio <= 0.03:
                score = 1.0
            elif 0.008 <= atr_ratio <= 0.04:
                score = 0.75
            elif atr_ratio < 0.008:
                score = 0.35
            else:
                score = 0.55
            components.append(("volatility:range-expanding", score, score >= BUY_QUALITY_SUPPORT_THRESHOLD))
        if snapshot.bollinger_middle is not None and snapshot.bollinger_upper is not None and snapshot.bollinger_lower is not None:
            if snapshot.close >= snapshot.bollinger_upper:
                score = 1.0
            elif snapshot.close >= snapshot.bollinger_middle:
                score = 0.75
            elif snapshot.close >= snapshot.bollinger_lower:
                score = 0.65
            else:
                score = 0.2
            components.append(("volatility:above-bollinger-mid", score, score >= BUY_QUALITY_SUPPORT_THRESHOLD))
        if not components:
            return QualitySlice(score=0.0, supportive_signals=0, component_count=0, reasons=())
        score = sum(component_score for _, component_score, _ in components) / len(components)
        reasons = tuple(reason for reason, _, supportive in components if supportive)
        supportive = sum(1 for _, _, is_supportive in components if is_supportive)
        return QualitySlice(score=score, supportive_signals=supportive, component_count=len(components), reasons=reasons)

    def _strength_quality_slice(self, snapshot: IndicatorSnapshot) -> QualitySlice:
        components: list[tuple[str, float, bool]] = []
        if snapshot.adx is not None and isfinite(snapshot.adx):
            if snapshot.adx >= 25:
                score = 1.0
            elif snapshot.adx >= 20:
                score = 0.8
            elif snapshot.adx >= 15:
                score = 0.5
            else:
                score = 0.2
            components.append(("strength:adx-supportive", score, score >= BUY_QUALITY_SUPPORT_THRESHOLD))
        if snapshot.plus_di is not None and snapshot.minus_di is not None:
            score = 1.0 if snapshot.plus_di >= snapshot.minus_di else 0.2
            components.append(("strength:directional-pressure", score, score >= BUY_QUALITY_SUPPORT_THRESHOLD))
        if not components:
            return QualitySlice(score=0.0, supportive_signals=0, component_count=0, reasons=())
        score = sum(component_score for _, component_score, _ in components) / len(components)
        reasons = tuple(reason for reason, _, supportive in components if supportive)
        supportive = sum(1 for _, _, is_supportive in components if is_supportive)
        return QualitySlice(score=score, supportive_signals=supportive, component_count=len(components), reasons=reasons)

    def _buy_signal_tier(
        self,
        entry_score: float,
        risk_score: float,
        quality_score: float,
        supportive_signals: int,
        session_risk: float,
        strong_exit_pressure: bool,
    ) -> SignalTier | None:
        pressure_penalty = BUY_TIER_STRONG_EXIT_PENALTY if strong_exit_pressure else 0.0
        high_risk_penalty = BUY_TIER_HIGH_RISK_PENALTY if risk_score >= 0.75 else 0.0
        opening_penalty = self._opening_session_penalty(session_risk)
        tier_quality_penalty = pressure_penalty + opening_penalty
        if (
            entry_score >= BUY_TIER_CONVICTION_ENTRY + opening_penalty
            and risk_score <= BUY_TIER_CONVICTION_MAX_RISK
            and quality_score >= BUY_TIER_CONVICTION_QUALITY + tier_quality_penalty
            and supportive_signals >= BUY_TIER_CONVICTION_MIN_SUPPORTING_SIGNALS
        ):
            return SignalTier.CONVICTION_BUY
        if (
            entry_score >= BUY_TIER_BALANCED_ENTRY + opening_penalty
            and risk_score <= BUY_TIER_BALANCED_MAX_RISK
            and quality_score >= BUY_TIER_BALANCED_QUALITY + tier_quality_penalty
            and supportive_signals >= BUY_TIER_BALANCED_MIN_SUPPORTING_SIGNALS
        ):
            return SignalTier.BALANCED_BUY
        if (
            entry_score >= BUY_TIER_OPPORTUNISTIC_ENTRY + opening_penalty
            and risk_score <= BUY_TIER_OPPORTUNISTIC_MAX_RISK
            and quality_score >= BUY_TIER_OPPORTUNISTIC_QUALITY + tier_quality_penalty + high_risk_penalty
            and supportive_signals >= BUY_TIER_OPPORTUNISTIC_MIN_SUPPORTING_SIGNALS
        ):
            return SignalTier.OPPORTUNISTIC_BUY
        if (
            entry_score >= BUY_TIER_SPECULATIVE_ENTRY + opening_penalty
            and risk_score <= BUY_TIER_SPECULATIVE_MAX_RISK
            and quality_score >= BUY_TIER_SPECULATIVE_QUALITY + tier_quality_penalty + high_risk_penalty
            and supportive_signals >= BUY_TIER_SPECULATIVE_MIN_SUPPORTING_SIGNALS
        ):
            return SignalTier.SPECULATIVE_BUY
        return None
