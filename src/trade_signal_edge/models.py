from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Mapping, Optional


DEFAULT_ENTRY_GATE_CAP = 0.52


class TradeState(str, Enum):
    FLAT = "FLAT"
    ENTRY_SIGNALLED = "ENTRY_SIGNALLED"
    ACCEPTED_OPEN = "ACCEPTED_OPEN"
    REJECTED = "REJECTED"
    EXPIRED = "EXPIRED"
    EXIT_SIGNALLED = "EXIT_SIGNALLED"
    CLOSED = "CLOSED"


class SignalAction(str, Enum):
    HOLD = "HOLD"
    BUY_ALERT = "BUY_ALERT"
    SELL_ALERT = "SELL_ALERT"


class SignalTier(str, Enum):
    CONVICTION_BUY = "conviction_buy"
    BALANCED_BUY = "balanced_buy"
    OPPORTUNISTIC_BUY = "opportunistic_buy"
    SPECULATIVE_BUY = "speculative_buy"


@dataclass(frozen=True, slots=True)
class Bar:
    symbol: str
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float


@dataclass(frozen=True, slots=True)
class IndicatorSnapshot:
    symbol: str
    timestamp: datetime
    close: float
    sma_fast: Optional[float] = None
    sma_slow: Optional[float] = None
    ema_fast: Optional[float] = None
    ema_slow: Optional[float] = None
    vwap: Optional[float] = None
    rsi: Optional[float] = None
    atr: Optional[float] = None
    plus_di: Optional[float] = None
    minus_di: Optional[float] = None
    adx: Optional[float] = None
    macd: Optional[float] = None
    macd_signal: Optional[float] = None
    macd_histogram: Optional[float] = None
    stochastic_k: Optional[float] = None
    stochastic_d: Optional[float] = None
    bollinger_middle: Optional[float] = None
    bollinger_upper: Optional[float] = None
    bollinger_lower: Optional[float] = None
    obv: Optional[float] = None
    obv_delta: Optional[float] = None
    relative_volume: Optional[float] = None
    volume_profile: Optional[float] = None


def default_signal_weights() -> dict[str, float]:
    return {
        "sma": 0.6,
        "ema": 1.0,
        "vwap": 1.2,
        "bollinger": 0.7,
        "rsi": 0.9,
        "atr": 0.5,
        "dm": 0.6,
        "macd": 1.0,
        "stochastic": 0.5,
        "obv": 0.9,
        "relative_volume": 1.2,
        "volume_profile": 0.8,
    }


def default_timeframe_weights() -> dict[str, float]:
    return dict(
        zip(
            TIMEFRAME_KEYS,
            (1.0, 0.85, 0.75, 0.6, 0.45, 0.3),
            strict=True,
        )
    )


@dataclass(frozen=True, slots=True)
class SignalConfig:
    entry_threshold: float = 0.7
    exit_threshold: float = 0.6
    entry_exit_margin: float = 0.1
    entry_gate_cap: float = DEFAULT_ENTRY_GATE_CAP
    buy_weights: Mapping[str, float] = field(default_factory=default_signal_weights)
    sell_weights: Mapping[str, float] = field(default_factory=default_signal_weights)
    buy_timeframe_weights: Mapping[str, float] = field(default_factory=default_timeframe_weights)
    sell_timeframe_weights: Mapping[str, float] = field(default_factory=default_timeframe_weights)
    entry_profile: Mapping[str, float] = field(default_factory=dict)
    exit_profile: Mapping[str, float] = field(default_factory=dict)
    optimizer_learning_rate: float = 0.12
    optimizer_bias_cap: float = 0.08


@dataclass(frozen=True, slots=True)
class SignalDecision:
    symbol: str
    timestamp: datetime
    entry_score: float
    exit_score: float
    action: SignalAction
    signal_tier: Optional[SignalTier] = None
    reasons: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class TradeWindow:
    window_id: str
    symbol: str
    opened_at: datetime
    closed_at: Optional[datetime] = None
    status: TradeState = TradeState.FLAT
TIMEFRAME_KEYS = ("1m", "5m", "10m", "15m", "30m", "60m")
