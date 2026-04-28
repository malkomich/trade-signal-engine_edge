from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Mapping, Optional


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


def default_signal_weights() -> dict[str, float]:
    return {
        "sma": 1.7,
        "ema": 1.7,
        "vwap": 1.0,
        "rsi": 0.8,
        "atr": 0.5,
        "dm": 1.1,
        "macd": 1.4,
        "stochastic": 0.6,
    }


def default_timeframe_weights() -> dict[str, float]:
    return {
        "1m": 1.0,
        "5m": 0.75,
        "15m": 0.5,
    }


@dataclass(frozen=True, slots=True)
class SignalConfig:
    entry_threshold: float = 0.65
    exit_threshold: float = 0.55
    entry_exit_margin: float = 0.05
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
    reasons: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class TradeWindow:
    window_id: str
    symbol: str
    opened_at: datetime
    closed_at: Optional[datetime] = None
    status: TradeState = TradeState.FLAT
