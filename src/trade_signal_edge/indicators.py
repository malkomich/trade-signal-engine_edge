from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np
import pandas as pd

from .models import Bar, IndicatorSnapshot


def _last_value(series: pd.Series) -> float | None:
    cleaned = series.dropna()
    if cleaned.empty:
        return None
    return float(cleaned.iloc[-1])


def _rma(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(alpha=1 / period, adjust=False).mean()


@dataclass(slots=True)
class IndicatorCalculator:
    fast_sma: int = 20
    slow_sma: int = 50
    fast_ema: int = 12
    slow_ema: int = 26
    rsi_period: int = 14
    atr_period: int = 14
    dmi_period: int = 14
    stochastic_period: int = 14
    stochastic_signal_period: int = 3
    macd_signal_period: int = 9
    bollinger_period: int = 20
    bollinger_stddev: float = 2.0
    volume_profile_period: int = 20
    volume_profile_bins: int = 8

    def compute(self, bars: Sequence[Bar]) -> IndicatorSnapshot:
        if not bars:
            raise ValueError("bars cannot be empty")

        frame = pd.DataFrame(
            [
                {
                    "symbol": bar.symbol,
                    "timestamp": bar.timestamp,
                    "open": bar.open,
                    "high": bar.high,
                    "low": bar.low,
                    "close": bar.close,
                    "volume": bar.volume,
                }
                for bar in bars
            ]
        )
        frame = frame.sort_values("timestamp").reset_index(drop=True)

        close = frame["close"]
        high = frame["high"]
        low = frame["low"]
        volume = frame["volume"]

        sma_fast = close.rolling(self.fast_sma).mean()
        sma_slow = close.rolling(self.slow_sma).mean()
        ema_fast = close.ewm(span=self.fast_ema, adjust=False).mean()
        ema_slow = close.ewm(span=self.slow_ema, adjust=False).mean()

        bollinger_middle = close.rolling(self.bollinger_period).mean()
        bollinger_std = close.rolling(self.bollinger_period).std(ddof=1)
        bollinger_upper = bollinger_middle + (self.bollinger_stddev * bollinger_std)
        bollinger_lower = bollinger_middle - (self.bollinger_stddev * bollinger_std)

        typical_price = (high + low + close) / 3.0
        cumulative_vwap = (typical_price * volume).cumsum() / volume.cumsum()

        obv = pd.Series(0.0, index=frame.index)
        close_delta = close.diff().fillna(0)
        obv_direction = np.sign(close_delta)
        obv = (obv_direction * volume).cumsum()
        obv_delta = obv.diff()

        volume_average = volume.rolling(self.bollinger_period).mean()
        relative_volume = volume / volume_average.replace(0, np.nan)
        relative_volume = relative_volume.replace([np.inf, -np.inf], np.nan)

        volume_profile = pd.Series(np.nan, index=frame.index)
        profile_window = self.volume_profile_period
        if profile_window <= 0:
            profile_window = 1
        window = frame.iloc[max(0, len(frame) - profile_window) : len(frame)]
        if not window.empty:
            price_low = float(window["low"].min())
            price_high = float(window["high"].max())
            if np.isfinite(price_low) and np.isfinite(price_high) and price_high > price_low:
                histogram, bin_edges = np.histogram(
                    window["close"].astype(float),
                    bins=max(2, self.volume_profile_bins),
                    range=(price_low, price_high),
                    weights=window["volume"].astype(float),
                )
                total_volume = float(histogram.sum())
                if total_volume > 0:
                    last_close = float(window["close"].iloc[-1])
                    bin_index = np.searchsorted(bin_edges, last_close, side="right") - 1
                    bin_index = int(np.clip(bin_index, 0, len(histogram) - 1))
                    volume_profile.iloc[-1] = histogram[bin_index] / total_volume

        delta = close.diff()
        gain = delta.clip(lower=0)
        loss = (-delta).clip(lower=0)
        avg_gain = _rma(gain, self.rsi_period)
        avg_loss = _rma(loss, self.rsi_period)
        rs = avg_gain / avg_loss.replace(0, np.nan)
        rsi = 100 - (100 / (1 + rs))
        rsi = rsi.fillna(100.0)

        prev_close = close.shift(1)
        tr_components = pd.concat(
            [
                high - low,
                (high - prev_close).abs(),
                (low - prev_close).abs(),
            ],
            axis=1,
        )
        true_range = tr_components.max(axis=1)
        atr = _rma(true_range, self.atr_period)

        up_move = high.diff()
        down_move = -low.diff()
        plus_dm = pd.Series(0.0, index=frame.index)
        minus_dm = pd.Series(0.0, index=frame.index)
        plus_dm[(up_move > down_move) & (up_move > 0)] = up_move[(up_move > down_move) & (up_move > 0)]
        minus_dm[(down_move > up_move) & (down_move > 0)] = down_move[(down_move > up_move) & (down_move > 0)]

        smoothed_plus_dm = _rma(plus_dm, self.dmi_period)
        smoothed_minus_dm = _rma(minus_dm, self.dmi_period)
        plus_di = 100 * (smoothed_plus_dm / atr.replace(0, np.nan))
        minus_di = 100 * (smoothed_minus_dm / atr.replace(0, np.nan))
        dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
        adx = _rma(dx.fillna(0), self.dmi_period)

        macd_line = ema_fast - ema_slow
        macd_signal = macd_line.ewm(span=self.macd_signal_period, adjust=False).mean()
        macd_histogram = macd_line - macd_signal

        lowest_low = low.rolling(self.stochastic_period).min()
        highest_high = high.rolling(self.stochastic_period).max()
        stochastic_k = 100 * (close - lowest_low) / (highest_high - lowest_low).replace(0, np.nan)
        stochastic_d = stochastic_k.rolling(self.stochastic_signal_period).mean()

        last = frame.iloc[-1]
        return IndicatorSnapshot(
            symbol=str(last["symbol"]),
            timestamp=last["timestamp"].to_pydatetime() if hasattr(last["timestamp"], "to_pydatetime") else last["timestamp"],
            close=float(last["close"]),
            sma_fast=_last_value(sma_fast),
            sma_slow=_last_value(sma_slow),
            ema_fast=_last_value(ema_fast),
            ema_slow=_last_value(ema_slow),
            vwap=_last_value(cumulative_vwap),
            rsi=_last_value(rsi),
            atr=_last_value(atr),
            plus_di=_last_value(plus_di),
            minus_di=_last_value(minus_di),
            adx=_last_value(adx),
            macd=_last_value(macd_line),
            macd_signal=_last_value(macd_signal),
            macd_histogram=_last_value(macd_histogram),
            stochastic_k=_last_value(stochastic_k),
            stochastic_d=_last_value(stochastic_d),
            bollinger_middle=_last_value(bollinger_middle),
            bollinger_upper=_last_value(bollinger_upper),
            bollinger_lower=_last_value(bollinger_lower),
            obv=_last_value(obv),
            obv_delta=_last_value(obv_delta),
            relative_volume=_last_value(relative_volume),
            volume_profile=_last_value(volume_profile),
        )
