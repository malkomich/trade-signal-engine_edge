from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import asdict, replace
from datetime import datetime, timezone
from typing import Any, get_args

from .api_client import ApiSessionClient
from .config import load_runtime_config
from .indicators import IndicatorCalculator
from .ingestion import ingest_bars, resample_bars
from .models import SignalAction, SignalConfig, SignalDecision, TradeState
from .publisher import HttpDecisionPublisher
from .providers import (
    ProviderName,
    build_provider,
    load_provider_config,
    provider_policies,
    resolve_provider_name,
    selected_provider_policy,
)
from .session_calendar import load_session_calendar
from .signal_engine import SignalEngine
from .state_machine import StateMachine
from .status_server import start_status_server


class ConfigError(RuntimeError):
    """Raised when the current runtime configuration cannot be used."""


TIMEFRAME_KEYS = ("1m", "5m", "15m")
SIGNAL_WEIGHT_KEYS = ("sma", "ema", "vwap", "rsi", "atr", "dm", "macd", "stochastic")


def _parse_int_env(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None:
        return default
    candidate = raw.strip()
    if not candidate:
        return default
    try:
        return int(candidate)
    except ValueError:
        return default


def _parse_float(value: Any, fallback: float) -> float:
    if value is None:
        return fallback
    if isinstance(value, bool):
        return fallback
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        candidate = value.strip()
        if not candidate:
            return fallback
        try:
            return float(candidate)
        except ValueError:
            return fallback
    return fallback


def _parse_symbols(value: Any, fallback: tuple[str, ...]) -> tuple[str, ...]:
    if isinstance(value, list):
        items = [str(item).strip().upper() for item in value]
    elif isinstance(value, str):
        items = [item.strip().upper() for item in value.split(",")]
    else:
        return fallback
    symbols = tuple(item for item in items if item)
    if not symbols:
        return fallback
    return tuple(dict.fromkeys(symbols))


def _config_fields_map(payload: object) -> dict[str, Any]:
    if not isinstance(payload, dict):
        return {}
    selected = payload.get("selected_version")
    if not isinstance(selected, dict):
        return {}
    fields = selected.get("fields")
    if not isinstance(fields, list):
        return {}
    field_map: dict[str, Any] = {}
    for item in fields:
        if not isinstance(item, dict):
            continue
        key = str(item.get("key") or "").strip()
        if not key:
            continue
        field_map[key] = item.get("value")
    return field_map


def _runtime_from_session_config(runtime, payload: object):
    field_map = _config_fields_map(payload)
    if not field_map:
        return runtime

    symbols = _parse_symbols(field_map.get("monitored_symbols"), runtime.symbols)
    benchmark_symbol = str(field_map.get("benchmark_symbol") or runtime.benchmark_symbol).strip().upper() or runtime.benchmark_symbol
    session_timezone = str(field_map.get("session_timezone") or runtime.session_timezone).strip() or runtime.session_timezone
    entry_threshold = _parse_float(
        field_map.get("buy_score_threshold", field_map.get("entry_score_threshold")),
        runtime.entry_threshold,
    )
    exit_threshold = _parse_float(
        field_map.get("sell_score_threshold", field_map.get("exit_score_threshold")),
        runtime.exit_threshold,
    )

    buy_signal_weights = dict(runtime.buy_signal_weights)
    sell_signal_weights = dict(runtime.sell_signal_weights)
    for key in SIGNAL_WEIGHT_KEYS:
        shared_weight_key = f"weight_{key}"
        buy_signal_weights[key] = _parse_float(
            field_map.get(f"buy_{shared_weight_key}", field_map.get(shared_weight_key)),
            buy_signal_weights.get(key, 0.0),
        )
        sell_signal_weights[key] = _parse_float(
            field_map.get(f"sell_{shared_weight_key}", field_map.get(shared_weight_key)),
            sell_signal_weights.get(key, 0.0),
        )

    buy_timeframe_weights = dict(runtime.buy_timeframe_weights)
    sell_timeframe_weights = dict(runtime.sell_timeframe_weights)
    for key in TIMEFRAME_KEYS:
        shared_weight_key = f"weight_{key}"
        buy_timeframe_weights[key] = _parse_float(
            field_map.get(f"buy_{shared_weight_key}", field_map.get(shared_weight_key)),
            buy_timeframe_weights.get(key, 0.0),
        )
        sell_timeframe_weights[key] = _parse_float(
            field_map.get(f"sell_{shared_weight_key}", field_map.get(shared_weight_key)),
            sell_timeframe_weights.get(key, 0.0),
        )

    return replace(
        runtime,
        symbol=symbols[0] if symbols else runtime.symbol,
        symbols=symbols,
        benchmark_symbol=benchmark_symbol,
        session_timezone=session_timezone,
        entry_threshold=entry_threshold,
        exit_threshold=exit_threshold,
        buy_signal_weights=buy_signal_weights,
        sell_signal_weights=sell_signal_weights,
        buy_timeframe_weights=buy_timeframe_weights,
        sell_timeframe_weights=sell_timeframe_weights,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a local signal evaluation.")
    parser.add_argument("--symbol", default=None)
    parser.add_argument("--bars", type=int, default=None)
    parser.add_argument("--provider", choices=get_args(ProviderName), default=None)
    parser.add_argument("--api-base-url", default=None)
    parser.add_argument("--watch", action="store_true", help="Keep the worker running and poll on an interval.")
    parser.add_argument("--interval-seconds", type=int, default=60, help="Polling interval used with --watch.")
    parser.add_argument("--http-port", type=int, default=_parse_int_env("EDGE_HTTP_PORT", 8081))
    args = parser.parse_args()

    if args.watch:
        _run_watch_loop(args)
        return

    try:
        runtime = load_runtime_config()
        _run_once(args, runtime)
    except (ConfigError, OSError, ValueError) as error:
        print(_json_error("config", error))
        raise SystemExit(1) from error
    except Exception as error:
        print(_json_error("runtime", error))
        raise SystemExit(1) from error


def _run_watch_loop(args: argparse.Namespace) -> None:
    interval = max(args.interval_seconds, 1)
    backoff = interval
    status_server = None
    try:
        try:
            status_server = start_status_server(max(args.http_port, 1))
        except Exception as error:
            print(_json_error("runtime", error))
            raise SystemExit(1) from error
        while True:
            try:
                runtime = load_runtime_config()
                report = _run_once(args, runtime)
                if report.get("session_active", True):
                    status_server.store.update_run(
                        session_active=True,
                        symbols=report.get("symbols") if isinstance(report.get("symbols"), list) else None,
                        symbol=report.get("symbol"),
                        provider=report.get("provider"),
                        action=report.get("action"),
                        next_state=report.get("next_state"),
                        last_error=report.get("error"),
                        decision_count=report.get("decision_count") if isinstance(report.get("decision_count"), int) else None,
                    )
                else:
                    status_server.store.update_run(
                        session_active=False,
                        symbols=report.get("symbols") if isinstance(report.get("symbols"), list) else None,
                        last_error=report.get("error"),
                        clear_details=False,
                    )
                backoff = interval
            except (OSError, ValueError, ConfigError) as error:
                print(_json_error("config", error))
                if status_server is not None:
                    status_server.store.update_run(session_active=False, last_error=str(error), clear_details=False)
                time.sleep(backoff)
                backoff = min(backoff * 2, 300)
                continue
            except Exception as error:
                print(_json_error("runtime", error))
                if status_server is not None:
                    status_server.store.update_run(session_active=False, last_error=str(error), clear_details=False)
                time.sleep(backoff)
                backoff = min(backoff * 2, 300)
                continue
            time.sleep(interval)
    finally:
        if status_server is not None:
            status_server.close()


def _run_once(args: argparse.Namespace, runtime) -> dict[str, object]:
    calendar = load_session_calendar()
    current_time = datetime.now(tz=timezone.utc)
    market_open = calendar.is_open(current_time)

    try:
        provider_config = load_provider_config(runtime)
        if args.provider is not None:
            provider_config.name = resolve_provider_name(args.provider)
        provider = build_provider(provider_config)
    except (ValueError, NotImplementedError) as error:
        raise ConfigError(str(error)) from error

    api_base_url = args.api_base_url or runtime.api_base_url
    session_client = ApiSessionClient(api_base_url) if api_base_url else None
    errors: list[str] = []
    session_config = None
    if session_client is not None:
        try:
            session_config = session_client.load_session_config(runtime.session_id)
        except Exception as error:
            errors.append(f"session config load failed: {error}")
    runtime = _runtime_from_session_config(runtime, session_config)

    symbol = (args.symbol or runtime.symbol).strip().upper()
    bars = args.bars or runtime.bars
    symbols = list(runtime.symbols)
    if args.symbol is not None:
        symbols = [symbol]

    indicator_calculator = IndicatorCalculator()
    signal_engine = SignalEngine(
        config=SignalConfig(
            entry_threshold=runtime.entry_threshold,
            exit_threshold=runtime.exit_threshold,
            buy_weights=runtime.buy_signal_weights,
            sell_weights=runtime.sell_signal_weights,
            buy_timeframe_weights=runtime.buy_timeframe_weights,
            sell_timeframe_weights=runtime.sell_timeframe_weights,
        )
    )
    open_symbols = set()
    if session_client is not None:
        try:
            open_symbols = session_client.load_open_symbols(runtime.session_id)
        except Exception as error:
            errors.append(f"open symbols load failed: {error}")
    benchmark_payload = _load_benchmark_snapshot(provider, indicator_calculator, runtime.benchmark_symbol, bars)
    if benchmark_payload is None:
        report = {
            "session_active": market_open,
            "error": f"benchmark {runtime.benchmark_symbol}: no history found",
            "symbol": symbols[0] if symbols else symbol,
            "symbols": symbols,
            "provider": provider_config.name,
            "action": None,
            "next_state": None,
            "decision_count": 0,
        }
        print(json.dumps(report, indent=2))
        return report
    benchmark_snapshot, benchmark_series = benchmark_payload
    benchmark_series_by_timeframe = {
        "1m": benchmark_series,
        "5m": resample_bars(benchmark_series, 5),
        "15m": resample_bars(benchmark_series, 15),
    }
    benchmark_snapshots_by_timeframe = {
        timeframe: indicator_calculator.compute(series)
        for timeframe, series in benchmark_series_by_timeframe.items()
        if series
    }
    publisher = HttpDecisionPublisher(api_base_url, runtime.session_id) if api_base_url else None

    decisions: list[dict[str, object]] = []
    try:
        benchmark_market_payload = _build_market_snapshot_payload(
            session_id=runtime.session_id,
            symbol=runtime.benchmark_symbol,
            bars_series=benchmark_series,
            snapshot=benchmark_snapshot,
            entry_score=0.0,
            exit_score=0.0,
            decision_action=None,
            next_state="BENCHMARK",
            benchmark_symbol=runtime.benchmark_symbol,
            regime="benchmark snapshot",
        )
        if session_client is not None:
            session_client.publish_market_snapshot(runtime.session_id, benchmark_market_payload)
    except Exception as error:
        errors.append(f"{runtime.benchmark_symbol}: market snapshot publish failed: {error}")
    for current_symbol in symbols:
        try:
            history = provider.history(current_symbol, bars)
        except Exception as error:
            errors.append(f"{current_symbol}: history load failed: {error}")
            continue
        if not history:
            errors.append(f"{current_symbol}: no history found")
            continue

        bars_series = ingest_bars(history)
        timeframe_series = {
            "1m": bars_series,
            "5m": resample_bars(bars_series, 5),
            "15m": resample_bars(bars_series, 15),
        }
        snapshots_by_timeframe = {
            timeframe: indicator_calculator.compute(series)
            for timeframe, series in timeframe_series.items()
            if series
        }
        snapshot = snapshots_by_timeframe.get("1m")
        if snapshot is None:
            errors.append(f"{current_symbol}: missing 1m snapshot")
            continue
        state = TradeState.ACCEPTED_OPEN if current_symbol in open_symbols else TradeState.FLAT
        if not market_open and state is TradeState.FLAT:
            continue
        timeframe_decisions: dict[str, SignalDecision] = {}
        for timeframe, timeframe_snapshot in snapshots_by_timeframe.items():
            benchmark_for_timeframe = benchmark_snapshots_by_timeframe.get(timeframe, benchmark_snapshot)
            timeframe_decisions[timeframe] = signal_engine.evaluate(timeframe_snapshot, state, benchmark_for_timeframe)

        decision = _combine_timeframe_decisions(
            current_symbol,
            snapshots_by_timeframe,
            timeframe_decisions,
            runtime.buy_timeframe_weights,
            runtime.sell_timeframe_weights,
            signal_engine,
            state,
        )
        if not market_open:
            if state is TradeState.ACCEPTED_OPEN:
                if decision.action.value != "SELL_ALERT":
                    decision = replace(
                        decision,
                        action=SignalAction.SELL_ALERT,
                        reasons=tuple(dict.fromkeys((*decision.reasons, "session-close exit"))),
                    )
        next_state = state
        if decision.action.value == "BUY_ALERT":
            next_state = StateMachine().transition(TradeState.FLAT, "entry_signal")
        elif decision.action.value == "SELL_ALERT" and state is TradeState.ACCEPTED_OPEN:
            next_state = StateMachine().transition(TradeState.ACCEPTED_OPEN, "exit_signal")
        if publisher is not None:
            try:
                publisher.publish(decision)
            except Exception as error:
                errors.append(f"{current_symbol}: decision publish failed: {error}")
        if session_client is not None:
            try:
                session_client.publish_market_snapshot(
                    runtime.session_id,
                    _build_market_snapshot_payload(
                        session_id=runtime.session_id,
                        symbol=current_symbol,
                        bars_series=bars_series,
                        snapshot=snapshot,
                        entry_score=decision.entry_score,
                        exit_score=decision.exit_score,
                        decision_action=decision.action.value,
                        next_state=next_state.value,
                        benchmark_symbol=runtime.benchmark_symbol,
                        regime=decision.reasons[-1] if decision.reasons else "live market session",
                    ),
                )
            except Exception as error:
                errors.append(f"{current_symbol}: market snapshot publish failed: {error}")
        decisions.append(
            {
                "symbol": current_symbol,
                "snapshot": asdict(snapshot),
                "decision": asdict(decision),
                "next_state": next_state.value,
                "timeframes": {timeframe: asdict(item) for timeframe, item in timeframe_decisions.items()},
            }
        )

    if errors and not decisions:
        report = {
            "session_active": market_open,
            "error": "; ".join(errors),
            "errors": errors,
            "symbol": symbol,
            "provider": provider_config.name,
            "action": None,
            "next_state": None,
        }
        print(json.dumps(report, indent=2))
        return report

    report = {
        "runtime": {
            "symbols": symbols,
            "benchmark_symbol": runtime.benchmark_symbol,
            "bars": bars,
            "api_base_url": api_base_url,
            "provider": provider_config.name,
            "session_id": runtime.session_id,
            "entry_threshold": runtime.entry_threshold,
            "exit_threshold": runtime.exit_threshold,
            "buy_signal_weights": runtime.buy_signal_weights,
            "sell_signal_weights": runtime.sell_signal_weights,
            "buy_timeframe_weights": runtime.buy_timeframe_weights,
            "sell_timeframe_weights": runtime.sell_timeframe_weights,
            "market_open": market_open,
        },
        "observability": {
            "log_level": runtime.log_level,
            "metrics_enabled": runtime.metrics_enabled,
            "secret_source": runtime.secret_source,
            "deployment_profile": runtime.deployment_profile,
        },
        "provider_policy": {
            "selected": asdict(selected_provider_policy(provider_config)),
            "matrix": [asdict(policy) for policy in provider_policies()],
        },
        "benchmark": asdict(benchmark_snapshot) if benchmark_snapshot is not None else None,
        "signals": decisions,
        "errors": errors,
    }
    print(json.dumps(report, default=str, indent=2))
    latest = decisions[-1] if decisions else None
    return {
        "session_active": market_open,
        "symbols": symbols,
        "decision_count": len(decisions),
        "symbol": symbols[0] if symbols else symbol,
        "provider": provider_config.name,
        "action": latest["decision"]["action"] if latest else None,
        "next_state": latest["next_state"] if latest else None,
        "latest_symbol": latest["symbol"] if latest else None,
        "errors": errors,
        "error": "; ".join(errors) if errors else None,
        "timestamp": _iso_timestamp(latest["snapshot"]["timestamp"]) if latest else datetime.now(tz=timezone.utc).isoformat(),
    }


def _json_error(kind: str, error: Exception) -> str:
    return json.dumps(
        {
            "error": {
                "kind": kind,
                "message": str(error),
            }
        },
        indent=2,
    )


def _load_benchmark_snapshot(provider, indicator_calculator: IndicatorCalculator, benchmark_symbol: str, bars: int):
    history = provider.history(benchmark_symbol, bars)
    if not history:
        return None
    bars_series = ingest_bars(history)
    return indicator_calculator.compute(bars_series), bars_series


def _combine_timeframe_decisions(
    symbol: str,
    snapshots_by_timeframe: dict[str, Any],
    timeframe_decisions: dict[str, SignalDecision],
    buy_timeframe_weights: dict[str, float],
    sell_timeframe_weights: dict[str, float],
    signal_engine: SignalEngine,
    state: TradeState,
) -> SignalDecision:
    entry_total = 0.0
    exit_total = 0.0
    buy_weight_sum = 0.0
    sell_weight_sum = 0.0
    reasons: list[str] = []
    primary_snapshot = snapshots_by_timeframe.get("1m")

    for timeframe in TIMEFRAME_KEYS:
        decision = timeframe_decisions.get(timeframe)
        if decision is None:
            continue
        buy_weight = buy_timeframe_weights.get(timeframe, 0.0)
        sell_weight = sell_timeframe_weights.get(timeframe, 0.0)
        if buy_weight <= 0 and sell_weight <= 0:
            continue
        buy_weight_sum += buy_weight
        sell_weight_sum += sell_weight
        entry_total += buy_weight * decision.entry_score
        exit_total += sell_weight * decision.exit_score
        if decision.reasons:
            reasons.append(f"{timeframe}:{'; '.join(decision.reasons)}")

    entry_score = entry_total / buy_weight_sum if buy_weight_sum > 0 else 0.0
    exit_score = exit_total / sell_weight_sum if sell_weight_sum > 0 else 0.0

    strong_exit_pressure = any(
        signal_engine.is_strong_exit_pressure(snapshot)
        for snapshot in snapshots_by_timeframe.values()
        if snapshot is not None
    )
    action, action_reasons = signal_engine.decide_action(
        entry_score,
        exit_score,
        state,
        primary_snapshot,
        strong_exit_pressure=strong_exit_pressure,
    )
    reasons.extend(action_reasons)

    deduped_reasons = list(dict.fromkeys(reason for reason in reasons if reason))

    timestamp = primary_snapshot.timestamp if primary_snapshot is not None else datetime.now(tz=timezone.utc)
    return SignalDecision(
        symbol=symbol,
        timestamp=timestamp,
        entry_score=round(entry_score, 4),
        exit_score=round(exit_score, 4),
        action=action,
        reasons=tuple(deduped_reasons),
    )


def _build_market_snapshot_payload(
    session_id: str,
    symbol: str,
    bars_series,
    snapshot,
    entry_score: float,
    exit_score: float,
    decision_action: str | None,
    next_state: str,
    benchmark_symbol: str,
    regime: str,
) -> dict[str, object]:
    latest_bar = bars_series[-1]
    payload = {
        "session_id": session_id,
        "symbol": symbol,
        "timestamp": snapshot.timestamp.isoformat(),
        "created_at": snapshot.timestamp.isoformat(),
        "updated_at": snapshot.timestamp.isoformat(),
        "open": latest_bar.open,
        "high": latest_bar.high,
        "low": latest_bar.low,
        "close": latest_bar.close,
        "volume": latest_bar.volume,
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
        "entry_score": entry_score,
        "exit_score": exit_score,
        "event_type": "market.snapshot",
        "signal_action": decision_action or "HOLD",
        "signal_state": next_state,
        "signal_regime": regime,
        "benchmark_symbol": benchmark_symbol,
        "reasons": [],
    }
    return payload


def _iso_timestamp(value: object) -> str:
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, str):
        return value
    return datetime.now(tz=timezone.utc).isoformat()
