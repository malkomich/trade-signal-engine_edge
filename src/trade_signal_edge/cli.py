from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import asdict
from datetime import datetime, timezone
from typing import get_args

from .api_client import ApiSessionClient
from .config import load_runtime_config
from .indicators import IndicatorCalculator
from .ingestion import ingest_bars
from .models import TradeState
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


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a local sample signal evaluation.")
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
    if not calendar.is_open(current_time):
        report = {
            "session_active": False,
            "timestamp": current_time.isoformat(),
            "timezone": calendar.timezone_name,
        }
        print(json.dumps(report, indent=2))
        return report

    symbol = (args.symbol or runtime.symbol).strip().upper()
    bars = args.bars or runtime.bars
    api_base_url = args.api_base_url or runtime.api_base_url

    try:
        provider_config = load_provider_config(runtime)
        if args.provider is not None:
            provider_config.name = resolve_provider_name(args.provider)
        provider = build_provider(provider_config)
    except (ValueError, NotImplementedError) as error:
        raise ConfigError(str(error)) from error

    symbols = list(runtime.symbols)
    if args.symbol is not None:
        symbols = [symbol]

    indicator_calculator = IndicatorCalculator()
    signal_engine = SignalEngine()
    session_client = ApiSessionClient(api_base_url) if api_base_url else None
    open_symbols = session_client.load_open_symbols(runtime.session_id) if session_client else set()
    benchmark_snapshot = _load_benchmark_snapshot(provider, indicator_calculator, runtime.benchmark_symbol, bars)
    publisher = HttpDecisionPublisher(api_base_url, runtime.session_id) if api_base_url else None

    decisions: list[dict[str, object]] = []
    errors: list[str] = []
    for current_symbol in symbols:
        history = provider.history(current_symbol, bars)
        if not history:
            errors.append(f"{current_symbol}: no history found")
            continue

        bars_series = ingest_bars(history)
        snapshot = indicator_calculator.compute(bars_series)
        state = TradeState.ACCEPTED_OPEN if current_symbol in open_symbols else TradeState.FLAT
        decision = signal_engine.evaluate(snapshot, state, benchmark_snapshot)
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
        decisions.append(
            {
                "symbol": current_symbol,
                "snapshot": asdict(snapshot),
                "decision": asdict(decision),
                "next_state": next_state.value,
            }
        )

    if errors and not decisions:
        report = {
            "session_active": True,
            "error": "; ".join(errors),
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
        "session_active": True,
        "symbols": [decision["symbol"] for decision in decisions],
        "decision_count": len(decisions),
        "symbol": latest["symbol"] if latest else symbol,
        "provider": provider_config.name,
        "action": latest["decision"]["action"] if latest else None,
        "next_state": latest["next_state"] if latest else None,
        "timestamp": latest["snapshot"]["timestamp"] if latest else datetime.now(tz=timezone.utc).isoformat(),
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
    return indicator_calculator.compute(bars_series)
