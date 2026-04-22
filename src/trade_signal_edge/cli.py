from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import asdict
from datetime import datetime, timezone
from typing import get_args

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
from .status_server import start_status_server
from .signal_engine import SignalEngine
from .state_machine import StateMachine
from .session_calendar import load_session_calendar


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
                        symbol=report.get("symbol"),
                        provider=report.get("provider"),
                        action=report.get("action"),
                        next_state=report.get("next_state"),
                        last_error=report.get("error"),
                    )
                else:
                    status_server.store.update_run(
                        session_active=False,
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

    symbol = args.symbol or runtime.symbol
    bars = args.bars or runtime.bars
    api_base_url = args.api_base_url or runtime.api_base_url

    try:
        provider_config = load_provider_config(runtime)
        if args.provider is not None:
            provider_config.name = resolve_provider_name(args.provider)
        provider = build_provider(provider_config)
    except (ValueError, NotImplementedError) as error:
        raise ConfigError(str(error)) from error

    history = provider.history(symbol, bars)
    if not history:
        report = {
            "session_active": True,
            "error": "No history found",
            "symbol": symbol,
            "provider": provider_config.name,
            "action": None,
            "next_state": None,
        }
        print(json.dumps(report, indent=2))
        return report

    bars_series = ingest_bars(history)
    indicator_calculator = IndicatorCalculator()
    snapshot = indicator_calculator.compute(bars_series)
    signal_engine = SignalEngine()
    decision = signal_engine.evaluate(snapshot, TradeState.FLAT)
    next_state = StateMachine().transition(TradeState.FLAT, "entry_signal") if decision.action.value == "BUY_ALERT" else TradeState.FLAT

    if api_base_url:
        HttpDecisionPublisher(api_base_url).publish(decision)

    print(
        json.dumps(
            {
                "runtime": {
                    "symbol": symbol,
                    "bars": bars,
                    "api_base_url": api_base_url,
                    "provider": provider_config.name,
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
                "snapshot": asdict(snapshot),
                "decision": asdict(decision),
                "next_state": next_state.value,
            },
            default=str,
            indent=2,
        )
    )
    return {
        "session_active": True,
        "symbol": symbol,
        "provider": provider_config.name,
        "action": decision.action.value,
        "next_state": next_state.value,
        "timestamp": snapshot.timestamp.isoformat(),
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
