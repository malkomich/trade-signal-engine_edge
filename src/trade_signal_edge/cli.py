from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from datetime import datetime, timezone
from typing import get_args

from .config import load_runtime_config
from .indicators import IndicatorCalculator
from .models import TradeState
from .publisher import HttpDecisionPublisher
from .providers import (
    ProviderName,
    build_provider,
    load_provider_config,
    provider_policies,
    selected_provider_policy,
    resolve_provider_name,
)
from .signal_engine import SignalEngine
from .state_machine import StateMachine
from .session_calendar import load_session_calendar


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a local sample signal evaluation.")
    parser.add_argument("--symbol", default=None)
    parser.add_argument("--bars", type=int, default=None)
    parser.add_argument("--provider", choices=get_args(ProviderName), default=None)
    parser.add_argument("--api-base-url", default=None)
    args = parser.parse_args()

    runtime = load_runtime_config()
    calendar = load_session_calendar()
    current_time = datetime.now(tz=timezone.utc)
    if not calendar.is_open(current_time):
        print(
            json.dumps(
                {
                    "session_active": False,
                    "timestamp": current_time.isoformat(),
                    "timezone": calendar.timezone_name,
                },
                indent=2,
            )
        )
        return

    symbol = args.symbol or runtime.symbol
    bars = args.bars or runtime.bars
    api_base_url = args.api_base_url or runtime.api_base_url

    try:
        provider_config = load_provider_config(runtime)
        if args.provider:
            provider_config.name = resolve_provider_name(args.provider)
        provider = build_provider(provider_config)
    except (ValueError, NotImplementedError) as error:
        print(
            json.dumps(
                {
                    "error": {
                        "kind": "config",
                        "message": str(error),
                    }
                },
                indent=2,
            )
        )
        raise SystemExit(1) from error

    bars_series = provider.history(symbol, bars)
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
