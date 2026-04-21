from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from datetime import datetime, timezone

from .config import load_runtime_config
from .indicators import IndicatorCalculator
from .models import TradeState
from .publisher import HttpDecisionPublisher
from .providers import build_provider, load_provider_selection, resolve_provider_name
from .signal_engine import SignalEngine
from .state_machine import StateMachine
from .session_calendar import load_session_calendar


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a local sample signal evaluation.")
    parser.add_argument("--symbol", default="AAPL")
    parser.add_argument("--bars", type=int, default=60)
    parser.add_argument("--provider", choices=["synthetic", "alpaca"], default=None)
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

    provider_selection = load_provider_selection()
    provider_selection.name = resolve_provider_name(args.provider or runtime.provider)
    provider = build_provider(provider_selection)

    symbol = args.symbol or runtime.symbol
    bars = args.bars or runtime.bars
    bars_series = provider.history(symbol, bars)
    indicator_calculator = IndicatorCalculator()
    snapshot = indicator_calculator.compute(bars_series)
    signal_engine = SignalEngine()
    decision = signal_engine.evaluate(snapshot, TradeState.FLAT)
    next_state = StateMachine().transition(TradeState.FLAT, "entry_signal") if decision.action.value == "BUY_ALERT" else TradeState.FLAT

    api_base_url = args.api_base_url or runtime.api_base_url
    if api_base_url:
        HttpDecisionPublisher(api_base_url).publish(decision)

    print(
        json.dumps(
            {
                "snapshot": asdict(snapshot),
                "decision": asdict(decision),
                "next_state": next_state.value,
            },
            default=str,
            indent=2,
        )
    )
