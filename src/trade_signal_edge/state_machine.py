from __future__ import annotations

from dataclasses import dataclass

from .models import TradeState


@dataclass(slots=True)
class StateMachine:
    def transition(self, current: TradeState, event: str) -> TradeState:
        transitions: dict[TradeState, dict[str, TradeState]] = {
            TradeState.FLAT: {
                "entry_signal": TradeState.ENTRY_SIGNALLED,
            },
            TradeState.ENTRY_SIGNALLED: {
                "accept_entry": TradeState.ACCEPTED_OPEN,
                "reject_entry": TradeState.REJECTED,
                "expire": TradeState.EXPIRED,
            },
            TradeState.ACCEPTED_OPEN: {
                "exit_signal": TradeState.EXIT_SIGNALLED,
                "close": TradeState.CLOSED,
            },
            TradeState.EXIT_SIGNALLED: {
                "accept_exit": TradeState.CLOSED,
                "hold_open": TradeState.ACCEPTED_OPEN,
            },
        }

        allowed = transitions.get(current, {})
        if event not in allowed:
            raise ValueError(f"invalid transition from {current.value} via {event}")
        return allowed[event]

