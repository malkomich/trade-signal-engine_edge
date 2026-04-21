import pytest

from trade_signal_edge.models import TradeState
from trade_signal_edge.state_machine import StateMachine


def test_state_machine_advances_through_valid_transitions() -> None:
    machine = StateMachine()

    assert machine.transition(TradeState.FLAT, "entry_signal") is TradeState.ENTRY_SIGNALLED
    assert machine.transition(TradeState.ENTRY_SIGNALLED, "accept_entry") is TradeState.ACCEPTED_OPEN
    assert machine.transition(TradeState.ACCEPTED_OPEN, "exit_signal") is TradeState.EXIT_SIGNALLED
    assert machine.transition(TradeState.EXIT_SIGNALLED, "accept_exit") is TradeState.CLOSED


def test_state_machine_rejects_invalid_transition() -> None:
    machine = StateMachine()

    with pytest.raises(ValueError):
        machine.transition(TradeState.FLAT, "accept_exit")

