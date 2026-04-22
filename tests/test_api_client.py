from __future__ import annotations

from trade_signal_edge.api_client import _clean_symbol


def test_clean_symbol_handles_null_and_non_string_values() -> None:
    assert _clean_symbol(None) == ""
    assert _clean_symbol(123) == ""
    assert _clean_symbol(" nvda ") == "NVDA"
