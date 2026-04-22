from __future__ import annotations

from unittest.mock import Mock, patch

from trade_signal_edge.api_client import _clean_symbol
from trade_signal_edge.api_client import ApiSessionClient


def test_load_open_symbols_accepts_accepted_open_status() -> None:
    payload = b'[{"status":"ACCEPTED_OPEN","symbol":" nvda "},{"status":"open","symbol":"aapl"}]'
    response = Mock()
    response.__enter__ = Mock(return_value=response)
    response.__exit__ = Mock(return_value=None)
    response.read = Mock(return_value=payload)

    with patch("trade_signal_edge.api_client.request.urlopen", return_value=response):
        symbols = ApiSessionClient("https://api.example.com").load_open_symbols("session-1")

    assert symbols == {"NVDA", "AAPL"}


def test_clean_symbol_handles_null_and_non_string_values() -> None:
    assert _clean_symbol(None) == ""
    assert _clean_symbol(123) == ""
    assert _clean_symbol(" nvda ") == "NVDA"
