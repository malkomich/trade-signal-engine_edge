from __future__ import annotations

from dataclasses import dataclass
from urllib.error import HTTPError, URLError
from unittest.mock import Mock, patch

from trade_signal_edge.api_client import _clean_symbol
from trade_signal_edge.api_client import ApiSessionClient
from trade_signal_edge.api_client import json_dumps


def test_load_open_symbols_accepts_accepted_open_status() -> None:
    payload = b'[{"status":"ACCEPTED_OPEN","symbol":" nvda "},{"status":"open","symbol":"aapl"}]'
    response = Mock()
    response.__enter__ = Mock(return_value=response)
    response.__exit__ = Mock(return_value=None)
    response.read = Mock(return_value=payload)

    with patch("trade_signal_edge.api_client.request.urlopen", return_value=response):
        symbols = ApiSessionClient("https://api.example.com").load_open_symbols("session-1")

    assert symbols == {"NVDA", "AAPL"}


def test_load_open_symbols_wraps_http_error() -> None:
    http_error = HTTPError("https://api.example.com", 500, "boom", hdrs=None, fp=None)

    with patch("trade_signal_edge.api_client.request.urlopen", side_effect=http_error):
        client = ApiSessionClient("https://api.example.com")

        try:
            client.load_open_symbols("session-1")
        except RuntimeError as error:
            assert "failed to load open windows: 500" in str(error)
        else:
            raise AssertionError("RuntimeError was not raised")


def test_load_open_symbols_wraps_url_error() -> None:
    with patch("trade_signal_edge.api_client.request.urlopen", side_effect=URLError("offline")):
        client = ApiSessionClient("https://api.example.com")

        try:
            client.load_open_symbols("session-1")
        except RuntimeError as error:
            assert str(error) == "failed to load open windows"
        else:
            raise AssertionError("RuntimeError was not raised")


def test_load_open_symbols_rejects_non_list_payload() -> None:
    response = Mock()
    response.__enter__ = Mock(return_value=response)
    response.__exit__ = Mock(return_value=None)
    response.read = Mock(return_value=b'{"not":"a list"}')

    with patch("trade_signal_edge.api_client.request.urlopen", return_value=response):
        symbols = ApiSessionClient("https://api.example.com").load_open_symbols("session-1")

    assert symbols == set()


def test_load_open_symbols_returns_empty_set_without_base_url() -> None:
    assert ApiSessionClient("").load_open_symbols("session-1") == set()


def test_load_open_symbols_url_encodes_session_id() -> None:
    captured = {}

    def fake_request(url: str, method: str = "GET", **kwargs):
        captured["url"] = url
        captured["method"] = method
        return Mock()

    response = Mock()
    response.__enter__ = Mock(return_value=response)
    response.__exit__ = Mock(return_value=None)
    response.read = Mock(return_value=b"[]")

    with patch("trade_signal_edge.api_client.request.Request", side_effect=fake_request), patch(
        "trade_signal_edge.api_client.request.urlopen",
        return_value=response,
    ):
        ApiSessionClient("https://api.example.com").load_open_symbols("session/with spaces?and#chars")

    assert captured["method"] == "GET"
    assert captured["url"] == "https://api.example.com/v1/sessions/session%2Fwith%20spaces%3Fand%23chars/windows"


def test_clean_symbol_handles_null_and_non_string_values() -> None:
    assert _clean_symbol(None) == ""
    assert _clean_symbol(123) == ""
    assert _clean_symbol(" nvda ") == "NVDA"


def test_json_dumps_handles_dataclass_types_without_unpacking_the_class() -> None:
    @dataclass
    class Dummy:
        value: str = "ok"

    assert json_dumps(Dummy).startswith('"')
