from __future__ import annotations

from datetime import datetime, timezone
import json
from unittest.mock import Mock, patch

from trade_signal_edge.models import SignalAction, SignalDecision
from trade_signal_edge.publisher import HttpDecisionPublisher


def test_http_decision_publisher_url_encodes_session_id() -> None:
    captured = {}

    def fake_request(url: str, data=None, headers=None, method=None, **kwargs):
        captured["url"] = url
        captured["data"] = data
        captured["headers"] = headers
        captured["method"] = method
        return Mock()

    response = Mock()
    response.__enter__ = Mock(return_value=response)
    response.__exit__ = Mock(return_value=None)
    response.read = Mock(return_value=b'{"window_id":"session:NVDA:decision-1"}')

    decision = SignalDecision(
        symbol="NVDA",
        timestamp=datetime(2026, 4, 22, 13, 30, tzinfo=timezone.utc),
        entry_score=0.82,
        exit_score=0.22,
        action=SignalAction.BUY_ALERT,
        signal_tier=None,
        reasons=("entry-qualified",),
    )

    with patch("trade_signal_edge.publisher.request.Request", side_effect=fake_request), patch(
        "trade_signal_edge.publisher.request.urlopen",
        return_value=response,
    ):
        result = HttpDecisionPublisher("https://api.example.com", "session/with spaces?and#chars").publish(decision)

    assert captured["method"] == "POST"
    assert captured["url"] == "https://api.example.com/v1/sessions/session%2Fwith%20spaces%3Fand%23chars/accept"
    assert b'"signal_tier": null' in captured["data"]
    assert json.loads(captured["data"].decode("utf-8"))["signal_tier"] is None
    assert result == {"window_id": "session:NVDA:decision-1"}
