from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol
from urllib import error, request
import json

from .models import SignalDecision
from .schema import EVENT_TYPE_DECISION_CREATED


class DecisionPublisher(Protocol):
    def publish(self, decision: SignalDecision) -> None:
        raise NotImplementedError


@dataclass(slots=True)
class HttpDecisionPublisher:
    base_url: str
    timeout_seconds: int = 10

    def publish(self, decision: SignalDecision) -> None:
        payload = {
            "event_type": EVENT_TYPE_DECISION_CREATED,
            "session_id": "local-session",
            "symbol": decision.symbol,
            "action": decision.action.value,
            "reason": "; ".join(decision.reasons) if decision.reasons else "signal-evaluated",
            "entry_score": decision.entry_score,
            "exit_score": decision.exit_score,
            "requested_by": "edge",
        }
        body = json.dumps(payload).encode("utf-8")
        req = request.Request(
            f"{self.base_url.rstrip('/')}/v1/decisions",
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with request.urlopen(req, timeout=self.timeout_seconds) as response:
                response.read()
        except error.HTTPError as exc:
            raise RuntimeError(f"decision publish failed with status {exc.code}") from exc
        except error.URLError as exc:
            raise RuntimeError("decision publish failed") from exc
