from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol
from urllib import error, parse, request
import json

from .models import SignalAction, SignalDecision


class DecisionPublisher(Protocol):
    def publish(self, decision: SignalDecision) -> dict[str, object] | None:
        raise NotImplementedError


@dataclass(slots=True)
class HttpDecisionPublisher:
    base_url: str
    session_id: str
    timeout_seconds: int = 10

    def publish(self, decision: SignalDecision) -> dict[str, object] | None:
        action = self._resolve_action(decision.action)
        if action is None:
            return None
        payload = {
            "symbol": decision.symbol,
            "reason": "; ".join(decision.reasons) if decision.reasons else "signal-evaluated",
            "entry_score": decision.entry_score,
            "exit_score": decision.exit_score,
            "requested_by": "edge",
        }
        body = json.dumps(payload).encode("utf-8")
        encoded_session_id = parse.quote(self.session_id, safe="")
        req = request.Request(
            f"{self.base_url.rstrip('/')}/v1/sessions/{encoded_session_id}/{action}",
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with request.urlopen(req, timeout=self.timeout_seconds) as response:
                raw = response.read()
        except error.HTTPError as exc:
            raise RuntimeError(f"decision publish failed with status {exc.code}") from exc
        except error.URLError as exc:
            raise RuntimeError("decision publish failed") from exc
        if not raw:
            return None
        try:
            payload = json.loads(raw.decode("utf-8"))
        except json.JSONDecodeError:
            return None
        return payload if isinstance(payload, dict) else None

    def _resolve_action(self, action: SignalAction) -> str | None:
        if action is SignalAction.BUY_ALERT:
            return "accept"
        if action is SignalAction.SELL_ALERT:
            return "exit"
        return None
