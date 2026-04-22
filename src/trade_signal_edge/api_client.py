from __future__ import annotations

from dataclasses import asdict, is_dataclass
from dataclasses import dataclass
import json
from json import loads
from urllib import error, parse, request


@dataclass(slots=True)
class ApiSessionClient:
    base_url: str
    timeout_seconds: int = 10

    def load_open_symbols(self, session_id: str) -> set[str]:
        if not self.base_url:
            return set()

        encoded_session_id = parse.quote(session_id, safe="")
        req = request.Request(f"{self.base_url.rstrip('/')}/v1/sessions/{encoded_session_id}/windows", method="GET")
        try:
            with request.urlopen(req, timeout=self.timeout_seconds) as response:
                payload = loads(response.read().decode("utf-8"))
        except error.HTTPError as exc:
            raise RuntimeError(f"failed to load open windows: {exc.code}") from exc
        except error.URLError as exc:
            raise RuntimeError("failed to load open windows") from exc

        if not isinstance(payload, list):
            return set()

        open_symbols: set[str] = set()
        for item in payload:
            if not isinstance(item, dict):
                continue
            status = item.get("status")
            if not isinstance(status, str):
                continue
            normalized_status = status.strip().lower()
            if normalized_status not in {"open", "accepted_open"}:
                continue
            symbol = _clean_symbol(item.get("symbol"))
            if symbol:
                open_symbols.add(symbol)
        return open_symbols

    def publish_market_snapshot(self, session_id: str, payload: object) -> None:
        if not self.base_url:
            return

        encoded_session_id = parse.quote(session_id, safe="")
        body = json_dumps(payload).encode("utf-8")
        req = request.Request(
            f"{self.base_url.rstrip('/')}/v1/sessions/{encoded_session_id}/market-snapshots",
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with request.urlopen(req, timeout=self.timeout_seconds) as response:
                response.read()
        except error.HTTPError as exc:
            raise RuntimeError(f"failed to save market snapshot: {exc.code}") from exc
        except error.URLError as exc:
            raise RuntimeError("failed to save market snapshot") from exc

def _clean_symbol(value: object) -> str:
    if value is None:
        return ""
    if not isinstance(value, str):
        return ""
    return value.strip().upper()


def json_dumps(payload: object) -> str:
    if is_dataclass(payload):
        payload = asdict(payload)
    return json.dumps(payload, default=str)
