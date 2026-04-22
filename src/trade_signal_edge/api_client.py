from __future__ import annotations

from dataclasses import dataclass
from json import loads
from urllib import error, request


@dataclass(slots=True)
class ApiSessionClient:
    base_url: str
    timeout_seconds: int = 10

    def load_open_symbols(self, session_id: str) -> set[str]:
        if not self.base_url:
            return set()

        req = request.Request(f"{self.base_url.rstrip('/')}/v1/sessions/{session_id}/windows", method="GET")
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
            if not isinstance(status, str) or status.strip().lower() != "open":
                continue
            symbol = _clean_symbol(item.get("symbol"))
            if symbol:
                open_symbols.add(symbol)
        return open_symbols


def _clean_symbol(value: object) -> str:
    if value is None:
        return ""
    if not isinstance(value, str):
        return ""
    return value.strip().upper()
