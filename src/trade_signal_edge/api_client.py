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
        except error.HTTPError:
            return set()
        except error.URLError as exc:
            raise RuntimeError("failed to load open windows") from exc

        if not isinstance(payload, list):
            return set()

        open_symbols: set[str] = set()
        for item in payload:
            if not isinstance(item, dict):
                continue
            if str(item.get("status", "")).lower() != "open":
                continue
            symbol = str(item.get("symbol", "")).strip().upper()
            if symbol:
                open_symbols.add(symbol)
        return open_symbols
