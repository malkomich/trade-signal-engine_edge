from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from html import escape
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from threading import Lock, Thread
from typing import Any
import json
from urllib.parse import urlsplit


@dataclass(slots=True)
class WorkerStatus:
    started_at: datetime = field(default_factory=lambda: datetime.now(tz=timezone.utc))
    last_run_at: datetime | None = None
    session_active: bool | None = None
    symbol: str | None = None
    provider: str | None = None
    action: str | None = None
    next_state: str | None = None
    last_error: str | None = None


@dataclass(slots=True)
class WorkerStatusStore:
    _status: WorkerStatus = field(default_factory=WorkerStatus)
    _lock: Lock = field(default_factory=Lock, repr=False)

    def update_run(
        self,
        *,
        session_active: bool,
        symbol: str | None = None,
        provider: str | None = None,
        action: str | None = None,
        next_state: str | None = None,
        last_error: str | None = None,
        clear_details: bool = True,
    ) -> None:
        with self._lock:
            self._status.session_active = session_active
            if clear_details:
                self._status.symbol = symbol
                self._status.provider = provider
                self._status.action = action
                self._status.next_state = next_state
            else:
                if symbol is not None:
                    self._status.symbol = symbol
                if provider is not None:
                    self._status.provider = provider
                if action is not None:
                    self._status.action = action
                if next_state is not None:
                    self._status.next_state = next_state
            self._status.last_error = _sanitize_error(last_error)
            self._status.last_run_at = datetime.now(tz=timezone.utc)

    def is_ready(self) -> bool:
        with self._lock:
            return self._status.last_run_at is not None

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            payload = asdict(self._status)
        payload = _public_snapshot(payload)
        return payload


def _public_snapshot(payload: dict[str, Any]) -> dict[str, Any]:
    payload["started_at"] = payload["started_at"].isoformat()
    if payload["last_run_at"] is not None:
        payload["last_run_at"] = payload["last_run_at"].isoformat()
    return payload


def _sanitize_error(value: Any) -> str | None:
    if value is None:
        return None
    message = " ".join(str(value).split())
    if not message:
        return None
    if len(message) <= 240:
        return message
    return f"{message[:237]}..."


@dataclass(slots=True)
class EdgeStatusServer:
    store: WorkerStatusStore
    server: ThreadingHTTPServer
    thread: Thread

    def close(self) -> None:
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=2)


def start_status_server(port: int) -> EdgeStatusServer:
    store = WorkerStatusStore()

    class Handler(BaseHTTPRequestHandler):
        def _send_json(self, status: int, payload: dict[str, Any]) -> None:
            encoded = json.dumps(payload, indent=2, default=str).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(encoded)))
            self.end_headers()
            self.wfile.write(encoded)

        def _send_html(self, status: int, payload: dict[str, Any]) -> None:
            body = _render_html(payload).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler API
            path = urlsplit(self.path).path
            if path in {"/", "/edge", "/healthz", "/readyz", "/status"}:
                snapshot = store.snapshot()
                if path == "/status":
                    self._send_json(200, snapshot)
                elif path == "/healthz":
                    self._send_json(200, {"status": "ok", "started_at": snapshot["started_at"]})
                elif path == "/readyz":
                    ready = store.is_ready()
                    self._send_json(
                        200 if ready else 503,
                        {"status": "ready" if ready else "starting", "started_at": snapshot["started_at"]},
                    )
                else:
                    self._send_html(200, snapshot)
                return
            self._send_json(404, {"error": "route not found"})

        def log_message(self, *_args: Any) -> None:  # noqa: D401 - keep the status server quiet
            return

    server = ThreadingHTTPServer(("0.0.0.0", port), Handler)
    server.daemon_threads = True
    thread = Thread(target=server.serve_forever, name="trade-signal-edge-status", daemon=True)
    thread.start()
    return EdgeStatusServer(store=store, server=server, thread=thread)


def _render_html(snapshot: dict[str, Any]) -> str:
    def render_value(value: Any) -> str:
        if value is None:
            return "n/a"
        if isinstance(value, bool):
            return "yes" if value else "no"
        return escape(str(value))

    rows = "\n".join(
        f"<tr><th>{escape(key.replace('_', ' '))}</th><td>{render_value(value)}</td></tr>"
        for key, value in snapshot.items()
    )
    return f"""<!doctype html>
<html lang="en">
  <head>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <title>Trade Signal Engine Edge</title>
    <style>
      body {{
        font-family: ui-sans-serif, system-ui, sans-serif;
        background: #0b1220;
        color: #e5eefc;
        margin: 0;
        padding: 32px;
      }}
      main {{
        max-width: 880px;
        margin: 0 auto;
      }}
      h1 {{
        margin: 0 0 12px;
        font-size: 2rem;
      }}
      p {{
        color: #aac0e6;
      }}
      table {{
        width: 100%;
        border-collapse: collapse;
        margin-top: 24px;
        background: rgba(255, 255, 255, 0.04);
        border-radius: 12px;
        overflow: hidden;
      }}
      th, td {{
        text-align: left;
        padding: 12px 16px;
        border-bottom: 1px solid rgba(255, 255, 255, 0.08);
      }}
      th {{
        width: 220px;
        text-transform: capitalize;
        color: #9db5dc;
      }}
      tr:last-child th,
      tr:last-child td {{
        border-bottom: none;
      }}
      code {{
        background: rgba(255, 255, 255, 0.08);
        padding: 2px 6px;
        border-radius: 6px;
      }}
    </style>
  </head>
  <body>
    <main>
      <h1>Trade Signal Engine Edge</h1>
      <p>The worker is running and the latest runtime snapshot is available below.</p>
      <p>Use <code>/status</code> for JSON and <code>/healthz</code> for a health probe.</p>
      <table>
        <tbody>
          {rows}
        </tbody>
      </table>
    </main>
  </body>
</html>"""
