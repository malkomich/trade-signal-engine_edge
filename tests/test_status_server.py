from __future__ import annotations

import json
import time
from urllib.error import HTTPError
from urllib.request import urlopen

from trade_signal_edge.status_server import start_status_server


def test_status_server_exposes_json_and_html() -> None:
    server = start_status_server(0)
    try:
        port = server.server.server_address[1]
        try:
            with urlopen(f"http://127.0.0.1:{port}/readyz", timeout=2) as response:
                assert response.status == 503
        except HTTPError as error:
            assert error.code == 503

        server.store.update_run(
            session_active=True,
            symbol="AAPL",
            provider="synthetic",
            action="BUY_ALERT",
            next_state="ENTRY_SIGNALLED",
        )

        for _ in range(20):
            try:
                with urlopen(f"http://127.0.0.1:{port}/status?format=json", timeout=2) as response:
                    payload = json.loads(response.read().decode("utf-8"))
                break
            except OSError:
                time.sleep(0.05)
        else:
            raise AssertionError("status server did not become ready")

        assert payload["symbol"] == "AAPL"
        assert payload["action"] == "BUY_ALERT"
        assert payload["last_error"] is None

        with urlopen(f"http://127.0.0.1:{port}/readyz", timeout=2) as response:
            assert response.status == 200
            ready_payload = json.loads(response.read().decode("utf-8"))
        assert ready_payload["status"] == "ready"

        with urlopen(f"http://127.0.0.1:{port}/edge?view=full", timeout=2) as response:
            body = response.read().decode("utf-8")
        assert "Trade Signal Engine Edge" in body
        assert "BUY_ALERT" in body
    finally:
        server.close()
