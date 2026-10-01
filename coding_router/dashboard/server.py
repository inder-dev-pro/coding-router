"""Lightweight HTTP server for the coding-router dashboard.

Serves the single-page dashboard and a JSON API consumed by the
front-end.  Runs in a daemon thread so it never blocks the main
routing process.
"""

from __future__ import annotations

import json
import threading
import time
from http.server import HTTPServer, BaseHTTPRequestHandler
from pathlib import Path
from typing import Any

from .tracker import RoutingTracker, get_tracker

_STATIC_DIR = Path(__file__).resolve().parent / "static"


class _DashboardHandler(BaseHTTPRequestHandler):
    """Handles GET requests for the SPA and API routes."""

    tracker: RoutingTracker  # set by DashboardServer before serving

    def log_message(self, format: str, *args: Any) -> None:
        """Suppress default stderr logging."""
        pass

    def _json_response(self, data: Any, status: int = 200) -> None:
        body = json.dumps(data).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(body)

    def _file_response(self, path: Path, content_type: str) -> None:
        if not path.exists():
            self.send_error(404)
            return
        body = path.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        path = self.path.split("?")[0]

        # ── API routes ────────────────────────────────────────────
        if path == "/api/summary":
            self._json_response(self.tracker.summary_stats())
        elif path == "/api/models":
            self._json_response(self.tracker.model_distribution())
        elif path == "/api/providers":
            self._json_response(self.tracker.provider_distribution())
        elif path == "/api/categories":
            self._json_response(self.tracker.category_distribution())
        elif path == "/api/volume":
            self._json_response(self.tracker.request_volume())
        elif path == "/api/events":
            self._json_response(self.tracker.recent_events())
        elif path == "/api/cost":
            self._json_response(self.tracker.cost_over_time())
        elif path == "/api/modes":
            self._json_response(self.tracker.mode_distribution())
        elif path == "/api/tiers":
            self._json_response(self.tracker.tier_distribution())
        elif path.startswith("/api/event/"):
            try:
                event_id = int(path.split("/")[-1])
                detail = self.tracker.event_detail(event_id)
                if detail:
                    self._json_response(detail)
                else:
                    self._json_response({"error": "not found"}, 404)
            except ValueError:
                self._json_response({"error": "invalid id"}, 400)

        # ── Static files ──────────────────────────────────────────
        elif path == "/" or path == "/index.html":
            self._file_response(_STATIC_DIR / "index.html", "text/html; charset=utf-8")
        elif path == "/style.css":
            self._file_response(_STATIC_DIR / "style.css", "text/css; charset=utf-8")
        elif path == "/app.js":
            self._file_response(_STATIC_DIR / "app.js", "application/javascript; charset=utf-8")
        else:
            self.send_error(404)


class DashboardServer:
    """Manages the dashboard HTTP server lifecycle."""

    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 3000,
        tracker: RoutingTracker | None = None,
    ) -> None:
        self.host = host
        self.port = port
        self.tracker = tracker or get_tracker()
        self._server: HTTPServer | None = None
        self._thread: threading.Thread | None = None

    @property
    def url(self) -> str:
        return f"http://{self.host}:{self.port}"

    def start(self) -> str:
        """Start the dashboard server in a background daemon thread.

        Returns the URL the dashboard is accessible at.
        """
        if self._server is not None:
            return self.url

        _DashboardHandler.tracker = self.tracker
        self._server = HTTPServer((self.host, self.port), _DashboardHandler)
        self._thread = threading.Thread(
            target=self._server.serve_forever,
            name="coding-router-dashboard",
            daemon=True,
        )
        self._thread.start()
        return self.url

    def stop(self) -> None:
        if self._server:
            self._server.shutdown()
            self._server = None
            self._thread = None
