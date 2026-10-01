"""Event tracker — records every routing decision to a local SQLite database.

The tracker is designed to be used as a singleton via ``get_tracker()``.
It is thread-safe and writes lazily so the routing hot-path is never blocked.
"""

from __future__ import annotations

import json
import sqlite3
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

_DB_DIR = Path.home() / ".cache" / "coding-router"
_DB_PATH = _DB_DIR / "dashboard.db"

_SCHEMA = """\
CREATE TABLE IF NOT EXISTS routing_events (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp   REAL    NOT NULL,
    query       TEXT    NOT NULL,
    query_preview TEXT  NOT NULL,
    category    TEXT    NOT NULL,
    classifier_model TEXT,
    routing_mode TEXT   NOT NULL,
    alpha       REAL,
    beta        REAL,
    selection_group TEXT NOT NULL,
    selection_reason TEXT,
    selected_key TEXT   NOT NULL,
    selected_model TEXT NOT NULL,
    selected_service TEXT NOT NULL,
    selected_tier TEXT  NOT NULL,
    similarity  REAL,
    reward      REAL,
    estimated_cost_usd REAL,
    input_price REAL,
    output_price REAL,
    estimated_input_tokens  INTEGER,
    estimated_output_tokens INTEGER,
    latency_ms  REAL,
    had_error   INTEGER NOT NULL DEFAULT 0,
    error_text  TEXT,
    ranked_models_json TEXT
);

CREATE INDEX IF NOT EXISTS idx_events_timestamp ON routing_events(timestamp);
CREATE INDEX IF NOT EXISTS idx_events_selected_key ON routing_events(selected_key);
CREATE INDEX IF NOT EXISTS idx_events_category ON routing_events(category);
CREATE INDEX IF NOT EXISTS idx_events_service ON routing_events(selected_service);
"""

# Singleton lock
_tracker_lock = threading.Lock()
_tracker_instance: RoutingTracker | None = None


class RoutingTracker:
    """Thread-safe event recorder backed by SQLite."""

    def __init__(self, db_path: Path | None = None) -> None:
        self._db_path = db_path or _DB_PATH
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._local = threading.local()
        # Ensure schema exists
        conn = self._conn()
        conn.executescript(_SCHEMA)
        conn.commit()

    def _conn(self) -> sqlite3.Connection:
        if not hasattr(self._local, "conn") or self._local.conn is None:
            self._local.conn = sqlite3.connect(
                str(self._db_path),
                check_same_thread=False,
            )
            self._local.conn.execute("PRAGMA journal_mode=WAL")
            self._local.conn.execute("PRAGMA synchronous=NORMAL")
        return self._local.conn

    def record(
        self,
        state: dict[str, Any],
        latency_ms: float,
    ) -> None:
        """Record a completed routing event."""
        selected = state.get("selected_model", {})
        query = state.get("user_query", "")
        preview = query[:120] + ("…" if len(query) > 120 else "")

        ranked_json = None
        if "ranked_models" in state:
            ranked_json = json.dumps(state["ranked_models"])

        conn = self._conn()
        conn.execute(
            """\
            INSERT INTO routing_events (
                timestamp, query, query_preview, category,
                classifier_model, routing_mode, alpha, beta,
                selection_group, selection_reason,
                selected_key, selected_model, selected_service, selected_tier,
                similarity, reward, estimated_cost_usd,
                input_price, output_price,
                estimated_input_tokens, estimated_output_tokens,
                latency_ms, had_error, error_text, ranked_models_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                time.time(),
                query,
                preview,
                state.get("classifier_category", "unknown"),
                state.get("classifier_model"),
                state.get("routing_mode", "mixed"),
                selected.get("alpha"),
                selected.get("beta"),
                state.get("selection_group", ""),
                state.get("selection_reason"),
                selected.get("catalog_key", "unknown"),
                selected.get("model", "unknown"),
                selected.get("service", "unknown"),
                selected.get("tier", "standard"),
                selected.get("similarity"),
                selected.get("reward"),
                selected.get("estimated_request_cost_usd"),
                selected.get("input_price_per_million"),
                selected.get("output_price_per_million"),
                state.get("estimated_input_tokens"),
                state.get("estimated_output_tokens"),
                latency_ms,
                1 if state.get("invocation_error") else 0,
                state.get("invocation_error"),
                ranked_json,
            ),
        )
        conn.commit()

    # ------------------------------------------------------------------
    # Query helpers used by the dashboard API
    # ------------------------------------------------------------------

    def _query(self, sql: str, params: tuple = ()) -> list[dict]:
        conn = self._conn()
        conn.row_factory = sqlite3.Row
        rows = conn.execute(sql, params).fetchall()
        conn.row_factory = None
        return [dict(row) for row in rows]

    def summary_stats(self) -> dict[str, Any]:
        """Aggregate numbers for the dashboard header cards."""
        rows = self._query(
            """\
            SELECT
                COUNT(*)                                    AS total_requests,
                COUNT(DISTINCT selected_service)            AS providers,
                COUNT(DISTINCT selected_key)                AS models_active,
                COALESCE(SUM(estimated_cost_usd), 0)        AS total_spend,
                COALESCE(AVG(latency_ms), 0)                AS avg_latency,
                SUM(CASE WHEN had_error = 1 THEN 1 ELSE 0 END) AS error_count,
                COUNT(CASE WHEN timestamp > ? THEN 1 END)  AS requests_today
            FROM routing_events
            """,
            (time.time() - 86400,),
        )
        return rows[0] if rows else {}

    def model_distribution(self) -> list[dict]:
        """Count of selections per model key."""
        return self._query(
            """\
            SELECT selected_key AS model, selected_service AS service, COUNT(*) AS count
            FROM routing_events
            GROUP BY selected_key
            ORDER BY count DESC
            """
        )

    def provider_distribution(self) -> list[dict]:
        """Count of selections per provider."""
        return self._query(
            """\
            SELECT selected_service AS provider, COUNT(*) AS count
            FROM routing_events
            GROUP BY selected_service
            ORDER BY count DESC
            """
        )

    def category_distribution(self) -> list[dict]:
        """Count of requests per classifier category."""
        return self._query(
            """\
            SELECT category, COUNT(*) AS count
            FROM routing_events
            GROUP BY category
            ORDER BY count DESC
            """
        )

    def request_volume(self, hours: int = 24) -> list[dict]:
        """Bucketed request counts for the volume chart."""
        cutoff = time.time() - hours * 3600
        return self._query(
            """\
            SELECT
                CAST((timestamp - ?) / 3600 AS INTEGER) AS hour_bucket,
                COUNT(*) AS count,
                MIN(timestamp) AS bucket_start
            FROM routing_events
            WHERE timestamp > ?
            GROUP BY hour_bucket
            ORDER BY hour_bucket
            """,
            (cutoff, cutoff),
        )

    def recent_events(self, limit: int = 50) -> list[dict]:
        """Most recent events for the live feed."""
        return self._query(
            """\
            SELECT
                id, timestamp, query_preview, category,
                routing_mode, selected_key, selected_service,
                selected_tier, similarity, reward,
                estimated_cost_usd, latency_ms, had_error, error_text
            FROM routing_events
            ORDER BY timestamp DESC
            LIMIT ?
            """,
            (limit,),
        )

    def cost_over_time(self, hours: int = 24) -> list[dict]:
        """Cumulative cost over the past N hours."""
        cutoff = time.time() - hours * 3600
        return self._query(
            """\
            SELECT timestamp, estimated_cost_usd,
                   SUM(estimated_cost_usd) OVER (ORDER BY timestamp) AS cumulative_cost
            FROM routing_events
            WHERE timestamp > ?
            ORDER BY timestamp
            """,
            (cutoff,),
        )

    def mode_distribution(self) -> list[dict]:
        """Count of requests per routing mode."""
        return self._query(
            """\
            SELECT routing_mode AS mode, COUNT(*) AS count
            FROM routing_events
            GROUP BY routing_mode
            ORDER BY count DESC
            """
        )

    def tier_distribution(self) -> list[dict]:
        """Count of advanced vs standard selections."""
        return self._query(
            """\
            SELECT selected_tier AS tier, COUNT(*) AS count
            FROM routing_events
            GROUP BY selected_tier
            ORDER BY count DESC
            """
        )

    def event_detail(self, event_id: int) -> dict | None:
        """Full detail for a single event."""
        rows = self._query(
            "SELECT * FROM routing_events WHERE id = ?", (event_id,)
        )
        return rows[0] if rows else None


def get_tracker(db_path: Path | None = None) -> RoutingTracker:
    """Return the global singleton tracker instance."""
    global _tracker_instance
    with _tracker_lock:
        if _tracker_instance is None:
            _tracker_instance = RoutingTracker(db_path)
        return _tracker_instance
