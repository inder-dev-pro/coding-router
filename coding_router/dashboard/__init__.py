"""Web dashboard for the coding-router.

Provides a local web server that visualises routing events,
model distribution, request volume, cost tracking, and a
live event stream — all from an in-process SQLite database.
"""

from .server import DashboardServer
from .tracker import RoutingTracker, get_tracker

__all__ = ["DashboardServer", "RoutingTracker", "get_tracker"]
