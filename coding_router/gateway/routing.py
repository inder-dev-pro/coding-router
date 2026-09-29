"""Bridge between GatewayRequest and the existing CodingRouter.

This is the ONLY place where the gateway touches the router's decision logic.
The existing router remains the single source of truth for model selection.
"""

from __future__ import annotations

import logging
import os
import time
from typing import Any

from coding_router.catalog import ModelProfile
from coding_router.router import CodingRouter, RouterConfig

from .errors import InvalidRequestError, RoutingError
from .models import GatewayRequest

log = logging.getLogger("coding-router.gateway.routing")

# Virtual model name — configurable via env.
VIRTUAL_MODEL = os.environ.get("CODING_ROUTER_MODEL_NAME", "coding-router")

# Singleton router instance — initialized once, reused for all requests.
_router: CodingRouter | None = None
_router_config: RouterConfig | None = None


def init_router(config: RouterConfig | None = None) -> CodingRouter:
    """Initialize the global router instance (called once at server startup)."""
    global _router, _router_config
    _router_config = config or RouterConfig()
    _router = CodingRouter(_router_config)
    return _router


def get_router() -> CodingRouter:
    if _router is None:
        raise RoutingError("Router not initialized. Call init_router() first.")
    return _router


def get_profiles() -> dict[str, ModelProfile]:
    """Return the catalog keyed by catalog_key."""
    return get_router().profile_by_key


def resolve_model(req: GatewayRequest) -> tuple[ModelProfile, dict[str, Any], float]:
    """Select a model for the given request.

    Returns:
        (profile, router_state, routing_latency_ms)

    If the client requested the virtual model (``coding-router``), the existing
    router decides.  If the client requested a specific catalog key or
    provider model ID, we look it up directly — bypassing the router.
    """
    router = get_router()
    requested = req.requested_model

    # ── Explicit model bypass ────────────────────────────────────────
    if requested != VIRTUAL_MODEL:
        # Try catalog_key first, then provider model ID
        profile = router.profile_by_key.get(requested)
        if profile is None:
            # Search by provider model ID
            for p in router.profiles:
                if p.model == requested:
                    profile = p
                    break
        if profile is None:
            raise InvalidRequestError(f"Model '{requested}' not found in catalog.", param="model")
        return profile, {"selection_reason": "explicit model selection"}, 0.0

    # ── Automatic routing via the existing router ────────────────────
    query = req.user_query_text()
    if not query.strip():
        raise InvalidRequestError("Cannot route: no user message content found.", param="messages")

    t0 = time.perf_counter()
    try:
        state = router.route(query, route_only=True)
    except Exception as exc:
        raise RoutingError(f"Router failed: {exc}") from exc
    latency_ms = (time.perf_counter() - t0) * 1000

    selected = state.get("selected_model", {})
    catalog_key = selected.get("catalog_key", "")
    profile = router.profile_by_key.get(catalog_key)
    if profile is None:
        raise RoutingError(f"Router selected unknown model: {catalog_key}")

    log.info(
        "Routed request=%s model=%s provider=%s category=%s mode=%s latency=%.0fms",
        req.request_id, catalog_key, profile.service,
        state.get("classifier_category", "?"),
        state.get("routing_mode", "?"),
        latency_ms,
    )
    return profile, state, latency_ms
