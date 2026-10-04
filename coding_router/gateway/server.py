"""FastAPI gateway server.

Brings together all protocol adapters, authentication, error handling,
and the router lifecycle.
"""

from __future__ import annotations

import logging
import os
import time
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from .auth import load_or_create_key, validate_bearer
from .errors import AuthenticationError, GatewayError, gateway_error_handler, generic_error_handler
import traceback
from .providers.base import close_client
from .routing import VIRTUAL_MODEL, get_profiles, init_router

from .protocols.openai_chat import router as openai_chat_router
from .protocols.openai_responses import router as openai_responses_router
from .protocols.anthropic_messages import router as anthropic_messages_router

log = logging.getLogger("coding-router.gateway")

# ---------------------------------------------------------------------------
# App state
# ---------------------------------------------------------------------------

_gateway_key: str | None = None
_auth_enabled: bool = True


# ---------------------------------------------------------------------------
# Lifespan — initialize router once, clean up on shutdown
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    global _gateway_key, _auth_enabled

    # Authentication setup
    _auth_enabled = os.environ.get("CODING_ROUTER_AUTH", "true").lower() not in ("false", "0", "no")
    if _auth_enabled:
        _gateway_key = load_or_create_key()

    # Initialize the router (loads embeddings, classifier, catalog — once)
    log.info("Initializing coding-router...")
    
    from .routing import RouterConfig
    from pathlib import Path
    
    kwargs = {}
    if "CODING_ROUTER_MODE" in os.environ:
        kwargs["routing_mode"] = os.environ["CODING_ROUTER_MODE"]
    if "CODING_ROUTER_ALPHA" in os.environ:
        kwargs["alpha"] = float(os.environ["CODING_ROUTER_ALPHA"])
    if "CODING_ROUTER_BETA" in os.environ:
        kwargs["beta"] = float(os.environ["CODING_ROUTER_BETA"])
    if "CODING_ROUTER_QUALITY_GAP" in os.environ:
        kwargs["quality_gap_threshold"] = float(os.environ["CODING_ROUTER_QUALITY_GAP"])
    if "CODING_ROUTER_CATALOG" in os.environ:
        kwargs["catalog_path"] = Path(os.environ["CODING_ROUTER_CATALOG"])
    if "CODING_ROUTER_USER_CATALOG" in os.environ:
        kwargs["user_catalog_path"] = Path(os.environ["CODING_ROUTER_USER_CATALOG"])
    if "CODING_ROUTER_POOL" in os.environ:
        kwargs["candidate_pool"] = os.environ["CODING_ROUTER_POOL"]
    if "CODING_ROUTER_INDEX" in os.environ:
        kwargs["index_path"] = Path(os.environ["CODING_ROUTER_INDEX"])
    if "CODING_ROUTER_EMBEDDING_MODEL" in os.environ:
        kwargs["embedding_model"] = os.environ["CODING_ROUTER_EMBEDDING_MODEL"]
    if "CODING_ROUTER_CLASSIFIER_BACKEND" in os.environ:
        kwargs["classifier_backend"] = os.environ["CODING_ROUTER_CLASSIFIER_BACKEND"]
    if "CODING_ROUTER_CLASSIFIER_MODEL_PATH" in os.environ:
        kwargs["classifier_model_path"] = os.environ["CODING_ROUTER_CLASSIFIER_MODEL_PATH"]
    if "CODING_ROUTER_CLASSIFIER_BASE_URL" in os.environ:
        kwargs["classifier_base_url"] = os.environ["CODING_ROUTER_CLASSIFIER_BASE_URL"]
    if "CODING_ROUTER_CLASSIFIER_MODEL" in os.environ:
        kwargs["classifier_model"] = os.environ["CODING_ROUTER_CLASSIFIER_MODEL"]
    if "CODING_ROUTER_MAX_OUTPUT_TOKENS" in os.environ:
        kwargs["target_max_tokens"] = int(os.environ["CODING_ROUTER_MAX_OUTPUT_TOKENS"])

    config = RouterConfig(**kwargs) # type: ignore
    init_router(config)
    log.info("Router ready. Catalog: %d models.", len(get_profiles()))

    yield

    # Shutdown
    await close_client()
    log.info("Gateway shutdown complete.")


# ---------------------------------------------------------------------------
# App creation
# ---------------------------------------------------------------------------

def create_app() -> FastAPI:
    app = FastAPI(
        title="coding-router gateway",
        description="Local OpenAI/Anthropic-compatible LLM gateway powered by coding-router.",
        version="0.1.0",
        lifespan=lifespan,
        docs_url="/docs",
        redoc_url=None,
    )

    # Exception handlers
    app.add_exception_handler(GatewayError, gateway_error_handler)  # type: ignore[arg-type]
    app.add_exception_handler(Exception, generic_error_handler)  # type: ignore[arg-type]

    # Authentication middleware
    @app.middleware("http")
    async def auth_middleware(request: Request, call_next):
        # Skip auth for health, docs, and openapi
        if request.url.path in ("/health", "/v1/health", "/docs", "/openapi.json"):
            return await call_next(request)
        if not _auth_enabled or _gateway_key is None:
            return await call_next(request)
        auth_header = request.headers.get("Authorization") or request.headers.get("x-api-key")
        if not validate_bearer(auth_header, _gateway_key):
            return JSONResponse(
                status_code=401,
                content={"error": {"message": "Invalid or missing API key.", "type": "authentication_error", "code": "invalid_api_key"}},
            )
        return await call_next(request)

    # Protocol routers
    app.include_router(openai_chat_router)
    app.include_router(openai_responses_router)
    app.include_router(anthropic_messages_router)

    # ---------------------------------------------------------------------------
    # Health
    # ---------------------------------------------------------------------------

    @app.get("/health")
    @app.get("/v1/health")
    async def health():
        return {"status": "ok", "service": "coding-router-gateway"}

    # ---------------------------------------------------------------------------
    # GET /v1/models
    # ---------------------------------------------------------------------------

    @app.get("/v1/models")
    async def list_models():
        models: list[dict[str, Any]] = []

        # Virtual router model
        models.append({
            "id": VIRTUAL_MODEL,
            "object": "model",
            "created": int(time.time()),
            "owned_by": "coding-router",
            "permission": [],
        })

        # Catalog models
        for key, profile in get_profiles().items():
            models.append({
                "id": key,
                "object": "model",
                "created": int(time.time()),
                "owned_by": profile.service,
                "permission": [],
            })

        return {"object": "list", "data": models}

    return app
