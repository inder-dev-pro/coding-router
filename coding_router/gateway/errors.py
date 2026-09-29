"""Gateway error types and FastAPI exception handlers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from fastapi import Request
from fastapi.responses import JSONResponse


# ---------------------------------------------------------------------------
# Error types
# ---------------------------------------------------------------------------

@dataclass
class GatewayError(Exception):
    """Base gateway error with HTTP status and protocol-compatible body."""
    message: str
    status_code: int = 500
    error_type: str = "server_error"
    param: str | None = None

    def __str__(self) -> str:
        return self.message


class AuthenticationError(GatewayError):
    def __init__(self, message: str = "Invalid or missing API key."):
        super().__init__(message=message, status_code=401, error_type="authentication_error")


class InvalidRequestError(GatewayError):
    def __init__(self, message: str, param: str | None = None):
        super().__init__(message=message, status_code=400, error_type="invalid_request_error", param=param)


class ModelNotFoundError(GatewayError):
    def __init__(self, model: str):
        super().__init__(
            message=f"Model '{model}' not found in the catalog.",
            status_code=404,
            error_type="model_not_found",
        )


class ProviderError(GatewayError):
    def __init__(self, message: str, status_code: int = 502):
        super().__init__(message=message, status_code=status_code, error_type="provider_error")


class ProviderKeyMissing(GatewayError):
    def __init__(self, service: str):
        super().__init__(
            message=f"API key for provider '{service}' is not configured.",
            status_code=502,
            error_type="provider_key_missing",
        )


class RoutingError(GatewayError):
    def __init__(self, message: str):
        super().__init__(message=message, status_code=500, error_type="routing_error")


# ---------------------------------------------------------------------------
# FastAPI exception handlers
# ---------------------------------------------------------------------------

def _openai_error_body(err: GatewayError) -> dict[str, Any]:
    """Format compatible with OpenAI's error schema."""
    body: dict[str, Any] = {
        "error": {
            "message": err.message,
            "type": err.error_type,
            "code": err.error_type,
        }
    }
    if err.param:
        body["error"]["param"] = err.param
    return body


async def gateway_error_handler(_request: Request, exc: GatewayError) -> JSONResponse:
    if exc.status_code >= 500:
        import traceback
        traceback.print_exc()
    return JSONResponse(status_code=exc.status_code, content=_openai_error_body(exc))


async def generic_error_handler(_request: Request, exc: Exception) -> JSONResponse:
    import traceback
    traceback.print_exc()
    return JSONResponse(
        status_code=500,
        content={
            "error": {
                "message": f"Internal gateway error: {str(exc)}",
                "type": "server_error",
                "code": "internal_error",
            }
        },
    )
