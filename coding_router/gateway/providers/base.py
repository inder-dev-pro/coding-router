"""LiteLLM-based provider adapter.

Replaces hand-rolled httpx adapters with litellm.acompletion() for unified
multi-provider support. Keeps the same invoke_provider / stream_provider
contract that all protocol adapters depend on.
"""

from __future__ import annotations

import json
import logging
import time
from typing import Any, AsyncIterator

import litellm

from ..errors import (
    InvalidRequestError,
    ProviderError,
    ProviderKeyMissing,
)
from ..models import (
    GatewayChoice,
    GatewayRequest,
    GatewayResponse,
    GatewayUsage,
    Message,
    ToolCall,
)
from coding_router.catalog import ModelProfile
from coding_router.invocation import _key_for

log = logging.getLogger("coding-router.gateway.provider")

# Suppress litellm's verbose default logging
litellm.suppress_debug_info = True

# ---------------------------------------------------------------------------
# Service → LiteLLM provider prefix mapping
# ---------------------------------------------------------------------------

_SERVICE_TO_LITELLM_PREFIX: dict[str, str] = {
    "OpenAI": "openai/",
    "Anthropic": "anthropic/",
    "Google": "gemini/",
    "DeepSeek": "deepseek/",
    "xAI": "xai/",
    "Mistral AI": "mistral/",
    "Alibaba Cloud (Qwen)": "openai/",  # Qwen via OpenAI-compat
    "Z.ai (Zhipu)": "openai/",          # Zhipu via OpenAI-compat
    "Moonshot AI": "openai/",            # Moonshot via OpenAI-compat
    "MiniMax": "openai/",               # MiniMax via OpenAI-compat
    "Self-hosted (Ollama / vLLM)": "ollama/",
}


def _litellm_model(profile: ModelProfile) -> str:
    """Derive the LiteLLM model identifier from a ModelProfile.

    E.g. service="Google", model="gemini-3-pro-preview" → "gemini/gemini-3-pro-preview"
         service="Anthropic", model="claude-3-opus-20240229" → "anthropic/claude-3-opus-20240229"
         service="OpenAI", model="gpt-4o" → "openai/gpt-4o"
    """
    prefix = _SERVICE_TO_LITELLM_PREFIX.get(profile.service, "openai/")
    return f"{prefix}{profile.model}"


# ---------------------------------------------------------------------------
# Build messages list for LiteLLM (OpenAI format — litellm handles translation)
# ---------------------------------------------------------------------------

def _build_messages(req: GatewayRequest) -> list[dict[str, Any]]:
    """Convert internal Message objects to OpenAI-format dicts.

    LiteLLM accepts OpenAI-format messages and translates them internally
    for Anthropic, Gemini, etc.
    """
    out: list[dict[str, Any]] = []

    if req.system:
        out.append({"role": "system", "content": req.system})

    for msg in req.messages:
        m: dict[str, Any] = {"role": msg.role}

        if msg.content is not None:
            m["content"] = msg.content

        if msg.tool_calls:
            m["tool_calls"] = [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {"name": tc.name, "arguments": tc.arguments},
                }
                for tc in msg.tool_calls
            ]

        if msg.tool_call_id:
            m["tool_call_id"] = msg.tool_call_id

        if msg.name:
            m["name"] = msg.name

        out.append(m)

    return out


def _build_tools(req: GatewayRequest) -> list[dict[str, Any]] | None:
    """Convert internal ToolDefinition objects to OpenAI-format tool dicts."""
    if not req.tools:
        return None
    return [
        {
            "type": "function",
            "function": {
                "name": t.name,
                "description": t.description,
                "parameters": t.parameters,
            },
        }
        for t in req.tools
    ]


# ---------------------------------------------------------------------------
# Build common kwargs for litellm.acompletion
# ---------------------------------------------------------------------------

def _build_kwargs(
    req: GatewayRequest,
    profile: ModelProfile,
    *,
    stream: bool = False,
) -> dict[str, Any]:
    """Build the keyword arguments dict for litellm.acompletion()."""
    model = _litellm_model(profile)
    messages = _build_messages(req)
    tools = _build_tools(req)

    kwargs: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "stream": stream,
    }

    # API key — pass explicitly if our resolver finds one (supports per-model overrides)
    api_key = _key_for(profile)
    if api_key:
        kwargs["api_key"] = api_key

    # Custom API base for self-hosted or non-standard endpoints
    if profile.service in (
        "Self-hosted (Ollama / vLLM)",
        "Alibaba Cloud (Qwen)",
        "Z.ai (Zhipu)",
        "Moonshot AI",
        "MiniMax",
    ):
        kwargs["api_base"] = profile.api_endpoint

    # Generation parameters
    if req.max_tokens is not None:
        kwargs["max_tokens"] = req.max_tokens
    if req.temperature is not None:
        kwargs["temperature"] = req.temperature
    if req.top_p is not None:
        kwargs["top_p"] = req.top_p
    if req.stop:
        kwargs["stop"] = req.stop

    # Tools
    if tools:
        kwargs["tools"] = tools
    if req.tool_choice is not None:
        kwargs["tool_choice"] = req.tool_choice

    return kwargs


# ---------------------------------------------------------------------------
# Error mapping: LiteLLM exceptions → GatewayError subclasses
# ---------------------------------------------------------------------------

def _handle_litellm_error(exc: Exception, profile: ModelProfile) -> None:
    """Map litellm exceptions to our GatewayError hierarchy and re-raise."""
    model_info = f"{profile.service}/{profile.model}"

    if isinstance(exc, litellm.AuthenticationError):
        raise ProviderKeyMissing(profile.service) from exc

    if isinstance(exc, litellm.ContextWindowExceededError):
        raise InvalidRequestError(
            f"Context window exceeded for {model_info}: {exc}",
            param="messages",
        ) from exc

    if isinstance(exc, litellm.RateLimitError):
        raise ProviderError(
            f"Rate limited by {model_info}: {exc}",
            status_code=429,
        ) from exc

    if isinstance(exc, litellm.BadRequestError):
        raise InvalidRequestError(
            f"Bad request to {model_info}: {exc}",
        ) from exc

    if isinstance(exc, litellm.NotFoundError):
        raise ProviderError(
            f"Model not found at provider {model_info}: {exc}",
            status_code=404,
        ) from exc

    if isinstance(exc, litellm.Timeout):
        raise ProviderError(
            f"Timeout calling {model_info}: {exc}",
            status_code=504,
        ) from exc

    if isinstance(exc, litellm.APIConnectionError):
        raise ProviderError(
            f"Connection error to {model_info}: {exc}",
            status_code=502,
        ) from exc

    if isinstance(exc, litellm.APIError):
        raise ProviderError(
            f"Provider error from {model_info}: {exc}",
            status_code=502,
        ) from exc

    # Unknown litellm error — wrap generically
    raise ProviderError(
        f"LLM call to {model_info} failed: {exc}",
        status_code=502,
    ) from exc


# ---------------------------------------------------------------------------
# Non-streaming dispatch
# ---------------------------------------------------------------------------

async def invoke_provider(
    req: GatewayRequest, profile: ModelProfile
) -> GatewayResponse:
    """Non-streaming LLM invocation via LiteLLM."""
    kwargs = _build_kwargs(req, profile, stream=False)

    t0 = time.perf_counter()
    try:
        response = await litellm.acompletion(**kwargs)
    except Exception as exc:
        _handle_litellm_error(exc, profile)
    latency = (time.perf_counter() - t0) * 1000

    # Parse LiteLLM response (already in OpenAI format)
    choices: list[GatewayChoice] = []
    for c in response.choices:
        msg_data = c.message
        tool_calls = None
        if msg_data.tool_calls:
            tool_calls = [
                ToolCall(
                    id=tc.id,
                    name=tc.function.name,
                    arguments=tc.function.arguments,
                )
                for tc in msg_data.tool_calls
            ]
        choices.append(
            GatewayChoice(
                index=c.index,
                message=Message(
                    role=msg_data.role or "assistant",
                    content=msg_data.content,
                    tool_calls=tool_calls,
                ),
                finish_reason=c.finish_reason,
            )
        )

    usage = None
    if response.usage:
        usage = GatewayUsage(
            prompt_tokens=response.usage.prompt_tokens or 0,
            completion_tokens=response.usage.completion_tokens or 0,
            total_tokens=response.usage.total_tokens or 0,
        )

    return GatewayResponse(
        request_id=req.request_id,
        model=profile.model,
        catalog_key=profile.catalog_key,
        provider=profile.service,
        choices=choices,
        usage=usage,
        provider_latency_ms=latency,
    )


# ---------------------------------------------------------------------------
# Streaming dispatch
# ---------------------------------------------------------------------------

async def stream_provider(
    req: GatewayRequest, profile: ModelProfile
) -> AsyncIterator[str]:
    """Streaming LLM invocation via LiteLLM.

    Yields OpenAI-compatible JSON chunk strings (without the ``data: `` prefix).
    The protocol adapters handle SSE framing.
    """
    kwargs = _build_kwargs(req, profile, stream=True)

    try:
        response = await litellm.acompletion(**kwargs)
    except Exception as exc:
        _handle_litellm_error(exc, profile)

    try:
        async for chunk in response:
            # LiteLLM streaming chunks are ModelResponse objects in OpenAI format.
            # Serialize to JSON string for the protocol adapters.
            chunk_dict = chunk.model_dump(exclude_none=True, exclude_unset=True)
            yield json.dumps(chunk_dict, separators=(",", ":"))
    except Exception as exc:
        _handle_litellm_error(exc, profile)


# ---------------------------------------------------------------------------
# Cleanup — no longer needed (LiteLLM manages its own connections)
# but kept for backward compat with server.py lifespan
# ---------------------------------------------------------------------------

async def close_client() -> None:
    """No-op — LiteLLM manages its own HTTP connections."""
    pass
