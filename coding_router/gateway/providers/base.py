"""Base provider adapter and dispatch logic.

Each provider adapter converts a GatewayRequest + ModelProfile into the
provider-native HTTP request, executes it, and yields either a single
GatewayResponse or an async iterator of raw SSE chunks.
"""

from __future__ import annotations

import json
import logging
import time
from typing import Any, AsyncIterator

import httpx

from ..errors import ProviderError, ProviderKeyMissing
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

# Shared httpx client — reused across requests, created lazily.
_client: httpx.AsyncClient | None = None


async def get_client() -> httpx.AsyncClient:
    global _client
    if _client is None or _client.is_closed:
        _client = httpx.AsyncClient(timeout=httpx.Timeout(connect=10, read=120, write=30, pool=10))
    return _client


async def close_client() -> None:
    global _client
    if _client is not None:
        await _client.aclose()
        _client = None


def _resolve_key(profile: ModelProfile) -> str | None:
    """Resolve provider API key using the existing library's logic."""
    return _key_for(profile)


# ---------------------------------------------------------------------------
# OpenAI-compatible provider (OpenAI, DeepSeek, xAI, Mistral, Ollama, etc.)
# ---------------------------------------------------------------------------

def _build_openai_messages(req: GatewayRequest) -> list[dict[str, Any]]:
    """Convert internal messages to OpenAI-compatible messages list."""
    out: list[dict[str, Any]] = []
    # Prepend system if present
    if req.system:
        out.append({"role": "system", "content": req.system})
    for msg in req.messages:
        m: dict[str, Any] = {"role": msg.role}
        if msg.content is not None:
            m["content"] = msg.content
        if msg.tool_calls:
            m["tool_calls"] = [
                {"id": tc.id, "type": "function", "function": {"name": tc.name, "arguments": tc.arguments}}
                for tc in msg.tool_calls
            ]
        if msg.tool_call_id:
            m["tool_call_id"] = msg.tool_call_id
        if msg.name:
            m["name"] = msg.name
        out.append(m)
    return out


def _build_openai_body(req: GatewayRequest, model_id: str) -> dict[str, Any]:
    """Build the request body for an OpenAI-compatible endpoint."""
    body: dict[str, Any] = {
        "model": model_id,
        "messages": _build_openai_messages(req),
        "stream": req.stream,
    }
    if req.max_tokens is not None:
        body["max_tokens"] = req.max_tokens
    if req.temperature is not None:
        body["temperature"] = req.temperature
    if req.top_p is not None:
        body["top_p"] = req.top_p
    if req.stop:
        body["stop"] = req.stop
    if req.tools:
        body["tools"] = [
            {"type": "function", "function": {"name": t.name, "description": t.description, "parameters": _sanitize_schema(t.parameters, _OPENAI_UNSUPPORTED_SCHEMA_KEYS)}}
            for t in req.tools
        ]
    if req.tool_choice is not None:
        body["tool_choice"] = req.tool_choice
    return body


async def invoke_openai_compat(
    req: GatewayRequest, profile: ModelProfile
) -> GatewayResponse:
    """Non-streaming invocation of an OpenAI-compatible provider."""
    key = _resolve_key(profile)
    headers: dict[str, str] = {"Content-Type": "application/json"}
    if key:
        headers["Authorization"] = f"Bearer {key}"

    url = f"{profile.api_endpoint.rstrip('/')}/chat/completions"
    body = _build_openai_body(req, profile.model)
    body["stream"] = False

    client = await get_client()
    t0 = time.perf_counter()
    try:
        resp = await client.post(url, json=body, headers=headers)
        resp.raise_for_status()
        data = resp.json()
    except httpx.HTTPStatusError as exc:
        detail = exc.response.text[:500] if exc.response else ""
        raise ProviderError(f"Provider returned {exc.response.status_code}: {detail}", status_code=502)
    except httpx.RequestError as exc:
        raise ProviderError(f"Provider request failed: {exc}")
    latency = (time.perf_counter() - t0) * 1000

    # Parse response
    choices: list[GatewayChoice] = []
    for c in data.get("choices", []):
        msg_data = c.get("message", {})
        tool_calls = None
        if msg_data.get("tool_calls"):
            tool_calls = [
                ToolCall(id=tc["id"], name=tc["function"]["name"], arguments=tc["function"]["arguments"])
                for tc in msg_data["tool_calls"]
            ]
        choices.append(GatewayChoice(
            index=c.get("index", 0),
            message=Message(role=msg_data.get("role", "assistant"), content=msg_data.get("content"), tool_calls=tool_calls),
            finish_reason=c.get("finish_reason"),
        ))
    usage_data = data.get("usage")
    usage = GatewayUsage(
        prompt_tokens=usage_data.get("prompt_tokens", 0),
        completion_tokens=usage_data.get("completion_tokens", 0),
        total_tokens=usage_data.get("total_tokens", 0),
    ) if usage_data else None

    return GatewayResponse(
        request_id=req.request_id,
        model=profile.model,
        catalog_key=profile.catalog_key,
        provider=profile.service,
        choices=choices,
        usage=usage,
        provider_latency_ms=latency,
    )


async def stream_openai_compat(
    req: GatewayRequest, profile: ModelProfile
) -> AsyncIterator[str]:
    """Streaming invocation of an OpenAI-compatible provider.

    Yields raw JSON strings (each is one SSE data chunk).
    The protocol adapter wraps these with ``data: `` + ``[DONE]``.
    """
    key = _resolve_key(profile)
    headers: dict[str, str] = {"Content-Type": "application/json"}
    if key:
        headers["Authorization"] = f"Bearer {key}"

    url = f"{profile.api_endpoint.rstrip('/')}/chat/completions"
    body = _build_openai_body(req, profile.model)
    body["stream"] = True

    client = await get_client()
    try:
        async with client.stream("POST", url, json=body, headers=headers) as resp:
            if resp.status_code != 200:
                error_body = await resp.aread()
                raise ProviderError(f"Provider returned {resp.status_code}: {error_body.decode()[:500]}")
            async for line in resp.aiter_lines():
                line = line.strip()
                if not line:
                    continue
                if line.startswith("data: "):
                    payload = line[6:]
                    if payload == "[DONE]":
                        break
                    yield payload
    except httpx.RequestError as exc:
        raise ProviderError(f"Streaming failed: {exc}")


# ---------------------------------------------------------------------------
# Anthropic provider
# ---------------------------------------------------------------------------

def _build_anthropic_messages(req: GatewayRequest) -> list[dict[str, Any]]:
    """Convert internal messages to Anthropic messages format."""
    out: list[dict[str, Any]] = []
    for msg in req.messages:
        m: dict[str, Any] = {"role": msg.role}
        # Anthropic: content can be string or list of blocks
        if msg.tool_calls:
            # Assistant message with tool_use blocks
            blocks: list[dict[str, Any]] = []
            if msg.content:
                blocks.append({"type": "text", "text": msg.content if isinstance(msg.content, str) else json.dumps(msg.content)})
            for tc in msg.tool_calls:
                args = json.loads(tc.arguments) if isinstance(tc.arguments, str) else tc.arguments
                blocks.append({"type": "tool_use", "id": tc.id, "name": tc.name, "input": args})
            m["content"] = blocks
        elif msg.role == "tool":
            # Tool result → Anthropic uses role=user with tool_result blocks
            m["role"] = "user"
            m["content"] = [{"type": "tool_result", "tool_use_id": msg.tool_call_id or "", "content": msg.content or ""}]
        else:
            m["content"] = msg.content or ""
        out.append(m)
    return out


def _build_anthropic_body(req: GatewayRequest, model_id: str) -> dict[str, Any]:
    body: dict[str, Any] = {
        "model": model_id,
        "messages": _build_anthropic_messages(req),
        "max_tokens": req.max_tokens or 4096,
        "stream": req.stream,
    }
    if req.system:
        body["system"] = req.system
    if req.temperature is not None:
        body["temperature"] = req.temperature
    if req.top_p is not None:
        body["top_p"] = req.top_p
    if req.stop:
        body["stop_sequences"] = req.stop
    if req.tools:
        body["tools"] = [
            {"name": t.name, "description": t.description, "input_schema": _sanitize_schema(t.parameters, _ANTHROPIC_UNSUPPORTED_SCHEMA_KEYS)}
            for t in req.tools
        ]
    if req.tool_choice is not None:
        if isinstance(req.tool_choice, str):
            body["tool_choice"] = {"type": req.tool_choice}
        else:
            body["tool_choice"] = req.tool_choice
    return body


async def invoke_anthropic(
    req: GatewayRequest, profile: ModelProfile
) -> GatewayResponse:
    """Non-streaming Anthropic invocation."""
    key = _resolve_key(profile)
    if not key:
        raise ProviderKeyMissing(profile.service)

    headers = {
        "Content-Type": "application/json",
        "x-api-key": key,
        "anthropic-version": "2023-06-01",
    }
    url = f"{profile.api_endpoint.rstrip('/')}/messages"
    body = _build_anthropic_body(req, profile.model)
    body["stream"] = False

    client = await get_client()
    t0 = time.perf_counter()
    try:
        resp = await client.post(url, json=body, headers=headers)
        resp.raise_for_status()
        data = resp.json()
    except httpx.HTTPStatusError as exc:
        detail = exc.response.text[:500] if exc.response else ""
        raise ProviderError(f"Anthropic returned {exc.response.status_code}: {detail}")
    except httpx.RequestError as exc:
        raise ProviderError(f"Anthropic request failed: {exc}")
    latency = (time.perf_counter() - t0) * 1000

    # Parse Anthropic response → GatewayResponse
    content_blocks = data.get("content", [])
    text_parts = []
    tool_calls = []
    for block in content_blocks:
        if block.get("type") == "text":
            text_parts.append(block.get("text", ""))
        elif block.get("type") == "tool_use":
            tool_calls.append(ToolCall(
                id=block["id"],
                name=block["name"],
                arguments=json.dumps(block.get("input", {})),
            ))

    msg = Message(
        role="assistant",
        content="\n".join(text_parts) if text_parts else None,
        tool_calls=tool_calls if tool_calls else None,
    )
    usage_data = data.get("usage", {})
    usage = GatewayUsage(
        prompt_tokens=usage_data.get("input_tokens", 0),
        completion_tokens=usage_data.get("output_tokens", 0),
        total_tokens=usage_data.get("input_tokens", 0) + usage_data.get("output_tokens", 0),
    )

    return GatewayResponse(
        request_id=req.request_id,
        model=profile.model,
        catalog_key=profile.catalog_key,
        provider=profile.service,
        choices=[GatewayChoice(index=0, message=msg, finish_reason=data.get("stop_reason", "end_turn"))],
        usage=usage,
        provider_latency_ms=latency,
    )


async def stream_anthropic(
    req: GatewayRequest, profile: ModelProfile
) -> AsyncIterator[dict[str, Any]]:
    """Streaming Anthropic invocation.

    Yields raw Anthropic SSE event dicts (type, data, etc.).
    """
    key = _resolve_key(profile)
    if not key:
        raise ProviderKeyMissing(profile.service)

    headers = {
        "Content-Type": "application/json",
        "x-api-key": key,
        "anthropic-version": "2023-06-01",
    }
    url = f"{profile.api_endpoint.rstrip('/')}/messages"
    body = _build_anthropic_body(req, profile.model)
    body["stream"] = True

    client = await get_client()
    try:
        async with client.stream("POST", url, json=body, headers=headers) as resp:
            if resp.status_code != 200:
                error_body = await resp.aread()
                raise ProviderError(f"Anthropic streaming returned {resp.status_code}: {error_body.decode()[:500]}")
            event_type = ""
            async for line in resp.aiter_lines():
                line = line.strip()
                if not line:
                    continue
                if line.startswith("event: "):
                    event_type = line[7:]
                elif line.startswith("data: "):
                    try:
                        data = json.loads(line[6:])
                        data["_event_type"] = event_type
                        yield data
                    except json.JSONDecodeError:
                        continue
    except httpx.RequestError as exc:
        raise ProviderError(f"Anthropic streaming failed: {exc}")


# ---------------------------------------------------------------------------
# Google Gemini provider
# ---------------------------------------------------------------------------

def _build_gemini_contents(req: GatewayRequest) -> list[dict[str, Any]]:
    """Convert internal messages to Gemini contents format."""
    contents: list[dict[str, Any]] = []
    for msg in req.messages:
        role = "user" if msg.role in ("user", "system", "tool") else "model"
        parts: list[dict[str, Any]] = []
        if isinstance(msg.content, str) and msg.content:
            parts.append({"text": msg.content})
        if msg.tool_calls:
            for tc in msg.tool_calls:
                args = json.loads(tc.arguments) if isinstance(tc.arguments, str) else tc.arguments
                parts.append({"functionCall": {"name": tc.name, "args": args}})
        if msg.role == "tool" and msg.tool_call_id:
            parts.append({"functionResponse": {"name": msg.name or "", "response": {"result": msg.content or ""}}})
        if parts:
            contents.append({"role": role, "parts": parts})
    return contents


# ---------------------------------------------------------------------------
# Tool schema sanitization per provider
# ---------------------------------------------------------------------------
# IDEs (Copilot, Cursor, etc.) send tool definitions with full JSON Schema
# fields that many LLM providers reject. Each provider has a different set
# of unsupported fields. We strip them recursively before forwarding.

# OpenAI is the most tolerant, but smaller OpenAI-compatible backends
# (Ollama, vLLM, MLX, DeepSeek, Mistral) can choke on $-prefixed meta keys.
_OPENAI_UNSUPPORTED_SCHEMA_KEYS = frozenset({
    "$comment", "$id", "$schema", "$defs", "$ref",
    "deprecated", "readOnly", "writeOnly",
    "examples", "const",
})

# Anthropic rejects many JSON Schema drafts and annotation keywords.
_ANTHROPIC_UNSUPPORTED_SCHEMA_KEYS = frozenset({
    "$comment", "$id", "$schema", "$defs", "$ref",
    "additionalProperties", "patternProperties", "unevaluatedProperties",
    "if", "then", "else",
    "default", "examples", "const", "title",
    "deprecated", "readOnly", "writeOnly",
})

# Google Gemini is the strictest — rejects almost all non-core keywords.
_GEMINI_UNSUPPORTED_SCHEMA_KEYS = frozenset({
    "$comment", "$id", "$schema", "$defs", "$ref",
    "additionalProperties", "patternProperties", "unevaluatedProperties",
    "if", "then", "else", "allOf", "anyOf", "oneOf", "not",
    "default", "examples", "const", "title",
    "deprecated", "readOnly", "writeOnly",
    "enumDescriptions", "propertyNames", "exclusiveMinimum", "exclusiveMaximum",
    "multipleOf", "minLength", "maxLength", "pattern",
    "minItems", "maxItems", "uniqueItems", "minProperties", "maxProperties",
    "dependentRequired", "dependentSchemas", "contains",
    "formatMaximum", "formatMinimum", "formatExclusiveMaximum", "formatExclusiveMinimum",
})


def _sanitize_schema(schema: Any, unsupported_keys: frozenset[str]) -> Any:
    """Recursively strip JSON Schema fields that a provider rejects."""
    if isinstance(schema, dict):
        new_schema = {}
        for k, v in schema.items():
            if k in unsupported_keys:
                continue
            if k == "properties" and isinstance(v, dict):
                # Do not strip property names that happen to match unsupported keys
                new_schema[k] = {
                    prop_name: _sanitize_schema(prop_schema, unsupported_keys)
                    for prop_name, prop_schema in v.items()
                }
            else:
                new_schema[k] = _sanitize_schema(v, unsupported_keys)
        return new_schema
    if isinstance(schema, list):
        return [_sanitize_schema(item, unsupported_keys) for item in schema]
    return schema


async def invoke_google(req: GatewayRequest, profile: ModelProfile) -> GatewayResponse:
    """Non-streaming Google Gemini invocation."""
    import urllib.parse
    key = _resolve_key(profile)
    if not key:
        raise ProviderKeyMissing(profile.service)

    url = f"{profile.api_endpoint.rstrip('/')}/models/{profile.model}:generateContent?key={urllib.parse.quote(key)}"
    body: dict[str, Any] = {"contents": _build_gemini_contents(req)}
    gen_config: dict[str, Any] = {}
    if req.max_tokens is not None:
        gen_config["maxOutputTokens"] = req.max_tokens
    if req.temperature is not None:
        gen_config["temperature"] = req.temperature
    if req.top_p is not None:
        gen_config["topP"] = req.top_p
    if req.stop:
        gen_config["stopSequences"] = req.stop
    if gen_config:
        body["generationConfig"] = gen_config
    if req.system:
        body["systemInstruction"] = {"parts": [{"text": req.system}]}
    if req.tools:
        body["tools"] = [{"functionDeclarations": [
            {"name": t.name, "description": t.description, "parameters": _sanitize_schema(t.parameters, _GEMINI_UNSUPPORTED_SCHEMA_KEYS)}
            for t in req.tools
        ]}]

    client = await get_client()
    t0 = time.perf_counter()
    try:
        resp = await client.post(url, json=body, headers={"Content-Type": "application/json"})
        resp.raise_for_status()
        data = resp.json()
    except httpx.HTTPStatusError as exc:
        detail = exc.response.text[:500] if exc.response else ""
        raise ProviderError(f"Google returned {exc.response.status_code}: {detail}")
    except httpx.RequestError as exc:
        raise ProviderError(f"Google request failed: {exc}")
    latency = (time.perf_counter() - t0) * 1000

    # Parse
    candidates = data.get("candidates", [{}])
    cand = candidates[0] if candidates else {}
    parts = cand.get("content", {}).get("parts", [])
    text_parts = [p["text"] for p in parts if "text" in p]
    tool_calls = [
        ToolCall(id=f"call_{i}", name=p["functionCall"]["name"], arguments=json.dumps(p["functionCall"].get("args", {})))
        for i, p in enumerate(parts) if "functionCall" in p
    ]

    msg = Message(
        role="assistant",
        content="\n".join(text_parts) if text_parts else None,
        tool_calls=tool_calls if tool_calls else None,
    )
    usage_data = data.get("usageMetadata", {})

    return GatewayResponse(
        request_id=req.request_id,
        model=profile.model,
        catalog_key=profile.catalog_key,
        provider=profile.service,
        choices=[GatewayChoice(index=0, message=msg, finish_reason=cand.get("finishReason", "stop"))],
        usage=GatewayUsage(
            prompt_tokens=usage_data.get("promptTokenCount", 0),
            completion_tokens=usage_data.get("candidatesTokenCount", 0),
            total_tokens=usage_data.get("totalTokenCount", 0),
        ),
        provider_latency_ms=latency,
    )


async def stream_google(req: GatewayRequest, profile: ModelProfile) -> AsyncIterator[str]:
    """Streaming Google Gemini — translates to OpenAI-style SSE chunks.

    Gemini streams via ``streamGenerateContent?alt=sse``.
    We translate each chunk into an OpenAI-compatible delta.
    """
    import urllib.parse
    key = _resolve_key(profile)
    if not key:
        raise ProviderKeyMissing(profile.service)

    url = f"{profile.api_endpoint.rstrip('/')}/models/{profile.model}:streamGenerateContent?alt=sse&key={urllib.parse.quote(key)}"
    body: dict[str, Any] = {"contents": _build_gemini_contents(req)}
    gen_config: dict[str, Any] = {}
    if req.max_tokens is not None:
        gen_config["maxOutputTokens"] = req.max_tokens
    if req.temperature is not None:
        gen_config["temperature"] = req.temperature
    if gen_config:
        body["generationConfig"] = gen_config
    if req.system:
        body["systemInstruction"] = {"parts": [{"text": req.system}]}
    if req.tools:
        body["tools"] = [{"functionDeclarations": [
            {"name": t.name, "description": t.description, "parameters": _sanitize_schema(t.parameters, _GEMINI_UNSUPPORTED_SCHEMA_KEYS)}
            for t in req.tools
        ]}]

    client = await get_client()
    try:
        async with client.stream("POST", url, json=body, headers={"Content-Type": "application/json"}) as resp:
            if resp.status_code != 200:
                error_body = await resp.aread()
                raise ProviderError(f"Google streaming returned {resp.status_code}: {error_body.decode()[:500]}")
            chunk_id = f"chatcmpl-{req.request_id[:16]}"
            created_ts = int(time.time())
            
            # Emit initial role chunk
            yield json.dumps({
                "id": chunk_id, "object": "chat.completion.chunk", "created": created_ts, "model": profile.model,
                "choices": [{"index": 0, "delta": {"role": "assistant", "content": ""}, "finish_reason": None}],
            }, separators=(",", ":"))
            
            async for line in resp.aiter_lines():
                line = line.strip()
                if not line or not line.startswith("data: "):
                    continue
                try:
                    data = json.loads(line[6:])
                except json.JSONDecodeError:
                    continue
                parts = data.get("candidates", [{}])[0].get("content", {}).get("parts", [])
                for p in parts:
                    if "text" in p:
                        # Translate to OpenAI streaming chunk
                        openai_chunk = {
                            "id": chunk_id,
                            "object": "chat.completion.chunk",
                            "created": created_ts,
                            "model": profile.model,
                            "choices": [{"index": 0, "delta": {"content": p["text"]}, "finish_reason": None}],
                        }
                        yield json.dumps(openai_chunk, separators=(",", ":"))
                    elif "functionCall" in p:
                        fc = p["functionCall"]
                        args_str = json.dumps(fc.get("args", {}))
                        openai_chunk = {
                            "id": chunk_id,
                            "object": "chat.completion.chunk",
                            "created": created_ts,
                            "model": profile.model,
                            "choices": [{"index": 0, "delta": {"tool_calls": [{"index": 0, "id": f"call_{chunk_id[9:]}", "type": "function", "function": {"name": fc["name"], "arguments": args_str}}]}, "finish_reason": None}],
                        }
                        yield json.dumps(openai_chunk, separators=(",", ":"))
            # Final chunk
            yield json.dumps({
                "id": chunk_id,
                "object": "chat.completion.chunk",
                "created": created_ts,
                "model": profile.model,
                "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
            }, separators=(",", ":"))
    except httpx.RequestError as exc:
        raise ProviderError(f"Google streaming failed: {exc}")


# ---------------------------------------------------------------------------
# Dispatch
# ---------------------------------------------------------------------------

async def invoke_provider(req: GatewayRequest, profile: ModelProfile) -> GatewayResponse:
    """Non-streaming dispatch to the appropriate provider."""
    if profile.service == "Anthropic":
        return await invoke_anthropic(req, profile)
    if profile.service == "Google":
        return await invoke_google(req, profile)
    return await invoke_openai_compat(req, profile)


async def stream_provider(
    req: GatewayRequest, profile: ModelProfile
) -> AsyncIterator[str]:
    """Streaming dispatch — yields OpenAI-compatible JSON chunks (strings).

    For non-OpenAI providers the adapter translates into OpenAI chunk format
    so the protocol adapters only need to handle one streaming shape.
    """
    if profile.service == "Anthropic":
        # Translate Anthropic SSE into OpenAI-style chunks
        chunk_id = f"chatcmpl-{req.request_id[:16]}"
        created_ts = int(time.time())
        async for event in stream_anthropic(req, profile):
            etype = event.get("_event_type", "")
            if etype == "message_start":
                yield json.dumps({
                    "id": chunk_id, "object": "chat.completion.chunk", "created": created_ts, "model": profile.model,
                    "choices": [{"index": 0, "delta": {"role": "assistant", "content": ""}, "finish_reason": None}],
                }, separators=(",", ":"))
            elif etype == "content_block_delta":
                delta = event.get("delta", {})
                if delta.get("type") == "text_delta":
                    yield json.dumps({
                        "id": chunk_id, "object": "chat.completion.chunk", "created": created_ts, "model": profile.model,
                        "choices": [{"index": 0, "delta": {"content": delta.get("text", "")}, "finish_reason": None}],
                    }, separators=(",", ":"))
                elif delta.get("type") == "input_json_delta":
                    yield json.dumps({
                        "id": chunk_id, "object": "chat.completion.chunk", "created": created_ts, "model": profile.model,
                        "choices": [{"index": 0, "delta": {"tool_calls": [{"index": 0, "function": {"arguments": delta.get("partial_json", "")}}]}, "finish_reason": None}],
                    }, separators=(",", ":"))
            elif etype == "content_block_start":
                cb = event.get("content_block", {})
                if cb.get("type") == "tool_use":
                    yield json.dumps({
                        "id": chunk_id, "object": "chat.completion.chunk", "created": created_ts, "model": profile.model,
                        "choices": [{"index": 0, "delta": {"tool_calls": [{"index": 0, "id": cb.get("id", ""), "type": "function", "function": {"name": cb.get("name", ""), "arguments": ""}}]}, "finish_reason": None}],
                    }, separators=(",", ":"))
            elif etype == "message_delta":
                yield json.dumps({
                    "id": chunk_id, "object": "chat.completion.chunk", "created": created_ts, "model": profile.model,
                    "choices": [{"index": 0, "delta": {}, "finish_reason": event.get("delta", {}).get("stop_reason", "stop")}],
                }, separators=(",", ":"))
        return

    if profile.service == "Google":
        async for chunk in stream_google(req, profile):
            yield chunk
        return

    # OpenAI-compatible: pass through directly
    async for chunk in stream_openai_compat(req, profile):
        yield chunk
