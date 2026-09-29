"""OpenAI Chat Completions protocol adapter.

POST /v1/chat/completions

Handles both streaming and non-streaming requests.
"""

from __future__ import annotations

import json
import logging
import time
import uuid
from typing import Any

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse, StreamingResponse

from ..errors import InvalidRequestError
from ..models import (
    ClientProtocol,
    GatewayRequest,
    GatewayResponse,
    Message,
    ToolCall,
    ToolDefinition,
)
from ..providers.base import invoke_provider, stream_provider
from ..routing import resolve_model
from ..streaming import sse_wrap

log = logging.getLogger("coding-router.gateway.openai_chat")
router = APIRouter()


def _parse_request(body: dict[str, Any]) -> GatewayRequest:
    """Convert an OpenAI Chat Completions request body into a GatewayRequest."""
    messages: list[Message] = []
    system_parts: list[str] = []

    for raw_msg in body.get("messages", []):
        role = raw_msg.get("role", "user")
        content = raw_msg.get("content")

        # System messages → accumulate into system field
        if role == "system" or role == "developer":
            if isinstance(content, str):
                system_parts.append(content)
            elif isinstance(content, list):
                for part in content:
                    if isinstance(part, dict) and part.get("type") == "text":
                        system_parts.append(part.get("text", ""))
            continue

        # Parse tool_calls in assistant messages
        tool_calls = None
        if raw_msg.get("tool_calls"):
            tool_calls = [
                ToolCall(
                    id=tc.get("id", f"call_{uuid.uuid4().hex[:8]}"),
                    name=tc.get("function", {}).get("name", ""),
                    arguments=tc.get("function", {}).get("arguments", "{}"),
                )
                for tc in raw_msg["tool_calls"]
            ]

        messages.append(Message(
            role=role,
            content=content,
            tool_calls=tool_calls,
            tool_call_id=raw_msg.get("tool_call_id"),
            name=raw_msg.get("name"),
        ))

    # Parse tools
    tools = None
    if body.get("tools"):
        tools = [
            ToolDefinition(
                name=t.get("function", {}).get("name", ""),
                description=t.get("function", {}).get("description", ""),
                parameters=t.get("function", {}).get("parameters", {}),
            )
            for t in body["tools"]
            if t.get("type") == "function"
        ]

    return GatewayRequest(
        protocol=ClientProtocol.OPENAI_CHAT,
        requested_model=body.get("model", "coding-router"),
        messages=messages,
        system="\n\n".join(system_parts) if system_parts else None,
        max_tokens=body.get("max_tokens") or body.get("max_completion_tokens"),
        temperature=body.get("temperature"),
        top_p=body.get("top_p"),
        stop=body.get("stop") if isinstance(body.get("stop"), list) else ([body["stop"]] if body.get("stop") else None),
        stream=body.get("stream", False),
        tools=tools,
        tool_choice=body.get("tool_choice"),
    )


def _format_response(resp: GatewayResponse) -> dict[str, Any]:
    """Format a GatewayResponse as an OpenAI Chat Completions response."""
    choices = []
    for ch in resp.choices:
        msg: dict[str, Any] = {"role": "assistant"}
        if ch.message:
            if ch.message.content is not None:
                msg["content"] = ch.message.content
            else:
                msg["content"] = None
            if ch.message.tool_calls:
                msg["tool_calls"] = [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {"name": tc.name, "arguments": tc.arguments},
                    }
                    for tc in ch.message.tool_calls
                ]
        choices.append({
            "index": ch.index,
            "message": msg,
            "finish_reason": ch.finish_reason or "stop",
        })

    result: dict[str, Any] = {
        "id": f"chatcmpl-{resp.request_id[:20]}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": resp.model,
        "choices": choices,
    }
    if resp.usage:
        result["usage"] = {
            "prompt_tokens": resp.usage.prompt_tokens,
            "completion_tokens": resp.usage.completion_tokens,
            "total_tokens": resp.usage.total_tokens,
        }
    return result


@router.post("/v1/chat/completions", response_model=None)
async def chat_completions(request: Request) -> JSONResponse | StreamingResponse:
    body = await request.json()

    if not body.get("messages"):
        raise InvalidRequestError("'messages' is required.", param="messages")

    req = _parse_request(body)
    profile, state, routing_latency = resolve_model(req)
    req.metadata["routing_state"] = state

    log.info(
        "req=%s protocol=openai_chat model=%s→%s stream=%s",
        req.request_id, req.requested_model, profile.catalog_key, req.stream,
    )

    if req.stream:
        async def generate():
            async for chunk in stream_provider(req, profile):
                yield f"data: {chunk}\n\n"
            yield "data: [DONE]\n\n"

        return StreamingResponse(generate(), media_type="text/event-stream")

    # Non-streaming
    t0 = time.perf_counter()
    resp = await invoke_provider(req, profile)
    resp.routing_latency_ms = routing_latency
    log.info(
        "req=%s completed model=%s provider_ms=%.0f routing_ms=%.0f",
        req.request_id, profile.catalog_key, resp.provider_latency_ms, routing_latency,
    )
    return JSONResponse(content=_format_response(resp))
