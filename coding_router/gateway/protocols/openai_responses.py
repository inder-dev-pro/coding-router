"""OpenAI Responses API protocol adapter.

POST /v1/responses

The Responses API is a newer OpenAI interface that uses a different
request/response shape from Chat Completions.  We normalize it into
GatewayRequest, route through the existing router, and translate back.
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
    Message,
    ToolCall,
    ToolDefinition,
    GatewayResponse,
)
from ..providers.base import invoke_provider, stream_provider
from ..routing import resolve_model

log = logging.getLogger("coding-router.gateway.openai_responses")
router = APIRouter()


def _parse_input(raw_input: Any) -> tuple[list[Message], str | None]:
    """Parse the Responses API 'input' field.

    The input can be:
      - A plain string (simple prompt)
      - A list of message objects (conversation)
    """
    messages: list[Message] = []
    system: str | None = None

    if isinstance(raw_input, str):
        messages.append(Message(role="user", content=raw_input))
        return messages, system

    if isinstance(raw_input, list):
        for item in raw_input:
            if not isinstance(item, dict):
                continue
            role = item.get("role", "user")
            # Handle various content shapes
            content = item.get("content")
            if isinstance(content, list):
                # Multi-part content — extract text
                text_parts = []
                for part in content:
                    if isinstance(part, dict):
                        if part.get("type") == "input_text":
                            text_parts.append(part.get("text", ""))
                        elif part.get("type") == "text":
                            text_parts.append(part.get("text", ""))
                content = "\n".join(text_parts) if text_parts else ""

            if role in ("system", "developer"):
                system = content if isinstance(content, str) else str(content or "")
                continue

            # Handle tool results
            if item.get("type") == "function_call_output":
                messages.append(Message(
                    role="tool",
                    content=item.get("output", ""),
                    tool_call_id=item.get("call_id", ""),
                ))
                continue

            messages.append(Message(role=role, content=content))

    return messages, system


def _parse_request(body: dict[str, Any]) -> GatewayRequest:
    """Convert a Responses API request into a GatewayRequest."""
    raw_input = body.get("input", "")
    messages, system = _parse_input(raw_input)

    # Instructions → system
    if body.get("instructions"):
        system = body["instructions"] if system is None else f"{system}\n\n{body['instructions']}"

    # Parse tools
    tools = None
    if body.get("tools"):
        tools = []
        for t in body["tools"]:
            if t.get("type") == "function":
                func = t.get("function", t)  # may be nested or flat
                tools.append(ToolDefinition(
                    name=func.get("name", t.get("name", "")),
                    description=func.get("description", t.get("description", "")),
                    parameters=func.get("parameters", t.get("parameters", {})),
                ))

    return GatewayRequest(
        protocol=ClientProtocol.OPENAI_RESPONSES,
        requested_model=body.get("model", "coding-router"),
        messages=messages,
        system=system,
        max_tokens=body.get("max_output_tokens"),
        temperature=body.get("temperature"),
        top_p=body.get("top_p"),
        stream=body.get("stream", False),
        tools=tools,
        tool_choice=body.get("tool_choice"),
    )


def _format_response(resp: GatewayResponse, req: GatewayRequest) -> dict[str, Any]:
    """Format a GatewayResponse as a Responses API response."""
    response_id = f"resp-{resp.request_id[:20]}"
    output: list[dict[str, Any]] = []

    for ch in resp.choices:
        if ch.message:
            # Text output
            if ch.message.content:
                output.append({
                    "type": "message",
                    "id": f"msg-{uuid.uuid4().hex[:16]}",
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": ch.message.content}],
                    "status": "completed",
                })
            # Tool calls
            if ch.message.tool_calls:
                for tc in ch.message.tool_calls:
                    output.append({
                        "type": "function_call",
                        "id": tc.id,
                        "call_id": tc.id,
                        "name": tc.name,
                        "arguments": tc.arguments,
                        "status": "completed",
                    })

    result: dict[str, Any] = {
        "id": response_id,
        "object": "response",
        "created_at": int(time.time()),
        "model": resp.model,
        "output": output,
        "status": "completed",
    }
    if resp.usage:
        result["usage"] = {
            "input_tokens": resp.usage.prompt_tokens,
            "output_tokens": resp.usage.completion_tokens,
            "total_tokens": resp.usage.total_tokens,
        }
    return result


def _format_stream_event(event_type: str, data: dict[str, Any]) -> str:
    """Format one Responses API streaming event."""
    return f"event: {event_type}\ndata: {json.dumps(data, separators=(',', ':'))}\n\n"


@router.post("/v1/responses", response_model=None)
async def responses(request: Request) -> JSONResponse | StreamingResponse:
    body = await request.json()

    if not body.get("input"):
        raise InvalidRequestError("'input' is required.", param="input")

    req = _parse_request(body)
    profile, state, routing_latency = resolve_model(req)

    log.info(
        "req=%s protocol=openai_responses model=%s→%s stream=%s",
        req.request_id, req.requested_model, profile.catalog_key, req.stream,
    )

    if req.stream:
        response_id = f"resp-{req.request_id[:20]}"

        async def generate():
            # response.created
            yield _format_stream_event("response.created", {
                "response": {"id": response_id, "object": "response", "status": "in_progress",
                             "model": profile.model, "output": []},
            })
            yield _format_stream_event("response.in_progress", {
                "response": {"id": response_id, "status": "in_progress"},
            })

            # Stream content
            item_id = f"msg-{uuid.uuid4().hex[:12]}"
            yield _format_stream_event("response.output_item.added", {
                "output_index": 0,
                "item": {"type": "message", "id": item_id, "role": "assistant",
                         "content": [], "status": "in_progress"},
            })
            yield _format_stream_event("response.content_part.added", {
                "output_index": 0, "content_index": 0,
                "part": {"type": "output_text", "text": ""},
            })

            async for chunk_json in stream_provider(req, profile):
                try:
                    chunk = json.loads(chunk_json)
                    delta = chunk.get("choices", [{}])[0].get("delta", {})
                    text = delta.get("content", "")
                    if text:
                        yield _format_stream_event("response.output_text.delta", {
                            "output_index": 0, "content_index": 0, "delta": text,
                        })
                except (json.JSONDecodeError, IndexError, KeyError):
                    continue

            yield _format_stream_event("response.output_text.done", {
                "output_index": 0, "content_index": 0, "text": "",
            })
            yield _format_stream_event("response.completed", {
                "response": {"id": response_id, "status": "completed", "model": profile.model},
            })

        return StreamingResponse(generate(), media_type="text/event-stream")

    # Non-streaming
    resp = await invoke_provider(req, profile)
    resp.routing_latency_ms = routing_latency
    return JSONResponse(content=_format_response(resp, req))
