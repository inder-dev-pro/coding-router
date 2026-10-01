"""Anthropic Messages protocol adapter.

POST /v1/messages

Handles both streaming and non-streaming Anthropic requests.
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

log = logging.getLogger("coding-router.gateway.anthropic")
router = APIRouter()


def _parse_request(body: dict[str, Any]) -> GatewayRequest:
    """Convert an Anthropic Messages request body into a GatewayRequest."""
    messages: list[Message] = []

    for raw_msg in body.get("messages", []):
        role = raw_msg.get("role", "user")
        content = raw_msg.get("content")

        # Content can be string or list of blocks
        if isinstance(content, list):
            text_parts: list[str] = []
            tool_calls: list[ToolCall] = []
            tool_results: list[dict[str, Any]] = []

            for block in content:
                if not isinstance(block, dict):
                    continue
                btype = block.get("type", "")
                if btype == "text":
                    text_parts.append(block.get("text", ""))
                elif btype == "tool_use":
                    tool_calls.append(ToolCall(
                        id=block.get("id", f"toolu_{uuid.uuid4().hex[:12]}"),
                        name=block.get("name", ""),
                        arguments=json.dumps(block.get("input", {})),
                    ))
                elif btype == "tool_result":
                    # Anthropic sends tool results as user messages with tool_result blocks
                    result_content = block.get("content", "")
                    if isinstance(result_content, list):
                        result_content = "\n".join(
                            p.get("text", "") for p in result_content if isinstance(p, dict) and p.get("type") == "text"
                        )
                    messages.append(Message(
                        role="tool",
                        content=str(result_content),
                        tool_call_id=block.get("tool_use_id", ""),
                    ))
                    continue

            if tool_calls:
                messages.append(Message(
                    role=role,
                    content="\n".join(text_parts) if text_parts else None,
                    tool_calls=tool_calls,
                ))
            elif text_parts:
                messages.append(Message(role=role, content="\n".join(text_parts)))
        else:
            messages.append(Message(role=role, content=content or ""))

    # Parse tools
    tools = None
    if body.get("tools"):
        tools = [
            ToolDefinition(
                name=t.get("name", ""),
                description=t.get("description", ""),
                parameters=t.get("input_schema", {}),
            )
            for t in body["tools"]
        ]

    return GatewayRequest(
        protocol=ClientProtocol.ANTHROPIC_MESSAGES,
        requested_model=body.get("model", "coding-router"),
        messages=messages,
        system=body.get("system") if isinstance(body.get("system"), str) else None,
        max_tokens=body.get("max_tokens", 4096),
        temperature=body.get("temperature"),
        top_p=body.get("top_p"),
        stop=body.get("stop_sequences"),
        stream=body.get("stream", False),
        tools=tools,
        tool_choice=body.get("tool_choice"),
        metadata=body.get("metadata", {}),
    )


def _format_response(resp: GatewayResponse) -> dict[str, Any]:
    """Format a GatewayResponse as an Anthropic Messages response."""
    content: list[dict[str, Any]] = []

    for ch in resp.choices:
        if ch.message:
            if ch.message.content:
                content.append({"type": "text", "text": ch.message.content})
            if ch.message.tool_calls:
                for tc in ch.message.tool_calls:
                    try:
                        inp = json.loads(tc.arguments)
                    except json.JSONDecodeError:
                        inp = {}
                    content.append({
                        "type": "tool_use",
                        "id": tc.id,
                        "name": tc.name,
                        "input": inp,
                    })

    stop_reason = "end_turn"
    if resp.choices and resp.choices[0].finish_reason:
        fr = resp.choices[0].finish_reason
        if fr in ("stop", "end_turn"):
            stop_reason = "end_turn"
        elif fr in ("tool_calls", "tool_use"):
            stop_reason = "tool_use"
        elif fr == "length":
            stop_reason = "max_tokens"
        else:
            stop_reason = "end_turn"

    result: dict[str, Any] = {
        "id": f"msg-{resp.request_id[:20]}",
        "type": "message",
        "role": "assistant",
        "content": content,
        "model": resp.model,
        "stop_reason": stop_reason,
        "stop_sequence": None,
    }
    if resp.usage:
        result["usage"] = {
            "input_tokens": resp.usage.prompt_tokens,
            "output_tokens": resp.usage.completion_tokens,
        }
    return result


@router.post("/v1/messages", response_model=None)
async def messages(request: Request) -> JSONResponse | StreamingResponse:
    body = await request.json()

    if not body.get("messages"):
        raise InvalidRequestError("'messages' is required.", param="messages")

    req = _parse_request(body)
    profile, state, routing_latency = resolve_model(req)

    log.info(
        "req=%s protocol=anthropic_messages model=%s→%s stream=%s",
        req.request_id, req.requested_model, profile.catalog_key, req.stream,
    )

    if req.stream:
        response_id = f"msg-{req.request_id[:20]}"

        async def generate():
            # message_start
            yield f"event: message_start\ndata: {json.dumps({'type': 'message_start', 'message': {'id': response_id, 'type': 'message', 'role': 'assistant', 'content': [], 'model': profile.model, 'stop_reason': None, 'usage': {'input_tokens': 0, 'output_tokens': 0}}}, separators=(',', ':'))}\n\n"

            # content_block_start
            yield f"event: content_block_start\ndata: {json.dumps({'type': 'content_block_start', 'index': 0, 'content_block': {'type': 'text', 'text': ''}}, separators=(',', ':'))}\n\n"

            async for chunk_json in stream_provider(req, profile):
                try:
                    chunk = json.loads(chunk_json)
                    choices = chunk.get("choices", [])
                    if not choices:
                        continue
                    delta = choices[0].get("delta", {})

                    text = delta.get("content", "")
                    if text:
                        yield f"event: content_block_delta\ndata: {json.dumps({'type': 'content_block_delta', 'index': 0, 'delta': {'type': 'text_delta', 'text': text}}, separators=(',', ':'))}\n\n"

                    # Tool call streaming
                    tool_calls = delta.get("tool_calls")
                    if tool_calls:
                        for tc in tool_calls:
                            if tc.get("id"):
                                # New tool call block
                                yield f"event: content_block_start\ndata: {json.dumps({'type': 'content_block_start', 'index': tc.get('index', 0) + 1, 'content_block': {'type': 'tool_use', 'id': tc['id'], 'name': tc.get('function', {}).get('name', ''), 'input': {}}}, separators=(',', ':'))}\n\n"
                            if tc.get("function", {}).get("arguments"):
                                yield f"event: content_block_delta\ndata: {json.dumps({'type': 'content_block_delta', 'index': tc.get('index', 0) + 1, 'delta': {'type': 'input_json_delta', 'partial_json': tc['function']['arguments']}}, separators=(',', ':'))}\n\n"

                except (json.JSONDecodeError, IndexError, KeyError):
                    continue

            # content_block_stop
            yield f"event: content_block_stop\ndata: {json.dumps({'type': 'content_block_stop', 'index': 0}, separators=(',', ':'))}\n\n"

            # message_delta
            yield f"event: message_delta\ndata: {json.dumps({'type': 'message_delta', 'delta': {'stop_reason': 'end_turn', 'stop_sequence': None}, 'usage': {'output_tokens': 0}}, separators=(',', ':'))}\n\n"

            # message_stop
            yield f"event: message_stop\ndata: {json.dumps({'type': 'message_stop'}, separators=(',', ':'))}\n\n"

        return StreamingResponse(generate(), media_type="text/event-stream")

    # Non-streaming
    resp = await invoke_provider(req, profile)
    resp.routing_latency_ms = routing_latency
    return JSONResponse(content=_format_response(resp))
