"""Internal request/response models for the gateway.

Every protocol adapter converts its wire format into a GatewayRequest,
and every provider adapter returns a GatewayResponse.  These are the
only types that cross between protocol → router → provider boundaries.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class ClientProtocol(str, Enum):
    OPENAI_CHAT = "openai_chat"
    OPENAI_RESPONSES = "openai_responses"
    ANTHROPIC_MESSAGES = "anthropic_messages"


@dataclass
class ToolDefinition:
    """A tool the client has declared (function-calling)."""
    name: str
    description: str = ""
    parameters: dict[str, Any] = field(default_factory=dict)


@dataclass
class ToolCall:
    """A tool call produced by the model."""
    id: str
    name: str
    arguments: str  # JSON string


@dataclass
class ToolResult:
    """A tool result supplied by the client."""
    tool_use_id: str
    content: str


@dataclass
class Message:
    """A single conversation message in the internal representation."""
    role: str  # "system", "user", "assistant", "tool"
    content: str | list[dict[str, Any]] | None = None
    tool_calls: list[ToolCall] | None = None
    tool_call_id: str | None = None  # for role="tool" messages
    name: str | None = None


@dataclass
class GatewayRequest:
    """Protocol-neutral internal request."""
    request_id: str = field(default_factory=lambda: f"gr-{uuid.uuid4().hex[:24]}")
    protocol: ClientProtocol = ClientProtocol.OPENAI_CHAT
    timestamp: float = field(default_factory=time.time)

    # Model selection
    requested_model: str = "coding-router"

    # Conversation
    messages: list[Message] = field(default_factory=list)
    system: str | None = None

    # Generation parameters
    max_tokens: int | None = None
    temperature: float | None = None
    top_p: float | None = None
    stop: list[str] | None = None
    stream: bool = False

    # Tools
    tools: list[ToolDefinition] | None = None
    tool_choice: str | dict[str, Any] | None = None

    # Metadata
    metadata: dict[str, Any] = field(default_factory=dict)

    def user_query_text(self) -> str:
        """Extract a plain-text query for the router's classifier.

        Takes the last user message content as the routing signal.
        Falls back to system prompt if no user messages exist.
        """
        for msg in reversed(self.messages):
            if msg.role == "user" and isinstance(msg.content, str) and msg.content.strip():
                # Copilot prepends massive codebase context to the user's prompt.
                # The actual user instruction is typically the very last paragraph.
                text = msg.content.strip()
                
                # Split by double newline to grab the final paragraph
                paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
                if paragraphs:
                    prompt = paragraphs[-1]
                    # Fallback to last 500 chars if the last paragraph is somehow huge
                    if len(prompt) > 500:
                        prompt = prompt[-500:]
                    return prompt
                    
                return text[-500:] if len(text) > 500 else text
        
        # Fall back to system if no user content
        if self.system:
            return self.system
        return ""


@dataclass
class GatewayChoice:
    """A single completion choice."""
    index: int = 0
    message: Message | None = None
    finish_reason: str | None = None


@dataclass
class GatewayUsage:
    """Token usage information."""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


@dataclass
class GatewayResponse:
    """Protocol-neutral internal response."""
    request_id: str = ""
    model: str = ""              # actual model used
    catalog_key: str = ""        # router catalog key
    provider: str = ""           # service name
    choices: list[GatewayChoice] = field(default_factory=list)
    usage: GatewayUsage | None = None
    finish_reason: str | None = None
    error: str | None = None

    # Timing
    routing_latency_ms: float = 0.0
    provider_latency_ms: float = 0.0
