"""SSE streaming helpers for the gateway."""

from __future__ import annotations

import json
from typing import Any, AsyncIterator


async def sse_wrap(chunks: AsyncIterator[str]) -> AsyncIterator[str]:
    """Wrap raw SSE data lines with the ``data: ...`` prefix + blank line."""
    async for chunk in chunks:
        yield f"data: {chunk}\n\n"
    yield "data: [DONE]\n\n"


def sse_json(obj: dict[str, Any]) -> str:
    """Serialize an object for inclusion in an SSE data line."""
    return json.dumps(obj, separators=(",", ":"))
