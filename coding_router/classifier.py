"""Local GGUF-based classifier using llama-cpp-python.

Replaces the vLLM HTTP client with direct local inference via a quantised
GGUF model.  The classifier categorises a coding request into one of
five routes used by the router's scoring pipeline.

Supports two backends:
  1. **Local GGUF** (default) — runs via llama-cpp-python, no server needed.
  2. **Remote vLLM** — connects to an OpenAI-compatible server (advanced use).
"""

from __future__ import annotations

import ast
import json
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any

from .quantize import ensure_model, llama_cpp_kwargs

# ---------------------------------------------------------------------------
# Route definitions
# ---------------------------------------------------------------------------

ROUTES = [
    {
        "name": "code_generation",
        "description": (
            "Generating new code snippets, functions, or boilerplate "
            "based on user prompts or requirements"
        ),
    },
    {
        "name": "bug_fixing",
        "description": (
            "Identifying and fixing errors or bugs in provided code "
            "across programming languages"
        ),
    },
    {
        "name": "performance_optimization",
        "description": (
            "Suggesting improvements to make code more efficient, "
            "readable, or scalable"
        ),
    },
    {
        "name": "api_help",
        "description": (
            "Assisting with understanding or integrating external "
            "APIs and libraries"
        ),
    },
    {
        "name": "programming",
        "description": (
            "Answering general programming questions, theory, or "
            "best practices"
        ),
    },
]
VALID_ROUTE_NAMES = frozenset(route["name"] for route in ROUTES)

TASK_INSTRUCTION = """
You are a helpful assistant designed to find the best suited route.
You are provided with route description within <routes></routes> XML tags:
<routes>
{routes}
</routes>
<conversation>
{conversation}
</conversation>
"""

FORMAT_PROMPT = """
Your task is to decide which route best suits the user's latest intent. Follow these rules:
1. If the latest intent is irrelevant or already fulfilled, return {"route": "other"}.
2. Analyze the route descriptions and find the best match.
3. Respond only with the exact JSON object {"route": "route_name"}.
"""


class ClassifierError(RuntimeError):
    pass


@dataclass(frozen=True)
class Classification:
    category: str
    raw_response: str
    model: str


# ---------------------------------------------------------------------------
# Response parsing
# ---------------------------------------------------------------------------


def parse_route(content: str) -> str | None:
    """Accept strict JSON and the single-quoted dict emitted by this model."""
    candidates = [content.strip()]
    start, end = content.find("{"), content.rfind("}")
    if start >= 0 and end > start:
        candidates.append(content[start : end + 1])
    for candidate in candidates:
        for parser in (json.loads, ast.literal_eval):
            try:
                value = parser(candidate)
                if isinstance(value, dict) and isinstance(value.get("route"), str):
                    return value["route"]
            except (ValueError, SyntaxError, json.JSONDecodeError):
                continue
    return None


# ---------------------------------------------------------------------------
# Local GGUF classifier (default)
# ---------------------------------------------------------------------------


class LocalClassifier:
    """Classify queries using the local GGUF model via llama-cpp-python.

    The model is downloaded automatically on first use (~600 MB, one-time).
    Inference runs on CPU by default, with Metal acceleration on macOS.
    """

    def __init__(
        self,
        model_path: str | None = None,
        n_ctx: int = 2048,
        n_threads: int | None = None,
        **extra_kwargs: Any,
    ) -> None:
        self._model_path = model_path
        self._n_ctx = n_ctx
        self._n_threads = n_threads
        self._extra_kwargs = extra_kwargs
        self._llm: Any = None

    def _load(self) -> Any:
        """Lazy-load the GGUF model on first classify() call."""
        if self._llm is not None:
            return self._llm

        try:
            from llama_cpp import Llama
        except ImportError as error:
            raise RuntimeError(
                "llama-cpp-python is required for local classification.  "
                "Install it with:  pip install llama-cpp-python"
            ) from error

        path = self._model_path or str(ensure_model())

        kwargs = llama_cpp_kwargs()
        kwargs["model_path"] = path
        kwargs["n_ctx"] = self._n_ctx
        kwargs["chat_format"] = "chatml"
        if self._n_threads is not None:
            kwargs["n_threads"] = self._n_threads
        kwargs.update(self._extra_kwargs)

        self._llm = Llama(**kwargs)
        return self._llm

    def classify(self, query: str) -> Classification:
        """Classify a coding query into one of the predefined routes."""
        llm = self._load()

        # Truncate query to avoid context window limits (e.g., Copilot context)
        max_len = 1500
        if len(query) > max_len:
            query = query[: max_len // 2] + "\n...[truncated]...\n" + query[-max_len // 2 :]

        prompt = TASK_INSTRUCTION.format(
            routes=json.dumps(ROUTES),
            conversation=json.dumps([{"role": "user", "content": query}]),
        ) + FORMAT_PROMPT

        result = llm.create_chat_completion(
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=100,
        )

        try:
            raw_response = result["choices"][0]["message"]["content"].strip()
        except (KeyError, IndexError, AttributeError) as error:
            raise ClassifierError(
                f"Unexpected llama.cpp response: {result}"
            ) from error

        category = parse_route(raw_response)
        if category not in VALID_ROUTE_NAMES:
            category = "programming"

        model_name = self._model_path or "adapted-arch-router-1.5B-Q4_K_M"
        return Classification(
            category=category,
            raw_response=raw_response,
            model=model_name,
        )


# ---------------------------------------------------------------------------
# Remote vLLM classifier (advanced / optional)
# ---------------------------------------------------------------------------


def _request_json(url: str, payload: dict[str, Any] | None, timeout: int) -> Any:
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8") if payload is not None else None,
        headers={"Content-Type": "application/json"},
        method="POST" if payload is not None else "GET",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.load(response)
    except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError) as error:
        raise ClassifierError(str(error)) from error


class VLLMClassifier:
    """Connect to an external vLLM server for classification.

    Use this when you already have a vLLM instance serving the classifier
    model and prefer to avoid loading the GGUF locally.
    """

    def __init__(
        self,
        base_url: str = "http://127.0.0.1:8080",
        model: str | None = None,
        timeout: int = 30,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = timeout

    def _model_id(self) -> str:
        if self.model:
            return self.model
        models = _request_json(
            f"{self.base_url}/models", None, self.timeout
        ).get("data", [])
        if not models or not models[0].get("id"):
            raise ClassifierError(
                "No model was returned from the vLLM /models endpoint."
            )
        return models[0]["id"]

    def classify(self, query: str) -> Classification:
        prompt = (
            TASK_INSTRUCTION.format(
                routes=json.dumps(ROUTES),
                conversation=json.dumps(
                    [{"role": "user", "content": query}]
                ),
            )
            + FORMAT_PROMPT
        )
        model = self._model_id()
        result = _request_json(
            f"{self.base_url}/chat/completions",
            {
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0,
                "max_tokens": 100,
            },
            self.timeout,
        )
        try:
            raw_response = result["choices"][0]["message"]["content"].strip()
        except (KeyError, IndexError, AttributeError) as error:
            raise ClassifierError(
                f"Unexpected vLLM completion response: {result}"
            ) from error
        category = parse_route(raw_response)
        if category not in VALID_ROUTE_NAMES:
            category = "programming"
        return Classification(
            category=category, raw_response=raw_response, model=model
        )
