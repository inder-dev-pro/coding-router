"""Tests for the LLM Gateway.

These tests start the gateway via FastAPI TestClient and mock the provider
network calls via httpx MockTransport.
"""

from __future__ import annotations

import json
import os
from unittest import mock

import httpx
import pytest
from fastapi.testclient import TestClient

from coding_router.catalog import ModelProfile
from coding_router.gateway.server import create_app
from coding_router.router import CodingRouter


@pytest.fixture(autouse=True)
def mock_router():
    """Mock the router initialization and execution to avoid loading actual models/indexes."""
    with mock.patch("coding_router.gateway.server.init_router"), \
         mock.patch("coding_router.gateway.routing.get_router") as mock_get_router:

        router = mock.Mock(spec=CodingRouter)
        profile_openai = ModelProfile(
            catalog_key="test-openai-model",
            catalog_group="test",
            model="gpt-test-model",
            service="OpenAI",
            api_endpoint="https://api.openai.fake/v1",
            feature="test",
            size="test",
            input_price=0.0,
            output_price=0.0,
            used_in=(),
            api_key_env="OPENAI_API_KEY",
        )
        profile_anthropic = ModelProfile(
            catalog_key="test-anthropic-model",
            catalog_group="test",
            model="claude-test-model",
            service="Anthropic",
            api_endpoint="https://api.anthropic.fake/v1",
            feature="test",
            size="test",
            input_price=0.0,
            output_price=0.0,
            used_in=(),
            api_key_env="ANTHROPIC_API_KEY",
        )
        router.profile_by_key = {
            "test-openai-model": profile_openai,
            "test-anthropic-model": profile_anthropic,
        }
        router.profiles = [profile_openai, profile_anthropic]

        # Default route response
        router.route.return_value = {
            "selected_model": router.profile_by_key["test-openai-model"].public_dict(),
            "classifier_category": "test_category",
            "routing_mode": "mixed",
        }

        mock_get_router.return_value = router
        yield router


@pytest.fixture
def mock_httpx_client():
    """Mock the global httpx client to intercept provider requests."""
    def handler(request: httpx.Request) -> httpx.Response:
        url_str = str(request.url)
        body = json.loads(request.read())

        if "openai.fake" in url_str:
            if body.get("stream"):
                content = "data: {\"id\":\"chatcmpl-test\",\"object\":\"chat.completion.chunk\",\"choices\":[{\"delta\":{\"content\":\"Hello from OpenAI!\"}}]}\n\ndata: [DONE]\n\n"
                return httpx.Response(200, content=content.encode("utf-8"))
            return httpx.Response(200, json={
                "id": "chatcmpl-test",
                "object": "chat.completion",
                "model": "gpt-test-model",
                "choices": [{"index": 0, "message": {"role": "assistant", "content": "Hello from OpenAI!"}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 10, "completion_tokens": 10, "total_tokens": 20}
            })

        if "anthropic.fake" in url_str:
            if body.get("stream"):
                content = (
                    "event: message_start\ndata: {\"type\":\"message_start\",\"message\":{\"id\":\"msg-test\",\"role\":\"assistant\"}}\n\n"
                    "event: content_block_delta\ndata: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"text_delta\",\"text\":\"Hello from Anthropic!\"}}\n\n"
                    "event: message_stop\ndata: {\"type\":\"message_stop\"}\n\n"
                )
                return httpx.Response(200, content=content.encode("utf-8"))
            return httpx.Response(200, json={
                "id": "msg-test",
                "type": "message",
                "role": "assistant",
                "model": "claude-test-model",
                "content": [{"type": "text", "text": "Hello from Anthropic!"}],
                "stop_reason": "end_turn",
                "usage": {"input_tokens": 10, "output_tokens": 10}
            })

        return httpx.Response(404, text="Not Found")

    transport = httpx.MockTransport(handler)
    client = httpx.AsyncClient(transport=transport)

    with mock.patch("coding_router.gateway.providers.base.get_client", return_value=client):
        yield client


@pytest.fixture
def client(mock_httpx_client):
    """Test client for the FastAPI gateway, with auth disabled."""
    with mock.patch.dict(os.environ, {"CODING_ROUTER_AUTH": "false"}):
        app = create_app()
        with TestClient(app) as test_client:
            yield test_client


def test_health(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_models(client):
    response = client.get("/v1/models")
    assert response.status_code == 200
    data = response.json()
    assert data["object"] == "list"
    ids = [m["id"] for m in data["data"]]
    assert "coding-router" in ids
    assert "test-openai-model" in ids


def test_openai_chat_completions_non_streaming(client, mock_router):
    response = client.post(
        "/v1/chat/completions",
        json={"model": "coding-router", "messages": [{"role": "user", "content": "Hi"}], "stream": False},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["choices"][0]["message"]["content"] == "Hello from OpenAI!"
    assert data["model"] == "gpt-test-model"

    # Verify router was called
    mock_router.route.assert_called_once_with("Hi", route_only=True)


def test_openai_chat_completions_streaming(client, mock_router):
    response = client.post(
        "/v1/chat/completions",
        json={"model": "coding-router", "messages": [{"role": "user", "content": "Hi"}], "stream": True},
    )
    assert response.status_code == 200
    assert "text/event-stream" in response.headers["content-type"]
    lines = [line.strip() for line in response.iter_lines() if line.strip()]
    assert any("Hello from OpenAI!" in line for line in lines)
    assert lines[-1] == "data: [DONE]"


def test_anthropic_messages_non_streaming(client, mock_router):
    # Change mock router to return the Anthropic model
    mock_router.route.return_value = {
        "selected_model": mock_router.profile_by_key["test-anthropic-model"].public_dict(),
        "classifier_category": "test_category",
        "routing_mode": "mixed",
    }
    # Mock anthropic API key presence
    with mock.patch("coding_router.gateway.providers.base._resolve_key", return_value="fake-anthropic-key"):
        response = client.post(
            "/v1/messages",
            json={"model": "coding-router", "messages": [{"role": "user", "content": "Hi"}], "stream": False},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["content"][0]["text"] == "Hello from Anthropic!"
        assert data["model"] == "claude-test-model"


def test_anthropic_messages_streaming(client, mock_router):
    mock_router.route.return_value = {
        "selected_model": mock_router.profile_by_key["test-anthropic-model"].public_dict(),
        "classifier_category": "test_category",
        "routing_mode": "mixed",
    }
    with mock.patch("coding_router.gateway.providers.base._resolve_key", return_value="fake-anthropic-key"):
        response = client.post(
            "/v1/messages",
            json={"model": "coding-router", "messages": [{"role": "user", "content": "Hi"}], "stream": True},
        )
        assert response.status_code == 200
        assert "text/event-stream" in response.headers["content-type"]
        lines = [line.strip() for line in response.iter_lines() if line.strip()]
        assert any("Hello from Anthropic!" in line for line in lines)
        assert any("message_start" in line for line in lines)
