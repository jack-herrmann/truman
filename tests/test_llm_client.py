"""Tests for LLMClient (provider wiring, NeMo config)."""
from __future__ import annotations

import os
from unittest.mock import patch

import pytest

from intuition.llm.client import LLMClient


def test_nemo_provider_requires_base_url():
    """NeMo provider raises ValueError when NIM_PROXY_BASE_URL and base_url are missing."""
    client = LLMClient(provider="nemo", model="meta/llama-3.1-8b-instruct")
    # Remove NIM_PROXY_BASE_URL so we test the missing-config path
    with patch.dict(os.environ, {"NIM_PROXY_BASE_URL": ""}, clear=False):
        with pytest.raises(ValueError, match="NIM_PROXY_BASE_URL|base_url"):
            client._ensure_client()


def test_nemo_provider_uses_base_url_kwarg():
    """NeMo provider uses base_url passed to constructor."""
    with patch("intuition.llm.client.openai") as mock_openai:
        mock_openai.AsyncOpenAI = lambda **kw: None
        client = LLMClient(
            provider="nemo",
            model="meta/llama-3.1-8b-instruct",
            base_url="http://nim.local:8000",
        )
        with patch.dict(os.environ, {"NIM_PROXY_BASE_URL": ""}, clear=False):
            client._ensure_client()
        mock_openai.AsyncOpenAI.assert_called_once()
        call_kw = mock_openai.AsyncOpenAI.call_args[1]
        assert call_kw["base_url"] == "http://nim.local:8000/v1"


def test_nemo_provider_uses_env_when_base_url_not_passed():
    """NeMo provider uses NIM_PROXY_BASE_URL when base_url kwarg not passed."""
    with patch("intuition.llm.client.openai") as mock_openai:
        mock_openai.AsyncOpenAI = lambda **kw: None
        client = LLMClient(provider="nemo", model="meta/llama-3.1-8b-instruct")
        with patch.dict(os.environ, {"NIM_PROXY_BASE_URL": "http://env-nim:9000"}, clear=False):
            client._ensure_client()
        call_kw = mock_openai.AsyncOpenAI.call_args[1]
        assert call_kw["base_url"] == "http://env-nim:9000/v1"
