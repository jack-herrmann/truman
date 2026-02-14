"""LLMClient — unified async interface for text generation."""

from __future__ import annotations

import json
import logging
import os
from typing import Any, TypeVar

from pydantic import BaseModel

logger = logging.getLogger(__name__)
T = TypeVar("T", bound=BaseModel)


class LLMClient:
    """Unified async LLM client with structured-output support.

    Supported providers:
      - "anthropic"  (paid — requires ANTHROPIC_API_KEY)
      - "openai"     (paid — requires OPENAI_API_KEY)
      - "gemini"     (free tier — requires GEMINI_API_KEY)
      - "groq"       (free tier — requires GROQ_API_KEY, OpenAI-compatible)
    """

    # Default models per provider
    DEFAULT_MODELS: dict[str, str] = {
        "anthropic": "claude-sonnet-4-20250514",
        "openai": "gpt-4o",
        "gemini": "gemini-2.0-flash",
        "groq": "llama-3.3-70b-versatile",
    }

    def __init__(self, provider: str = "anthropic", model: str | None = None,
                 max_tokens: int = 4096, temperature: float = 0.7, **kwargs: Any) -> None:
        self.provider = provider
        self.model = model or self.DEFAULT_MODELS.get(provider, "claude-sonnet-4-20250514")
        self.max_tokens = max_tokens
        self.temperature = temperature
        self._client: Any = None

    # ------------------------------------------------------------------
    # Client initialisation
    # ------------------------------------------------------------------

    def _ensure_client(self) -> None:
        if self._client is not None:
            return
        if self.provider == "anthropic":
            import anthropic
            self._client = anthropic.AsyncAnthropic()
        elif self.provider == "openai":
            import openai
            self._client = openai.AsyncOpenAI()
        elif self.provider == "gemini":
            from google import genai
            api_key = os.environ.get("GEMINI_API_KEY", "")
            self._client = genai.Client(api_key=api_key)
        elif self.provider == "groq":
            # Groq exposes an OpenAI-compatible API
            import openai
            self._client = openai.AsyncOpenAI(
                base_url="https://api.groq.com/openai/v1",
                api_key=os.environ.get("GROQ_API_KEY", ""),
            )
        else:
            raise ValueError(f"Unsupported LLM provider: {self.provider}")

    # ------------------------------------------------------------------
    # Text generation
    # ------------------------------------------------------------------

    async def generate(self, system: str, messages: list[dict[str, str]],
                       temperature: float | None = None, max_tokens: int | None = None) -> str:
        self._ensure_client()
        temp = temperature if temperature is not None else self.temperature
        tokens = max_tokens if max_tokens is not None else self.max_tokens

        if self.provider == "anthropic":
            response = await self._client.messages.create(
                model=self.model, max_tokens=tokens, system=system,
                messages=messages, temperature=temp)
            return response.content[0].text

        if self.provider == "gemini":
            return await self._generate_gemini(system, messages, temp, tokens)

        # OpenAI and Groq share the same code path
        oai_messages = [{"role": "system", "content": system}] + messages
        response = await self._client.chat.completions.create(
            model=self.model, messages=oai_messages, temperature=temp, max_tokens=tokens)
        return response.choices[0].message.content

    async def _generate_gemini(self, system: str, messages: list[dict[str, str]],
                               temperature: float, max_tokens: int) -> str:
        """Generate text using the Google GenAI SDK (async)."""
        from google.genai import types
        contents = self._messages_to_gemini_contents(messages)
        config = types.GenerateContentConfig(
            system_instruction=system,
            temperature=temperature,
            max_output_tokens=max_tokens,
        )
        response = await self._client.aio.models.generate_content(
            model=self.model, contents=contents, config=config,
        )
        return response.text

    # ------------------------------------------------------------------
    # Structured output
    # ------------------------------------------------------------------

    async def generate_structured(self, system: str, messages: list[dict[str, str]],
                                  response_model: type[T], temperature: float | None = None) -> T:
        self._ensure_client()
        temp = temperature if temperature is not None else 0.4
        schema = response_model.model_json_schema()
        schema = self._inline_refs(schema)

        if self.provider == "anthropic":
            return await self._structured_anthropic(system, messages, response_model, schema, temp)
        if self.provider == "gemini":
            return await self._structured_gemini(system, messages, response_model, schema, temp)
        # OpenAI / Groq
        return await self._structured_openai(system, messages, response_model, schema, temp)

    async def _structured_anthropic(self, system, messages, response_model, schema, temperature):
        tool = {"name": "respond", "description": "Produce the structured response.",
                "input_schema": schema}
        response = await self._client.messages.create(
            model=self.model, max_tokens=self.max_tokens, system=system,
            messages=messages, tools=[tool],
            tool_choice={"type": "tool", "name": "respond"}, temperature=temperature)
        for block in response.content:
            if block.type == "tool_use":
                return response_model.model_validate(block.input)
        raise RuntimeError("No tool_use block in response")

    async def _structured_openai(self, system, messages, response_model, schema, temperature):
        func = {"name": "respond", "description": "Produce the structured response.",
                "parameters": schema}
        oai_messages = [{"role": "system", "content": system}] + messages
        response = await self._client.chat.completions.create(
            model=self.model, messages=oai_messages,
            tools=[{"type": "function", "function": func}],
            tool_choice={"type": "function", "function": {"name": "respond"}},
            temperature=temperature)
        call = response.choices[0].message.tool_calls[0]
        data = json.loads(call.function.arguments)
        return response_model.model_validate(data)

    async def _structured_gemini(self, system, messages, response_model, schema, temperature):
        """Structured output via Gemini's native JSON schema mode."""
        from google.genai import types

        contents = self._messages_to_gemini_contents(messages)
        config = types.GenerateContentConfig(
            system_instruction=system,
            temperature=temperature,
            max_output_tokens=self.max_tokens,
            response_mime_type="application/json",
            response_schema=schema,
        )
        response = await self._client.aio.models.generate_content(
            model=self.model, contents=contents, config=config,
        )
        data = json.loads(response.text)
        return response_model.model_validate(data)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _messages_to_gemini_contents(messages: list[dict[str, str]]) -> list[dict]:
        """Convert OpenAI-style messages to Gemini contents format."""
        contents: list[dict] = []
        for msg in messages:
            role = "user" if msg["role"] == "user" else "model"
            contents.append({"role": role, "parts": [{"text": msg["content"]}]})
        return contents

    @staticmethod
    def _inline_refs(schema: dict) -> dict:
        defs = schema.pop("$defs", {})
        if not defs:
            return schema
        def _resolve(obj):
            if isinstance(obj, dict):
                if "$ref" in obj:
                    ref_name = obj["$ref"].split("/")[-1]
                    return _resolve(defs.get(ref_name, obj))
                return {k: _resolve(v) for k, v in obj.items()}
            if isinstance(obj, list):
                return [_resolve(item) for item in obj]
            return obj
        return _resolve(schema)
