"""LLMClient — unified async interface for text generation."""

from __future__ import annotations

import json
import logging
from typing import Any, TypeVar

from pydantic import BaseModel

logger = logging.getLogger(__name__)
T = TypeVar("T", bound=BaseModel)


class LLMClient:
    """Unified async LLM client with structured-output support."""

    def __init__(self, provider: str = "anthropic", model: str = "claude-sonnet-4-20250514",
                 max_tokens: int = 4096, temperature: float = 0.7, **kwargs: Any) -> None:
        self.provider = provider
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        self._client: Any = None

    def _ensure_client(self) -> None:
        if self._client is not None:
            return
        if self.provider == "anthropic":
            import anthropic
            self._client = anthropic.AsyncAnthropic()
        elif self.provider == "openai":
            import openai
            self._client = openai.AsyncOpenAI()
        else:
            raise ValueError(f"Unsupported LLM provider: {self.provider}")

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
        oai_messages = [{"role": "system", "content": system}] + messages
        response = await self._client.chat.completions.create(
            model=self.model, messages=oai_messages, temperature=temp, max_tokens=tokens)
        return response.choices[0].message.content

    async def generate_structured(self, system: str, messages: list[dict[str, str]],
                                  response_model: type[T], temperature: float | None = None) -> T:
        self._ensure_client()
        temp = temperature if temperature is not None else 0.4
        schema = response_model.model_json_schema()
        schema = self._inline_refs(schema)
        if self.provider == "anthropic":
            return await self._structured_anthropic(system, messages, response_model, schema, temp)
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
