"""Server-side Gemma client using OpenRouter's OpenAI-compatible API."""

from __future__ import annotations

import os
from collections.abc import Sequence
from typing import Any

from dotenv import load_dotenv
from openai import AsyncOpenAI, OpenAIError


load_dotenv()


class GemmaConfigurationError(RuntimeError):
    """Raised when required server-side Gemma settings are missing."""


class GemmaProviderError(RuntimeError):
    """Raised when the model provider cannot complete a request."""


def _settings() -> tuple[str, str]:
    api_key = os.getenv("OPENROUTER_API_KEY", "").strip()
    model = os.getenv("GEMMA_MODEL", "").strip()
    if not api_key:
        raise GemmaConfigurationError("OPENROUTER_API_KEY is not configured")
    if not model:
        raise GemmaConfigurationError("GEMMA_MODEL is not configured")
    return api_key, model


async def generate_answer(messages: Sequence[dict[str, Any]]) -> str:
    """Generate one grounded Sol response with the configured Gemma model."""
    api_key, model = _settings()
    client = AsyncOpenAI(
        api_key=api_key,
        base_url="https://openrouter.ai/api/v1",
        default_headers={
            "HTTP-Referer": os.getenv("OPENROUTER_SITE_URL", "http://localhost:3000"),
            "X-Title": os.getenv("OPENROUTER_APP_NAME", "iHeat Sol"),
        },
    )

    try:
        completion = await client.chat.completions.create(
            model=model,
            messages=list(messages),
            temperature=0.2,
            max_tokens=700,
        )
    except OpenAIError as exc:
        raise GemmaProviderError("Gemma request failed") from exc
    finally:
        await client.close()

    answer = completion.choices[0].message.content if completion.choices else None
    if not answer:
        raise GemmaProviderError("Gemma returned an empty response")
    return answer

