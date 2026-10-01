"""Async wrapper around the OpenAI-compatible chat completions API."""
from __future__ import annotations

from openai import (
    APIConnectionError,
    APIStatusError,
    APITimeoutError,
    AsyncOpenAI,
    BadRequestError,
)

from src.core.config import Settings


class LLMError(Exception):
    """Transport/backend failure with a short, UI-friendly message."""


class LLMClient:
    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._client = AsyncOpenAI(
            base_url=settings.base_url,
            api_key=settings.api_key,
            timeout=settings.timeout_s,
            max_retries=settings.max_retries,
        )

    async def complete(
        self,
        *,
        system: str,
        user: str,
        temperature: float,
        model: str | None = None,
    ) -> str:
        """Return the raw text of a single chat completion."""
        kwargs = dict(
            model=model or self._settings.model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            temperature=temperature,
            max_tokens=self._settings.max_tokens,
        )
        try:
            try:
                response = await self._client.chat.completions.create(
                    **kwargs, response_format={"type": "json_object"}
                )
            except BadRequestError:
                response = await self._client.chat.completions.create(**kwargs)
        except APITimeoutError as exc:
            raise LLMError("TIMEOUT: NO RESPONSE FROM NODE") from exc
        except APIConnectionError as exc:
            raise LLMError(f"LINK FAILURE: {self._settings.base_url}") from exc
        except APIStatusError as exc:
            raise LLMError(f"BACKEND ERROR: HTTP {exc.status_code}") from exc
        except Exception as exc:  # noqa: BLE001 - never let transport errors escape
            raise LLMError(f"UNEXPECTED FAULT: {type(exc).__name__}") from exc

        content = response.choices[0].message.content if response.choices else None
        if not content or not content.strip():
            raise LLMError("EMPTY RESPONSE FROM NODE")
        return content

    async def aclose(self) -> None:
        await self._client.close()
