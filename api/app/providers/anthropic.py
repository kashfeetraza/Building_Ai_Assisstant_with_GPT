"""Anthropic Claude via the official SDK."""

from collections.abc import AsyncIterator

import anthropic
from anthropic import AsyncAnthropic
from langchain_anthropic import ChatAnthropic

from app.providers.base import Provider, ProviderError, to_langchain_messages
from app.schemas import ChatMessage, ModelInfo

DEFAULT_MAX_TOKENS = 8192


class AnthropicProvider(Provider):
    id = "anthropic"
    label = "Anthropic Claude"
    local = False

    def __init__(self, api_key: str | None, default_model: str = "", timeout: float = 120.0):
        self._default_model = default_model
        self._client = (
            AsyncAnthropic(api_key=api_key, timeout=timeout, max_retries=1) if api_key else None
        )

    @property
    def configured(self) -> bool:
        return self._client is not None

    @property
    def default_model(self) -> str:
        return self._default_model

    def _require_client(self) -> AsyncAnthropic:
        if self._client is None:
            raise ProviderError(f"{self.label} is not configured (missing API key)")
        return self._client

    async def list_models(self) -> list[ModelInfo]:
        client = self._require_client()
        try:
            models = [m async for m in client.models.list(limit=100)]
        except anthropic.APIError as exc:
            raise ProviderError(f"{self.label}: could not list models ({exc})") from exc
        return [
            ModelInfo(id=m.id, name=m.display_name or m.id, provider=self.id, local=False)
            for m in models
        ]

    async def stream_chat(
        self,
        model: str,
        messages: list[ChatMessage],
        temperature: float | None = None,
    ) -> AsyncIterator[str]:
        client = self._require_client()
        try:
            kwargs: dict = {
                "model": model,
                "api_key": client.api_key,
                "timeout": client.timeout or 120.0,
                "max_retries": 0,
                "max_tokens": DEFAULT_MAX_TOKENS,
            }
            if temperature is not None:
                kwargs["temperature"] = min(temperature, 1.0)
            chat_model = ChatAnthropic(**kwargs)
            async for chunk in chat_model.astream(to_langchain_messages(messages)):
                content = chunk.content
                if isinstance(content, str) and content:
                    yield content
        except Exception as exc:
            # Keep provider-specific details and credentials out of user-facing errors.
            raise ProviderError(f"{self.label}: chat request failed") from exc

    async def aclose(self) -> None:
        if self._client is not None:
            await self._client.close()
