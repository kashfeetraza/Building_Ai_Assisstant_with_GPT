"""Cloud providers that speak the OpenAI Chat Completions protocol.

Covers OpenAI, Google Gemini, xAI Grok and Meta Llama API via their
OpenAI-compatible endpoints, so one implementation serves all four.
"""

from collections.abc import AsyncIterator, Callable

import openai
from openai import AsyncOpenAI
from langchain_openai import ChatOpenAI

from app.providers.base import Provider, ProviderError, to_langchain_messages
from app.schemas import ChatMessage, ModelInfo

ModelFilter = Callable[[str], bool]


def _chat_capable_openai(model_id: str) -> bool:
    blocked = (
        "embedding",
        "whisper",
        "tts",
        "dall-e",
        "image",
        "moderation",
        "audio",
        "realtime",
        "transcribe",
        "search",
        "davinci",
        "babbage",
    )
    return model_id.startswith(("gpt", "o", "chatgpt")) and not any(b in model_id for b in blocked)


def _chat_capable_generic(model_id: str) -> bool:
    blocked = ("embed", "image", "imagen", "veo", "tts", "audio", "vision-only", "aqa")
    return not any(b in model_id.lower() for b in blocked)


class OpenAICompatibleProvider(Provider):
    local = False

    def __init__(
        self,
        provider_id: str,
        label: str,
        api_key: str | None,
        base_url: str,
        *,
        default_model: str = "",
        timeout: float = 120.0,
        model_filter: ModelFilter = _chat_capable_generic,
    ) -> None:
        self.id = provider_id
        self.label = label
        self._default_model = default_model
        self._filter = model_filter
        self._client = (
            AsyncOpenAI(api_key=api_key, base_url=base_url, timeout=timeout, max_retries=1)
            if api_key
            else None
        )

    @property
    def configured(self) -> bool:
        return self._client is not None

    @property
    def default_model(self) -> str:
        return self._default_model

    def _require_client(self) -> AsyncOpenAI:
        if self._client is None:
            raise ProviderError(f"{self.label} is not configured (missing API key)")
        return self._client

    async def list_models(self) -> list[ModelInfo]:
        client = self._require_client()
        try:
            page = await client.models.list()
            ids = [m.id.removeprefix("models/") async for m in page]
        except openai.APIError as exc:
            if self._default_model:  # listing unsupported: still offer the configured model
                ids = [self._default_model]
            else:
                raise ProviderError(f"{self.label}: could not list models ({exc})") from exc

        ids = sorted({i for i in ids if self._filter(i)} | ({self._default_model} - {""}))
        return [ModelInfo(id=i, name=i, provider=self.id, local=False) for i in ids]

    async def stream_chat(
        self,
        model: str,
        messages: list[ChatMessage],
        temperature: float | None = None,
    ) -> AsyncIterator[str]:
        if self._client is None:
            raise ProviderError(f"{self.label} is not configured (missing API key)")
        try:
            chat_model = ChatOpenAI(
                model=model,
                api_key=self._client.api_key,
                base_url=str(self._client.base_url),
                timeout=self._client.timeout or 120.0,
                max_retries=0,
                **({"temperature": temperature} if temperature is not None else {}),
            )
            async for chunk in chat_model.astream(to_langchain_messages(messages)):
                content = chunk.content
                if isinstance(content, str) and content:
                    yield content
        except Exception as exc:
            # Never expose raw SDK/HTTP exception details or credentials.
            raise ProviderError(f"{self.label}: chat request failed") from exc

    async def aclose(self) -> None:
        if self._client is not None:
            await self._client.close()


def openai_model_filter() -> ModelFilter:
    return _chat_capable_openai
