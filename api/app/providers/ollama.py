"""Local models served by Ollama via the official Python SDK."""

from collections.abc import AsyncIterator

import httpx
from langchain_ollama import ChatOllama
from ollama import AsyncClient, ResponseError

from app.providers.base import Provider, ProviderError, to_langchain_messages
from app.schemas import ChatMessage, ModelInfo


class OllamaProvider(Provider):
    id = "ollama"
    label = "Ollama (local)"
    local = True

    def __init__(self, host: str, enabled: bool = True, timeout: float = 120.0) -> None:
        self._host = host
        self._enabled = enabled
        self._client = AsyncClient(host=host, timeout=timeout)

    @property
    def configured(self) -> bool:
        return self._enabled and bool(self._host)

    async def list_models(self) -> list[ModelInfo]:
        try:
            response = await self._client.list()
        except (httpx.HTTPError, ConnectionError, ResponseError) as exc:
            raise ProviderError(f"Ollama is not reachable at {self._host}") from exc

        models: list[ModelInfo] = []
        for item in response.models:
            if not item.model:
                continue
            details = item.details
            # Embedding-only models cannot chat, so keep them out of the dropdown.
            families = (details.families or []) if details else []
            if any("bert" in f for f in families) or "embed" in item.model:
                continue
            models.append(
                ModelInfo(
                    id=item.model,
                    name=item.model.removesuffix(":latest"),
                    provider=self.id,
                    local=True,
                    size_bytes=item.size,
                    parameter_size=details.parameter_size if details else None,
                    family=details.family if details else None,
                )
            )
        return sorted(models, key=lambda m: m.name)

    async def stream_chat(
        self,
        model: str,
        messages: list[ChatMessage],
        temperature: float | None = None,
    ) -> AsyncIterator[str]:
        try:
            chat_model = ChatOllama(
                model=model,
                base_url=self._host,
                temperature=temperature if temperature is not None else 0.7,
            )
            async for chunk in chat_model.astream(to_langchain_messages(messages)):
                content = chunk.content
                if isinstance(content, str) and content:
                    yield content
        except (httpx.HTTPError, ConnectionError, ResponseError) as exc:
            raise ProviderError(f"Ollama is not reachable at {self._host}") from exc
        except Exception as exc:
            # LangChain integrations may wrap transport/model errors differently.
            raise ProviderError("Ollama could not complete the chat request") from exc
