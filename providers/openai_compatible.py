"""OpenAICompatibleProvider — base for any OpenAI-compatible API.

DeepSeek, Qwen, Together, and custom providers inherit from this class.
Only ``base_url``, ``api_key``, and ``name`` change between them.
"""

from __future__ import annotations

from core.config_loader import ProviderConfig
from core.models import ModelResponse
from providers.base import ModelProviderPort


class OpenAICompatibleProvider(ModelProviderPort):
    """Base provider for OpenAI-compatible ``/chat/completions`` APIs."""

    def __init__(self, config: ProviderConfig) -> None:
        self._base_url = (config.base_url or "").rstrip("/")
        self._api_key = config.api_key or ""

    @property
    def name(self) -> str:
        return "openai_compatible"

    # ------------------------------------------------------------------
    # Core
    # ------------------------------------------------------------------

    async def complete(
        self,
        prompt: str,
        model: str,
        temperature: float = 0.7,
        timeout: int = 30,
        system_prompt: str | None = None,
    ) -> ModelResponse:
        messages: list[dict] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        payload = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
        }

        t0 = self._timer()
        data = await self._post(
            f"{self._base_url}/chat/completions",
            payload,
            headers=self._auth_headers(),
            timeout=timeout,
        )
        latency = (self._timer() - t0) * 1000

        return ModelResponse(
            content=data["choices"][0]["message"]["content"],
            model=model,
            provider=self.name,
            tokens_used=data.get("usage", {}).get("total_tokens"),
            latency_ms=latency,
        )

    async def health_check(self) -> bool:
        try:
            await self._get(
                f"{self._base_url}/models",
                headers=self._auth_headers(),
            )
            return True
        except Exception:
            return False

    async def list_models(self) -> list[str]:
        data = await self._get(
            f"{self._base_url}/models",
            headers=self._auth_headers(),
        )
        return [m["id"] for m in data.get("data", [])]

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _auth_headers(self) -> dict:
        headers: dict = {"Content-Type": "application/json"}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"
        return headers
