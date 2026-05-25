"""OllamaProvider — local inference via Ollama REST API.

Endpoint: http://localhost:11434 (configurable).
No API key required.
"""

from __future__ import annotations

from core.config_loader import ProviderConfig
from core.models import ModelResponse
from providers.base import ModelProviderPort


class OllamaProvider(ModelProviderPort):
    """Provider for Ollama (local models)."""

    def __init__(self, config: ProviderConfig) -> None:
        self._base_url = (config.base_url or "http://localhost:11434").rstrip("/")

    @property
    def name(self) -> str:
        return "ollama"

    async def complete(
        self,
        prompt: str,
        model: str,
        temperature: float = 0.7,
        timeout: int = 30,
        system_prompt: str | None = None,
    ) -> ModelResponse:
        payload: dict = {
            "model": model,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": temperature},
        }
        if system_prompt:
            payload["system"] = system_prompt

        t0 = self._timer()
        data = await self._post(
            f"{self._base_url}/api/generate", payload, timeout=timeout
        )
        latency = (self._timer() - t0) * 1000

        return ModelResponse(
            content=data["response"],
            model=model,
            provider=self.name,
            tokens_used=data.get("eval_count"),
            latency_ms=latency,
        )

    async def health_check(self) -> bool:
        try:
            await self._get(f"{self._base_url}/api/tags")
            return True
        except Exception:
            return False

    async def list_models(self) -> list[str]:
        data = await self._get(f"{self._base_url}/api/tags")
        return [m["name"] for m in data.get("models", [])]
