"""ModelProviderPort — abstract interface for model providers.

ALL implementations must inherit from this class.
The provider ``name`` is used for logging and metrics.
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod

import httpx
from loguru import logger

from core.config_loader import ProviderConfig
from core.exceptions import ProviderError
from core.models import ModelResponse


class ModelProviderPort(ABC):
    """Abstract port for LLM provider adapters."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Short provider identifier (e.g. ``'ollama'``, ``'deepseek'``)."""

    @abstractmethod
    async def complete(
        self,
        prompt: str,
        model: str,
        temperature: float = 0.7,
        timeout: int = 30,
        system_prompt: str | None = None,
    ) -> ModelResponse:
        """Generate a completion. Raises ``ProviderError`` on failure."""

    @abstractmethod
    async def health_check(self) -> bool:
        """Return ``True`` if the provider is reachable."""

    @abstractmethod
    async def list_models(self) -> list[str]:
        """Return names of models available on this provider."""

    # ------------------------------------------------------------------
    # HTTP helpers (shared by all concrete providers)
    # ------------------------------------------------------------------

    async def _post(
        self,
        url: str,
        json: dict,
        headers: dict | None = None,
        timeout: int = 30,
    ) -> dict:
        try:
            async with httpx.AsyncClient() as client:
                resp = await client.post(
                    url, json=json, headers=headers, timeout=timeout
                )
                resp.raise_for_status()
                return resp.json()
        except httpx.HTTPStatusError as exc:
            raise ProviderError(
                f"[{self.name}] HTTP {exc.response.status_code}: "
                f"{exc.response.text[:200]}"
            ) from exc
        except httpx.RequestError as exc:
            raise ProviderError(
                f"[{self.name}] Request failed: {exc}"
            ) from exc

    async def _get(
        self,
        url: str,
        headers: dict | None = None,
        timeout: int = 10,
    ) -> dict:
        try:
            async with httpx.AsyncClient() as client:
                resp = await client.get(
                    url, headers=headers, timeout=timeout
                )
                resp.raise_for_status()
                return resp.json()
        except httpx.RequestError as exc:
            raise ProviderError(
                f"[{self.name}] Request failed: {exc}"
            ) from exc

    @staticmethod
    def _timer() -> float:
        """Return a high-resolution timestamp (seconds)."""
        return time.perf_counter()
