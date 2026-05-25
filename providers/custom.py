"""CustomProvider — user-configurable OpenAI-compatible endpoint."""

from __future__ import annotations

from core.config_loader import ProviderConfig
from providers.openai_compatible import OpenAICompatibleProvider


class CustomProvider(OpenAICompatibleProvider):

    def __init__(self, config: ProviderConfig) -> None:
        super().__init__(config)

    @property
    def name(self) -> str:
        return "custom"
