"""DeepSeekProvider — DeepSeek API (OpenAI-compatible)."""

from __future__ import annotations

from core.config_loader import ProviderConfig
from providers.openai_compatible import OpenAICompatibleProvider

_DEFAULT_URL = "https://api.deepseek.com/v1"


class DeepSeekProvider(OpenAICompatibleProvider):

    def __init__(self, config: ProviderConfig) -> None:
        super().__init__(config)
        if not self._base_url:
            self._base_url = _DEFAULT_URL

    @property
    def name(self) -> str:
        return "deepseek"
