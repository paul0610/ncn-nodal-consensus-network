"""ProviderFactory — creates the right provider from config."""

from __future__ import annotations

from core.config_loader import Config, ProviderConfig
from core.exceptions import ProviderError
from providers.base import ModelProviderPort
from providers.custom import CustomProvider
from providers.deepseek import DeepSeekProvider
from providers.ollama import OllamaProvider
from providers.openai_compatible import OpenAICompatibleProvider
from providers.qwen import QwenProvider
from providers.together import TogetherProvider

PROVIDER_REGISTRY: dict[str, type[ModelProviderPort]] = {
    "ollama": OllamaProvider,
    "openai": OpenAICompatibleProvider,
    "deepseek": DeepSeekProvider,
    "qwen": QwenProvider,
    "together": TogetherProvider,
    "custom": CustomProvider,
}

# Providers that require an API key to authenticate. Ollama runs locally
# and does not need one; `custom` may or may not depending on base_url.
PROVIDERS_REQUIRING_KEY: frozenset[str] = frozenset(
    {"openai", "deepseek", "qwen", "together"}
)

# Suggested environment variable name per provider (for error messages).
ENV_VAR_HINTS: dict[str, str] = {
    "openai": "OPENAI_API_KEY",
    "deepseek": "DEEPSEEK_API_KEY",
    "qwen": "DASHSCOPE_API_KEY",
    "together": "TOGETHER_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
}


class ProviderFactory:
    """Instantiate model providers by name using shared config."""

    def __init__(self, config: Config) -> None:
        self.config = config

    def create(self, provider_name: str) -> ModelProviderPort:
        cls = PROVIDER_REGISTRY.get(provider_name)
        if cls is None:
            raise ProviderError(
                f"Provider '{provider_name}' not recognised. "
                f"Available: {list(PROVIDER_REGISTRY.keys())}"
            )
        provider_config: ProviderConfig = getattr(
            self.config.providers, provider_name, ProviderConfig()
        )
        return cls(provider_config)


# ---------------------------------------------------------------------------
# Key auditing
# ---------------------------------------------------------------------------

def audit_provider_keys(config: Config) -> list[dict]:
    """Audit providers actually used by the swarm and flag missing API keys.

    Returns a list of dicts, one per provider IN USE:

        [{"provider": str, "required_key": bool, "has_key": bool,
          "env_var_hint": str | None, "used_by_roles": list[str],
          "ok": bool}]

    "ok" is False when ``required_key`` is True but ``has_key`` is False —
    meaning the provider is called by the swarm but will fail at runtime.
    """
    # Collect providers referenced by swarm nodes, grouped by role.
    used: dict[str, set[str]] = {}
    for node in config.swarm.nodes:
        used.setdefault(node.provider, set()).add(node.role)

    result: list[dict] = []
    for provider_name, roles in used.items():
        provider_cfg: ProviderConfig | None = getattr(
            config.providers, provider_name, None
        )
        api_key = (provider_cfg.api_key if provider_cfg else None) or ""
        has_key = bool(api_key.strip())
        required = provider_name in PROVIDERS_REQUIRING_KEY

        result.append({
            "provider": provider_name,
            "required_key": required,
            "has_key": has_key,
            "env_var_hint": ENV_VAR_HINTS.get(provider_name),
            "used_by_roles": sorted(roles),
            "ok": (not required) or has_key,
        })

    # Sort: missing keys first (so warnings surface at the top of logs/UI)
    result.sort(key=lambda r: (r["ok"], r["provider"]))
    return result
