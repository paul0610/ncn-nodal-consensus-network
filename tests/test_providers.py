"""Sprint 3 checkpoint — model providers tests.

CHECKPOINT: call Ollama with llama3.2:1b and get a response.
"""

from __future__ import annotations

from unittest.mock import AsyncMock

import httpx
import pytest

from core.config_loader import Config, ProviderConfig
from core.exceptions import ProviderError
from core.models import ModelResponse
from providers.base import ModelProviderPort
from providers.custom import CustomProvider
from providers.deepseek import DeepSeekProvider
from providers.factory import PROVIDER_REGISTRY, ProviderFactory
from providers.ollama import OllamaProvider
from providers.openai_compatible import OpenAICompatibleProvider
from providers.qwen import QwenProvider
from providers.together import TogetherProvider


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def _ollama_config() -> ProviderConfig:
    return ProviderConfig(base_url="http://localhost:11434")


def _oai_config(url: str = "https://api.example.com/v1") -> ProviderConfig:
    return ProviderConfig(base_url=url, api_key="test-key")


def _ollama_reachable() -> bool:
    try:
        r = httpx.get("http://localhost:11434/api/tags", timeout=2)
        return r.status_code == 200
    except Exception:
        return False


SKIP_OLLAMA = not _ollama_reachable()


# ------------------------------------------------------------------
# ModelProviderPort interface
# ------------------------------------------------------------------

class TestProviderInterface:
    def test_cannot_instantiate_abc(self):
        with pytest.raises(TypeError):
            ModelProviderPort()  # type: ignore[abstract]

    def test_ollama_is_provider(self):
        p = OllamaProvider(_ollama_config())
        assert isinstance(p, ModelProviderPort)

    def test_openai_compatible_is_provider(self):
        p = OpenAICompatibleProvider(_oai_config())
        assert isinstance(p, ModelProviderPort)


# ------------------------------------------------------------------
# OllamaProvider (mocked HTTP)
# ------------------------------------------------------------------

class TestOllamaProvider:
    async def test_name(self):
        p = OllamaProvider(_ollama_config())
        assert p.name == "ollama"

    async def test_complete(self):
        p = OllamaProvider(_ollama_config())
        p._post = AsyncMock(return_value={
            "response": "Hello world!",
            "eval_count": 42,
        })

        result = await p.complete("say hello", "llama3.2:1b")

        assert isinstance(result, ModelResponse)
        assert result.content == "Hello world!"
        assert result.model == "llama3.2:1b"
        assert result.provider == "ollama"
        assert result.tokens_used == 42
        assert result.latency_ms is not None

        # Verify correct URL was called
        call_args = p._post.call_args
        assert "/api/generate" in call_args[0][0]

    async def test_complete_with_system_prompt(self):
        p = OllamaProvider(_ollama_config())
        p._post = AsyncMock(return_value={
            "response": "I am an assistant.",
            "eval_count": 10,
        })

        await p.complete(
            "who are you?", "llama3.2:1b",
            system_prompt="You are a helpful assistant.",
        )

        payload = p._post.call_args[0][1]
        assert payload["system"] == "You are a helpful assistant."

    async def test_complete_sends_temperature(self):
        p = OllamaProvider(_ollama_config())
        p._post = AsyncMock(return_value={"response": "ok", "eval_count": 1})

        await p.complete("test", "m", temperature=0.3)

        payload = p._post.call_args[0][1]
        assert payload["options"]["temperature"] == 0.3

    async def test_health_check_ok(self):
        p = OllamaProvider(_ollama_config())
        p._get = AsyncMock(return_value={"models": []})
        assert await p.health_check() is True

    async def test_health_check_fail(self):
        p = OllamaProvider(_ollama_config())
        p._get = AsyncMock(side_effect=ProviderError("unreachable"))
        assert await p.health_check() is False

    async def test_list_models(self):
        p = OllamaProvider(_ollama_config())
        p._get = AsyncMock(return_value={
            "models": [
                {"name": "llama3.2:1b"},
                {"name": "qwen2.5:0.5b"},
            ]
        })

        models = await p.list_models()
        assert "llama3.2:1b" in models
        assert "qwen2.5:0.5b" in models


# ------------------------------------------------------------------
# OpenAICompatibleProvider (mocked HTTP)
# ------------------------------------------------------------------

class TestOpenAICompatible:
    async def test_name(self):
        p = OpenAICompatibleProvider(_oai_config())
        assert p.name == "openai_compatible"

    async def test_complete(self):
        p = OpenAICompatibleProvider(_oai_config())
        p._post = AsyncMock(return_value={
            "choices": [{"message": {"content": "Hi there."}}],
            "usage": {"total_tokens": 25},
        })

        result = await p.complete("hello", "gpt-4")

        assert result.content == "Hi there."
        assert result.tokens_used == 25
        assert result.provider == "openai_compatible"

        # Verify auth header was sent
        call_kwargs = p._post.call_args
        headers = call_kwargs[1]["headers"] if "headers" in call_kwargs[1] else call_kwargs[0][2]
        assert "Bearer test-key" in str(headers)

    async def test_complete_with_system_prompt(self):
        p = OpenAICompatibleProvider(_oai_config())
        p._post = AsyncMock(return_value={
            "choices": [{"message": {"content": "ok"}}],
        })

        await p.complete("hi", "m", system_prompt="Be brief.")

        payload = p._post.call_args[0][1]
        assert payload["messages"][0] == {
            "role": "system", "content": "Be brief."
        }
        assert payload["messages"][1] == {
            "role": "user", "content": "hi"
        }

    async def test_health_check(self):
        p = OpenAICompatibleProvider(_oai_config())
        p._get = AsyncMock(return_value={"data": []})
        assert await p.health_check() is True

    async def test_list_models(self):
        p = OpenAICompatibleProvider(_oai_config())
        p._get = AsyncMock(return_value={
            "data": [{"id": "model-a"}, {"id": "model-b"}]
        })
        models = await p.list_models()
        assert models == ["model-a", "model-b"]


# ------------------------------------------------------------------
# Subclass providers
# ------------------------------------------------------------------

class TestSubclassProviders:
    def test_deepseek_defaults(self):
        p = DeepSeekProvider(ProviderConfig())
        assert p.name == "deepseek"
        assert "deepseek.com" in p._base_url

    def test_qwen_defaults(self):
        p = QwenProvider(ProviderConfig())
        assert p.name == "qwen"
        assert "dashscope" in p._base_url

    def test_together_defaults(self):
        p = TogetherProvider(ProviderConfig())
        assert p.name == "together"
        assert "together.xyz" in p._base_url

    def test_custom_name(self):
        p = CustomProvider(_oai_config("https://my-api.local/v1"))
        assert p.name == "custom"
        assert p._base_url == "https://my-api.local/v1"

    def test_all_are_providers(self):
        for cls in (DeepSeekProvider, QwenProvider, TogetherProvider, CustomProvider):
            p = cls(ProviderConfig(base_url="http://x"))
            assert isinstance(p, ModelProviderPort)

    async def test_deepseek_complete(self):
        p = DeepSeekProvider(ProviderConfig(api_key="dk-test"))
        p._post = AsyncMock(return_value={
            "choices": [{"message": {"content": "deep answer"}}],
            "usage": {"total_tokens": 15},
        })
        result = await p.complete("question", "deepseek-chat")
        assert result.content == "deep answer"
        assert result.provider == "deepseek"


# ------------------------------------------------------------------
# Factory
# ------------------------------------------------------------------

class TestProviderFactory:
    def test_registry_has_expected_providers(self):
        expected = {"ollama", "openai", "deepseek", "qwen", "together", "custom"}
        assert expected.issubset(set(PROVIDER_REGISTRY.keys()))

    def test_create_ollama(self):
        factory = ProviderFactory(Config())
        p = factory.create("ollama")
        assert isinstance(p, OllamaProvider)

    def test_create_deepseek(self):
        factory = ProviderFactory(Config())
        p = factory.create("deepseek")
        assert isinstance(p, DeepSeekProvider)

    def test_create_unknown_raises(self):
        factory = ProviderFactory(Config())
        with pytest.raises(ProviderError, match="not recognised"):
            factory.create("nonexistent")

    def test_create_all_registered(self):
        factory = ProviderFactory(Config())
        for name in PROVIDER_REGISTRY:
            p = factory.create(name)
            assert isinstance(p, ModelProviderPort)


# ------------------------------------------------------------------
# CHECKPOINT — real Ollama call (skipped if Ollama not running)
# ------------------------------------------------------------------

@pytest.mark.skipif(SKIP_OLLAMA, reason="Ollama not running on localhost:11434")
class TestOllamaIntegration:
    async def test_health_check(self):
        p = OllamaProvider(_ollama_config())
        assert await p.health_check() is True

    async def test_list_models(self):
        p = OllamaProvider(_ollama_config())
        models = await p.list_models()
        assert isinstance(models, list)

    async def test_complete_real(self):
        """CHECKPOINT: call Ollama with a model and get a response."""
        p = OllamaProvider(_ollama_config())
        models = await p.list_models()
        if not models:
            pytest.skip("No models pulled in Ollama")

        model = models[0]
        result = await p.complete(
            prompt="Reply with exactly one word: hello",
            model=model,
            temperature=0.1,
            timeout=120,
        )

        assert isinstance(result, ModelResponse)
        assert len(result.content) > 0
        assert result.model == model
        assert result.provider == "ollama"
        assert result.latency_ms > 0
