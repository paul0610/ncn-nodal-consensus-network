"""Tests for provider API key auditing (audit_provider_keys)."""

from __future__ import annotations

from core.config_loader import (
    Config,
    ProviderConfig,
    ProvidersConfig,
    SwarmConfig,
    SwarmNodeConfig,
)
from providers.factory import audit_provider_keys


class TestAuditProviderKeys:
    def test_ollama_only_is_always_ok(self):
        """Ollama is local — no key needed."""
        cfg = Config(
            swarm=SwarmConfig(
                nodes=[
                    SwarmNodeConfig(
                        provider="ollama", model="llama3.2:1b",
                        count=1, role="extractor",
                    ),
                ],
            ),
        )
        audit = audit_provider_keys(cfg)
        assert len(audit) == 1
        assert audit[0]["provider"] == "ollama"
        assert audit[0]["required_key"] is False
        assert audit[0]["ok"] is True

    def test_deepseek_without_key_is_flagged(self):
        """DeepSeek requires a key; audit should flag when missing."""
        cfg = Config(
            providers=ProvidersConfig(
                deepseek=ProviderConfig(base_url="https://api.deepseek.com", api_key=""),
            ),
            swarm=SwarmConfig(
                nodes=[
                    SwarmNodeConfig(
                        provider="deepseek", model="deepseek-chat",
                        count=1, role="synthesizer",
                    ),
                ],
            ),
        )
        audit = audit_provider_keys(cfg)
        assert len(audit) == 1
        assert audit[0]["provider"] == "deepseek"
        assert audit[0]["required_key"] is True
        assert audit[0]["has_key"] is False
        assert audit[0]["ok"] is False
        assert audit[0]["env_var_hint"] == "DEEPSEEK_API_KEY"
        assert "synthesizer" in audit[0]["used_by_roles"]

    def test_deepseek_with_key_is_ok(self):
        cfg = Config(
            providers=ProvidersConfig(
                deepseek=ProviderConfig(
                    base_url="https://api.deepseek.com",
                    api_key="sk-test-1234",
                ),
            ),
            swarm=SwarmConfig(
                nodes=[
                    SwarmNodeConfig(
                        provider="deepseek", model="deepseek-chat",
                        count=1, role="synthesizer",
                    ),
                ],
            ),
        )
        audit = audit_provider_keys(cfg)
        assert audit[0]["has_key"] is True
        assert audit[0]["ok"] is True

    def test_only_used_providers_are_audited(self):
        """Providers not used by any swarm node should not appear."""
        cfg = Config(
            providers=ProvidersConfig(
                deepseek=ProviderConfig(api_key=""),  # NOT used
                qwen=ProviderConfig(api_key=""),       # NOT used
            ),
            swarm=SwarmConfig(
                nodes=[
                    SwarmNodeConfig(
                        provider="ollama", model="llama3.2:1b",
                        count=1, role="extractor",
                    ),
                ],
            ),
        )
        audit = audit_provider_keys(cfg)
        providers_in_audit = {a["provider"] for a in audit}
        assert providers_in_audit == {"ollama"}

    def test_mixed_providers_some_missing(self):
        """Multiple providers, some with keys some without — sort missing first."""
        cfg = Config(
            providers=ProvidersConfig(
                deepseek=ProviderConfig(api_key="sk-valid"),
                together=ProviderConfig(api_key=""),  # missing
            ),
            swarm=SwarmConfig(
                nodes=[
                    SwarmNodeConfig(
                        provider="deepseek", model="deepseek-chat",
                        count=1, role="synthesizer",
                    ),
                    SwarmNodeConfig(
                        provider="together", model="llama-3.3",
                        count=1, role="extractor",
                    ),
                    SwarmNodeConfig(
                        provider="ollama", model="llama3.2:1b",
                        count=1, role="critic",
                    ),
                ],
            ),
        )
        audit = audit_provider_keys(cfg)
        providers = [a["provider"] for a in audit]
        # "together" has missing key, should come first
        assert providers[0] == "together"
        assert audit[0]["ok"] is False
        # All three providers present
        assert {"together", "deepseek", "ollama"} == set(providers)

    def test_role_aggregation_when_provider_reused(self):
        """When the same provider serves multiple roles, collect all roles."""
        cfg = Config(
            providers=ProvidersConfig(
                deepseek=ProviderConfig(api_key=""),
            ),
            swarm=SwarmConfig(
                nodes=[
                    SwarmNodeConfig(
                        provider="deepseek", model="deepseek-chat",
                        count=1, role="extractor",
                    ),
                    SwarmNodeConfig(
                        provider="deepseek", model="deepseek-chat",
                        count=1, role="synthesizer",
                    ),
                ],
            ),
        )
        audit = audit_provider_keys(cfg)
        assert len(audit) == 1
        assert set(audit[0]["used_by_roles"]) == {"extractor", "synthesizer"}


# Note: endpoint test coverage lives in test_interfaces.py when an api_client
# fixture is available. The unit tests above cover the core audit logic.
