"""NCN configuration loader — reads config.yaml and validates with Pydantic."""

from __future__ import annotations

import os
import re
from pathlib import Path

import yaml
from pydantic import BaseModel

from core.exceptions import ConfigFileNotFoundError, ConfigValidationError


# ---------------------------------------------------------------------------
# Config section models
# ---------------------------------------------------------------------------

class SystemConfig(BaseModel):
    name: str = "NCN"
    version: str = "0.1.0"
    log_level: str = "INFO"
    log_file: str = "logs/ncn.log"
    data_dir: str = "data/"


class CLIConfig(BaseModel):
    enabled: bool = True
    history_file: str = ".ncn_history"


class APIConfig(BaseModel):
    enabled: bool = True
    host: str = "0.0.0.0"
    port: int = 8000
    cors_enabled: bool = True
    cors_origins: list[str] = ["*"]
    api_key_required: bool = False


class InterfaceConfig(BaseModel):
    cli: CLIConfig = CLIConfig()
    api: APIConfig = APIConfig()


class SwarmNodeConfig(BaseModel):
    provider: str
    model: str
    count: int
    role: str
    temperature: float | None = None
    api_key: str | None = None
    terms_accepted: bool = False


class SwarmConfig(BaseModel):
    max_concurrent: int = 10
    request_timeout: int = 30
    retry_attempts: int = 2
    temperature: float = 0.7
    nodes: list[SwarmNodeConfig] = []


class ProviderConfig(BaseModel):
    base_url: str | None = None
    api_key: str | None = None


class ProvidersConfig(BaseModel):
    ollama: ProviderConfig = ProviderConfig(base_url="http://localhost:11434")
    openai: ProviderConfig = ProviderConfig()
    anthropic: ProviderConfig = ProviderConfig()
    deepseek: ProviderConfig = ProviderConfig()
    qwen: ProviderConfig = ProviderConfig()
    together: ProviderConfig = ProviderConfig()
    custom: ProviderConfig = ProviderConfig()


class ConsensusConfig(BaseModel):
    verification_threshold: float = 0.66
    discard_threshold: float = 0.33
    validators_per_claim: int = 5
    auto_judge_uncertain: bool = False
    tenth_man_enabled: bool = True
    tenth_man_triggers_above: float = 0.80


class ReputationConfig(BaseModel):
    enabled: bool = True
    elite_ratio: float = 0.20
    elite_vote_multiplier: float = 2.0
    decay_per_session: float = 0.01
    history_file: str = "data/reputation.json"


class SourceAuthorityConfig(BaseModel):
    enabled: bool = True
    weights: dict[str, float] = {
        "user_direct": 1.00,
        "coach_model": 0.90,
        "consensus": 0.75,
        "web_verified": 0.60,
        "web_raw": 0.45,
        "slm_single": 0.30,
    }
    correction_enabled: bool = True
    authority_gap_for_correction: int = 2
    confidence_penalty_on_correction: float = 0.30
    confidence_bonus_on_high_authority: float = 0.10
    source_type_authority: dict[str, str] = {
        "text": "user_direct",
        "pdf": "user_direct",
        "web_search": "web_raw",
        "wikipedia": "web_raw",
    }
    model_authority: dict[str, str] = {
        "deepseek": "coach_model",
    }


class KuzuConfig(BaseModel):
    path: str = "data/graph/"


class FalkorDBConfig(BaseModel):
    host: str = "localhost"
    port: int = 6379


class Neo4jConfig(BaseModel):
    uri: str = "bolt://localhost:7687"
    user: str = "neo4j"
    password: str | None = None


class CustomGraphConfig(BaseModel):
    connection_string: str | None = None


class GraphConfig(BaseModel):
    adapter: str = "kuzu"
    kuzu: KuzuConfig = KuzuConfig()
    falkordb: FalkorDBConfig = FalkorDBConfig()
    neo4j: Neo4jConfig = Neo4jConfig()
    custom: CustomGraphConfig = CustomGraphConfig()
    node_types: list[str] = [
        "persona", "concepto", "evento", "lugar", "organizacion",
    ]
    namespaces_enabled: bool = True
    default_namespace: str = "general"
    cross_namespace_relations: bool = True
    temperature_boost: float = 0.10
    temperature_decay: float = 0.05
    cold_node_threshold: float = 0.02
    confidence_boost: float = 0.05
    confidence_decay: float = 0.15
    uncertain_threshold: float = 0.30


class NamespaceConfig(BaseModel):
    auto_create: bool = True
    min_nodes_to_create: int = 5


class OntologyConfig(BaseModel):
    auto_extend: bool = True
    min_confidence: float = 0.70
    validators: int = 3
    max_custom_types: int = 100
    dedup_threshold: float = 0.85
    namespace: NamespaceConfig = NamespaceConfig()


class RetrievalConfig(BaseModel):
    model: str = "paraphrase-multilingual-MiniLM-L12-v2"
    top_k_nodes: int = 5
    hop_depth: int = 2
    token_budget: int = 800
    cache_enabled: bool = True
    cache_ttl_minutes: int = 30
    # Relevance Gate: nodes below this cosine similarity are treated
    # as irrelevant to the query. Prevents cross-contamination when
    # the graph has no information about the actual topic.
    # Calibrated for paraphrase-multilingual-MiniLM-L12-v2:
    #   noise floor ~0.49 → threshold 0.55 blocks all noise
    #   real signal ~0.89 → comfortable margin above threshold
    relevance_threshold: float = 0.55


class InternetSourceConfig(BaseModel):
    type: str
    provider: str | None = None
    enabled: bool = True
    languages: list[str] | None = None
    api_key: str | None = None
    tags: list[str] | None = None
    min_score: int | None = None
    categories: list[str] | None = None
    content: list[str] | None = None
    feeds: list[str] | None = None
    urls: list[str] | None = None


class InternetConfig(BaseModel):
    search_mode: str = "auto"
    graph_confidence_threshold: float = 0.60
    cache_enabled: bool = True
    cache_ttl_minutes: int = 60
    sources: list[InternetSourceConfig] = []


class PDFConfig(BaseModel):
    enabled: bool = True
    max_size_mb: int = 50
    extract_images: bool = False


class IngestionConfig(BaseModel):
    pdf: PDFConfig = PDFConfig()
    internet: InternetConfig = InternetConfig()


class TeacherConfig(BaseModel):
    provider: str = "deepseek"
    model: str = "deepseek-chat"
    api_key: str | None = None


class KnowledgeBootstrapConfig(BaseModel):
    enabled: bool = False
    terms_accepted: bool = False
    teacher: TeacherConfig = TeacherConfig()
    topics: list[str] = []
    questions_per_topic: int = 20
    store_verified_only: bool = True


class ContinuousLearningConfig(BaseModel):
    enabled: bool = False
    interval_hours: int = 24
    max_claims_per_session: int = 100
    sources: list[str] = []
    sub_queries_per_topic: int = 5
    max_pages_per_query: int = 3
    scrape_full_pages: bool = True
    rate_limit_seconds: float = 2.0
    page_timeout: int = 15
    max_page_chars: int = 5000


class ExperimentalConfig(BaseModel):
    metrics_enabled: bool = True
    metrics_file: str = "data/metrics.jsonl"
    ab_testing: bool = False
    ab_config_b: str | None = None


class AgentConfig(BaseModel):
    enabled: bool = False
    verify_plan: bool = False
    allow_shell: bool = True
    allow_file_ops: bool = True
    command_timeout: int = 30
    max_output_bytes: int = 65536
    dangerous_patterns: list[str] = [
        "rm -rf", "format ", "mkfs", "dd if=",
        ":(){ :|:& };:", "> /dev/sd",
        "del /f /s /q", "rd /s /q",
    ]
    confirm_dangerous: bool = True
    memory_namespace: str = "agent_memory"
    skill_namespace: str = "skills"
    action_log_namespace: str = "action_logs"


# ---------------------------------------------------------------------------
# Root config
# ---------------------------------------------------------------------------

class Config(BaseModel):
    system: SystemConfig = SystemConfig()
    interface: InterfaceConfig = InterfaceConfig()
    swarm: SwarmConfig = SwarmConfig()
    providers: ProvidersConfig = ProvidersConfig()
    consensus: ConsensusConfig = ConsensusConfig()
    reputation: ReputationConfig = ReputationConfig()
    graph: GraphConfig = GraphConfig()
    ontology: OntologyConfig = OntologyConfig()
    retrieval: RetrievalConfig = RetrievalConfig()
    ingestion: IngestionConfig = IngestionConfig()
    knowledge_bootstrap: KnowledgeBootstrapConfig = KnowledgeBootstrapConfig()
    continuous_learning: ContinuousLearningConfig = ContinuousLearningConfig()
    source_authority: SourceAuthorityConfig = SourceAuthorityConfig()
    agent: AgentConfig = AgentConfig()
    experimental: ExperimentalConfig = ExperimentalConfig()

    @property
    def internet(self) -> InternetConfig:
        """Convenience accessor — mirrors self.ingestion.internet."""
        return self.ingestion.internet


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------

_ENV_VAR_PATTERN = re.compile(r"\$\{(\w+)\}")


def _resolve_env_vars(raw: str) -> str:
    """Replace ``${VAR}`` placeholders with environment values."""

    def _replacer(match: re.Match) -> str:
        value = os.environ.get(match.group(1))
        return value if value is not None else ""

    return _ENV_VAR_PATTERN.sub(_replacer, raw)


def load_config(path: str | Path = "config.yaml") -> Config:
    """Load *config.yaml*, resolve env-vars and validate with Pydantic."""
    path = Path(path)
    if not path.exists():
        raise ConfigFileNotFoundError(f"Config file not found: {path}")

    raw = path.read_text(encoding="utf-8")
    resolved = _resolve_env_vars(raw)
    data = yaml.safe_load(resolved) or {}

    try:
        return Config(**data)
    except Exception as exc:
        raise ConfigValidationError(f"Invalid config: {exc}") from exc
