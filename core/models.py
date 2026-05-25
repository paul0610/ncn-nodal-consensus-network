"""NCN shared Pydantic models."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from enum import Enum
from uuid import uuid4

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class ClaimStatus(str, Enum):
    PENDING = "pending"
    VERIFIED = "verified"
    UNCERTAIN = "uncertain"
    DISCARDED = "discarded"


class NodeRole(str, Enum):
    EXTRACTOR = "extractor"
    CRITIC = "critic"
    TENTH_MAN = "tenth_man"
    JUDGE = "judge"
    SYNTHESIZER = "synthesizer"
    PLANNER = "planner"


class SourceAuthority(str, Enum):
    """Trust hierarchy — higher value = more authoritative."""
    USER_DIRECT = "user_direct"      # user typed / ingested directly
    COACH_MODEL = "coach_model"      # DeepSeek or large model verification
    CONSENSUS = "consensus"          # multi-model agreement
    WEB_VERIFIED = "web_verified"    # internet source verified by consensus
    WEB_RAW = "web_raw"              # internet source not yet verified
    SLM_SINGLE = "slm_single"       # single small model claim (lowest)


# Ordered from highest to lowest for rank comparison
SOURCE_AUTHORITY_RANK: list[str] = [
    "user_direct", "coach_model", "consensus",
    "web_verified", "web_raw", "slm_single",
]


def authority_rank(auth: str) -> int:
    """Return rank (0 = highest, 5 = lowest). Unknown defaults to lowest."""
    try:
        return SOURCE_AUTHORITY_RANK.index(auth)
    except ValueError:
        return len(SOURCE_AUTHORITY_RANK) - 1


# ---------------------------------------------------------------------------
# Graph models
# ---------------------------------------------------------------------------

class GraphNode(BaseModel):
    node_id: str = Field(default_factory=lambda: str(uuid4()))
    name: str
    node_type: str
    aliases: list[str] = []
    namespace: str = "general"
    confidence: float = 1.0
    temperature: float = 0.5
    embedding: list[float] | None = None
    source: str | None = None
    source_authority: str = "slm_single"
    verified_by: list[str] = []
    uncertain: bool = False
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    last_accessed: datetime = Field(default_factory=lambda: datetime.now(UTC))
    metadata: dict = {}


class GraphRelation(BaseModel):
    relation_id: str = Field(default_factory=lambda: str(uuid4()))
    source_node_id: str
    target_node_id: str
    predicate: str
    weight: float = 1.0
    confidence: float = 1.0
    temporal: bool = False
    date: str | None = None
    contested: bool = False
    source: str | None = None
    source_authority: str = "slm_single"
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


class GraphStats(BaseModel):
    total_nodes: int = 0
    total_relations: int = 0
    namespaces: list[str] = []
    node_types: list[str] = []


# ---------------------------------------------------------------------------
# Consensus models
# ---------------------------------------------------------------------------

class Claim(BaseModel):
    claim_id: str = Field(default_factory=lambda: str(uuid4()))
    subject: str
    predicate: str
    object: str
    confidence: float
    temporal: bool = False
    date: str | None = None
    source_text: str | None = None
    source_model: str = ""
    source_authority: SourceAuthority = SourceAuthority.SLM_SINGLE
    status: ClaimStatus = ClaimStatus.PENDING
    is_counter_claim: bool = False
    challenges_claim_id: str | None = None


class Vote(BaseModel):
    node_id: str
    claim_id: str
    vote: bool
    confidence: float
    reason: str | None = None


class CorrectionEvent(BaseModel):
    """Tracks when a high-authority source corrects a low-authority node."""
    event_id: str = Field(default_factory=lambda: str(uuid4()))
    corrected_node_id: str
    corrected_node_name: str
    old_confidence: float
    new_confidence: float
    correcting_authority: str
    corrected_authority: str
    correcting_claim_id: str = ""
    reason: str = ""
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))


class ConsensusResult(BaseModel):
    verified_claims: list[Claim] = []
    uncertain_claims: list[Claim] = []
    discarded_claims: list[Claim] = []
    synthesized_answer: str = ""
    model_performances: dict[str, float] = {}
    session_id: str = Field(default_factory=lambda: str(uuid4()))
    processing_time_ms: float = 0.0
    corrections: list[CorrectionEvent] = []


# ---------------------------------------------------------------------------
# Context models
# ---------------------------------------------------------------------------

class GraphContext(BaseModel):
    nodes: list[GraphNode] = []
    serialized_context: str = ""
    avg_confidence: float = 0.0
    nodes_used: list[str] = []
    # Relevance Gate: maximum cosine similarity between query and retrieved nodes
    # 0.0 = no relevant nodes found (possible cross-contamination risk)
    max_relevance: float = 0.0


class TokenMetrics(BaseModel):
    """Token usage comparison: NCN graph context vs hypothetical .md approach."""
    context_tokens: int = 0
    context_chars: int = 0
    nodes_retrieved: int = 0
    estimated_md_tokens: int = 0
    reduction_pct: float = 0.0


class QueryResponse(BaseModel):
    answer: str
    claims: list[Claim] = []
    nodes_used: list[str] = []
    nodes_created: list[str] = []
    token_metrics: TokenMetrics | None = None


class BootstrapResult(BaseModel):
    topics_processed: int = 0
    claims_stored: int = 0


class RawContent(BaseModel):
    text: str
    source: str
    source_type: str


class NodeResponse(BaseModel):
    node_id: str
    role: str
    content: str
    model: str
    provider: str


class ModelResponse(BaseModel):
    content: str
    model: str
    provider: str
    tokens_used: int | None = None
    latency_ms: float | None = None


# ---------------------------------------------------------------------------
# Agent models (Phase 2)
# ---------------------------------------------------------------------------

class ActionStep(BaseModel):
    """One step in an action plan."""
    step_id: str = Field(default_factory=lambda: str(uuid4()))
    action_type: str          # "shell", "file_read", "file_write", "skill"
    command: str
    description: str = ""
    depends_on: list[str] = []
    timeout: int = 30


class ActionResult(BaseModel):
    """Result of executing one action step."""
    step_id: str = ""
    action_type: str = ""
    command: str = ""
    exit_code: int = 0
    stdout: str = ""
    stderr: str = ""
    success: bool = True
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))
    duration_ms: float = 0.0


class ActionPlan(BaseModel):
    """A full plan of action steps to execute."""
    plan_id: str = Field(default_factory=lambda: str(uuid4()))
    goal: str = ""
    steps: list[ActionStep] = []
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


class AgentResponse(BaseModel):
    """Full response from the agent after plan-verify-execute cycle."""
    goal: str = ""
    plan: ActionPlan = ActionPlan()
    results: list[ActionResult] = []
    summary: str = ""
    success: bool = True
