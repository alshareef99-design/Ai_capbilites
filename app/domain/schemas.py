from __future__ import annotations

from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator

from app.config import get_settings
from app.core.language import Language


class Capability(str, Enum):
    SEARCH = "search"
    QA = "qa"
    SUMMARIZE = "summarize"
    ANALYZE = "analyze"
    EXTRACT = "extract"
    DECIDE = "decide"
    AUTO = "auto"


class SourceChunk(BaseModel):
    content: str
    source: str
    score: float | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class DocumentFilters(BaseModel):
    department: str | None = Field(default=None, max_length=80)
    language: Language | None = None
    doc_type: str | None = Field(default=None, max_length=80)
    document_ids: list[str] = Field(default_factory=list)


class GraphStep(BaseModel):
    name: str
    detail: str
    status: Literal["ok", "retry", "skip", "warn"] = "ok"


class MetricsPayload(BaseModel):
    latency_ms: float
    retrieval_ms: float = 0.0
    generation_ms: float = 0.0
    input_tokens: int = 0
    output_tokens: int = 0
    estimated_cost_usd: float = 0.0
    chunks_retrieved: int = 0
    language: str = "unknown"
    node_timings_ms: dict[str, int] = Field(default_factory=dict)
    demo_mode: bool = False


class EvaluationScore(BaseModel):
    groundedness: float = Field(ge=0, le=1)
    relevance: float = Field(ge=0, le=1)
    language_quality: float = Field(ge=0, le=1)
    overall: float = Field(ge=0, le=1)
    notes: list[str] = Field(default_factory=list)


class SearchHit(BaseModel):
    rank: int
    title: str
    snippet: str
    score: float
    doc_id: str
    chunk_id: str
    department: str | None = None
    doc_type: str | None = None
    language: str | None = None
    source: str


class RiskItem(BaseModel):
    title: str
    severity: Literal["low", "medium", "high"]
    detail: str
    evidence: str | None = None


class ObligationItem(BaseModel):
    party: str
    obligation: str
    deadline: str | None = None


class Stakeholder(BaseModel):
    name: str
    role: str | None = None


class AnalysisBrief(BaseModel):
    title: str
    overview: str
    themes: list[str] = Field(default_factory=list)
    risks: list[RiskItem] = Field(default_factory=list)
    obligations: list[ObligationItem] = Field(default_factory=list)
    stakeholders: list[Stakeholder] = Field(default_factory=list)
    gaps: list[str] = Field(default_factory=list)
    recommended_next_steps: list[str] = Field(default_factory=list)


class ExtractedField(BaseModel):
    name: str
    value: str
    evidence: str | None = None


class ExtractionRecord(BaseModel):
    title: str
    entities: list[str] = Field(default_factory=list)
    dates: list[str] = Field(default_factory=list)
    amounts: list[str] = Field(default_factory=list)
    clauses: list[str] = Field(default_factory=list)
    fields: list[ExtractedField] = Field(default_factory=list)


class DecisionOption(BaseModel):
    name: str
    summary: str
    pros: list[str] = Field(default_factory=list)
    cons: list[str] = Field(default_factory=list)
    estimated_cost: str | None = None


class DecisionBrief(BaseModel):
    question: str
    recommendation: str
    confidence: float = Field(ge=0, le=1)
    rationale: str
    options: list[DecisionOption] = Field(default_factory=list)
    risks_if_accepted: list[str] = Field(default_factory=list)
    open_questions: list[str] = Field(default_factory=list)


class SummaryBrief(BaseModel):
    title: str
    executive_summary: str
    bullets: list[str] = Field(default_factory=list)
    key_facts: list[str] = Field(default_factory=list)


class RunRequest(BaseModel):
    query: str = Field(..., min_length=1)
    capability: Capability = Capability.AUTO
    language: Language | None = None
    top_k: int | None = Field(default=None, ge=1, le=20)
    filters: DocumentFilters | None = None
    extra_text: str | None = Field(default=None, max_length=24_000)
    evaluate: bool = False
    include_sources: bool = True

    @field_validator("query")
    @classmethod
    def validate_query(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("query must not be empty")
        max_len = get_settings().max_query_length
        if len(cleaned) > max_len:
            raise ValueError(f"query exceeds max length of {max_len}")
        return cleaned


class RunResponse(BaseModel):
    id: str
    capability: Capability
    language: Language
    answer: str
    hits: list[SearchHit] = Field(default_factory=list)
    summary: SummaryBrief | None = None
    analysis: AnalysisBrief | None = None
    extraction: ExtractionRecord | None = None
    decision: DecisionBrief | None = None
    sources: list[SourceChunk] = Field(default_factory=list)
    steps: list[GraphStep] = Field(default_factory=list)
    metrics: MetricsPayload
    evaluation: EvaluationScore | None = None
    refused: bool = False
    grounded: bool = True
    error: str | None = None


class IngestResponse(BaseModel):
    documents_indexed: int
    chunks_indexed: int
    reset: bool = False


class DocumentInfo(BaseModel):
    id: str
    title: str
    department: str
    doc_type: str
    language: str
    source: str
    chunks: int


class HealthResponse(BaseModel):
    status: str
    mode: str
    model: str
    document_count: int
    chunk_count: int
    capabilities: list[str]
