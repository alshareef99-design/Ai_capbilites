from __future__ import annotations

import hashlib
import time
from uuid import uuid4

from app.config import Settings, get_settings
from app.core.evaluation import evaluate_answer, hits_to_sources
from app.core.language import Language, reply_language
from app.core.metrics import UsageMetrics
from app.domain.schemas import (
    AnalysisBrief,
    Capability,
    DecisionBrief,
    DocumentFilters,
    DocumentInfo,
    ExtractionRecord,
    GraphStep,
    HealthResponse,
    IngestResponse,
    MetricsPayload,
    RunRequest,
    RunResponse,
    SearchHit,
    SummaryBrief,
)
from app.graphs.pipeline import run_capability
from app.rag.index import get_index, load_index
from app.rag.ingest import ingest_knowledge, ingest_text
from app.repositories.runs import RunRepository


def ensure_knowledge_loaded() -> None:
    index = get_index()
    if index.documents:
        return
    if load_index():
        return
    ingest_knowledge(reset=True)


def _query_hash(query: str) -> str:
    return hashlib.sha256(query.encode("utf-8")).hexdigest()[:12]


def _to_hits(raw: list[dict]) -> list[SearchHit]:
    hits: list[SearchHit] = []
    for item in raw:
        snippet = " ".join((item.get("text") or "").split())
        hits.append(
            SearchHit(
                rank=int(item.get("rank") or len(hits) + 1),
                title=item.get("title") or item.get("doc_id") or "Untitled",
                snippet=snippet[:280],
                score=float(item.get("score") or 0),
                doc_id=item.get("doc_id") or "unknown",
                chunk_id=item.get("chunk_id") or "unknown",
                department=item.get("department"),
                doc_type=item.get("doc_type"),
                language=item.get("language"),
                source=item.get("source") or "unknown",
            )
        )
    return hits


def run(request: RunRequest, repo: RunRepository | None = None, settings: Settings | None = None) -> RunResponse:
    settings = settings or get_settings()
    ensure_knowledge_loaded()
    language = request.language or reply_language(request.query)
    filters = request.filters or DocumentFilters()
    started = time.perf_counter()
    raw = run_capability(
        request.query,
        capability=request.capability,
        language=language,
        top_k=request.top_k,
        department=filters.department,
        doc_type=filters.doc_type,
        filter_language=filters.language.value if filters.language else None,
        document_ids=filters.document_ids or None,
        extra_text=request.extra_text,
        settings=settings,
    )
    latency_ms = int((time.perf_counter() - started) * 1000)
    tracker = raw.get("_tracker")
    hits_raw = raw.get("graded_hits") or raw.get("hits") or []
    sources = hits_to_sources(hits_raw) if request.include_sources else []
    capability = Capability(raw.get("capability") or request.capability.value)
    resolved_language = Language(raw.get("language") or language.value)
    metrics = UsageMetrics(
        latency_ms=latency_ms,
        retrieval_ms=0,
        generation_ms=0,
        input_tokens=getattr(tracker, "prompt_tokens", 0) if tracker else 0,
        output_tokens=getattr(tracker, "completion_tokens", 0) if tracker else 0,
        chunks_retrieved=len(hits_raw),
        language=resolved_language.value,
        extra={
            "node_timings_ms": raw.get("node_timings_ms") or {},
            "demo_mode": settings.demo_mode,
        },
    )
    metrics.estimate_cost()

    answer = raw.get("answer") or ""
    evaluation = None
    if request.evaluate:
        evaluation = evaluate_answer(request.query, answer, sources, resolved_language)

    response = RunResponse(
        id=str(uuid4()),
        capability=capability,
        language=resolved_language,
        answer=answer,
        hits=_to_hits(hits_raw) if capability == Capability.SEARCH or request.include_sources else [],
        summary=SummaryBrief.model_validate(raw["summary"]) if raw.get("summary") else None,
        analysis=AnalysisBrief.model_validate(raw["analysis"]) if raw.get("analysis") else None,
        extraction=ExtractionRecord.model_validate(raw["extraction"]) if raw.get("extraction") else None,
        decision=DecisionBrief.model_validate(raw["decision"]) if raw.get("decision") else None,
        sources=sources,
        steps=[GraphStep(**step) for step in raw.get("steps") or []],
        metrics=MetricsPayload(
            latency_ms=metrics.latency_ms,
            retrieval_ms=metrics.retrieval_ms,
            generation_ms=metrics.generation_ms,
            input_tokens=metrics.input_tokens,
            output_tokens=metrics.output_tokens,
            estimated_cost_usd=metrics.estimated_cost_usd,
            chunks_retrieved=metrics.chunks_retrieved,
            language=metrics.language,
            node_timings_ms=metrics.extra.get("node_timings_ms") or {},
            demo_mode=settings.demo_mode,
        ),
        evaluation=evaluation,
        refused=bool(raw.get("refused")),
        grounded=bool(raw.get("grounded", True)),
        error=raw.get("error"),
    )
    if repo is not None:
        repo.save(
            response.id,
            response.capability.value,
            response.language.value,
            _query_hash(request.query),
            response.grounded,
            response.refused,
            int(response.metrics.latency_ms),
            response.metrics.input_tokens + response.metrics.output_tokens,
            response.metrics.estimated_cost_usd,
        )
    return response


def ingest(reset: bool = False) -> IngestResponse:
    docs, chunks = ingest_knowledge(reset=reset)
    return IngestResponse(documents_indexed=docs, chunks_indexed=chunks, reset=reset)


def ingest_upload(
    text: str,
    filename: str,
    department: str = "general",
    doc_type: str = "policy",
) -> tuple[str, int]:
    title = filename.rsplit(".", 1)[0].replace("_", " ").replace("-", " ").strip() or "Uploaded document"
    return ingest_text(text, title=title, department=department, doc_type=doc_type, source=filename)


def list_documents() -> list[DocumentInfo]:
    ensure_knowledge_loaded()
    return [
        DocumentInfo(
            id=doc.id,
            title=doc.title,
            department=doc.department,
            doc_type=doc.doc_type,
            language=doc.language,
            source=doc.source,
            chunks=doc.chunks,
        )
        for doc in get_index().list_documents()
    ]


def health() -> HealthResponse:
    settings = get_settings()
    ensure_knowledge_loaded()
    index = get_index()
    return HealthResponse(
        status="ok",
        mode="demo" if settings.demo_mode else "live",
        model=settings.openai_model,
        document_count=len(index.documents),
        chunk_count=len(index.chunks),
        capabilities=[c.value for c in Capability if c != Capability.AUTO] + [Capability.AUTO.value],
    )
