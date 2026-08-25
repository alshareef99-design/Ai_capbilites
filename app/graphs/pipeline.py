from __future__ import annotations

from typing import Any, Literal, TypedDict

from langgraph.graph import END, START, StateGraph

from app.config import Settings, get_settings
from app.core.language import Language, reply_language
from app.core.safety import is_unsafe, safety_refusal
from app.domain.schemas import (
    AnalysisBrief,
    Capability,
    DecisionBrief,
    ExtractionRecord,
    SummaryBrief,
)
from app.graphs import demo
from app.graphs.llm import UsageTracker, build_chat_model, invoke_structured, invoke_text
from app.prompts.templates import format_context, system_prompt, user_prompt
from app.rag.index import get_index

MAX_STEPS_DETAIL = 240


class GraphState(TypedDict, total=False):
    query: str
    capability: str
    language: str
    top_k: int
    department: str
    doc_type: str
    filter_language: str
    document_ids: list[str]
    extra_text: str
    hits: list[dict]
    graded_hits: list[dict]
    answer: str
    summary: dict
    analysis: dict
    extraction: dict
    decision: dict
    steps: list[dict]
    refused: bool
    grounded: bool
    node_timings_ms: dict[str, int]
    error: str


_CAPABILITY_HINTS: dict[Capability, tuple[str, ...]] = {
    Capability.SEARCH: ("search", "find", "locate", "ابحث", "بحث", "أين أجد"),
    Capability.SUMMARIZE: ("summarize", "summary", "tldr", "لخص", "ملخص", "اختصر"),
    Capability.ANALYZE: ("analyze", "analysis", "risks", "حلل", "تحليل", "المخاطر"),
    Capability.EXTRACT: ("extract", "entities", "fields", "استخرج", "استخراج", "الحقول"),
    Capability.DECIDE: (
        "should we",
        "recommend",
        "decision",
        "vs",
        "or stay",
        "هل يجب",
        "توصية",
        "قرار",
        "أيهما",
    ),
}


def _append_step(state: GraphState, name: str, detail: str, status: str = "ok") -> list[dict]:
    steps = list(state.get("steps") or [])
    steps.append({"name": name, "detail": detail[:MAX_STEPS_DETAIL], "status": status})
    return steps


def classify_capability(query: str, requested: str | None = None) -> Capability:
    if requested and requested != Capability.AUTO.value:
        try:
            return Capability(requested)
        except ValueError:
            pass
    lowered = query.lower()
    scores: dict[Capability, int] = {}
    for capability, hints in _CAPABILITY_HINTS.items():
        scores[capability] = sum(1 for hint in hints if hint in lowered)
    best = max(scores, key=scores.get)
    if scores[best] > 0:
        return best
    if "?" in query or query.strip().startswith(("what", "when", "how", "who", "كم", "ما ", "هل ", "متى")):
        return Capability.QA
    return Capability.SEARCH


def _hits_to_search_answer(query: str, hits: list[dict], language: Language) -> str:
    return demo.demo_search_answer(query, hits, language)


def build_graph(settings: Settings | None = None, tracker: UsageTracker | None = None):
    settings = settings or get_settings()
    tracker = tracker or UsageTracker(model=settings.openai_model)
    llm = None if settings.demo_mode else build_chat_model(settings)

    def safety_node(state: GraphState) -> dict[str, Any]:
        query = state["query"]
        language = Language(state.get("language") or reply_language(query).value)
        if is_unsafe(query) or is_unsafe(state.get("extra_text") or ""):
            return {
                "refused": True,
                "grounded": True,
                "answer": safety_refusal(language),
                "capability": state.get("capability") or Capability.QA.value,
                "language": language.value,
                "hits": [],
                "graded_hits": [],
                "steps": _append_step(state, "Safety", "Request blocked by the safety filter.", "warn"),
            }
        return {
            "refused": False,
            "language": language.value,
            "steps": _append_step(state, "Safety", "Request passed the safety filter."),
        }

    def classify_node(state: GraphState) -> dict[str, Any]:
        if state.get("refused"):
            return {}
        capability = classify_capability(state["query"], state.get("capability"))
        return {
            "capability": capability.value,
            "steps": _append_step(state, "Classify", f"Routed to {capability.value}."),
        }

    def retrieve_node(state: GraphState) -> dict[str, Any]:
        if state.get("refused"):
            return {}
        top_k = int(state.get("top_k") or settings.top_k)
        extra = (state.get("extra_text") or "").strip()
        hits = get_index().search(
            state["query"],
            top_k=top_k,
            department=state.get("department") or None,
            language=state.get("filter_language") or None,
            doc_type=state.get("doc_type") or None,
            document_ids=state.get("document_ids") or None,
        )
        if extra:
            hits = [
                {
                    "rank": 0,
                    "score": 1.0,
                    "text": extra[:4000],
                    "doc_id": "pasted",
                    "chunk_id": "pasted_0",
                    "title": "Pasted text",
                    "department": "adhoc",
                    "doc_type": "note",
                    "language": state.get("language") or "en",
                    "source": "paste",
                }
            ] + hits
            for i, hit in enumerate(hits, start=1):
                hit["rank"] = i
        detail = f"Retrieved {len(hits)} passage{'s' if len(hits) != 1 else ''}."
        return {
            "hits": hits,
            "steps": _append_step(state, "Retrieve", detail),
        }

    def grade_node(state: GraphState) -> dict[str, Any]:
        if state.get("refused"):
            return {}
        hits = list(state.get("hits") or [])
        threshold = settings.retrieval_threshold
        kept = [h for h in hits if (h.get("score") or 0) >= threshold or h.get("doc_id") == "pasted"]
        if not kept and hits:
            kept = hits[:1]
            status = "warn"
            detail = "Weak retrieval — keeping the top passage."
        elif not kept:
            status = "warn"
            detail = "No passages passed the relevance grade."
        else:
            status = "ok"
            detail = f"Kept {len(kept)} of {len(hits)} passages (threshold {threshold})."
        return {
            "graded_hits": kept,
            "steps": _append_step(state, "Grade", detail, status),
        }

    def dispatch_node(state: GraphState) -> dict[str, Any]:
        if state.get("refused"):
            return {}
        capability = Capability(state.get("capability") or Capability.QA.value)
        language = Language(state.get("language") or "en")
        hits = list(state.get("graded_hits") or state.get("hits") or [])
        extra = state.get("extra_text") or ""
        query = state["query"]
        context = format_context(hits, extra_text=None)

        if capability == Capability.SEARCH:
            answer = _hits_to_search_answer(query, hits, language)
            return {
                "answer": answer,
                "grounded": bool(hits),
                "steps": _append_step(state, "Search", f"Ranked {len(hits)} hits."),
            }

        if settings.demo_mode or llm is None:
            return _demo_dispatch(state, capability, language, query, hits, extra)

        system = system_prompt(capability, language)
        user = user_prompt(query, context)
        try:
            if capability == Capability.QA:
                answer = invoke_text(llm, system, user, tracker)
                grounded = bool(hits) and "do not know" not in answer.lower() and "لا أعرف" not in answer
                return {
                    "answer": answer,
                    "grounded": grounded,
                    "steps": _append_step(state, "Generate", "Grounded Q&A answer."),
                }
            if capability == Capability.SUMMARIZE:
                brief = invoke_structured(llm, SummaryBrief, system, user, tracker)
                return {
                    "summary": brief.model_dump(),
                    "answer": brief.executive_summary,
                    "grounded": True,
                    "steps": _append_step(state, "Summarize", "Executive summary produced."),
                }
            if capability == Capability.ANALYZE:
                brief = invoke_structured(llm, AnalysisBrief, system, user, tracker)
                return {
                    "analysis": brief.model_dump(),
                    "answer": brief.overview,
                    "grounded": True,
                    "steps": _append_step(state, "Analyze", "Structured analysis produced."),
                }
            if capability == Capability.EXTRACT:
                record = invoke_structured(llm, ExtractionRecord, system, user, tracker)
                return {
                    "extraction": record.model_dump(),
                    "answer": f"Extracted {len(record.fields)} fields, {len(record.entities)} entities.",
                    "grounded": True,
                    "steps": _append_step(state, "Extract", "Structured fields extracted."),
                }
            brief = invoke_structured(llm, DecisionBrief, system, user, tracker)
            return {
                "decision": brief.model_dump(),
                "answer": brief.recommendation,
                "grounded": True,
                "steps": _append_step(state, "Decide", "Recommendation produced."),
            }
        except Exception as exc:
            fallback = _demo_dispatch(state, capability, language, query, hits, extra)
            fallback["steps"] = _append_step(
                state,
                "Generate",
                f"LLM failed ({exc}); used deterministic fallback.",
                "skip",
            )
            return fallback

    def route_after_safety(state: GraphState) -> Literal["classify", "end"]:
        return "end" if state.get("refused") else "classify"

    graph = StateGraph(GraphState)
    graph.add_node("safety", safety_node)
    graph.add_node("classify", classify_node)
    graph.add_node("retrieve", retrieve_node)
    graph.add_node("grade", grade_node)
    graph.add_node("dispatch", dispatch_node)
    graph.add_edge(START, "safety")
    graph.add_conditional_edges("safety", route_after_safety, {"classify": "classify", "end": END})
    graph.add_edge("classify", "retrieve")
    graph.add_edge("retrieve", "grade")
    graph.add_edge("grade", "dispatch")
    graph.add_edge("dispatch", END)
    return graph.compile()


def _demo_dispatch(
    state: GraphState,
    capability: Capability,
    language: Language,
    query: str,
    hits: list[dict],
    extra: str,
) -> dict[str, Any]:
    if capability == Capability.QA:
        answer, grounded = demo.demo_qa_answer(query, hits, language, extra or None)
        return {
            "answer": answer,
            "grounded": grounded,
            "steps": _append_step(state, "Generate", "Demo Q&A from retrieved passages."),
        }
    if capability == Capability.SUMMARIZE:
        brief = demo.demo_summary(query, hits, language, extra or None)
        return {
            "summary": brief.model_dump(),
            "answer": brief.executive_summary,
            "grounded": True,
            "steps": _append_step(state, "Summarize", "Demo executive summary."),
        }
    if capability == Capability.ANALYZE:
        brief = demo.demo_analysis(query, hits, language, extra or None)
        return {
            "analysis": brief.model_dump(),
            "answer": brief.overview,
            "grounded": True,
            "steps": _append_step(state, "Analyze", "Demo document analysis."),
        }
    if capability == Capability.EXTRACT:
        record = demo.demo_extraction(hits, extra or None)
        return {
            "extraction": record.model_dump(),
            "answer": f"Extracted {len(record.fields)} fields, {len(record.entities)} entities.",
            "grounded": True,
            "steps": _append_step(state, "Extract", "Demo structured extraction."),
        }
    brief = demo.demo_decision(query, hits, language, extra or None)
    return {
        "decision": brief.model_dump(),
        "answer": brief.recommendation,
        "grounded": True,
        "steps": _append_step(state, "Decide", "Demo decision brief."),
    }


def run_capability(
    query: str,
    *,
    capability: Capability = Capability.AUTO,
    language: Language | None = None,
    top_k: int | None = None,
    department: str | None = None,
    doc_type: str | None = None,
    filter_language: str | None = None,
    document_ids: list[str] | None = None,
    extra_text: str | None = None,
    settings: Settings | None = None,
) -> dict[str, Any]:
    settings = settings or get_settings()
    tracker = UsageTracker(model=settings.openai_model)
    graph = build_graph(settings=settings, tracker=tracker)
    initial: GraphState = {
        "query": query.strip(),
        "capability": capability.value,
        "language": (language or reply_language(query)).value,
        "top_k": top_k or settings.top_k,
        "department": department or "",
        "doc_type": doc_type or "",
        "filter_language": filter_language or "",
        "document_ids": document_ids or [],
        "extra_text": extra_text or "",
        "hits": [],
        "graded_hits": [],
        "answer": "",
        "steps": [],
        "refused": False,
        "grounded": True,
        "node_timings_ms": {},
    }
    result = graph.invoke(initial, {"recursion_limit": 12})
    result["_tracker"] = tracker
    return result
