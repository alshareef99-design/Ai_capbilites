from __future__ import annotations

import re
from typing import Any

from app.core.language import Language, detect_language
from app.domain.schemas import EvaluationScore, SourceChunk

_ARABIC_RE = re.compile(r"[\u0600-\u06FF]")
_UNCERTAIN_EN = re.compile(r"\b(i don'?t know|do not know|not in (the )?knowledge|insufficient)\b", re.I)
_UNCERTAIN_AR = re.compile(r"(لا\s*أعرف|غير\s*وارد|السياق\s*غير\s*كاف)")


def token_set(text: str) -> set[str]:
    return {t for t in re.findall(r"[\w\u0600-\u06FF]+", text.lower()) if len(t) > 2}


def groundedness_score(answer: str, sources: list[SourceChunk]) -> float:
    if not answer.strip():
        return 0.0
    if _UNCERTAIN_EN.search(answer) or _UNCERTAIN_AR.search(answer):
        return 0.85
    if not sources:
        return 0.2
    context = " ".join(s.content for s in sources)
    answer_tokens = token_set(answer)
    context_tokens = token_set(context)
    if not answer_tokens:
        return 0.0
    overlap = len(answer_tokens & context_tokens) / len(answer_tokens)
    return round(min(1.0, max(0.0, overlap * 1.25)), 3)


def relevance_score(question: str, answer: str) -> float:
    q_tokens = token_set(question)
    a_tokens = token_set(answer)
    if not q_tokens or not a_tokens:
        return 0.0
    overlap = len(q_tokens & a_tokens) / len(q_tokens)
    length_bonus = 0.1 if 20 <= len(answer) <= 1600 else 0.0
    return round(min(1.0, overlap + length_bonus), 3)


def language_quality_score(answer: str, expected: Language) -> float:
    detected = detect_language(answer)
    if expected == Language.ARABIC:
        arabic_ratio = len(_ARABIC_RE.findall(answer)) / max(len(answer), 1)
        base = 0.9 if detected in {Language.ARABIC, Language.MIXED} else 0.4
        return round(min(1.0, base + arabic_ratio), 3)
    if expected == Language.ENGLISH:
        return 0.9 if detected == Language.ENGLISH else 0.55
    return 0.75


def evaluate_answer(
    question: str,
    answer: str,
    sources: list[SourceChunk],
    language: Language,
) -> EvaluationScore:
    g = groundedness_score(answer, sources)
    r = relevance_score(question, answer)
    lq = language_quality_score(answer, language)
    overall = round((0.45 * g) + (0.35 * r) + (0.20 * lq), 3)
    notes: list[str] = []
    if g < 0.5:
        notes.append("Low groundedness — answer may not be supported by retrieved context.")
    if r < 0.35:
        notes.append("Low relevance — answer may not address the question.")
    if lq < 0.6:
        notes.append("Language mismatch or weak bilingual quality.")
    if not notes:
        notes.append("Scores within acceptable range.")
    return EvaluationScore(
        groundedness=g,
        relevance=r,
        language_quality=lq,
        overall=overall,
        notes=notes,
    )


def hits_to_sources(hits: list[dict[str, Any]]) -> list[SourceChunk]:
    return [
        SourceChunk(
            content=h.get("text") or "",
            source=h.get("source") or h.get("doc_id") or "unknown",
            score=h.get("score"),
            metadata={
                "doc_id": h.get("doc_id"),
                "chunk_id": h.get("chunk_id"),
                "title": h.get("title"),
                "department": h.get("department"),
                "doc_type": h.get("doc_type"),
                "language": h.get("language"),
            },
        )
        for h in hits
    ]
