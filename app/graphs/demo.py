"""Deterministic stand-ins so every capability runs without an API key."""

from __future__ import annotations

import re
from collections import Counter

from app.core.language import Language, is_arabic
from app.domain.schemas import (
    AnalysisBrief,
    DecisionBrief,
    DecisionOption,
    ExtractedField,
    ExtractionRecord,
    ObligationItem,
    RiskItem,
    Stakeholder,
    SummaryBrief,
)
from app.rag.tokenize import tokenize

_SENTENCE_SPLIT = re.compile(r"(?<=[\n.!?؟])\s+")
_AMOUNT = re.compile(
    r"(?:SAR|USD|ريال)\s*[\d,]+(?:\.\d+)?|[\d,]+(?:\.\d+)?\s*(?:SAR|USD|ريال|million|مليون)",
    re.I,
)
_DATE = re.compile(
    r"\b\d{1,2}\s+(?:days?|day|months?|years?|يوم(?:اً)?|أيام|شهر|أشهر|سنة|سنوات)\b|"
    r"\b\d{1,2}/\d{1,2}/\d{2,4}\b|"
    r"\b20\d{2}\b",
    re.I,
)
_PERCENT = re.compile(r"\b\d+(?:\.\d+)?\s*%")
_ENTITY = re.compile(
    r"\b(?:AWS|Azure|Microsoft|ISO\s*27001|SOC\s*2|MFA|SLA|GDPR|NCA|SAMA|"
    r"Procurement|Legal|Finance|Security|HR|"
    r"مايكروسوفت|أزور|أمازون)\b",
    re.I,
)
_OBLIGATION = re.compile(
    r"((?:shall|must|required to|يلتزم|يجب|يُشترط)[^.!?\n]{8,140})",
    re.I,
)
_RISK_HINTS = (
    "risk",
    "penalty",
    "liability",
    "breach",
    "confidential",
    "terminate",
    "lock-in",
    "single vendor",
    "مخاطر",
    "جزاء",
    "مسؤولية",
    "إنهاء",
    "سري",
    "اختراق",
)


def _sentences(text: str) -> list[str]:
    parts = [re.sub(r"\s+", " ", s).strip(" -•") for s in _SENTENCE_SPLIT.split(text)]
    return [p for p in parts if 24 <= len(p) <= 280]


def _joined(hits: list[dict], extra_text: str | None = None) -> str:
    parts = [h.get("text") or "" for h in hits]
    if extra_text:
        parts.insert(0, extra_text)
    return "\n".join(parts)


def _best_title(hits: list[dict], fallback: str) -> str:
    return (hits[0].get("title") if hits else None) or fallback


def _relevant_sentences(query: str, corpus: str, limit: int = 4) -> list[str]:
    q_tokens = set(tokenize(query))
    ranked: list[tuple[int, str]] = []
    for sentence in _sentences(corpus):
        overlap = len(q_tokens & set(tokenize(sentence)))
        if overlap:
            ranked.append((overlap, sentence))
    ranked.sort(key=lambda item: item[0], reverse=True)
    unique: list[str] = []
    seen: set[str] = set()
    for _, sentence in ranked:
        key = sentence[:80]
        if key in seen:
            continue
        seen.add(key)
        unique.append(sentence)
        if len(unique) >= limit:
            break
    return unique


def demo_search_answer(query: str, hits: list[dict], language: Language) -> str:
    if not hits:
        if is_arabic(language):
            return "لم يُعثر على وثائق ذات صلة في المعرفة المؤسسية."
        return "No relevant enterprise documents were found."
    lines = []
    if is_arabic(language):
        lines.append(f"نتائج البحث المؤسسي عن «{query}»:")
    else:
        lines.append(f"Enterprise search results for “{query}”:")
    for hit in hits[:5]:
        snippet = re.sub(r"\s+", " ", hit.get("text") or "")[:180]
        lines.append(f"- {hit.get('title')} ({hit.get('department')}): {snippet}")
    return "\n".join(lines)


def demo_qa_answer(query: str, hits: list[dict], language: Language, extra_text: str | None = None) -> tuple[str, bool]:
    corpus = _joined(hits, extra_text)
    if not corpus.strip() or not hits and not extra_text:
        if is_arabic(language):
            return "لا أعرف. هذا غير وارد في المعرفة المؤسسية المتاحة.", False
        return "I do not know. That is not in the available enterprise knowledge.", False
    sentences = _relevant_sentences(query, corpus, limit=3)
    if not sentences:
        if is_arabic(language):
            return "السياق غير كافٍ للإجابة على هذا السؤال من الوثائق المتاحة.", False
        return "The retrieved context is not sufficient to answer from the available documents.", False
    if is_arabic(language):
        prefix = "استناداً إلى الوثائق المؤسسية: "
    else:
        prefix = "Based on the enterprise documents: "
    return prefix + " ".join(sentences), True


def demo_summary(query: str, hits: list[dict], language: Language, extra_text: str | None = None) -> SummaryBrief:
    corpus = _joined(hits, extra_text)
    title = _best_title(hits, "Document summary" if not is_arabic(language) else "ملخص الوثيقة")
    sentences = _relevant_sentences(query, corpus, limit=5) or _sentences(corpus)[:5]
    bullets = sentences[:4]
    facts = []
    for pattern in (_AMOUNT, _PERCENT, _DATE):
        facts.extend(m.group(0) for m in pattern.finditer(corpus))
    facts = list(dict.fromkeys(facts))[:6]
    if is_arabic(language):
        executive = " ".join(sentences[:2]) or "لا يوجد نص كافٍ لإعداد ملخص."
        return SummaryBrief(
            title=title,
            executive_summary=executive,
            bullets=bullets or ["لا توجد نقاط مستخرجة."],
            key_facts=facts or ["لا توجد أرقام أو تواريخ صريحة."],
        )
    executive = " ".join(sentences[:2]) or "There is not enough text to summarize."
    return SummaryBrief(
        title=title,
        executive_summary=executive,
        bullets=bullets or ["No bullets could be extracted."],
        key_facts=facts or ["No explicit figures or dates found."],
    )


def demo_analysis(query: str, hits: list[dict], language: Language, extra_text: str | None = None) -> AnalysisBrief:
    corpus = _joined(hits, extra_text)
    title = _best_title(hits, "Document analysis" if not is_arabic(language) else "تحليل الوثيقة")
    overview = " ".join(_relevant_sentences(query, corpus, limit=2) or _sentences(corpus)[:2])
    themes = [t for t, _ in Counter(tokenize(corpus)).most_common(8) if len(t) > 3][:5]
    risks: list[RiskItem] = []
    for sentence in _sentences(corpus):
        lower = sentence.lower()
        if any(hint in lower for hint in _RISK_HINTS):
            severity = "high" if any(w in lower for w in ("liability", "penalty", "breach", "مسؤولية", "جزاء")) else "medium"
            risks.append(RiskItem(title=sentence[:72], severity=severity, detail=sentence, evidence=sentence[:160]))
        if len(risks) >= 4:
            break
    obligations = []
    for match in _OBLIGATION.findall(corpus)[:4]:
        obligations.append(ObligationItem(party="Named party", obligation=match.strip(), deadline=None))
    stakeholders = [
        Stakeholder(name=m.group(0), role=None) for m in _ENTITY.finditer(corpus)
    ]
    unique_names = []
    seen = set()
    for item in stakeholders:
        key = item.name.lower()
        if key not in seen:
            seen.add(key)
            unique_names.append(item)
    if is_arabic(language):
        gaps = ["لم تُحدد بعض المهل أو الملاك بوضوح."] if "موعد" not in corpus else []
        next_steps = ["راجع الالتزامات عالية الخطورة مع الشؤون القانونية.", "أكد الأرقام مع مالك الوثيقة."]
        if not risks:
            risks = [RiskItem(title="فجوات توثيق", severity="low", detail="لم تُستخرج مخاطر صريحة من النص.", evidence=None)]
        return AnalysisBrief(
            title=title,
            overview=overview or "لا يوجد ملخص كافٍ.",
            themes=themes,
            risks=risks,
            obligations=obligations,
            stakeholders=unique_names[:6],
            gaps=gaps,
            recommended_next_steps=next_steps,
        )
    gaps = ["Some owners or deadlines are not explicit."] if "deadline" not in corpus.lower() else []
    next_steps = [
        "Review high-severity obligations with Legal.",
        "Confirm figures with the document owner.",
    ]
    if not risks:
        risks = [RiskItem(title="Documentation gaps", severity="low", detail="No explicit risks were extracted from the text.", evidence=None)]
    return AnalysisBrief(
        title=title,
        overview=overview or "Insufficient text for analysis.",
        themes=themes,
        risks=risks,
        obligations=obligations,
        stakeholders=unique_names[:6],
        gaps=gaps,
        recommended_next_steps=next_steps,
    )


def demo_extraction(hits: list[dict], extra_text: str | None = None) -> ExtractionRecord:
    corpus = _joined(hits, extra_text)
    title = _best_title(hits, "Extracted record")
    entities = list(dict.fromkeys(m.group(0) for m in _ENTITY.finditer(corpus)))[:10]
    dates = list(dict.fromkeys(m.group(0) for m in _DATE.finditer(corpus)))[:8]
    amounts = list(dict.fromkeys(m.group(0) for m in _AMOUNT.finditer(corpus)))[:8]
    percents = list(dict.fromkeys(m.group(0) for m in _PERCENT.finditer(corpus)))
    amounts = list(dict.fromkeys(amounts + percents))[:10]
    clauses = [m.strip() for m in _OBLIGATION.findall(corpus)][:6]
    fields = []
    if amounts:
        fields.append(ExtractedField(name="amounts", value="; ".join(amounts[:4]), evidence=amounts[0]))
    if dates:
        fields.append(ExtractedField(name="timeframes", value="; ".join(dates[:4]), evidence=dates[0]))
    if entities:
        fields.append(ExtractedField(name="entities", value="; ".join(entities[:6]), evidence=entities[0]))
    return ExtractionRecord(
        title=title,
        entities=entities,
        dates=dates,
        amounts=amounts,
        clauses=clauses,
        fields=fields,
    )


def _option_block(name: str, corpus: str) -> DecisionOption:
    sentences = [s for s in _sentences(corpus) if name.lower() in s.lower()]
    summary = sentences[0] if sentences else f"{name} is mentioned as a candidate in the knowledge base."
    cost_match = _AMOUNT.search(" ".join(sentences) or corpus)
    return DecisionOption(
        name=name,
        summary=summary,
        pros=[s for s in sentences if any(w in s.lower() for w in ("lower", "existing", "compliant", "أقل", "قائم", "متوافق"))][:2]
        or (["Aligned with current estate"] if "existing" in corpus.lower() or name.lower() in corpus.lower() else ["Documented option"]),
        cons=[s for s in sentences if any(w in s.lower() for w in ("lock-in", "cost", "risk", "تكلفة", "مخاطر", "ارتباط"))][:2]
        or ["Residual implementation risk remains."],
        estimated_cost=cost_match.group(0) if cost_match else None,
    )


def demo_decision(query: str, hits: list[dict], language: Language, extra_text: str | None = None) -> DecisionBrief:
    corpus = _joined(hits, extra_text)
    names = list(dict.fromkeys(m.group(0) for m in _ENTITY.finditer(corpus)))
    option_names = [n for n in names if n.lower() in {"aws", "azure", "microsoft"} or n in {"أمازون", "أزور", "مايكروسوفت"}]
    if "on-prem" in corpus.lower() or "on premises" in corpus.lower() or "الموقع" in corpus:
        option_names.append("On-premises")
    option_names = list(dict.fromkeys(option_names))[:4] or ["Status quo", "Proposed change"]
    options = [_option_block(name, corpus) for name in option_names]
    recommend_azure = bool(
        re.search(r"recommend(?:s|ed)?\s+azure", corpus, re.I)
        or ("توصي" in corpus and "أزور" in corpus)
    )
    recommendation = options[0].name
    if recommend_azure:
        for option in options:
            if option.name.lower() in {"azure", "أزور"}:
                recommendation = option.name
                break
        else:
            recommendation = "Azure" if not is_arabic(language) else "أزور"
    else:
        named = re.search(
            r"(?:recommend(?:s|ed)?|توصي)\s+([A-Za-z\u0600-\u06FF][A-Za-z0-9\u0600-\u06FF\-]{2,40})",
            corpus,
            re.I,
        )
        if named:
            recommendation = named.group(1).strip()
    rationale_sentences = _relevant_sentences(query, corpus, limit=2) or _sentences(corpus)[:2]
    rationale = " ".join(rationale_sentences) if rationale_sentences else options[0].summary
    confidence = 0.84 if recommend_azure else 0.62
    if is_arabic(language):
        return DecisionBrief(
            question=query,
            recommendation=recommendation,
            confidence=confidence,
            rationale=rationale,
            options=options,
            risks_if_accepted=["تحقق من الارتباط بمورد واحد والالتزامات التعاقدية قبل التنفيذ."],
            open_questions=["هل رُصدت ميزانية السنة المالية بالكامل؟"],
        )
    return DecisionBrief(
        question=query,
        recommendation=recommendation,
        confidence=confidence,
        rationale=rationale,
        options=options,
        risks_if_accepted=["Validate vendor lock-in and contractual liability before execution."],
        open_questions=["Is the full-year budget already reserved?"],
    )
