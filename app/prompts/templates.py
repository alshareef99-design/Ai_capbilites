"""Bilingual prompt templates — grounded, Arabic-first, no invented policy."""

from app.core.language import Language, is_arabic
from app.domain.schemas import Capability

SYSTEM_SHARED_EN = """You are Diwan, an enterprise knowledge assistant.
Use ONLY the provided context. If the context is insufficient, say so clearly.
Never invent policies, numbers, dates, SLAs, or legal obligations.
Cite facts that appear in the context. Be concise and professional."""

SYSTEM_SHARED_AR = """أنت ديوان، مساعد معرفة مؤسسي.
استخدم السياق المرفق فقط. إذا كان السياق غير كافٍ فصرّح بذلك بوضوح.
لا تختلق سياسات أو أرقاماً أو تواريخ أو التزامات قانونية.
انقل الحقائق الواردة في السياق. كن موجزاً ومهنياً وأجب بالعربية الفصحى."""

SYSTEM_SEARCH_EN = SYSTEM_SHARED_EN + "\nRank and explain why each passage is relevant. Do not answer beyond the hits."
SYSTEM_QA_EN = SYSTEM_SHARED_EN + "\nAnswer the question directly. If unknown, say you do not know."
SYSTEM_SUMMARIZE_EN = SYSTEM_SHARED_EN + "\nWrite an executive summary, then bullets and key facts from the context only."
SYSTEM_ANALYZE_EN = SYSTEM_SHARED_EN + "\nProduce a structured analysis: themes, risks, obligations, stakeholders, gaps, next steps."
SYSTEM_EXTRACT_EN = SYSTEM_SHARED_EN + "\nExtract entities, dates, amounts, and clauses. Use exact wording from the context."
SYSTEM_DECIDE_EN = SYSTEM_SHARED_EN + "\nCompare options found in the context. Recommend one with a confidence score and rationale."

SYSTEM_SEARCH_AR = SYSTEM_SHARED_AR + "\nرتّب المقاطع واشرح سبب صلتها. لا تجب خارج النتائج."
SYSTEM_QA_AR = SYSTEM_SHARED_AR + "\nأجب عن السؤال مباشرة. إذا لم تعرف فقل إنك لا تعرف."
SYSTEM_SUMMARIZE_AR = SYSTEM_SHARED_AR + "\nاكتب ملخصاً تنفيذياً ثم نقاطاً وحقائق مفتاحية من السياق فقط."
SYSTEM_ANALYZE_AR = SYSTEM_SHARED_AR + "\nقدّم تحليلاً منظماً: محاور، مخاطر، التزامات، أصحاب مصلحة، فجوات، خطوات تالية."
SYSTEM_EXTRACT_AR = SYSTEM_SHARED_AR + "\nاستخرج الكيانات والتواريخ والمبالغ والبنود بنصها الوارد في السياق."
SYSTEM_DECIDE_AR = SYSTEM_SHARED_AR + "\nقارن الخيارات الواردة في السياق. أوصِ بخيار واحد مع درجة ثقة ومبرر."

USER_TEMPLATE = """Context:
{context}

Request:
{query}

Instructions:
- Use only the context.
- If the context is empty or irrelevant, abstain.
- Match the user's language.
"""


def system_prompt(capability: Capability, language: Language) -> str:
    arabic = is_arabic(language)
    mapping = {
        Capability.SEARCH: (SYSTEM_SEARCH_AR, SYSTEM_SEARCH_EN),
        Capability.QA: (SYSTEM_QA_AR, SYSTEM_QA_EN),
        Capability.SUMMARIZE: (SYSTEM_SUMMARIZE_AR, SYSTEM_SUMMARIZE_EN),
        Capability.ANALYZE: (SYSTEM_ANALYZE_AR, SYSTEM_ANALYZE_EN),
        Capability.EXTRACT: (SYSTEM_EXTRACT_AR, SYSTEM_EXTRACT_EN),
        Capability.DECIDE: (SYSTEM_DECIDE_AR, SYSTEM_DECIDE_EN),
        Capability.AUTO: (SYSTEM_QA_AR, SYSTEM_QA_EN),
    }
    ar, en = mapping.get(capability, (SYSTEM_QA_AR, SYSTEM_QA_EN))
    return ar if arabic else en


def user_prompt(query: str, context: str) -> str:
    return USER_TEMPLATE.format(context=context or "(no context)", query=query)


def format_context(hits: list[dict], extra_text: str | None = None, limit: int = 6) -> str:
    parts: list[str] = []
    if extra_text and extra_text.strip():
        parts.append(f"[Pasted text]\n{extra_text.strip()[:6000]}")
    for hit in hits[:limit]:
        title = hit.get("title") or hit.get("doc_id")
        dept = hit.get("department") or ""
        header = f"[{title} | {dept} | {hit.get('chunk_id')}]".strip()
        parts.append(f"{header}\n{hit.get('text') or ''}")
    return "\n\n".join(parts) if parts else ""
