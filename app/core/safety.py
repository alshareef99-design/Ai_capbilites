"""Lightweight safety filter for demo and production pre-checks."""

from __future__ import annotations

import re

from app.config import get_settings
from app.core.language import Language, is_arabic

_BLOCKED_PATTERNS = [
    r"\b(make\s+a\s+bomb|how\s+to\s+hack|credit\s+card\s+dump|bypass\s+mfa\s+without)\b",
    r"(صنع\s*قنبلة|اختراق\s*حساب|سرقة\s*بطاقة)",
]


def is_unsafe(text: str) -> bool:
    settings = get_settings()
    if not settings.enable_safety_filter:
        return False
    lowered = text.lower()
    return any(re.search(pat, lowered, flags=re.IGNORECASE) for pat in _BLOCKED_PATTERNS)


def safety_refusal(language: Language) -> str:
    if is_arabic(language):
        return (
            "عذراً، لا يمكنني المساعدة في هذا الطلب لأنه يخالف سياسات السلامة. "
            "يرجى إعادة صياغة السؤال ضمن الاستخدام المؤسسي المسموح."
        )
    return (
        "Sorry, I cannot help with that request because it violates safety policies. "
        "Please rephrase within allowed enterprise use."
    )


def normalize_query(text: str) -> str:
    return " ".join(text.strip().split())
