from __future__ import annotations

import re
from enum import Enum


class Language(str, Enum):
    ARABIC = "ar"
    ENGLISH = "en"
    MIXED = "mixed"
    UNKNOWN = "unknown"


_ARABIC_RE = re.compile(r"[\u0600-\u06FF]")
_LATIN_RE = re.compile(r"[A-Za-z]")


def detect_language(text: str) -> Language:
    arabic = len(_ARABIC_RE.findall(text))
    latin = len(_LATIN_RE.findall(text))
    if arabic == 0 and latin == 0:
        return Language.UNKNOWN
    if arabic > 0 and latin > 0:
        if arabic >= latin * 0.3:
            return Language.ARABIC
        return Language.MIXED
    if arabic > 0:
        return Language.ARABIC
    return Language.ENGLISH


def reply_language(text: str) -> Language:
    detected = detect_language(text)
    if detected in {Language.ARABIC, Language.MIXED}:
        return Language.ARABIC
    return Language.ENGLISH


def is_arabic(language: Language) -> bool:
    return language in {Language.ARABIC, Language.MIXED}
