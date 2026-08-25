from __future__ import annotations

import math
import re
from collections import Counter

TOKEN_RE = re.compile(r"[\u0600-\u06FF]+|[A-Za-z0-9]+(?:[.-][A-Za-z0-9]+)*")

STOPWORDS = {
    "the", "a", "an", "and", "or", "of", "to", "in", "on", "for", "with", "by",
    "is", "are", "was", "were", "be", "been", "this", "that", "these", "those",
    "it", "its", "as", "at", "from", "not", "no", "if", "then", "than", "but",
    "we", "you", "they", "their", "our", "will", "shall", "must", "may", "can",
    "should", "into", "about", "over", "under", "per", "any", "all", "such",
    "من", "في", "على", "إلى", "الى", "عن", "مع", "هذا", "هذه", "ذلك", "تلك",
    "التي", "الذي", "أو", "او", "و", "لا", "ما", "أن", "ان", "إن", "كان",
    "يكون", "تم", "يتم", "كل", "بين", "بعد", "قبل", "غير", "دون", "حسب",
}


def tokenize(text: str) -> list[str]:
    tokens: list[str] = []
    for raw in TOKEN_RE.findall(text.lower()):
        token = raw.strip(".-")
        if len(token) < 2 or token in STOPWORDS:
            continue
        if token.startswith("ال") and len(token) > 4:
            token = token[2:]
        tokens.append(token)
    return tokens


def term_frequency(tokens: list[str]) -> dict[str, float]:
    counts = Counter(tokens)
    if not counts:
        return {}
    norm = math.sqrt(sum(v * v for v in counts.values()))
    if norm == 0:
        return {}
    return {term: count / norm for term, count in counts.items()}


def cosine(a: dict[str, float], b: dict[str, float]) -> float:
    if not a or not b:
        return 0.0
    if len(a) > len(b):
        a, b = b, a
    return float(sum(weight * b.get(term, 0.0) for term, weight in a.items()))
