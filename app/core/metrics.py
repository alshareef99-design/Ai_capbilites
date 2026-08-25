from __future__ import annotations

import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Iterator

from app.config import get_settings


@dataclass
class UsageMetrics:
    latency_ms: float = 0.0
    retrieval_ms: float = 0.0
    generation_ms: float = 0.0
    input_tokens: int = 0
    output_tokens: int = 0
    estimated_cost_usd: float = 0.0
    chunks_retrieved: int = 0
    language: str = "unknown"
    extra: dict = field(default_factory=dict)

    def estimate_cost(self) -> float:
        settings = get_settings()
        cost = (self.input_tokens / 1_000_000) * settings.input_cost_per_1m + (
            self.output_tokens / 1_000_000
        ) * settings.output_cost_per_1m
        self.estimated_cost_usd = round(cost, 6)
        return self.estimated_cost_usd

    def to_dict(self) -> dict:
        return {
            "latency_ms": round(self.latency_ms, 2),
            "retrieval_ms": round(self.retrieval_ms, 2),
            "generation_ms": round(self.generation_ms, 2),
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "estimated_cost_usd": self.estimated_cost_usd,
            "chunks_retrieved": self.chunks_retrieved,
            "language": self.language,
            **self.extra,
        }


@contextmanager
def timed_section(metrics: UsageMetrics, field_name: str) -> Iterator[None]:
    start = time.perf_counter()
    try:
        yield
    finally:
        elapsed_ms = (time.perf_counter() - start) * 1000
        setattr(metrics, field_name, elapsed_ms)
