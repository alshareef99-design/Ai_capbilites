from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from app.config import Settings, get_settings


@dataclass
class UsageTracker:
    prompt_tokens: int = 0
    completion_tokens: int = 0
    model: str = "gpt-4o-mini"
    events: list[dict] = field(default_factory=list)

    def add(self, prompt: int, completion: int) -> None:
        self.prompt_tokens += prompt
        self.completion_tokens += completion
        self.events.append({"prompt": prompt, "completion": completion})

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens


def build_chat_model(settings: Settings | None = None) -> ChatOpenAI:
    settings = settings or get_settings()
    kwargs: dict[str, Any] = {
        "model": settings.openai_model,
        "api_key": settings.openai_api_key,
        "temperature": settings.temperature,
        "timeout": 45,
        "max_retries": 2,
    }
    if settings.openai_base_url:
        kwargs["base_url"] = settings.openai_base_url
    return ChatOpenAI(**kwargs)


def record_usage(tracker: UsageTracker, message: object) -> None:
    usage = getattr(message, "usage_metadata", None) or {}
    tracker.add(
        int(usage.get("input_tokens") or 0),
        int(usage.get("output_tokens") or 0),
    )


def invoke_text(llm: Any, system: str, user: str, tracker: UsageTracker, max_tokens: int = 700) -> str:
    result = llm.invoke(
        [SystemMessage(content=system), HumanMessage(content=user)],
        max_tokens=max_tokens,
    )
    record_usage(tracker, result)
    return (result.content or "").strip()


def invoke_structured(llm: Any, schema: type, system: str, user: str, tracker: UsageTracker):
    structured = llm.with_structured_output(schema, include_raw=True)
    result = structured.invoke([SystemMessage(content=system), HumanMessage(content=user)])
    raw = result.get("raw") if isinstance(result, dict) else None
    parsed = result.get("parsed") if isinstance(result, dict) else result
    if raw is not None:
        record_usage(tracker, raw)
    if parsed is None:
        raise ValueError(f"Model did not return a valid {schema.__name__}")
    return parsed
