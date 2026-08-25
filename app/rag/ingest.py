from __future__ import annotations

from pathlib import Path

import yaml

from app.config import get_settings
from app.core.language import detect_language
from app.rag.index import get_index, persist_index, reset_index

FRONTMATTER_DELIM = "---"


def parse_markdown(path: Path) -> tuple[dict, str]:
    raw = path.read_text(encoding="utf-8")
    meta: dict = {}
    body = raw
    if raw.startswith(FRONTMATTER_DELIM):
        parts = raw.split(FRONTMATTER_DELIM, 2)
        if len(parts) >= 3:
            meta = yaml.safe_load(parts[1]) or {}
            body = parts[2].strip()
    return meta, body


def ingest_knowledge(reset: bool = False) -> tuple[int, int]:
    settings = get_settings()
    index = reset_index() if reset else get_index()
    if reset:
        index.reset()
    knowledge_dir = settings.knowledge_dir
    if not knowledge_dir.exists():
        persist_index()
        return 0, 0

    docs = 0
    chunks = 0
    for path in sorted(knowledge_dir.rglob("*")):
        if path.suffix.lower() not in {".md", ".txt"}:
            continue
        meta, body = parse_markdown(path)
        if not body.strip():
            continue
        language = str(meta.get("language") or detect_language(body).value)
        doc_id, n_chunks = index.add_document(
            body,
            title=str(meta.get("title") or path.stem.replace("_", " ").title()),
            department=str(meta.get("department") or "general"),
            doc_type=str(meta.get("doc_type") or "policy"),
            language=language,
            source=str(path.relative_to(knowledge_dir)),
            doc_id=str(meta.get("id") or path.stem),
        )
        docs += 1
        chunks += n_chunks
        _ = doc_id
    persist_index()
    return docs, chunks


def ingest_text(
    text: str,
    *,
    title: str,
    department: str = "general",
    doc_type: str = "policy",
    language: str | None = None,
    source: str = "upload",
    doc_id: str | None = None,
) -> tuple[str, int]:
    index = get_index()
    resolved_language = language or detect_language(text).value
    doc_id, n_chunks = index.add_document(
        text,
        title=title,
        department=department,
        doc_type=doc_type,
        language=resolved_language,
        source=source,
        doc_id=doc_id,
    )
    persist_index()
    return doc_id, n_chunks
