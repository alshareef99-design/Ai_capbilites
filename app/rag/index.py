from __future__ import annotations

import math
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from app.config import get_settings
from app.rag.chunking import split_text
from app.rag.tokenize import cosine, term_frequency, tokenize


@dataclass
class StoredDocument:
    id: str
    title: str
    department: str
    doc_type: str
    language: str
    source: str
    text: str
    chunks: int = 0


@dataclass
class StoredChunk:
    chunk_id: str
    doc_id: str
    text: str
    title: str
    department: str
    doc_type: str
    language: str
    source: str
    tfidf: dict[str, float] = field(default_factory=dict)


class DocumentIndex:
    def __init__(self) -> None:
        self.documents: dict[str, StoredDocument] = {}
        self.chunks: list[StoredChunk] = []
        self.idf: dict[str, float] = {}
        self._df: dict[str, int] = {}

    def reset(self) -> None:
        self.documents.clear()
        self.chunks.clear()
        self.idf.clear()
        self._df.clear()

    def add_document(
        self,
        text: str,
        *,
        title: str,
        department: str = "general",
        doc_type: str = "policy",
        language: str = "en",
        source: str = "upload",
        doc_id: str | None = None,
    ) -> tuple[str, int]:
        settings = get_settings()
        resolved_id = doc_id or str(uuid.uuid4())[:10]
        pieces = split_text(text, settings.chunk_size, settings.chunk_overlap)
        if not pieces:
            raise ValueError("Document contains no readable text.")

        if resolved_id in self.documents:
            self.chunks = [chunk for chunk in self.chunks if chunk.doc_id != resolved_id]

        self.documents[resolved_id] = StoredDocument(
            id=resolved_id,
            title=title,
            department=department,
            doc_type=doc_type,
            language=language,
            source=source,
            text=text,
            chunks=len(pieces),
        )
        for i, piece in enumerate(pieces):
            self.chunks.append(
                StoredChunk(
                    chunk_id=f"{resolved_id}_{i}",
                    doc_id=resolved_id,
                    text=piece,
                    title=title,
                    department=department,
                    doc_type=doc_type,
                    language=language,
                    source=source,
                )
            )
        self._rebuild_vectors()
        return resolved_id, len(pieces)

    def _rebuild_vectors(self) -> None:
        self._df = {}
        tokenized = [tokenize(chunk.text) for chunk in self.chunks]
        for tokens in tokenized:
            for term in set(tokens):
                self._df[term] = self._df.get(term, 0) + 1
        n_docs = max(len(self.chunks), 1)
        self.idf = {
            term: math.log((n_docs + 1) / (df + 1)) + 1.0 for term, df in self._df.items()
        }
        for chunk, tokens in zip(self.chunks, tokenized, strict=True):
            tf = term_frequency(tokens)
            chunk.tfidf = {term: weight * self.idf.get(term, 1.0) for term, weight in tf.items()}

    def search(
        self,
        query: str,
        *,
        top_k: int = 6,
        department: str | None = None,
        language: str | None = None,
        doc_type: str | None = None,
        document_ids: list[str] | None = None,
    ) -> list[dict[str, Any]]:
        query_tokens = tokenize(query)
        query_tf = term_frequency(query_tokens)
        query_vec = {term: weight * self.idf.get(term, 1.0) for term, weight in query_tf.items()}
        scored: list[tuple[float, StoredChunk]] = []
        allowed = set(document_ids or [])
        for chunk in self.chunks:
            if department and chunk.department.lower() != department.lower():
                continue
            if language and chunk.language != language:
                continue
            if doc_type and chunk.doc_type.lower() != doc_type.lower():
                continue
            if allowed and chunk.doc_id not in allowed:
                continue
            score = cosine(query_vec, chunk.tfidf)
            if score > 0:
                scored.append((score, chunk))
        scored.sort(key=lambda item: item[0], reverse=True)
        results: list[dict[str, Any]] = []
        for rank, (score, chunk) in enumerate(scored[:top_k], start=1):
            results.append(
                {
                    "rank": rank,
                    "score": round(float(score), 4),
                    "text": chunk.text,
                    "doc_id": chunk.doc_id,
                    "chunk_id": chunk.chunk_id,
                    "title": chunk.title,
                    "department": chunk.department,
                    "doc_type": chunk.doc_type,
                    "language": chunk.language,
                    "source": chunk.source,
                }
            )
        return results

    def list_documents(self) -> list[StoredDocument]:
        return list(self.documents.values())

    def snapshot(self) -> dict[str, Any]:
        return {
            "documents": [asdict(doc) for doc in self.documents.values()],
            "chunks": [asdict(chunk) for chunk in self.chunks],
        }


_index: DocumentIndex | None = None


def get_index() -> DocumentIndex:
    global _index
    if _index is None:
        _index = DocumentIndex()
    return _index


def reset_index() -> DocumentIndex:
    global _index
    _index = DocumentIndex()
    return _index


def persist_index(path: Path | None = None) -> None:
    import json

    settings = get_settings()
    target = path or settings.index_dir
    target.mkdir(parents=True, exist_ok=True)
    (target / "index.json").write_text(
        json.dumps(get_index().snapshot(), ensure_ascii=False),
        encoding="utf-8",
    )


def load_index(path: Path | None = None) -> bool:
    import json

    settings = get_settings()
    target = path or settings.index_dir / "index.json"
    if not target.exists():
        return False
    payload = json.loads(target.read_text(encoding="utf-8"))
    index = reset_index()
    for doc in payload.get("documents") or []:
        stored = StoredDocument(**{k: v for k, v in doc.items() if k in StoredDocument.__dataclass_fields__})
        index.documents[stored.id] = stored
    for chunk in payload.get("chunks") or []:
        stored_chunk = StoredChunk(
            **{k: v for k, v in chunk.items() if k in StoredChunk.__dataclass_fields__}
        )
        index.chunks.append(stored_chunk)
    index._rebuild_vectors()
    return True
