from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, File, Form, Header, HTTPException, Query, Request, UploadFile

from app.config import Settings, get_settings
from app.domain.schemas import (
    Capability,
    DocumentInfo,
    HealthResponse,
    IngestResponse,
    RunRequest,
    RunResponse,
)
from app.repositories.runs import RunRepository
from app.services import capabilities as svc

router = APIRouter()


def get_repo(settings: Annotated[Settings, Depends(get_settings)]) -> RunRepository:
    return RunRepository(settings.database_path)


def _require_api_key(settings: Settings, api_key: str | None) -> None:
    expected = settings.diwan_api_key.strip()
    if expected and api_key != expected:
        raise HTTPException(status_code=401, detail="Missing or invalid API key.")


@router.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return svc.health()


@router.get("/documents", response_model=list[DocumentInfo])
def documents() -> list[DocumentInfo]:
    return svc.list_documents()


@router.post("/ingest", response_model=IngestResponse)
def ingest_corpus(
    settings: Annotated[Settings, Depends(get_settings)],
    reset: bool = False,
    x_api_key: Annotated[str | None, Header()] = None,
) -> IngestResponse:
    _require_api_key(settings, x_api_key)
    return svc.ingest(reset=reset)


@router.post("/ingest/upload", response_model=IngestResponse)
async def ingest_upload(
    request: Request,
    settings: Annotated[Settings, Depends(get_settings)],
    file: UploadFile = File(...),
    department: str = Form(default="general"),
    doc_type: str = Form(default="policy"),
    x_api_key: Annotated[str | None, Header()] = None,
) -> IngestResponse:
    _require_api_key(settings, x_api_key)
    request.app.state.limiter.check(_client(request))
    if not file.filename or not file.filename.lower().endswith((".md", ".txt")):
        raise HTTPException(status_code=400, detail="Upload a .txt or .md file.")
    payload = await file.read()
    if len(payload) > settings.max_upload_bytes:
        raise HTTPException(status_code=413, detail="File is too large.")
    try:
        text = payload.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise HTTPException(status_code=400, detail="File must be UTF-8 text.") from exc
    if not text.strip():
        raise HTTPException(status_code=400, detail="The file contains no readable text.")
    _, chunks = svc.ingest_upload(text, file.filename, department=department, doc_type=doc_type)
    return IngestResponse(documents_indexed=1, chunks_indexed=chunks, reset=False)


@router.post("/run", response_model=RunResponse)
def run(
    payload: RunRequest,
    request: Request,
    settings: Annotated[Settings, Depends(get_settings)],
    repo: Annotated[RunRepository, Depends(get_repo)],
    x_api_key: Annotated[str | None, Header()] = None,
) -> RunResponse:
    _require_api_key(settings, x_api_key)
    request.app.state.limiter.check(_client(request))
    return svc.run(payload, repo=repo, settings=settings)


def _with_capability(payload: RunRequest, capability: Capability) -> RunRequest:
    return payload.model_copy(update={"capability": capability})


@router.post("/search", response_model=RunResponse)
def search(payload: RunRequest, request: Request, settings: Annotated[Settings, Depends(get_settings)], repo: Annotated[RunRepository, Depends(get_repo)], x_api_key: Annotated[str | None, Header()] = None) -> RunResponse:
    return run(_with_capability(payload, Capability.SEARCH), request, settings, repo, x_api_key)


@router.post("/ask", response_model=RunResponse)
def ask(payload: RunRequest, request: Request, settings: Annotated[Settings, Depends(get_settings)], repo: Annotated[RunRepository, Depends(get_repo)], x_api_key: Annotated[str | None, Header()] = None) -> RunResponse:
    return run(_with_capability(payload, Capability.QA), request, settings, repo, x_api_key)


@router.post("/summarize", response_model=RunResponse)
def summarize(payload: RunRequest, request: Request, settings: Annotated[Settings, Depends(get_settings)], repo: Annotated[RunRepository, Depends(get_repo)], x_api_key: Annotated[str | None, Header()] = None) -> RunResponse:
    return run(_with_capability(payload, Capability.SUMMARIZE), request, settings, repo, x_api_key)


@router.post("/analyze", response_model=RunResponse)
def analyze(payload: RunRequest, request: Request, settings: Annotated[Settings, Depends(get_settings)], repo: Annotated[RunRepository, Depends(get_repo)], x_api_key: Annotated[str | None, Header()] = None) -> RunResponse:
    return run(_with_capability(payload, Capability.ANALYZE), request, settings, repo, x_api_key)


@router.post("/extract", response_model=RunResponse)
def extract(payload: RunRequest, request: Request, settings: Annotated[Settings, Depends(get_settings)], repo: Annotated[RunRepository, Depends(get_repo)], x_api_key: Annotated[str | None, Header()] = None) -> RunResponse:
    return run(_with_capability(payload, Capability.EXTRACT), request, settings, repo, x_api_key)


@router.post("/decide", response_model=RunResponse)
def decide(payload: RunRequest, request: Request, settings: Annotated[Settings, Depends(get_settings)], repo: Annotated[RunRepository, Depends(get_repo)], x_api_key: Annotated[str | None, Header()] = None) -> RunResponse:
    return run(_with_capability(payload, Capability.DECIDE), request, settings, repo, x_api_key)


@router.get("/runs")
def list_runs(
    repo: Annotated[RunRepository, Depends(get_repo)],
    limit: int = Query(default=20, ge=1, le=100),
    offset: int = Query(default=0, ge=0),
) -> dict:
    items, total = repo.list_runs(limit=limit, offset=offset)
    return {"items": items, "total": total, "limit": limit, "offset": offset}


@router.get("/metrics")
def metrics(repo: Annotated[RunRepository, Depends(get_repo)]) -> dict:
    return repo.metrics()


def _client(request: Request) -> str:
    from app.api.deps import client_key

    return client_key(request)
