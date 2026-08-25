"""Diwan — bilingual enterprise AI capabilities API and studio."""

from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from app.api.deps import RateLimiter
from app.api.routes import router
from app.config import get_settings
from app.core.logging import configure_logging, get_logger
from app.repositories.runs import RunRepository
from app.services.capabilities import ensure_knowledge_loaded

configure_logging()
logger = get_logger("diwan")

STATIC_DIR = Path(__file__).resolve().parent / "static"


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    app.state.limiter = RateLimiter(settings.rate_limit_per_minute)
    app.state.repo = RunRepository(settings.database_path)
    ensure_knowledge_loaded()
    logger.info("startup mode=%s model=%s", "demo" if settings.demo_mode else "live", settings.openai_model)
    yield
    logger.info("shutdown")


app = FastAPI(
    title="Diwan — Enterprise AI Capabilities",
    description=(
        "البحث المؤسسي، الأسئلة والأجوبة، التلخيص، تحليل الوثائق، الاستخراج، ودعم القرار. "
        "Grounded LangGraph workflows over an enterprise knowledge base."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

_settings = get_settings()
app.add_middleware(
    CORSMiddleware,
    allow_origins=_settings.allowed_origins,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type", "X-API-Key"],
)

app.include_router(router, prefix="/v1")

if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.exception_handler(Exception)
async def unhandled_error(_: Request, exc: Exception) -> JSONResponse:
    if isinstance(exc, HTTPException):
        raise exc
    logger.exception("Unhandled error: %s", exc)
    return JSONResponse(status_code=500, content={"detail": "Something went wrong. Try again."})


@app.get("/")
def studio() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")
