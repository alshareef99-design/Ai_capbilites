from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from app.config import get_settings
from app.rag.index import reset_index
from app.rag.ingest import ingest_knowledge


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("INDEX_DIR", str(tmp_path / "index"))
    monkeypatch.setenv("DATABASE_PATH", str(tmp_path / "diwan.db"))
    monkeypatch.setenv("OPENAI_API_KEY", "")
    monkeypatch.setenv("DIWAN_API_KEY", "")
    get_settings.cache_clear()
    reset_index()
    ingest_knowledge(reset=True)
    from app.main import app

    with TestClient(app) as test_client:
        yield test_client
    get_settings.cache_clear()
    reset_index()
