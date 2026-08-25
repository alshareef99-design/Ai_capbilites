from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any


class RunRepository:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._init()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS runs (
                    id TEXT PRIMARY KEY,
                    capability TEXT NOT NULL,
                    language TEXT NOT NULL,
                    query_hash TEXT NOT NULL,
                    grounded INTEGER NOT NULL,
                    refused INTEGER NOT NULL,
                    latency_ms INTEGER NOT NULL,
                    tokens INTEGER NOT NULL,
                    cost_usd REAL NOT NULL,
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                )
                """
            )

    def save(
        self,
        run_id: str,
        capability: str,
        language: str,
        query_hash: str,
        grounded: bool,
        refused: bool,
        latency_ms: int,
        tokens: int,
        cost_usd: float,
    ) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO runs (
                    id, capability, language, query_hash, grounded, refused,
                    latency_ms, tokens, cost_usd
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    run_id,
                    capability,
                    language,
                    query_hash,
                    int(grounded),
                    int(refused),
                    latency_ms,
                    tokens,
                    cost_usd,
                ),
            )

    def list_runs(self, limit: int = 20, offset: int = 0) -> tuple[list[dict[str, Any]], int]:
        with self._connect() as conn:
            total = conn.execute("SELECT COUNT(*) FROM runs").fetchone()[0]
            rows = conn.execute(
                """
                SELECT id, capability, language, grounded, refused, latency_ms,
                       tokens, cost_usd, created_at
                FROM runs
                ORDER BY created_at DESC
                LIMIT ? OFFSET ?
                """,
                (limit, offset),
            ).fetchall()
        return [dict(row) for row in rows], int(total)

    def metrics(self) -> dict[str, Any]:
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT COUNT(*) AS runs,
                       COALESCE(AVG(latency_ms), 0) AS avg_latency_ms,
                       COALESCE(SUM(tokens), 0) AS tokens,
                       COALESCE(SUM(cost_usd), 0) AS cost_usd,
                       COALESCE(AVG(grounded), 0) AS grounded_rate
                FROM runs
                """
            ).fetchone()
        return dict(row)
