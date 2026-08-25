# Diwan / ديوان — Enterprise AI Capabilities

Bilingual (Arabic + English) LangGraph service for:

| Capability | Arabic | What it does |
|---|---|---|
| **Enterprise search** | البحث المؤسسي | Ranked passages from policies and contracts, with citations |
| **Q&A** | الأسئلة والأجوبة | Grounded answers; abstains when the corpus does not support the claim |
| **Summarization** | التلخيص | Executive summary, bullets, and key facts |
| **Document analysis** | تحليل الوثائق | Themes, risks, obligations, stakeholders, gaps |
| **Extraction** | الاستخراج | Entities, amounts, dates, clauses as structured JSON |
| **Decision support** | دعم القرار | Options, trade-offs, recommendation, and confidence |

The graph is the same for every request:

```text
safety → classify → retrieve → grade → capability node
```

Retrieval is a bilingual TF-IDF index over Markdown knowledge. No embedding download is required. Set `OPENAI_API_KEY` to swap the deterministic demo generator for a live model.

## Architecture

```text
Studio UI  ──POST /v1/run──►  FastAPI
                                │
                                ▼
                     LangGraph compiled graph
          safety → classify → retrieve → grade → dispatch
                                │
                                ▼
              DocumentIndex (TF-IDF) + SQLite run log
```

- **UI** — operator studio at `/` (RTL when the query is Arabic)
- **Domain** — `RunRequest` / `RunResponse` plus structured briefs for summary, analysis, extraction, and decisions
- **Services** — `app/services/capabilities.py` orchestrates the graph and metrics
- **Data** — in-process index persisted under `data/index`; run history in SQLite
- **Prompts** — bilingual templates in `app/prompts/templates.py`, not inside graph nodes

## Quick start

Python 3.12.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
uvicorn app.main:app --reload
```

Open [http://localhost:8000](http://localhost:8000). Click a sample chip, then **Run**.

API docs: [http://localhost:8000/docs](http://localhost:8000/docs)

Demo mode is the default (no API key). To use a live model:

```bash
# .env
OPENAI_API_KEY=sk-...
OPENAI_MODEL=gpt-4o-mini
```

Any OpenAI-compatible server works via `OPENAI_BASE_URL`.

## Example requests

**Search**

```bash
curl -s http://localhost:8000/v1/search \
  -H 'Content-Type: application/json' \
  -d '{"query":"vendor onboarding KYC"}'
```

**Arabic Q&A**

```bash
curl -s http://localhost:8000/v1/ask \
  -H 'Content-Type: application/json' \
  -d '{"query":"كم يوم الإجازة السنوية للموظف؟","evaluate":true}'
```

**Decision support**

```bash
curl -s http://localhost:8000/v1/decide \
  -H 'Content-Type: application/json' \
  -d '{"query":"Should we migrate production to Azure or stay on-premises?"}'
```

**Auto-route** (`classify` picks the capability)

```bash
curl -s http://localhost:8000/v1/run \
  -H 'Content-Type: application/json' \
  -d '{"query":"لخص سياسة المشتريات","capability":"auto"}'
```

## Tests and eval

```bash
pytest -q
python -m app.eval.runner
```

Eval cases live in `data/eval_cases.yaml`. They check grounded Q&A, abstention, extraction of the vendor SLA, and the Azure recommendation.

## Deploy

```bash
docker compose up --build
```

Optional operator key: set `DIWAN_API_KEY` and send `X-API-Key` on mutating routes. Per-IP rate limiting is on by default.

## Project map

```text
app/graphs/pipeline.py   LangGraph: safety → classify → retrieve → grade → dispatch
app/graphs/demo.py       Deterministic stand-in for demo mode
app/prompts/templates.py Bilingual system prompts
app/rag/                 Chunking, TF-IDF index, ingest
app/services/            API orchestration + metrics
app/static/              Studio UI
data/knowledge/{ar,en}   Sample enterprise corpus
```

## Design notes

- **Grounding over fluency**: Q&A abstains when retrieval is empty or off-topic.
- **Arabic-first**: mixed queries reply in Arabic; Arabic system prompts use MSA.
- **Structured outputs**: analysis, extraction, and decisions are Pydantic models, not free text.
- **Measure what you ship**: every `/v1/run` returns latency, tokens, and estimated USD cost.
- **Secrets stay out of logs**: the run store keeps a query hash, not the raw question.

## Extending

- Swap TF-IDF for FAISS/pgvector by replacing `app/rag/index.py`.
- Add PDF/DOCX loaders in `ingest_upload` once those packages are approved.
- Persist metrics to Prometheus using `MetricsPayload`.
- Turn on LangSmith with `LANGCHAIN_TRACING_V2` when running live models.
