# CLAUDE.md

SmartDesk AI: first-line IT/HR help desk agent (KB answers, ticket creation, ticket status). **`SPEC.md` is the source of truth.** Read the relevant section before building a feature, and cite FR ids (e.g. FR-G12) in commits and tests. If code and spec disagree, stop and propose the spec change. Edit SPEC.md only after I approve. Don't silently diverge.

## Stack
Python 3.12 · LangGraph + LangChain · OpenAI (`gpt-5.4-mini` main, `gpt-4o-mini` light, via `get_llm(task)`) · ChromaDB + rank_bm25 + bge-reranker-base · GitHub Issues (REST first) · SQLite (checkpointer, email-to-ticket map, long-term memory) · Streamlit · Redis (optional) · Presidio · Ragas + pytest. Full decisions table: SPEC.md §2. Layout: SPEC.md §6.

## Commands
Nothing is built yet. These are the intended entry points; update this list as they appear.
```
uv python install 3.12                   # once, if 3.12 is missing
uv venv --python 3.12 .venv              # create the environment (once)
source .venv/bin/activate                # switch it on (each new terminal)
uv pip install -r requirements.txt
python -m spacy download en_core_web_lg  # spaCy model for Presidio (pip does not install it)
python -m smartdesk_ai.rag.ingest        # build/refresh indexes (idempotent)
python -m smartdesk_ai.agent             # CLI chat (--debug shows chunks and scores)
streamlit run smartdesk_ai/ui/app.py     # UI + Monitoring tab
python scripts/seed_tickets.py           # idempotent seed tickets
pytest                                   # unit tests (mock mode, no network)
python -m evals.run                      # Ragas + retrieval ablations
```

## Conventions
- Config in `config.yaml` (no secrets): model IDs, tier map, prices, thresholds, prompt versions. Secrets only via env vars; keep `.env.example` current, never commit `.env`.
- Never hardcode a model ID or threshold. Get the LLM through `get_llm(task)`.
- Prompts live in `agent/prompts/` as versioned files. Edit by adding a new version.
- All tool calls go through the registry + executor (`tools/`), never called directly by agents. Ticket access goes only through the `tickets/` client wrapper.
- Each retrieval stage (BM25, dense, fusion, rerank) is a separate swappable function.
- Raw emails and unredacted PII must never reach logs or traces. Names are reduced to "John D." (FR-H1, FR-G11).
- Only `thread_id` goes in LangGraph `config["configurable"]`; it is persisted to the checkpoint DB, so never put an email or other PII there (FR-G11, PLAN §12.1).
- Every Pydantic type stored in graph state must be registered with the checkpoint serializer (PLAN §12.1).
- Never enable LangSmith via `LANGSMITH_TRACING=true` auto-tracing; use only the explicitly configured, masked client from `ops/tracing` (PLAN §12).
- Never put an email or other PII in trace user IDs, session IDs, tags, span names or error messages; redact before data reaches any trace (FR-G11).
- Optional components (reranker, Presidio, Redis, tracing) must degrade gracefully, never crash startup (FR-F7).
- Typed code with Pydantic schemas; small modules with meaningful names. All code and docs must be the user's own work (no copied code).

## Workflow rules
- Core first: SPEC §3A to 3G, then bonus. Build the single-path agent before multi-agent fan-out (§3I).
- Work one task from TASKS.md at a time: write its test first, implement, run pytest, then mark it done. Don’t start the next task unprompted.
- Write evals (SPEC §7) before the code they test. Eval questions must be paraphrased, never copied from indexed KB text (FR-A18).
- Tests use mock mode for GitHub and LLM; no test may create real issues or spend API budget.
- Never create a ticket without explicit confirmation (FR-B2, G12). Never invent ticket details (FR-B13).
- Real calls (OpenAI, GitHub, public tickets repo) cost money or are public: confirm with the user before running anything beyond mocks.
- Keep PLAN.md and TASKS.md current as work progresses. Commit in small steps, only when asked.
- Deadline: Oct 23, 2026; core target Oct 18. Default branch is `main`.

