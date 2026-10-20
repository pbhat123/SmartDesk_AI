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
python scripts/smoke_ui.py               # real Streamlit server + browser, mock mode, temp data (about 15 s); needs Playwright
scripts/install_hooks.sh                 # once per clone: pre-commit hook runs pytest and blocks a commit on any failure
python -m evals.run                      # Ragas + retrieval ablations
python -m evals.tier_comparison          # right-sizing comparison: prints the estimate and stops (--mock is free, --confirm-spend runs it live)
python -m smartdesk_ai.ops.drift         # drift check on the golden set: shows the cost estimate, asks to confirm (--estimate only prints it)
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
- **Definition of done: unit tests pass AND `scripts/smoke_ui.py` passes. Run it before reporting any UI or agent-flow task as done.**
- Core first: SPEC §3A to 3G, then bonus. Build the single-path agent before multi-agent fan-out (§3I).
- Work through the tasks I name, in order. For each task: write its test first, implement, run pytest, tick the box, and commit it separately with its FR ids. Stop at the end of the named range, or at the first failure you can't fix, or at any [CONFIRM] step. Tick and commit only when pytest passes (`pytest -q && git commit ...`); never commit on red, not even to amend later.
- Write evals (SPEC §7) before the code they test. Eval questions must be paraphrased, never copied from indexed KB text (FR-A18).
- Tests use mock mode for GitHub and LLM; no test may create real issues or spend API budget.
- Never create a ticket without explicit confirmation (FR-B2, G12). Never invent ticket details (FR-B13).
- Real calls (OpenAI, GitHub, public tickets repo) cost money or are public: confirm with the user before running anything beyond mocks.
- Keep PLAN.md and TASKS.md current as work progresses. Commit each completed task separately with its FR ids, as part of a named batch. Never push without asking.
- Deadline: Oct 23, 2026; core target Oct 18. Default branch is `main`.


## Working rules (added 2026-10-18)
- Every paid run: show an estimate first, use a spend ceiling plus the watchdog (`evals/judge_run.py`), and record spend in the spend ledger (`data/spend_ledger.jsonl`). Report the ledger cost, not only the tracker's.
- Commits go through the pre-commit hook (`scripts/install_hooks.sh`). Never bypass it with `--no-verify`.
- Every LLM call goes through `FallbackChat` (guarded by `tests/test_llm_observed.py`).
- Retrieval stays hybrid + rerank (see `evals/README.md`, "Retrieval ablation").
- Active versions: answer prompt v4 (workers are v4 verbatim), grounding judge v3, ticket_summary_v2 (v1 with a paraphrased title example, approved 2026-10-20), orchestrator_v9 (v8 with paraphrased examples, approved 2026-10-20; final rerun with v9 done, PLAN 14.9) plus the `merge_fragments` guard, `semantic_cache_cutoff` 0.97, escalation threshold 0.40. The post-Phase-7 and post-Phase-8 final runs are done (PLAN 14.6, 14.9). Changing any of these after the final runs needs my approval, because it means rerunning gap, safety and Ragas. Prompts change only by adding a new version file.
- PII checks in tests use `tests/pii_helpers` (whole-token match only, never a bare substring such as `"98765" in blob`).
- Never run git checkout/restore/reset on files with uncommitted work. For mutation tests use git stash or a git worktree.
- Never create or edit files in the repo while a commit hook is running; wait for it to finish. Use the scratchpad for drafts.
- Paid runs go through `evals/live_guard.py` (`LiveRun`: ledger source `eval`, ceiling = 2x the estimate, watchdog; `Checkpoint` for resume). A multi-step batch lives in a driver with its stop rules in code (`python -m evals.post_phase7`, `--resume` or `--fresh`). Changing a stop rule needs my approval first. Result tables print "not run" for any file older than the batch.
- Never put golden or eval questions in a prompt (examples are same-shape paraphrases), and eval questions stay out of the KB text (FR-A18).
- After a failing eval, inspect saved results first (free) and classify each failure before changing code, a prompt or a rule. No spend and no new prompt version until I approve.
- Commit messages go in a file in the scratchpad (`git commit -F`), never an inline heredoc. The hook takes about 2.5 minutes (pytest, plus smoke for UI changes): run the commit with a long timeout and wait for it. macOS has no `timeout` command. The hook now kills pytest or smoke after 600 s (`scripts/run_with_limit.py`) and pytest has a 60 s per-test timeout (`pytest-timeout`; mark a test that needs longer `@pytest.mark.slow` plus `@pytest.mark.timeout(N)`). While waiting on any hook or background run, check progress about every 10 minutes and report a hang rather than waiting silently. A hang found once: ChromaDB's process-wide client cache stalls the native start after about 450 clients, so `tests/conftest.py` clears it after each test module.
