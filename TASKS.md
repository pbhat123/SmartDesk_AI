# TASKS.md

Derived from PLAN.md section 7, against SPEC.md draft v3 (approved). Core milestone is SPEC sections 3A to 3G, due 10-18. Safety and ops items H1, H2, H3, H8, H13 are due 10-19.

Rules (CLAUDE.md): Work through the tasks the user names, in order, per CLAUDE.md: test first, implement, run pytest, tick the box, commit each task separately. Stop at the end of the named range, at the first unfixable failure, or at any [CONFIRM] step. Tests run in mock mode (no network, no real issues, no API spend). Anything beyond mocks (OpenAI, GitHub, public tickets repo) needs explicit confirmation first, marked **[CONFIRM]**.

Conventions: paths are relative to the repo root; `sd/` means `smartdesk_ai/`. "Test" is the acceptance test that must pass before the box is ticked; the FR ids in each row are cited in commits and tests.

---

## Phase 0: Scaffold + spikes (10-13 to 10-14)
Exit: `pip install` works, `pytest` runs, app starts with no keys.

| Done | ID | Task | FRs | Files | Acceptance test |
|---|---|---|---|---|---|
| [x] | T0.1 | Repo scaffold: package dirs per SPEC section 6, `requirements.txt` (Ragas pinned), `pytest.ini`, `.gitignore` check | F3, F5 | `sd/**/__init__.py`, `requirements.txt`, `pytest.ini`, `tests/test_smoke.py` | Fresh venv: `pip install -r requirements.txt` succeeds; `pytest` collects and passes the smoke test |
| [x] | T0.2 | `.env.example` and secrets-via-env loader | F3, F5 | `.env.example`, `sd/config/env.py`, `tests/test_env.py` | `.env.example` lists OPENAI_API_KEY, GITHUB_TOKEN, tickets repo name and optional LANGSMITH/LANGFUSE/REDIS keys with no real values; `.env` is gitignored; loader works with all keys absent |
| [x] | T0.3 | Config loader for `config.yaml` (typed, Pydantic): model IDs, tier map, windows, prices, budgets, rate limits, thresholds, prompt versions, `input.max_chars`, injection threshold, `budget.eval_bypass` | H16 | `config.yaml`, `sd/config/settings.py`, `tests/test_config.py` | Config loads and validates; a missing or mistyped key fails with a clear error; no model ID or threshold appears hardcoded outside config (grep test) |
| [x] | T0.4 | Versioned prompt loader (`<name>_v<N>.md`, active version from config) | H14 | `sd/agent/prompts/`, `sd/agent/prompt_loader.py`, `tests/test_prompt_loader.py` | Loader returns text and version for the configured version; changing the version in config changes what is loaded and reported; unknown version raises |
| [x] | T0.5 | Component health registry (reranker, Presidio, Redis, tracing) with degraded-mode reporting | F7 | `sd/ops/health.py`, `tests/test_health.py` | Forcing a component to fail records a degraded mode and a warning, raises nothing; registry lists all degraded modes |
| [x] | T0.6 | App skeleton: `ChatService` with no-op input/output guard hooks, CLI entry `python -m smartdesk_ai.agent` | F6, F7 | `sd/agent/__main__.py`, `sd/agent/chat_service.py`, `tests/test_startup.py` | App starts and answers a stub reply with no keys, no Redis, no tracing keys |
| [x] | T0.7 | Spike: LangGraph `interrupt` + `SqliteSaver` + Pydantic state + `Send`; findings written to PLAN.md | G12, H10, I4 | `spikes/langgraph_spike.py`, `tests/test_spike_langgraph.py` | Graph pauses at interrupt, resumes with a new process using the same thread ID, parallel `Send` branches join; result noted in PLAN.md |
| [x] | T0.8 | Spike: Langfuse and LangSmith masking hooks (mock/no keys) | G11 | `spikes/tracing_masking_spike.py`, `tests/test_spike_masking.py` | A raw email passed through each SDK's masking hook arrives masked; if masking cannot be guaranteed, the finding says to send only locally redacted spans |
| [x] | T0.9 | **[CONFIRM]** Spike: one real `gpt-5.4-mini` call to verify params (`reasoning_effort="none"`), model IDs, prices | H16, H17 | `spikes/openai_param_check.py`, notes in PLAN.md | One call succeeds with the configured params; any param/price/ID discrepancy is recorded as a config change proposal. Skipped without your OK |

---

## Phase 1: KB + eval data (10-13 to 10-15)
Exit: 30+ entries (aim 50), 40+ KB Qs, 15+ gaps, 30+ intents, `GAPS.md`, `SOURCES.md`. LLM-generated content costs money: **[CONFIRM]** before generating.

| Done | ID | Task | FRs | Files | Acceptance test |
|---|---|---|---|---|---|
| [x] | T1.1 | KB entry schema and validator (`##` section + YAML block: id, domain, topic, title, owner, last_reviewed) | A1 | `sd/rag/kb_schema.py`, `tests/test_kb_schema.py` | Validator accepts a good entry; rejects missing field, bad domain (not IT/HR), duplicate id; domain is read per entry, not per file |
| [x] | T1.2 | Fictional company profile + KB Markdown files (30+ entries, aim 50, concrete details) **[CONFIRM if LLM-generated]** | A1, A9 | `kb/*.md` | `tests/test_kb_content.py`: at least 30 valid entries, unique ids, each has portal/extension/number-style detail fields, mix of simple/how-to/policy types |
| [x] | T1.3 | KB domain and topic coverage per document section 2.1.1 (IT Support, HR Policies, Onboarding, Payroll; equipment allocation tagged IT) | A8 | `kb/*.md` | Test lists every required topic and asserts at least one entry per topic, with IT equipment allocation as `IT` and the rest of onboarding as `HR` |
| [x] | T1.4 | `kb/qa.json` Q&A pairs (question, answer, category, type, entry_id) **[CONFIRM if LLM-generated]** | A10 | `kb/qa.json`, `tests/test_qa_json.py` | Every pair has all fields, `type` in {simple, how-to, policy interpretation}, and an `entry_id` that exists; JSON is not indexed (checked in T2.x ingest test) |
| [x] | T1.5 | `kb/SOURCES.md`: each public source, licence, and adaptation notes | A10 | `kb/SOURCES.md`, `tests/test_sources_doc.py` | File exists and lists a source and licence per adapted item, or states none were used |
| [x] | T1.6 | `kb/GAPS.md` and deliberate gaps (monitor flickering, office parking, cafeteria menu, desk booking) | A2 | `kb/GAPS.md`, `tests/test_gaps_doc.py` | Each listed gap topic has no matching KB entry (keyword check against entry titles/topics) |
| [x] | T1.7 | KB eval set: 40+ paraphrased questions with expected entry ids and reference answers | A18 | `evals/kb_questions.jsonl`, `tests/test_eval_kb_set.py` | 40+ rows; every `expected_entry_id` exists; no question shares a long n-gram span with indexed text (leakage check); dev/test split field present (F-17) |
| [x] | T1.8 | Gap set: 15+ questions incl. grey-zone workplace cases | A2, D9 | `evals/gap_questions.jsonl`, `tests/test_eval_gap_set.py` | 15+ rows; includes parking, cafeteria, desk booking, monitor flickering; dev/test split present |
| [x] | T1.9 | Intent set: 30+ utterances with expected intent, route, tool or no tool, incl. confidently ambiguous and out-of-scope | D1, D8, D9, G2 | `evals/intents.jsonl`, `tests/test_eval_intents_set.py` | 30+ rows; all six intents appear; schema valid; out-of-scope rows include weather, recipes, trivia, coding help |
| [x] | T1.10 | Category set: utterances with expected IT/HR/MISC | B7, D7 | `evals/categories.jsonl`, `tests/test_eval_categories_set.py` | Schema valid; all three categories present; MISC rows are facilities-style |
| [x] | T1.11 | Compound set: 10+ multi-part questions (some with a gap part, some misroutable) | I4 to I7 (data only) | `evals/compound_questions.jsonl`, `tests/test_eval_compound_set.py` | 10+ rows; at least 2 with a gap part, 2 misroutable; each part tagged with expected domain and answerable flag |
| [x] | T1.12 | Scenario scripts: KB answer, ticket create (yes/no/edit), status (0/1/2+), API down, degraded mode; flow scripts A/B/C | SPEC section 7 | `evals/scenarios/*.yaml`, `tests/test_eval_scenarios_set.py` | Scripts parse against a schema; flows A, B, C match document section 8 steps |
| [x] | T1.13 | Safety set: 20+ adversarial inputs with expected behaviour | H3 (data only) | `evals/safety_set.jsonl`, `tests/test_eval_safety_set.py` | 20+ rows covering injection, PII, names, toxic, off-topic, skip-confirmation; each has an expected behaviour |

---

## Phase 2: RAG (10-14 to 10-16)
Exit: ingest twice gives same chunk count; Recall@k table for 4 strategies; chunking ablation; threshold tuned.

| Done | ID | Task | FRs | Files | Acceptance test |
|---|---|---|---|---|---|
| [x] | T2.1 | Heading-based chunker: one chunk per entry, split above ~600 tokens with ~15% overlap, parent id kept, "Document > Section" header prepended | A3 | `sd/rag/chunking.py`, `tests/test_chunking.py` | Short entry gives 1 chunk; long entry gives pieces that all keep parent `id` with ~15% overlap; header prepended to embedded text |
| [x] | T2.2 | Ingest pipeline: parse, chunk, embed (Chroma), build BM25; idempotent; prunes deleted/renamed entries; JSON Q&A not indexed | A11 | `sd/rag/ingest.py`, `tests/test_ingest.py` | Ingest reads only the policy files (`company.md`, `it_support.md`, `onboarding.md`, `hr_policies.md`, `payroll_comp.md`); `GAPS.md`, `SOURCES.md` and `qa.json` are never indexed (test asserts the file list and that their content is absent from both indexes); ingest twice gives the same chunk count; deleting an entry removes its chunks |
| [x] | T2.3 | BM25 retriever stage (top-20, optional domain filter) | A6 | `sd/rag/retrievers.py`, `tests/test_bm25.py` | Returns top-20 ranked chunks; domain filter restricts results; callable alone |
| [x] | T2.4 | Dense retriever stage (top-20, optional domain filter) | A6 | `sd/rag/retrievers.py`, `tests/test_dense.py` | Same as T2.3 for dense; embeddings mocked in tests |
| [x] | T2.5 | RRF fusion (k=60) and dedupe by `chunk_id` | A6 | `sd/rag/fusion.py`, `tests/test_fusion.py` | Known rankings fuse to the hand-computed RRF order; no `chunk_id` appears twice in the output |
| [x] | T2.6 | Cross-encoder rerank (bge-reranker-base) with sigmoid-normalised 0..1 scores, then collapse to entries (best piece wins), top-5 full parent entries | A6, A15 (signals 1-2) | `sd/rag/rerank.py`, `sd/rag/pipeline.py`, `tests/test_rerank.py` | Scores in [0,1]; no `chunk_id` twice in reranker input; no entry id twice in final top-5; output carries full parent text; reranker mocked |
| [x] | T2.7 | Reranker fallback: skip rerank, keep RRF order; the escalation threshold is unaffected (SPEC v3.1) | F7 (reranker), A15 | `sd/rag/pipeline.py`, `sd/ops/health.py`, `tests/test_reranker_fallback.py` | With reranker forced to fail, pipeline still returns results in RRF order (`ordered_by="rrf"`), `top_cosine`, threshold and `answerable` equal the working-reranker values, degraded mode is recorded |
| [x] | T2.8 | Answerable signals 1-2 (no chunks, top dense cosine below the one escalation threshold) | A15, A5 | `sd/rag/answerable.py`, `tests/test_answerable.py` | Zero chunks gives `answerable=False`; top dense cosine below `thresholds.escalation` gives False; reranker scores never change the result; there is no second threshold |
| [x] | T2.9 | Staleness: hash per KB file; only changed file's chunks flagged; re-ingest re-embeds flagged, clears flags | H6 | `sd/rag/staleness.py`, `sd/rag/ingest.py`, `tests/test_staleness.py` | Editing one file flags only its chunks; re-ingest clears flags and removes deleted-entry chunks; cache-clear event fires |
| [x] | T2.10 | Retrieval ablation harness: Recall@1/3/5 for BM25, dense, hybrid, hybrid+rerank | A7 | `evals/retrieval_eval.py`, `tests/test_retrieval_eval.py` | `python -m evals.run --only retrieval` prints the 4-strategy table (mock embeddings in test; real run after KB exists) |
| [x] | T2.11 | Chunking ablation: heading-based vs fixed 512/50 | A17 | `evals/chunking_ablation.py`, `tests/test_chunking_ablation.py` | Both chunkers run on the KB set and a Recall@k comparison table is produced |
| [x] | T2.12 | Threshold tuning on the dev half (top dense cosine); report on held-out test half | A5, A15 | `evals/tune_threshold.py`, `config.yaml`, `tests/test_tune_threshold.py` | Script picks the value that best separates KB from gap on dev, reports test-half separation, and records a chosen value's dev and test numbers in `evals/results`; chosen value (0.45) written to config as provisional until T6.4 (needs real embeddings, **[CONFIRM]**, done) |

---

## Phase 3: LLM, tools, tickets (10-15 to 10-17)
Exit: executor tests (retry x3, no retry on bad args, no create before confirmation, lost response no duplicate); seed script idempotent (mock first, live after **[CONFIRM]**).

| Done | ID | Task | FRs | Files | Acceptance test |
|---|---|---|---|---|---|
| [x] | T3.1 | Task-to-tier map and `get_llm(task)` factory (mock LLM in tests) | H17, H16 | `sd/llm/factory.py`, `sd/llm/tiers.py`, `tests/test_get_llm.py` | Each task returns its configured tier's model; changing one task's tier in config changes only that task; tasks include CONFIRM_PARSE and CHITCHAT on Light |
| [x] | T3.2 | LLM retry with backoff and Main/Light outage fallback; no fallback to Main at budget level >= 1; prompt-size check against Light window | F2 | `sd/llm/retry.py`, `sd/llm/fallback.py`, `tests/test_llm_fallback.py` | Main failure falls back to Light and vice versa; both failing gives a graceful message; level>=1 never falls back to Main; oversize prompt is not sent to Light |
| [x] | T3.3 | Tool definitions with typed args: `search_kb(query, domain)`, `create_ticket`, `get_tickets` | G1 | `sd/tools/definitions.py`, `tests/test_tool_definitions.py` | Each tool has name, description, Pydantic args; `domain` accepts IT/HR/None only |
| [x] | T3.4 | Tool registry (name to function) | G3 | `sd/tools/registry.py`, `tests/test_registry.py` | Lookup returns the function; unknown name errors; registry is not exposed to the LLM as a tool |
| [x] | T3.5 | Executor: retry max 3 attempts, short backoff, transient-only, then "unavailable" | G4 | `sd/tools/executor.py`, `tests/test_executor_retry.py` | Failing mock: transient error attempts exactly 3 then returns unavailable; bad args attempted once, not retried |
| [x] | T3.6 | Executor guards: schema validation, `create_ticket` only after confirmation, missing/invalid email error | G5 | `sd/tools/executor.py`, `tests/test_executor_guards.py` | `create_ticket` without a confirmation flag does not run; invalid email returns an ask-for-email error; schema errors return validation error |
| [x] | T3.7 | Duplicate protection: request ID in issue body, written to SQLite before the call, list-issues check before retry | G6 | `sd/tools/executor.py`, `sd/tickets/sqlite_map.py`, `tests/test_duplicate_protection.py` | Mock where the create response is lost then retried: exactly one issue exists; check uses list-issues, not search |
| [x] | T3.8 | `search_kb` tool implementation returning chunks, scores, `answerable` | G7 | `sd/tools/search_kb.py`, `tests/test_search_kb_tool.py` | Result carries chunks, scores and `answerable` from signals 1-2; domain filter honoured |
| [x] | T3.9 | Per-call metrics in executor (tool, agent, outcome, attempts, latency per attempt and total) | G9 | `sd/ops/metrics.py`, `sd/tools/executor.py`, `tests/test_tool_metrics.py` | After a failing-mock run the record shows outcome, attempts used, and latencies |
| [x] | T3.10 | Tickets client wrapper: interface + mock mode + REST implementation (hides REST/MCP) | B12, B10 | `sd/tickets/github_client.py`, `sd/tickets/schemas.py`, `tests/test_tickets_client.py` | Agent code imports only the wrapper; mock mode returns realistic dummy data; REST impl is exercised with a mocked HTTP layer only |
| [x] | T3.11 | Issue format: title, body with `Employee: <email>` and request ID, category/priority labels, status labels; labels created once | B11, B3 | `sd/tickets/github_client.py`, `scripts/setup_labels.py`, `tests/test_issue_format.py` | Built issue has exact `Employee: <email>` line, request ID, correct labels; category only IT/HR/MISC; priority only low/medium/high/critical |
| [x] | T3.12 | No-invention rule: ticket ID, link and status only from GitHub response; missing email asked for, never guessed | B13 | `sd/tickets/github_client.py`, `sd/tools/executor.py`, `tests/test_no_invented_details.py` | Mocked response with specific id/link/status: reply shows exactly those; no email provided gives ask-for-email |
| [x] | T3.13 | SQLite email-to-issue map | C2, C10, B3 | `sd/tickets/sqlite_map.py`, `tests/test_sqlite_map.py` | Create stores email to issue number; lookup returns all issue numbers for an email; persists across reopen |
| [ ] | T3.14 | Status mapping from state + labels | C6, C9 | `sd/tickets/status.py`, `tests/test_status_mapping.py` | Table test: open/no label = Open; open+in-progress = In Progress; closed+resolved = Resolved; closed = Closed; open+both = In Progress; open+resolved only = Open |
| [ ] | T3.15 | Latest team update from most recent issue comment | C9 | `sd/tickets/github_client.py`, `tests/test_latest_comment.py` | Issue with several comments returns the newest; none gives no update |
| [ ] | T3.16 | Search fallback: find `Employee: <email>` in issue bodies and backfill the map | C11 | `sd/tickets/search_fallback.py`, `tests/test_search_fallback.py` | Hand-created mock issue not in map is found by email and then appears in the map |
| [ ] | T3.17 | Seed script (idempotent, via the same client, writes map rows, all statuses, team comment, fake emails for 0/1/2+) | C8 | `scripts/seed_tickets.py`, `tests/test_seed_tickets.py` | Mock mode: run twice gives the same issue count; each seeded email's tickets found; every status present. Live run after **[CONFIRM]** |
| [ ] | T3.18 | GitHub unreachable handling: polite message, suggest retry | F1 | `sd/tickets/github_client.py`, `sd/tools/executor.py`, `tests/test_github_unreachable.py` | Mock network error after 3 attempts yields unavailable result mapped to a polite retry-later message |

---

## Phase 4: Single-path agent, core milestone (10-16 to 10-18)
Exit: flows A/B/C scripted; 0/1/2+ status; confirm yes/no/edit; routing set; CLI `--debug` shows chunks and scores.

| Done | ID | Task | FRs | Files | Acceptance test |
|---|---|---|---|---|---|
| [ ] | T4.1 | SQLite checkpointer for session state + pending draft; restart-safe | H10, G12 | `sd/memory/checkpointer.py`, `tests/test_checkpointer_restart.py` | Restart keeps pending draft and history for a thread ID |
| [ ] | T4.2 | Graph state (Pydantic): email, last topic, pending draft, `awaiting`, messages; thread ID per session | D2, G12 | `sd/agent/state.py`, `tests/test_state.py` | State round-trips through the SQLite checkpointer; two thread IDs never mix |
| [ ] | T4.3 | Intent classifier with confidence and `awaiting` input; six intents | D1, D2 | `sd/agent/intents.py`, `tests/test_intent_classifier.py` | Over `evals/intents.jsonl` (mock LLM fixtures) route matches expected; "yes" with `awaiting=offer_reply` routes correctly; new intent mid-flow clears pending offer |
| [ ] | T4.4 | Routing: classifier first; fallback to model tool pick on error/invalid/low confidence; mapping `search_kb`/`create_ticket`/`get_tickets`/no tool | D8, G2 | `sd/agent/router.py`, `tests/test_routing.py` | Forced classifier error gives route from tool pick; confident `ambiguous` gets exactly one clarifying question and no fallback; no tool maps to chitchat |
| [ ] | T4.5 | Scope boundary: workplace gaps go to KB path; unrelated gets deflection with no ticket | D9, D3 | `sd/agent/router.py`, `tests/test_scope_boundary.py` | Parking/cafeteria/desk booking end in a ticket offer; weather/recipes/trivia end in deflection with no ticket offer; unsure defaults to in-domain |
| [ ] | T4.6 | Answer step: answer from retrieved context only; "I don't know" when insufficient (signal 3 from the LLM) | A4, A15 (signal 3) | `sd/agent/answer.py`, `sd/agent/prompts/answer_v1.md`, `tests/test_answer_step.py` | Insufficient context yields an "I don't know" reply; the LLM-reports-insufficient flag escalates even when scores pass |
| [ ] | T4.7 | Citations on every KB answer | A13 | `sd/agent/answer.py`, `tests/test_citations.py` | Every KB answer in the mock eval run names at least one entry id that was retrieved |
| [ ] | T4.8 | KB-first rule: retrieval always attempted before a ticket offer | A14 | `sd/agent/graph.py`, `tests/test_kb_first.py` | Trace of any ticket offer shows a prior `search_kb` call |
| [ ] | T4.9 | Whole-KB retry once before ticket offer (core: search whole KB) | A19 (trivial), A14 | `sd/agent/graph.py`, `tests/test_whole_kb_retry.py` | An unanswerable question triggers exactly one extra no-domain search before the offer |
| [ ] | T4.10 | Escalation handoff: offer plus email request merged; known email gives plain yes/no; refusal ends flow | A16 | `sd/agent/ticket_agent.py`, `tests/test_escalation_handoff.py` | Email given counts as agreeing; known email gives yes/no question; "no" ends flow with no ticket |
| [ ] | T4.11 | Retrieval debug output (chunks + scores) in CLI `--debug` | A12 | `sd/agent/__main__.py`, `tests/test_debug_flag.py` | `--debug` output lists chunks with scores and the generated answer for a sample query |
| [ ] | T4.12 | Ticket flow: collect email, summary, description; reuse session email | B1, B5, C1 | `sd/agent/ticket_agent.py`, `tests/test_ticket_collect.py` | Known email is not re-asked; missing email is asked; invalid email re-asked |
| [ ] | T4.13 | Description summarises conversation so far; employee can add details | B6 | `sd/agent/ticket_agent.py`, `sd/agent/prompts/ticket_summary_v1.md`, `tests/test_ticket_description.py` | Description contains only what the employee said; added details appear in the next summary |
| [ ] | T4.14 | Category inference (IT/HR/MISC) and priority inference (Low/Medium/High/Critical, never asked) | B7 | `sd/agent/ticket_agent.py`, `tests/test_category_priority.py` | Over `evals/categories.jsonl` category matches; priority only accepts four values; no separate priority question |
| [ ] | T4.15 | Confirmation summary with all fields | B8 | `sd/agent/ticket_agent.py`, `tests/test_confirmation_summary.py` | Summary contains title, description, category, email, priority if set, and asks for explicit confirmation |
| [ ] | T4.16 | Approval gate via LangGraph interrupt; yes/no/edit; unclear asked once then cancel | B2, G12 | `sd/agent/graph.py`, `sd/agent/confirm.py`, `tests/test_approval_gate.py` | Proposed `create_ticket` pauses graph; "yes" creates exactly one ticket; "no" none; edit shows updated summary first; pause survives several turns; second unclear reply cancels |
| [ ] | T4.17 | CONFIRM_PARSE (Light-Model) with exact yes/no short-circuit | G12, H17 | `sd/agent/confirm.py`, `tests/test_confirm_parse.py` | Exact "yes"/"no" make zero model calls; free-text reply uses the CONFIRM_PARSE task |
| [ ] | T4.18 | Post-creation reply: ID, link, "ask for updates anytime" | B4, B9 | `sd/agent/ticket_agent.py`, `tests/test_post_creation.py` | Mocked response: reply contains exactly the returned ID and link and the updates note |
| [ ] | T4.19 | Scripted ticket scenario | B2, B4 (AC) | `tests/test_ticket_scenarios.py` | Scenario creates exactly one issue after "yes" and none after "no" |
| [ ] | T4.20 | Duplicate ticket request check (near-identical open title for same email) | F4 | `sd/agent/ticket_agent.py`, `tests/test_duplicate_request.py` | Near-identical open ticket triggers a "open another?" question before the gate |
| [ ] | T4.21 | Edge cases: empty input, invalid email, over `input.max_chars` | F4 | `sd/agent/chat_service.py`, `tests/test_edge_cases.py` | Empty gets a prompt to type; over-length gets polite shorten request; invalid email re-asked |
| [ ] | T4.22 | Status flow: trigger phrases route to status | C5 | `sd/agent/status_flow.py`, `tests/test_status_routing.py` | The three example phrases route to ticket_status via the intent set |
| [ ] | T4.23 | Status lookup: map then live GitHub fetch | C2 | `sd/agent/status_flow.py`, `tests/test_status_lookup.py` | Seeded mock: lookup returns status and latest comment fetched live |
| [ ] | T4.24 | Multiple tickets: list briefly, user picks | C3 | `sd/agent/status_flow.py`, `tests/test_status_multiple.py` | 2+ tickets produce a pick list (`awaiting=ticket_pick`); "the second one" selects it |
| [ ] | T4.25 | No tickets: clear message | C4 | `sd/agent/status_flow.py`, `tests/test_status_none.py` | 0 tickets says so clearly and offers nothing invented |
| [ ] | T4.26 | Single ticket display: ID, title, status, latest team update | C7 | `sd/agent/status_flow.py`, `tests/test_status_single.py` | 1 ticket reply contains all four fields (flow C) |
| [ ] | T4.27 | Tone and small talk: greetings, thanks, goodbyes, empathy before ticket offer | D4, D5 | `sd/agent/chitchat.py`, `sd/agent/prompts/chitchat_v1.md`, `tests/test_chitchat.py` | Small-talk intents get a reply with no tool call; ticket offer is preceded by an acknowledgement (mock-checked phrase class) |
| [ ] | T4.28 | Multi-turn: follow-ups use context; email remembered whole session | D6 | `sd/agent/graph.py`, `tests/test_multi_turn.py` | Multi-turn script never re-asks for email; "what about sick leave?" resolves using prior topic |
| [ ] | T4.29 | Single agent graph with all three capabilities; domain classification consistent | D7 | `sd/agent/graph.py`, `tests/test_single_graph.py` | One compiled graph handles KB, create, status; category set agreement threshold met on mock fixtures |
| [ ] | T4.30 | Flows A, B, C end to end in mock mode | A (AC), B (AC), C (AC) | `tests/test_flows_abc.py` | Scripted flows A, B, C pass: A answers password reset then goodbye; B offers ticket and returns ID; C lists one open ticket with team update |
| [ ] | T4.31 | **[CONFIRM]** One live run: CLI chat, create one ticket in the tickets repo, check status | B3, C2 | none (manual) | Your OK first; one real ticket created after explicit "yes", status check returns it; result noted in PLAN.md |

---

## Phase 5: Safety + ops, core-mapped (10-18 to 10-19)
Exit: safety set leakage 0, breach 0; budget and rate-limit ACs; masked logs. Due 10-19.

| Done | ID | Task | FRs | Files | Acceptance test |
|---|---|---|---|---|---|
| [ ] | T5.1 | Presidio PII detection and redaction, plus regex fallback (email, phone, card, government ID) | H1, F7 (Presidio) | `sd/guardrails/pii.py`, `tests/test_pii.py` | Injected phone/card/ID redacted before LLM call, log, trace, ticket body; with Presidio forced to fail, regex still redacts a phone number and degraded mode is logged |
| [ ] | T5.2 | Name reduction on user input only ("John Doe" to "John D."; single name unchanged) | H1, G11 | `sd/guardrails/names.py`, `tests/test_names.py` | Two full names both reduced in reply, log, trace, ticket body; KB text unaltered |
| [ ] | T5.3 | Moderation (OpenAI Moderation API, mocked) on input | H1 | `sd/guardrails/input_guard.py`, `tests/test_input_moderation.py` | Toxic message gives polite refusal, reason logged, no tool call |
| [ ] | T5.4 | Prompt-injection check: heuristic score, LLM classifier only above low threshold | H1 | `sd/guardrails/injection.py`, `tests/test_injection.py` | High-score input triggers classifier and refusal; low-score input makes zero classifier calls |
| [ ] | T5.5 | Wire input guard into `ChatService` before the graph (replace no-op hook) | H1 | `sd/agent/chat_service.py`, `tests/test_input_guard_wiring.py` | Blocked message never reaches the graph; email is allowed and masked in logs |
| [ ] | T5.6 | Output guard: Moderation, non-name PII scan, Pydantic schema validation on final reply | H2 | `sd/guardrails/output_guard.py`, `tests/test_output_guard.py` | Output with PII is redacted; toxic output replaced; invalid structured output rejected |
| [ ] | T5.7 | Runtime grounding judge (Main-Model) inside the answering step; ungrounded gives insufficient-context signal | H2 | `sd/agent/answer.py`, `sd/agent/prompts/grounding_judge_v1.md`, `tests/test_grounding_judge.py` | Unsupported draft answer is replaced by the whole-KB retry and ticket-offer path |
| [ ] | T5.8 | Safety metrics runner: PII leakage rate and policy breach rate | H3 | `evals/safety_eval.py`, `tests/test_safety_eval.py` | Over `evals/safety_set.jsonl` (mock LLM): leakage 0 and breach 0 |
| [ ] | T5.9 | Local trace log with masking (no raw email in logs/traces) | G8, G11, H9 (local part) | `sd/ops/trace_log.py`, `tests/test_trace_log.py` | Every tool call and retry attempt logged; grep of log finds no raw email |
| [ ] | T5.10 | Per-tool summary: count, success rate, failures by type, retry rate, p50/p95 | G10 (data layer) | `sd/ops/metrics.py`, `tests/test_tool_summary.py` | Scripted run with one failing mock tool yields correct failure/retry counts and percentiles |
| [ ] | T5.11 | Context compaction at 75% of active window (tiktoken); keeps email, draft, last turns; sliced summarisation; Light window after budget switch | H5 | `sd/ops/compaction.py`, `tests/test_compaction.py` | With small windows in config, long script triggers at 75%, keeps email and draft, triggers against Light window after a switch |
| [ ] | T5.12 | Cost tracking per request/task/agent/tier/step | H7 | `sd/ops/cost.py`, `tests/test_cost.py` | Token counts times config prices recorded at each level; eval cost tracked under task `eval` |
| [ ] | T5.13 | Budget cutoffs (day counter, timezone from config): level 1 $5 all-Light, level 2 $10 unavailable, eval bypass flag | H8, F2 | `sd/ops/budget.py`, `tests/test_budget.py` | Mocked spend: crossing $5 switches every task to Light and logs; crossing $10 returns unavailable message with no LLM call; counter resets on a new day; bypass flag works |
| [ ] | T5.14 | Rate limiting: three rolling 1-hour counters in SQLite, session and email keys, bypass flag | H13 | `sd/ops/rate_limit.py`, `tests/test_rate_limit.py` | 101st question, 41st creation, 41st lookup each give the usage-limit message with real wait time, independently; survives restart; bypass flag turns off |
| [ ] | T5.15 | Masked user ID and session ID in LLM call records | H9 | `sd/ops/trace_log.py`, `tests/test_llm_records.py` | Record has agent, prompt text+version, completion, model, tier, tokens, per-step latency, session ID; prompt/completion pass through redaction |
| [ ] | T5.16 | Tracing integrations with masking hooks, optional (Langfuse/LangSmith) | H9, F6, G11 | `sd/ops/tracing.py`, `tests/test_tracing_optional.py` | No keys: local log only, app runs; with mock keys: spans carry masked data only |


---

## Phase 6: Evals, UI, cache, bonus 1 to 4 (10-19 to 10-21)
Exit: `python -m evals.run` produces Ragas + ablation tables; Streamlit chat + Monitoring tab; cache AC passes with and without Redis.

| Done | ID | Task | FRs | Files | Acceptance test |
|---|---|---|---|---|---|
| [ ] | T6.1 | Eval runner `python -m evals.run` with run record (prompt versions, model IDs) | H11, H14 | `evals/run.py`, `tests/test_eval_runner.py` | Mock run writes a result file containing prompt versions and model IDs |
| [ ] | T6.2 | RAG metrics: answer correctness, precision/recall/F1 at k, hallucination rate, citation accuracy, factual precision (Ragas pinned) **[CONFIRM]** for real judge runs | H12 | `evals/rag_metrics.py`, `tests/test_rag_metrics.py` | Mock judge: metrics computed for a tiny set; citation accuracy counts only retrieved+supporting ids; README notes any substituted Ragas metric |
| [ ] | T6.3 | Agent metrics: completion rate, task accuracy, steps, latency, cost per task, error rate | H11 | `evals/agent_metrics.py`, `tests/test_agent_metrics.py` | Scenario scripts produce each metric; values match hand-computed fixtures |
| [ ] | T6.4 | Gap-set pass rate and escalation precision/recall | A5, SPEC section 7 | `evals/gap_eval.py`, `tests/test_gap_pass_rate.py` | Pytest over the gap set (mock) gives 100% pass: "I don't know" or ticket offer, never a made-up answer or deflection. With the real LLM, measure end-to-end gap pass rate (target 100%) and KB answer rate at the chosen threshold; re-tune if needed. **[CONFIRM]** |
| [ ] | T6.5 | Streamlit chat UI with retrieval-debug panel | A12 (UI), SPEC section 4 item 3 | `sd/ui/app.py`, `tests/test_ui_chat.py` | Streamlit `AppTest`: send a message, see a reply, citations, and chunks+scores panel |
| [ ] | T6.6 | Monitoring tab: tool stats, degraded modes, cost by tier, stale chunks, agent metrics | G10, F7, H4, H7, H6, H11 | `sd/ui/monitoring.py`, `tests/test_ui_monitoring.py` | After a scripted run with a failing mock tool the tab shows failure count, retry count, latency percentiles, degraded modes, and no raw email |
| [ ] | T6.7 | HITL counters: approvals, rejections, edits at the gate | H4 | `sd/ops/hitl_counters.py`, `tests/test_hitl_counters.py` | Scripted yes/no/edit runs increment the matching counters shown in the tab |
| [ ] | T6.8 | Drift monitoring: on-demand command + tab button with cost estimate and confirmation; chart over time | H15 | `sd/ops/drift.py`, `sd/ui/monitoring.py`, `tests/test_drift.py` | Mock run stores Faithfulness, Recall@k, escalation rate, avg rerank score with timestamp/versions/models; button asks for confirmation; adds a chart point. Real run **[CONFIRM]** |
| [ ] | T6.9 | Exact-match cache on rewritten sub-question (KB answers only) | Section 4 cache AC, H7 | `sd/ops/cache.py`, `tests/test_cache_exact.py` | Repeat of same rewritten sub-question skips retrieval, worker and judge; ticket and status replies never cached |
| [ ] | T6.10 | Semantic cache with config cut-off; embedding cost counted | Section 4 cache AC, H7 | `sd/ops/cache.py`, `tests/test_cache_semantic.py` | Paraphrase above cut-off hits; below cut-off misses; one embedding cost recorded per semantic lookup |
| [ ] | T6.11 | Redis backend with in-memory fallback; clear on KB change | F6, F7 (Redis), H6 | `sd/ops/cache.py`, `tests/test_cache_backends.py` | Whole cache test suite passes with and without Redis; re-ingest after a KB edit clears the cache |
| [ ] | T6.12 | Right-sizing comparison: tier map vs all-Main on KB set and scenarios **[CONFIRM]** | H17 | `evals/tier_comparison.py`, `tests/test_tier_comparison.py` | Mock run outputs cost per task, faithfulness, correctness side by side for both configurations |

---

## Phase 7: Multi-agent, bonus 5 (10-21 to 10-22)

| Done | ID | Task | FRs | Files | Acceptance test |
|---|---|---|---|---|---|
| [ ] | T7.1 | Agent nodes/subgraphs with own versioned prompts: Orchestrator, IT worker, HR worker, Synthesizer, Ticket agent | I1 | `sd/agent/{orchestrator,workers,synthesizer,ticket_agent}.py`, `sd/agent/prompts/*` | Each node loads its own prompt file and version; graph compiles |
| [ ] | T7.2 | Domain mapping: IT worker (IT support + equipment), HR worker (HR, onboarding except equipment, payroll); MISC runs `answer_part` with no filter | I2 | `sd/agent/workers.py`, `tests/test_domain_mapping.py` | Equipment question goes to IT worker; payroll to HR; a MISC gap question reaches the ticket agent with category MISC |
| [ ] | T7.3 | Orchestrator rewrites parts into standalone sub-questions; context isolation | I3 | `sd/agent/orchestrator.py`, `tests/test_context_isolation.py` | Worker LLM inputs contain only the sub-question, its chunks and domain prompt (no history); "what about sick leave?" is rewritten standalone |
| [ ] | T7.4 | Parallel fan-out with `Send`, joined before synthesizer; workers call `search_kb` deterministically | I4, G2 | `sd/agent/graph.py`, `tests/test_fanout.py` | Compound IT+HR question runs both workers in parallel and synthesizer once; answer cites both domains |
| [ ] | T7.5 | Single-part pass-through (skip synthesizer LLM call; output guard still runs) | I5 | `sd/agent/graph.py`, `tests/test_passthrough.py` | Single-domain question calls exactly one worker and no synthesizer call; output guard still invoked |
| [ ] | T7.6 | Full whole-KB retry per part; misroute recovery | I6, A19 (full) | `sd/agent/workers.py`, `tests/test_misroute_retry.py` | Deliberately misrouted question is answered after one no-domain retry; still-unanswerable part is marked unanswered |
| [ ] | T7.7 | Partial answers: answered parts with citations, unanswered named, ticket offer for that part only | I7, B6 | `sd/agent/synthesizer.py`, `tests/test_partial_answers.py` | Compound question with one gap part answers the other, names the gap, and the ticket draft covers only that part |
| [ ] | T7.8 | Ticket agent sequential behind the gate, owns create/get tools, never parallel with another ticket action | I8 | `sd/agent/ticket_agent.py`, `tests/test_ticket_agent_seq.py` | Ticket agent receives only state, conversation summary and unanswered part; no concurrent ticket actions in a fan-out run |
| [ ] | T7.9 | Per-agent spans with tier and cost in traces | I9, H7, H9 | `sd/ops/trace_log.py`, `tests/test_agent_spans.py` | A compound run shows each agent as its own span with tier and cost |
| [ ] | T7.10 | Compound set eval (misroutes, gap parts) | I4 to I7 | `evals/compound_eval.py`, `tests/test_compound_eval.py` | Over `evals/compound_questions.jsonl` (mock): all ACs in I (single, compound, gap part, misroute, MISC) pass |
| [ ] | T7.11 | Model right-sizing re-run across all agents | H17 | `evals/tier_comparison.py` | Updated comparison includes worker/synthesizer tiers (**[CONFIRM]** for real run) |

---

## Phase 8: Long-term memory, bonus 6 (10-22)

| Done | ID | Task | FRs | Files | Acceptance test |
|---|---|---|---|---|---|
| [ ] | T8.1 | Long-term per-email store: ticket numbers and last category only | H10 | `sd/memory/long_term.py`, `tests/test_long_term_store.py` | Store holds only numbers and last category; title/status never stored; emails masked in logs |
| [ ] | T8.2 | Cross-session recall on status check (title/status fetched live) | H10, C1 | `sd/agent/status_flow.py`, `tests/test_cross_session.py` | Session 1 creates a ticket; new session/thread with same email recalls it without naming it; no session-1 messages in session 2 context; different email recalls nothing |

---

## Phase 9: Docs + delivery (10-22 to 10-23)

| Done | ID | Task | FRs | Files | Acceptance test |
|---|---|---|---|---|---|
| [ ] | T9.1 | README: setup, env vars, populate KB, run, approach, chunking rationale + ablation, test results | E | `README.md`, `tests/test_readme.py` | README contains each required section heading; ablation table present |
| [ ] | T9.2 | README additions: example conversations, models and task-to-tier map, hardware requirements, MCP notes, mock mode, seed script, GitHub setup | E | `README.md` | Test greps for each required item; fake emails only |
| [ ] | T9.3 | Architecture diagram (single source `docs/architecture.md`, embedded in README, includes all section 3I agents) | E | `docs/architecture.md`, `README.md` | Mermaid block present; names Orchestrator, IT worker, HR worker, Synthesizer, Ticket agent; README embeds it |
| [ ] | T9.4 | README: GitHub-integration/mock disclosure (B10 note) | B10 | `README.md` | README states clearly whether live GitHub or mock is used and how to enable mock mode |
| [ ] | T9.5 | `scripts/load_test.py` (throughput, requests per second) | H11 | `scripts/load_test.py`, `tests/test_load_test.py` | Mock-mode run prints requests/sec; result feeds Monitoring tab and README |
| [ ] | T9.6 | Section 8 deliverables check: public repos, `.env.example`, `config.yaml`, no secrets, KB files | F3, F5 | `tests/test_deliverables.py` | Test confirms no keys/`.env` committed, required files present |
| [ ] | T9.7 | Fresh-clone run with no tracing keys and no Redis | F6, F7 | none (manual) + `tests/test_startup.py` | Fresh clone: `pip install -r requirements.txt` then run succeeds in degraded mode |
| [ ] | T9.8 | Demo video (5 min: KB answer, ticket creation, status, cross-session memory clip) | SPEC section 8, H10 | none (manual) | Recorded; uses fake emails; includes memory clip |

---

## Cut order if behind (PLAN section 7)
Stretch (SPEC section 5), GitHub MCP, one of LangSmith/Langfuse, semantic cache (keep exact-match), multi-agent fan-out (keep single-path), drift chart.
