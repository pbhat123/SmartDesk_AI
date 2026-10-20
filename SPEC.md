# SPEC.md: SmartDesk_AI

Status: draft v3.1. Items marked `TBD` need a decision. Defaults are proposals.

Change log, v2 to v3 (rulings on PLAN.md section 9):
| Id | Sections changed |
|---|---|
| F-01 | FR-H5, AC after FR-H17 (compaction) |
| F-02 | Section 2 Candidates row, FR-A6, AC after FR-A19 |
| F-03 | FR-G6 |
| F-04 | FR-H10, cross-session memory AC |
| F-05 | FR-I2 |
| F-06 | Section 4 cache rules and cache AC, FR-H7 |
| F-07 | Section 2 Input guardrails row, FR-H1 |
| F-08 | FR-D8 |
| F-09 | FR-F2 |
| F-10 | FR-D2, FR-G12 |
| F-11 | FR-H17 table, FR-G12 |
| F-12 | FR-G2 |
| F-13 | FR-H2 |
| F-14 | FR-F4 |
| F-15 | FR-B7 |
| F-16 | FR-A16 |
| F-17 | FR-A5, section 9 |
| F-18 | FR-A15 |
| F-19 | FR-A1 |
| F-20 | Section 9 |
| F-21 | FR-A10, section 6 |
| F-22 | FR-H8 |
| F-23 | FR-H13 |
| F-24 | FR-H15 |
| F-25 | FR-H1 |
| F-26 | FR-F4, FR-H16 |
| F-27 | FR-C6 |
| F-28 | FR-G11 |
| F-29 | FR-G11 |
| F-30 | FR-H1, FR-H2 |
| F-31 | Section 2 model notes |
| F-32 | FR-H12 |
| F-33 | Section 3E, FR-A11, FR-H6, section 4 cache rules |

Change log, v3 to v3.1 (after the first real retrieval run, October 2026):
| Id | Sections changed |
|---|---|
| F-34 | FR-A5, FR-A15, FR-F7, FR-H16, section 9: one escalation threshold, applied to the top dense cosine similarity; the reranker only orders candidates; `retrieval.strategy` switch |

## 1. Goal
First-line IT and HR help desk agent. It answers from a knowledge base (KB), escalates by creating a ticket when unsure, and reports ticket status. Optimize for the rubric first (100 marks), extras second.

Constraints from the project document:
- The agent tries the KB first and escalates only when genuinely unsure. The ticket must capture enough context that the human agent does not need to ask the same questions again (document section 1.1).
- Individual work. Code and docs must be entirely your own, and plagiarism checks are run (FAQ Q8). The user states that full use of Claude Code or any other platform is allowed.
- Python is used (Python or TypeScript are both accepted, FAQ Q9). Running locally with clone, install, run is sufficient (FAQ Q10).

## 2. Decisions
| Area | Default | Notes |
|---|---|---|
| Language | Python 3.12 | |
| Orchestration | LangGraph with LangChain components (decided) | State machine with conditional edges, parallel fan-out via the Send API |
| Agent architecture | Orchestrator, parallel IT and HR worker subagents, synthesizer, ticket agent (decided) | Section 3I |
| LLM | OpenAI (program key) | Provider behind one factory function, `get_llm(task)` |
| Model tiers | Main-Model `gpt-5.4-mini`, Light-Model `gpt-4o-mini`, IDs in `config.yaml` (decided) | Task-to-tier map in config (FR-H17) |
| Embeddings | OpenAI text-embedding-3-small (decided) | |
| Chunking | One chunk = one KB entry (a Markdown `##` section), 200 to 500 tokens, split only above ~600 tokens with ~15% overlap, "Document > Section" header prepended (decided) | FR-A3. JSON Q&A is not indexed |
| Retrieval | BM25 + dense + reranker (hybrid) | Baselines kept for ablation |
| Sparse retriever | rank_bm25 (decided) | Chroma has no built-in BM25 |
| Fusion | Reciprocal rank fusion (RRF, k=60) | Weighted-score fusion kept as ablation |
| Candidates | Top-20 from each retriever | Fused list deduplicated by `chunk_id`, reranked, then collapsed by `entry_id` (best piece wins) to the top-5 entries |
| Reranker | Cross-encoder bge-reranker-base via sentence-transformers (decided) | Falls back to RRF order if it fails to load (FR-F7) |
| Vector store | ChromaDB (decided) | |
| Ticketing | GitHub Issues, public tickets repo (decided) | PAT in env var. Risks of a public repo accepted |
| Ticket categories | IT, HR, MISC (decided) | MISC is a ticket category only, used when neither IT nor HR fits (FR-B7) |
| Email-to-ticket map | SQLite, plus GitHub search fallback (decided) | Issues has no email field (FR-C10, FR-C11) |
| UI | Streamlit (decided) | CLI always works |
| Company | Fictional company named SmartDesk AI (decided) | Realistic policies, not generic |
| KB data source | Option C: public HR datasets adapted to the fictional company plus LLM-generated IT documents (decided) | Document section 3.3 recommends combining both |
| KB format | Markdown (one file per policy area, indexed) plus JSON Q&A (not indexed, feeds evals) (decided) | FAQ Q18 |
| Ticket access | GitHub REST API only (decided 2026-10-20: MCP is not used) | The REST API is allowed (FAQ Q12) |
| Authentication | None | Employee is identified by the email given in conversation (FAQ Q19) |
| Routing | Intent classifier first; the model's tool pick only if the classifier errors or is low-confidence (decided) | FR-D8 |
| Evaluation | Ragas for RAG metrics, pytest for deterministic checks | DeepEval skipped, it overlaps Ragas. The judge model is the provided OpenAI key |
| Tracing | LangSmith and Langfuse (both kept) | Both do tracing, so keep the setup light |
| Cache | Exact-match plus semantic cache on Redis | KB answers only. Redis is optional with an in-memory fallback (FR-F6) |
| Tool calling | Schema-defined tools (Pydantic), a registry mapping names to functions, and a code-level executor | Code adds retries and guards (section 3G) |
| Human-in-the-loop | LangGraph interrupt with a checkpointer | Approval gate before `create_ticket` (FR-G12) |
| Input guardrails | Presidio (PII), OpenAI Moderation API (toxicity), heuristics plus an LLM classifier (prompt injection), the orchestrator's `out_of_scope` result (topic filter) (decided) | Section 3H. Presidio falls back to regex if it fails to load (FR-F7) |
| Output guardrails | Runtime grounding check, OpenAI Moderation API, PII scan, Pydantic schema validation (decided) | Section 3H |
| Names | Every person name is reduced to first name plus last initial ("John Doe" becomes "John D.") (decided) | FR-H1 |
| Budget | Daily spend: $5 cutoff 1 switches everything to Light-Model, a further $5 (total $10) cutoff 2 stops service (decided) | Amounts in config (FR-H8) |
| Rate limits | Rolling 1-hour window per session: 100 questions, 40 ticket creations, 40 ticket lookups (decided) | Independent counters. Evals and tests bypass (FR-H13) |
| Memory | Short-term: LangGraph checkpointer on SQLite. Long-term: per-email store in SQLite (decided) | FR-H10 |
| Prompt versioning | Versioned prompt files in `agent/prompts/`, one per agent (decided) | FR-H14 |
| Drift monitoring | On-demand golden-set re-run, charted in the Monitoring tab (decided) | FR-H15 |

### Model reference (checked October 12, 2026, OpenAI model pages; re-check before final runs)
| Model | Tier | Context window | Max output | Input / 1M | Cached input / 1M | Output / 1M |
|---|---|---|---|---|---|---|
| `gpt-5.4-mini` | Main-Model | 400,000 | 128,000 | $0.75 | $0.075 | $4.50 |
| `gpt-4o-mini` | Light-Model | 128,000 | 16,384 | $0.15 | $0.075 | $0.60 |
| `text-embedding-3-small` | Embeddings | n/a | n/a | $0.02 | n/a | n/a |

Notes: these values go into the price table and context window sizes in `config.yaml` (FR-H16). `gpt-5.4-mini` is a reasoning model; reasoning tokens are billed as output, so cost tracking (FR-H7) counts them, and its reasoning effort is set in config (default `none`). Compaction (FR-H5) triggers at 300,000 tokens on Main-Model and 96,000 on Light-Model. Neither model is available on the API free tier. The model IDs, prices and the `reasoning_effort` value were verified in the T0.9 spike (October 2026).

## 3. Functional requirements and acceptance criteria

### A. KB and RAG (25)
- FR-A1: 30+ entries (aim for 50) across IT, HR, onboarding, payroll. An entry is one `##` section of a Markdown policy file, with a YAML metadata block directly under the `##` heading holding the fields `id` (e.g. `HR-LEAVE-03`), `domain` (`IT` or `HR`), `topic`, `title`, `owner`, `last_reviewed`. Domain is set per entry, not per file (for example IT equipment allocation sits in an otherwise HR onboarding file). File-level front matter, if used, holds only the policy area name. JSON Q&A pairs carry `question`, `answer`, `category`, `type` (simple, how-to, policy interpretation), and the `entry_id` of the section they came from.
- FR-A2: Deliberate gaps (e.g. monitor flickering, office parking, cafeteria menu, desk booking). List them in `kb/GAPS.md`.
- FR-A3: Chunking (decided): one chunk per entry. Target 200 to 500 tokens. A section above ~600 tokens is split with ~15% overlap and every piece keeps the parent `id`. Each chunk is embedded with a "Document > Section" header prepended. Justification for the README: policy sections are self-contained answers, so heading-based chunks keep each answer whole and make citations map one to one to entries. Backed by the ablation in FR-A17.
- FR-A4: Answers use retrieved context only. If context is insufficient, say so.
- FR-A5: Escalation rule: no chunks, or top dense cosine similarity below the threshold, or LLM reports insufficient context. Threshold tuned on the eval set (pick the value that best separates the KB set from the gap set). Tune on a dev half of the KB and gap sets and report final numbers on the held-out test half, so results are not inflated.
- FR-A6: Hybrid pipeline: run BM25 and dense retrieval (top-20 each), fuse with RRF (k=60), deduplicate by `chunk_id`, rerank with a cross-encoder, then collapse pieces of the same entry (the best-scoring piece wins) and pass the top-5 entries to the LLM with the full parent entry text. Each stage is a separate, swappable function so every strategy can be evaluated alone. Retrieval accepts an optional `domain` filter (`IT`, `HR`, or none for the whole KB).
- FR-A7: Ablation: Recall@k (k=1,3,5) reported for BM25, dense, hybrid (RRF), and hybrid plus rerank on `evals/kb_questions.jsonl`. Results table goes in the README.
- FR-A8: KB covers every domain and example topic in document section 2.1.1. IT Support: password reset, VPN configuration, software installation requests, MFA setup, guest Wi-Fi, email on mobile. HR Policies: leave (casual, sick, earned), WFH guidelines, reimbursement, code of conduct, anti-harassment, performance review cycle. Onboarding: day-one checklist, IT equipment allocation, benefits enrollment, buddy programme, probation. Payroll and Comp: pay-day schedule, tax declaration, salary structure, stock-option vesting. Domain tags: IT Support and IT equipment allocation are `IT`; everything else is `HR`.
- FR-A9: KB quality (document section 3.4): at least 30 entries. Realistic, not generic copy-paste, with a fictional company profile (name, industry, size, location) and concrete details such as portal URLs, extension numbers, step-by-step procedures, and numbers. Diverse, with simple factual, procedural how-to, and policy-interpretation entries. Cleanly structured and easy to ingest.
- FR-A10: KB sourcing (document sections 3.1 to 3.3): public datasets (Hugging Face, Kaggle) are adapted to the fictional company (rename, tweak numbers, add categories), not used as-is (FAQ Q6). LLM-generated content: policy documents per category (3 to 5 sub-topics each), then Q&A pairs as JSON with `entry_id`. Final files are stored in the repo: Markdown (one per policy area) and the JSON Q&A array. Each public source and its licence are recorded in `kb/SOURCES.md`, and adapted text is rewritten, not copied.
- FR-A11: Indexing pipeline: a re-runnable script ingests the Markdown files, chunks them, embeds them, and builds both the dense index and the BM25 index. JSON Q&A is not indexed. Re-running does not create duplicates, and chunks of deleted or renamed entries are removed. The README says how to run it, and sample queries verify it works.
- FR-A12: Retrieval demonstration: for any query, the system can show the retrieved chunks with their scores and the answer generated from them (CLI debug flag and UI panel). Use it in the demo video.
- FR-A13: Citations: every KB answer names the KB entry or entries it came from.
- FR-A14: KB first: retrieval is always attempted before a ticket is offered. Escalate only when genuinely unsure.
- FR-A15: Score and signals: signal 2 compares the top dense cosine similarity of the retrieved chunks with one escalation threshold (config). It is the same whether or not the reranker is available. The reranker only orders the candidates (FR-A6): its scores stay sigmoid-normalized to 0 to 1 and are shown in the retrieval demo (FR-A12), but they are never compared with a threshold. Decided after the first real run (October 2026): bge-reranker-base gives no gain over dense alone on this KB (Recall@1 0.74 against 0.92) and scores correct paraphrased policy questions very low in absolute terms (median 0.19 on the KB set), so it separates answerable from unanswerable questions worse than dense cosine (dev AUC 0.76 against 0.83; test 0.92 against 0.99). Decided: escalate if any one signal fires (zero chunks, top dense cosine below threshold, LLM reports insufficient context), because a wrong answer loses marks (FAQ Q11).
- FR-A16: Escalation handoff: when the agent decides it cannot answer, it says so, offers to create a ticket, and starts the ticket flow (section B) only if the employee agrees. Decided: the offer and the email request are merged in one message (as in document flow B). Giving the email counts as agreeing; if the email is already known in the session, the offer is a plain yes or no question. A refusal ends the flow. The confirmation before creating the ticket (FR-B2) stays.
- FR-A17: Chunking ablation: Recall@k for heading-based chunks versus fixed 512-token chunks with 50 overlap, on the KB set. Results in the README.
- FR-A18: No eval leakage: questions in `evals/kb_questions.jsonl` are paraphrased, never copied from indexed text.
- FR-A19: Whole-KB retry: before any ticket offer, a sub-question that failed in its domain is retried once with no domain filter (FR-I6).
- AC: every KB question in `evals/kb_questions.jsonl` retrieves the expected entry in top-k. Every question in `evals/gap_questions.jsonl` ends in "I don't know" or a ticket offer.
- AC (added): re-running ingest twice leaves the same number of chunks. The retrieval demo shows chunks and scores for a sample query. Every KB answer in the eval run includes a citation. A gap question leads to a ticket offer, and the ticket flow starts only after the employee agrees. No chunk ID appears twice in the reranker input, and no entry ID appears twice in the final top-5. The chunking ablation table exists in the README.

### B. Ticket creation (20)
- FR-B1: Collect email, summary, description (mandatory). Infer category. Priority optional.
- FR-B2: Show a ticket summary and require explicit "yes" before creating. Never create silently.
- FR-B3: Create the GitHub issue (labels for category and priority). Store email to issue number in SQLite.
- FR-B4: Return the ticket ID and link.
- FR-B5: Mandatory fields: employee email, issue summary (a short one-line title, e.g. "Unable to connect to VPN from home"), and issue description. If the email is already known in the session the agent does not ask again, otherwise it asks explicitly.
- FR-B6: The description is a summary of the conversation so far, and the employee can add more details. The ticket carries enough context that the human agent does not need to ask the same questions again. For a partly answered compound question, the description covers only the unanswered part (FR-I7).
- FR-B7: Category (recommended): `IT`, `HR`, or `MISC`, inferred from the conversation. MISC is used only when neither IT nor HR fits (e.g. office parking, cafeteria). Priority (optional): Low, Medium, High, or Critical. Decided: priority is implemented; it is inferred from the conversation and shown in the confirmation summary, never asked as a separate question, and the employee can change it at the approval gate (FR-G12).
- FR-B8: The confirmation summary shows title, description, category, email, and priority if set, then asks for explicit confirmation (for example "Shall I go ahead and create this?").
- FR-B9: After creation the agent confirms it, gives the ticket ID and a link, and tells the employee they can ask for updates anytime.
- FR-B10: If the live GitHub integration cannot be made to work, the ticket API is mocked with realistic dummy data and this is documented clearly in the README (partial marks only, FAQ Q7). A mock mode is also used for tests.
- FR-B11: GitHub issue format: the title is the issue summary, the body is the description plus an exact line `Employee: <email>` and the request ID (FR-G6), and labels carry the category (`IT`, `HR`, `MISC`) and the priority (`low`, `medium`, `high`, `critical`). The status labels are `in-progress` and `resolved` (FR-C6). All labels are created once in the tickets repo. The ticket ID is the issue number and the link is the issue URL. The tickets repo is public (decided).
- FR-B12: A thin client wrapper in `tickets/` hides how GitHub is called from the agent graph. The only implementation is the REST API (a mock client stands in for it in mock mode); MCP is not used (decided 2026-10-20).
- FR-B13: No invented ticket details: the ticket ID, link, and status come only from GitHub's response, the ticket summary contains only what the employee said, and the agent never guesses an email.
- AC: scripted scenario creates exactly one issue after confirmation, none after "no".
- AC (added): the confirmation summary contains all listed fields. Priority only accepts the four listed values. Category only accepts IT, HR, MISC. The reply after creation contains the ticket ID and link.
- AC (added): a test with a mocked GitHub response checks that the reply shows exactly the returned ticket ID, link, and status, and that a missing email is asked for, not guessed.

### C. Ticket status (15)
- FR-C1: Reuse email from session, else ask.
- FR-C2: Look up via SQLite, then fetch status and latest comments from GitHub.
- FR-C3: Multiple tickets: list briefly, let the user pick.
- FR-C4: No tickets: say so clearly.
- FR-C5: Trigger phrases such as "What is the status of my ticket?", "Did anyone look at my VPN issue?", and "Any updates on my monitor ticket?" route to the status flow.
- FR-C6: Status is reported as Open, In Progress, Resolved, or Closed, with any comments or updates from the support team. Decided mapping on GitHub: Open = open issue with no status label. In Progress = open issue with the `in-progress` label. Resolved = closed issue with the `resolved` label. Closed = closed issue without it. Edge cases: an open issue with both labels is In Progress; an open issue with only `resolved` is Open.
- FR-C7: Single ticket: show ID, title, status, and the latest team update (as in document flow C).
- FR-C8: Test data: `scripts/seed_tickets.py` creates issues through the same `tickets/` client the agent uses and writes the matching SQLite map rows. Fixed fake emails cover 0, 1, and 2+ tickets. Seeded tickets include every status (labels set, some issues closed) and a team comment, so each status appears in the demo. The script is idempotent (FAQ Q17).
- FR-C9: Team updates: the status comes from the issue state and labels, and the latest team update from the most recent issue comment written by the repo's team. Only comments whose GitHub `author_association` is OWNER, MEMBER or COLLABORATOR count; comments from anyone else on the public issue are ignored. The comment text is untrusted data and is shown to the model only as quoted text, never as instructions.
- FR-C10: Email lookup: GitHub Issues has no email field, so a local SQLite map from email to issue numbers is saved on creation and used on status check (FAQ Q13). The email is also written in the issue body (FR-B11).
- FR-C11: Search fallback: every status lookup also searches the tickets repo for issues whose body contains `Employee: <email>`, merges them with the map results (one entry per issue number), and backfills the map with any numbers it was missing. This covers tickets created by hand in GitHub. The map stays the main path: GitHub search can lag a minute behind a new issue and can fail, so if it does, the map results are still returned.
- AC: tests for 0, 1, and 2+ tickets.
- AC (added): after running the seed script, the status check finds each seeded email's tickets. A ticket created by hand in GitHub (not in the map) is found through the search fallback and then appears in the map.

### D. Orchestration and routing (15)
- FR-D1: Intents: kb_query, create_ticket, ticket_status, smalltalk, out_of_scope, ambiguous.
- FR-D2: Session memory: email, last topic, pending ticket draft, and a dialog state `awaiting` (none, offer_reply, email, ticket_pick) that is passed to the intent classifier so a reply such as "yes" or "the second one" is routed correctly. If the employee changes topic mid-flow, the new intent wins and any pending offer is cleared.
- FR-D3: Out-of-scope and ambiguous: polite deflect or one clarifying question.
- FR-D4: Tone is polite, professional, and empathetic throughout (for example acknowledging the problem before offering a ticket).
- FR-D5: Greetings, small talk, thanks, and goodbyes are handled gracefully.
- FR-D6: Multi-turn: follow-up questions are answered using the conversation context, the agent does not lose the topic, and the email is remembered for the whole session (FAQ Q16).
- FR-D7: All three capabilities (KB query, ticket creation, status check) run in a single agent graph with routing by intent. Domain classification (IT, HR, MISC) is consistent, since consistently wrong routing lowers the score (FAQ Q15).
- FR-D8: Routing order (decided): the orchestrator's intent classifier decides the route and returns an intent with a confidence. If it confidently returns `ambiguous`, the agent asks one clarifying question (a valid result, not a failure). Only if the classifier errors, returns invalid output, or its confidence is below the routing threshold in config does the route come from the model's tool pick (FR-G2). Once the route is set, the model fills in the tool arguments. Tool-pick fallback mapping: `search_kb` routes to kb_query, `create_ticket` to create_ticket, `get_tickets` to ticket_status, and no tool to the chitchat route (small talk, deflection, or one clarifying question). The tool pick is used for routing only; execution still goes through the ticket agent and the executor.
- FR-D9: Scope boundary (decided): anything workplace-related (IT, HR, facilities, office, equipment) is in-domain and takes the KB path; a gap there gets "I don't know" plus a ticket offer. Anything unrelated to work (weather, recipes, trivia, general coding help) is out of scope and gets a polite deflection with no ticket. When unsure, treat the message as in-domain.
- AC: intent test set with expected routes. Multi-turn script never re-asks for email.
- AC (added): workplace gap questions (parking, cafeteria, desk booking) end in a ticket offer, never a deflection. Unrelated questions end in a deflection, never a ticket offer. A test with a forced classifier error shows the route coming from the model's tool pick. A confidently ambiguous message gets exactly one clarifying question.

### E. Code and docs (15)
- README: setup, env var list, populate KB, run, approach, chunking rationale and ablation, test results.
- Architecture diagram (Mermaid) showing components and data flow, including every agent in section 3I. The single source is `docs/architecture.md`, embedded in the README.
- Modular layout (section 6). Code is well-structured, modular, and readable with meaningful names.
- The README also includes: run commands; recorded example conversations for a KB answer, a ticket creation, and a status check (guiding step 17); the models used and the task-to-tier map; hardware requirements, always (CPU is sufficient; about 1 GB download for the reranker and about 0.5 GB for the spaCy model used by Presidio) (FAQ Q2); the line "MCP: not used; direct GitHub REST API (allowed per FAQ Q12)"; how to enable mock mode; how to run the seed script.
- The README documents the GitHub setup: create the public tickets repo, create a token with Issues access limited to that repo, set `GITHUB_TOKEN`, run the seed script, and the line "MCP: not used; direct GitHub REST API (allowed per FAQ Q12)".

### F. Robustness (10)
- FR-F1: GitHub unreachable: polite message, suggest retry.
- FR-F2: LLM failure: retry with backoff. If Main-Model still fails, the call goes to Light-Model, and the reverse. If both fail, show a graceful message. At budget level 1 or higher (FR-H8) a call never falls back to Main-Model. Before a Main-Model to Light-Model fallback, the prompt size is checked against Light-Model's context window. A synthesizer failure is not an outage for the employee: the answered parts are shown joined (FR-I7).
- FR-F3: Secrets only via env vars. `.env.example` committed, `.env` ignored.
- FR-F4: Edge cases: empty input, invalid email, very long input (over `input.max_chars` in config, default 4000, gets a polite request to shorten), duplicate ticket request (if the same email already has an open ticket with a near-identical title, tell the employee and ask whether to open another; this is separate from the request-ID protection in FR-G6). The duplicate question pauses the graph with its own interrupt, not an `awaiting` value; its reply follows the same rules as the approval gate (FR-G12): an exact yes or no is accepted, an unclear reply is asked once more, and a second unclear reply cancels.
- FR-F5: `.env.example` lists `OPENAI_API_KEY`, `GITHUB_TOKEN`, and the tickets repo name. Optional keys: `LANGSMITH_API_KEY`, `LANGFUSE_PUBLIC_KEY`, `LANGFUSE_SECRET_KEY`, `REDIS_URL`. No real values are committed.
- FR-F6: Tracing (LangSmith, Langfuse) and Redis are optional. The app runs after a plain clone with no tracing keys and no Redis, using an in-memory cache fallback.
- FR-F7: Graceful degradation (decided): everything is installed and used by default. If a component fails to load, the app logs a warning, shows the degraded mode in the Monitoring tab, and keeps running with a substitute. Reranker unavailable: skip reranking and keep the RRF order. The escalation threshold is unchanged, because it already applies to the top dense cosine similarity (FR-A15). Presidio unavailable: regex-based PII checks for email, phone, card number, and government ID (name reduction is skipped and logged). Redis unavailable: in-memory cache. Tracing keys missing: local log only.
- AC (added): with the reranker and Presidio forced to fail, the app starts, answers a KB question, still redacts a phone number, and logs both degraded modes.

### G. Tool calling and execution (supports sections A to F and I)
- FR-G1: Three tools, each with a name, a description, and typed arguments (Pydantic or JSON schema): `search_kb(query, domain)` where domain is `IT`, `HR`, or none (whole KB), used by the IT and HR workers and by the orchestrator for MISC and whole-KB retries; `create_ticket` and `get_tickets`, used only by the ticket agent.
- FR-G2: Within a route, the model chooses the tool and its arguments from the conversation and the tool definitions. This tool pick also serves as the routing fallback (FR-D8). For small talk, out-of-scope, and ambiguous messages no tool is called: the agent replies or asks one clarifying question. IT and HR workers do not pick a tool: they call `search_kb` directly through the executor with their fixed domain. The model chooses tools only in the routing fallback (FR-D8) and in the ticket agent.
- FR-G3: A registry maps each tool name to its Python function. A code-level executor, not a tool the LLM sees, runs every tool call through the registry.
- FR-G4: Retries: the executor makes at most 3 total attempts per tool call, with a short backoff. It retries only transient failures (network errors, timeouts, rate limits, 5xx responses), never bad arguments or a missing email. After the third failed attempt it returns an "unavailable" error, and the agent tells the employee politely and suggests trying again later (FR-F1).
- FR-G5: Guards: the executor validates arguments against the schema before running a function. `create_ticket` runs only after the employee confirmed the summary (FR-B2). A missing or invalid email returns an error that tells the model to ask for it.
- FR-G6: Duplicate protection: each `create_ticket` call carries a unique request ID in the issue body. The executor writes the request ID to SQLite before the call. Before a retry, it checks whether an issue with that ID already exists, using the list-issues endpoint (not GitHub search, which can lag, see FR-C11) and the SQLite row, so a lost response never creates a second ticket.
- FR-G7: `search_kb` returns the chunks, their scores, and an `answerable` flag computed from the first two escalation signals (no chunks, score below the threshold). The third signal, the LLM reporting insufficient context, comes from the worker's answer. Any one signal leads to the whole-KB retry (FR-I6) and then the ticket offer (FR-A15, FR-A16).
- FR-G8: Every tool call and every retry attempt appears in the traces (LangSmith, Langfuse).
- FR-G9: Per-call metrics: the executor records the tool name, the calling agent, the outcome (success, transient failure, validation error, unavailable), the attempts used, and the latency per attempt and in total.
- FR-G10: Per-tool summary: call count, success rate, failures by type, retry rate, and latency p50 and p95. It is visible in the Langfuse and LangSmith dashboards and in a Monitoring tab in the Streamlit app, fed by a local log, so it works without keys.
- FR-G11: Privacy: raw employee emails never appear in the local log or in the traces. They are masked. (The issue body in the public tickets repo holds the raw email by design, FR-B11.) Tracing integrations must use each SDK's input and output masking hook (verified in the first spike); if masking cannot be guaranteed, only locally redacted spans are sent. The checkpointer and the long-term store are data stores, not logs, and hold raw emails by necessity.
- FR-G12: Confirmation mechanism: the approval gate uses LangGraph's human-in-the-loop interrupt with a checkpointer (SQLite-backed, see FR-H10, so the pause also survives a restart). When the ticket agent proposes `create_ticket`, the graph pauses before the tool runs and shows the ticket summary. The employee's reply resumes it: yes runs the tool, no cancels, and an edit updates the summary and asks again. Each chat session has its own thread ID, so the pause survives across turns and sessions never mix. An unclear or unrelated reply at the gate is asked once more with the same summary; a second unclear reply cancels, and nothing is created. The reply is parsed by Light-Model (task CONFIRM_PARSE, FR-H17); an exact "yes" or "no" is accepted without a model call.
- AC: the intent set (section 7) records the expected route and tool, or no tool, for each utterance, and a test checks the choice. Tests with a failing mock confirm that a transient failure is retried up to 3 attempts and then returns "unavailable", that bad arguments are not retried, that `create_ticket` does not run before confirmation, and that a lost create response never creates a duplicate.
- AC (added): after a scripted run with one failing mock tool, the Monitoring tab shows that tool's failure count, retry count, and latency percentiles, and no raw email appears in the log or the traces.
- AC (added): a test confirms that a proposed `create_ticket` pauses the graph, that "yes" resumes it and creates exactly one ticket, that "no" creates none, that an edit shows an updated summary before anything is created, and that the pause survives several chat turns.

### H. Operations: AIOps, LLMOps, AgentOps
Safety and guardrails
- FR-H1: Input guardrails run on every user message before the agent graph. PII detection with Microsoft Presidio. Toxicity with the OpenAI Moderation API. Prompt-injection check with heuristics plus an LLM classifier that runs only when the heuristic score is above a low threshold set in config. There is no separate topic-filter call: the topic filter is the orchestrator's `out_of_scope` intent result inside the graph, following the scope boundary in FR-D9 (workplace gaps are never deflected). A blocked message gets a polite refusal, the reason is logged, and no tool runs. The employee's email is an allowed field and is masked in logs and traces (FR-G11). Every person name (the employee and anyone else mentioned) is reduced to first name plus last initial ("John Doe" becomes "John D."); a single name stays as it is. Names are reduced on user input only; retrieved KB text is not altered. Any other detected PII (phone, government ID, card number) is redacted before the LLM call, the logs, the traces, and the ticket body. Outage rules: if the Moderation API is unavailable the message is allowed and a warning is logged (fail open); if the injection classifier is unavailable or returns a malformed reply the message is blocked (fail closed; the classifier is only reached above the heuristic threshold). Team-update text from ticket comments (FR-C9) goes through the same injection check, and a flagged update is replaced by a fixed notice.
- FR-H2: Output guardrails run on every reply before it is shown. A runtime hallucination check compares the answer's claims with the retrieved chunks (LLM judge). The judge runs once per part inside the answering step, so an ungrounded part goes to the whole-KB retry (FR-I6) and the ticket offer. If the judge call fails because both model tiers are unavailable, that is an outage (FR-F2): the agent replies with the polite try-again-later message and makes no ticket offer; only a malformed or schema-invalid judge reply counts as ungrounded. Unsupported claims count as the "insufficient context" signal (FR-A15), so the agent replaces the answer with the "I don't know" and ticket-offer path. A toxicity filter uses the Moderation API. A PII scan runs on the output and redacts non-name PII only. Phone numbers that appear verbatim in the KB (the company switchboard and payroll lines) are not redacted. Schema validation uses Pydantic on tool arguments and structured model outputs. The final reply check covers Moderation, the PII scan and schema validation only.
- FR-H3: Safety metrics on `evals/safety_set.jsonl`. PII leakage rate: the share of test cases where unmasked PII appears in a reply, log, or trace. Policy breach rate: the share of test cases where the agent breaks a rule (answers outside the KB, creates a ticket without confirmation, follows an injected instruction, reveals its prompt). Target for both: 0.
- FR-H4: HITL checks: besides the ticket approval gate (FR-G12), the Monitoring tab counts approvals, rejections, and edits at that gate.

Context, cost, and logging
- FR-H5: Context compaction: when the orchestrator's context fills 75% (configurable) of the active model's context window, older turns are summarized into a running summary. The email, the pending ticket draft, and the last few turns are kept as they are. After a budget switch (FR-H8) compaction triggers against Light-Model's window. Workers and the synthesizer get only their isolated inputs (FR-I3), so compaction mainly applies to the orchestrator. Tokens are counted with tiktoken. The compaction summary is made in slices that fit Light-Model's window; Main-Model may be used instead, only at budget level 0, when the history to summarize exceeds Light-Model's window. Tests set small window sizes in config so the trigger can be reached.
- FR-H6: Staleness: ingest stores a hash per KB file. When a file changes, only the chunks from that file are flagged stale. The flag is listed in the Monitoring tab. Re-running ingest re-embeds the flagged chunks, removes chunks of deleted entries, clears the flag, and clears the answer cache (the same event as "the KB changes" in section 4).
- FR-H7: Cost tracking: token counts times a price table in `config.yaml`, recorded per request, per task (one conversation goal such as a KB answer or a ticket flow), per agent, per model tier, and per agent step (LLM call, embedding call, tool call). An exact-match cache hit adds no retrieval, worker or judge cost; a semantic cache hit costs one embedding call, which is counted.
- FR-H8: Budget cutoffs, per day, with amounts in `config.yaml`. Cutoff 1 at $5 of spend: all new LLM calls use Light-Model, overriding the task-to-tier map. Cutoff 2 at a further $5 (total $10): no LLM calls, and the agent shows a graceful "service temporarily unavailable" message. The daily counter resets at the start of each day (timezone in config). Both cutoffs are logged. Evals use a separate flag in config (`budget.eval_bypass`, off by default) so Ragas judge runs and ablations are not stopped by the daily cutoff; their cost is tracked separately under the task name `eval`. With the flag on, spend under the task `eval` is not added to the daily counter and eval runs are never cut off, while the chat service keeps both cutoffs; with the flag off, eval spend counts like any other. The daily counter is stored in SQLite, so a restart does not reset it.
- FR-H9: Trace logging: every user request and reply, every agent step, every tool call and result, and every LLM call and reply is logged locally and sent to Langfuse and LangSmith when keys exist. Each LLM record holds the agent name, the prompt text and prompt version, the completion text, the model name, version, and tier, input and output token counts, latency (total and per step), the session ID, and a masked user ID. Prompt and completion text pass through the redaction in FR-H1.

Memory, metrics, limits
- FR-H10: Memory store. Short-term: the LangGraph checkpointer on SQLite keeps session state and the pending ticket draft, so a conversation survives a restart. Long-term: a per-email SQLite store holds past ticket numbers and the last category used, and nothing else. Ticket title and status are fetched live from GitHub by number, never stored. It is used across sessions, for example to offer the employee's earlier tickets on a status check. Emails in it follow FR-G11 for logs. This is the document's cross-session memory bonus.
- FR-H11: Agent metrics, shown in the Monitoring tab and in the README: task completion rate, task accuracy (against the scenario scripts), steps to completion, end-to-end latency, cost per task, error rate, and throughput (requests per second, from `scripts/load_test.py`).
- FR-H12: RAG metrics, reported on the KB set: accuracy (Ragas answer correctness against the reference answer), retrieval precision, recall, and F1 at k against the expected entry IDs, hallucination rate (the share of answers with unsupported claims, from Ragas Faithfulness), citation accuracy (the share of cited entry IDs that were retrieved and support the answer), and factual precision (Ragas FactualCorrectness in precision mode). The Ragas version is pinned in `requirements.txt`; if a named metric does not exist in that version, the closest metric is used and the README says which.
- FR-H13: Rate limiting: a rolling 1-hour window, keyed by session ID; ticket lookups are limited per session and per email, and both must be within the limit. Three independent counters: 100 questions, 40 ticket creations, 40 ticket lookups. Ticket creations count successful creations. Counters are stored in SQLite so they survive a restart. Every user message counts toward the question counter, including mistypes and small talk. When a counter is full the agent replies "Usage limit reached. Please try again in N minutes." with the real wait time. Evals and tests bypass the limits through a config flag that is off by default. Order of checks for each message: rate limit, then budget cutoff (FR-H8), then the input guard (FR-H1); empty and over-long messages are rejected before any of them and do not count. Ticket lookups count only the employee's own status checks, not the duplicate check made while creating a ticket, and only successful ones; a lookup needs both its session counter and its email counter within the limit. Emails in the counters are stored only as a salted hash (the `MASK_SALT` env var, FR-G11).

AIOps
- FR-H14: Prompt version tracking: prompts are files in `agent/prompts/`, one per agent and purpose, named with a version (for example `synthesizer_v2.md`). The active version per prompt is set in `config.yaml`. The version is logged on every LLM call (FR-H9), shown in traces, and stored with every eval run so results compare across versions.
- FR-H15: Drift monitoring: an on-demand command and a Monitoring tab button re-run the golden eval set and store Faithfulness, Recall@k, escalation rate, and average rerank score with a timestamp, the prompt versions, and the model IDs. The tab charts them over time. The golden set is the KB set plus the gap set. The button shows an estimated cost and asks for confirmation before running, since the run calls the judge model. Drift faithfulness is the runtime grounding-judge accept rate (FR-H2) read from the run's own trace, not the Ragas faithfulness.
- FR-H16: Configuration: `config.yaml` (no secrets) holds the Main-Model ID (`gpt-5.4-mini`) and the Light-Model ID (`gpt-4o-mini`), the task-to-tier map (FR-H17), context window sizes, the price table, budget cutoffs, rate limits, the compaction trigger, the escalation threshold (on top dense cosine, FR-A15), the retrieval strategy `retrieval.strategy` (`hybrid_rerank` by default, as in FR-A6; `dense`, `hybrid` and `dense_rerank` are switches for ablations and fallback), the routing confidence threshold (FR-D8), the semantic cache cut-off, the LLM retry attempts and backoff (FR-F2), the tool retry `tools.max_attempts` and `tools.backoff_seconds` (FR-G4), the tickets mode `modes.tickets` (`mock` or `live`, FR-B10) and the GitHub client settings `github.*` (API base URL, API version, timeout, page size), the active prompt versions, the orchestrator's maximum number of sub-questions per message (`orchestrator.max_parts`) and the most tokens of earlier conversation it reads (`orchestrator.history_max_tokens`, default 4000), the maximum input length (`input.max_chars`), the injection-heuristic threshold, and the eval budget flag. Changing a model ID there is the only change needed to swap models.
- FR-H17: Model right-sizing (decided): each LLM task has a fixed tier in the task-to-tier map, and `get_llm(task)` returns that tier's model. Default map:

  | Task | Tier |
  |---|---|
  | Orchestrator (intent classification, decomposition, query rewriting) | Main-Model |
  | IT and HR worker answers (RAG) | Main-Model |
  | Synthesizer | Main-Model |
  | Grounding judge (FR-H2) | Main-Model |
  | Tool selection and argument filling | Light-Model |
  | Ticket agent (summary drafting) | Light-Model |
  | Prompt-injection classifier | Light-Model |
  | Context compaction summary | Light-Model |
  | Ticket confirmation reply parsing (yes, no, edit) | Light-Model |
  | Small talk, deflection and clarifying-question replies | Light-Model |

  Tiers are fixed per task, not scored per query. Budget cutoff 1 (FR-H8) and outage fallback (FR-F2) override the map. The tier is logged on every call, and the Monitoring tab shows calls and cost by tier. Comparison: run the KB set and scenario scripts once with the map and once with every task on Main-Model, then report cost per task and Ragas Faithfulness and answer correctness side by side in the README. The comparison requires the two model IDs to differ, which they do (`gpt-5.4-mini` and `gpt-4o-mini`).
- Hallucination and tool reliability are covered by FR-H2, FR-H12, section 7, and FR-G9, FR-G10.
- AC: the safety set shows PII leakage rate 0 and policy breach rate 0. A test with injected PII shows it redacted in the reply, log, trace, and ticket body. A test with "John Doe" and a second full name shows both as "John D." style in the reply, log, trace, and ticket body. A test with a toxic message shows a refusal and no tool call. A test with an unsupported draft answer shows it replaced by the ticket-offer path.
- AC: with a mocked spend counter, crossing $5 switches every task to Light-Model and logs it, crossing $10 returns the unavailable message with no LLM call, and the counter resets on a new day.
- AC: a long scripted conversation, run with small test window sizes set in config, triggers compaction at 75% of the active model's window, keeps the email and pending draft, and triggers against Light-Model's window after a switch. Editing one KB file flags only that file's chunks, and re-ingest clears them.
- AC: the 101st question, the 41st ticket creation, and the 41st ticket lookup in an hour each get the usage-limit message with a wait time, independently. The bypass flag turns the limits off for tests.
- AC: every LLM call record contains the agent name, prompt text and version, completion, model name and tier, token counts, per-step latency, and session ID. Changing a prompt version changes the logged version. Changing a task's tier in config changes the model used for that task and nothing else. The drift button adds a data point to the chart. A restart keeps the pending ticket draft and the per-email ticket history.
- AC (cross-session memory, FR-H10): session 1 creates a ticket for an email. A new session with a new thread ID and an empty conversation is then started, and the same email is given. The agent recalls that ticket without the employee naming it (the ID comes from the long-term store; title and status are fetched live from GitHub), and no messages from session 1 appear in session 2's context. A different email in the new session recalls nothing from the first.

### I. Multi-agent architecture (bonus, specified)
- FR-I1: Agents: Orchestrator, IT worker, HR worker, Synthesizer, Ticket agent. Each is a LangGraph node (or subgraph) with its own versioned prompt (FR-H14).
- FR-I2: Domain mapping (decided): the IT worker handles IT Support and IT equipment allocation (laptops, devices). The HR worker handles HR policies, onboarding (except equipment), and payroll and comp. MISC questions (no clear IT or HR fit, e.g. facilities) go to no worker: the orchestrator runs the same answering step with no domain filter (a worker without a domain), which answers if a match passes, and otherwise the ticket agent takes over with category MISC. The synthesizer only merges multi-part results.
- FR-I3: Context isolation (decided): the orchestrator holds the full conversation (subject to compaction, FR-H5) and rewrites each part of the message into a standalone sub-question ("what about sick leave?" becomes "How many sick leaves do I get per year?"). Each worker receives only its sub-question, its retrieved chunks, and its domain prompt. The synthesizer receives the original message plus each answered sub-question with its answer text; it never sees citations or unanswered parts. The ticket agent receives the session state (email, pending draft), a conversation summary, and the unanswered part if any. The orchestrator sees ticket status replies, here and in compaction summaries, only as a fixed placeholder, because they quote support-team comments (FR-C9). It reads the compacted summary plus the most recent turns that fit `orchestrator.history_max_tokens` (config, default 4000), newest first, masked; a turn that does not fit is dropped whole, except that the newest turn is cut to fit, so the size of this Main call stays flat however long the conversation gets.
- FR-I4: Parallel fan-out: for a compound question, the orchestrator splits it into sub-questions, each tagged with a domain, and dispatches them in parallel (LangGraph Send API). Results are joined before the synthesizer. A message is split only when its questions need different KB entries or belong to different domains; clauses about one topic stay one part. It is split into at most `orchestrator.max_parts` sub-questions (config, default 4); the rest are named in a fixed note and not answered.
- FR-I5: Single-part pass-through: a question with one part in one domain goes to one worker only, and the synthesizer's LLM call is skipped. Output guardrails (FR-H2) still run.
- FR-I6: Whole-KB retry: if a worker's part is unanswerable (any escalation signal, FR-A15), that sub-question is retried once with `search_kb` and no domain filter. If it is still unanswerable, the part is marked unanswered. This also recovers from IT/HR misroutes (FAQ Q15).
- FR-I7: Partial answers (decided): when some parts are answered and some are not, the reply gives the answered parts first, with their citations, then names each part the KB cannot answer, then offers a ticket for that part only. The naming and the offer are fixed wording produced by code, not by a model, and the offer is merged with the email request (FR-A16). The synthesizer LLM runs only when two or more parts were answered; one answered part is shown as its worker wrote it. The synthesizer's text is used only if every number and entry ID in it appears in the worker answers and it passes the fact check of FR-B13 (numbers and codes, URLs, emails, IDs, capitalised names) against them; otherwise, and when the synthesizer call fails, the reply is the answered parts joined in question order. Citations are built by code from the workers' citations. The ticket's summary and description cover only the unanswered parts (FR-B6) and may use facts from anything the employee said (FR-B13).
- FR-I8: Ticket agent: owns `create_ticket` and `get_tickets`, runs sequentially behind the approval gate (FR-G12), and never runs in parallel with another ticket action. It drafts the summary (FR-B6), infers category and priority, and handles the status flow (section C).
- FR-I9: Traces show each agent as its own span, with its tier and cost (FR-H7, FR-H9).
- AC: a single-domain question calls exactly one worker and no synthesizer LLM call. A compound IT plus HR question calls both workers in parallel and the synthesizer once, and the answer cites entries from both domains. A compound question with one gap part answers the other part, names the unanswered part, and offers a ticket whose draft covers only that part. A deliberately misrouted question is answered after the whole-KB retry. A MISC gap question reaches the ticket agent with category MISC. Worker LLM inputs contain no conversation history beyond the sub-question.

## 4. Bonus (cap +10), in priority order
1. Hybrid search with measured improvement over each baseline (Recall@k table in README).
2. Eval pipeline (Ragas, plus LangSmith and Langfuse tracing): faithfulness, answer relevance, context precision and recall, latency.
3. UI (Streamlit).
4. Caching: exact-match plus semantic cache on Redis (rules and acceptance test below).
5. Multi-agent architecture with specialised IT and HR sub-agents. Specified in section 3I.
6. Advanced memory: long-term, cross-session memory via a database. Specified in FR-H10.

Notes:
- Bonus marks are capped: the total never exceeds 100. They are a cushion if marks are lost elsewhere, or a differentiator.
- Item 3 is user-facing deployment (Streamlit, Gradio, Chainlit, or a web app). Deploying the agent also counts as a bonus (FAQ Q10), but running locally is sufficient.
- Start bonus items only after the core in section 3 (A to G) is solid (document section 12). Section 3I touches routing, so build the single-path version first and add fan-out once the core passes.
- Cache rules: cache only KB answers, never ticket-creation or status replies. Set the semantic cache cut-off (config) from evidence: at least 0.10 above the highest cosine similarity between question pairs that need different answers, so a hit is never a different question (currently 0.97, evidence in `evals/results/cache_cutoff.json`, which includes near-miss pairs). The cache key is the orchestrator's rewritten standalone sub-question, not the raw message, so follow-ups such as "what about sick leave?" are never mixed up. A hit skips retrieval, the worker and the judge; the orchestrator call still happens. Clear the cache whenever the KB changes (the same event as re-running ingest, FR-H6). Redis is optional, with an in-memory fallback.
- Cache AC: a repeated question (same rewritten sub-question) is an exact-match hit that skips retrieval, worker and judge calls and adds no cost beyond the orchestrator call. A paraphrase above the cut-off is a semantic hit. A question below the cut-off is a miss. Ticket-creation and status replies are never cached. Changing a KB file and re-running ingest clears the cache. With Redis unavailable, the in-memory fallback passes the same test.

## 5. Stretch (not graded, recruiter story)
- ContextOps loop: staleness flags, gap log from escalations, decision traces from resolved tickets, eval gate before KB publish.
- ITIL 5 mapping: Service Desk, Service Request, Incident, Knowledge, Problem, Service Level.
- Screenshot input (vision) for IT issues, with PII blur.
Do these only after sections 3 and 4 pass.

## 6. Repo layout
```
smartdesk_ai/
  agent/        graph, state, intents, orchestrator, workers (it, hr), synthesizer, ticket_agent, prompts/ (versioned files)
  rag/          ingest, chunking, retrievers, fusion, rerank
  tickets/      github_client, sqlite_map, search_fallback, schemas
  llm/          provider factory (get_llm, tiers), retry, fallback
  tools/        tool definitions, registry, executor (retries, guards)
  guardrails/   input and output checks, PII redaction, name reduction
  ops/          budget, rate limits, trace log, metrics, drift, compaction, cache
  memory/       sqlite checkpointer, long-term store
  ui/           app, monitoring tab
kb/             Markdown policy files, qa.json, GAPS.md, SOURCES.md
evals/          kb_questions, gap_questions, compound_questions, intents, categories, scenarios, safety_set
scripts/        seed_tickets.py, load_test.py
tests/
docs/           architecture.md (Mermaid)
SPEC.md  CLAUDE.md  PLAN.md  TASKS.md  README.md  .env.example  config.yaml
```

## 7. Evals (write before implementing)
- KB set: 40+ paraphrased questions (simple, how-to, policy interpretation) with expected entry IDs and a reference answer for each (some Ragas metrics need one). Built from `kb/qa.json` without copying indexed text (FR-A18).
- Gap set: 15+ questions not in KB, including workplace grey-zone cases (parking, cafeteria, desk booking, monitor flickering). Pass = "I don't know" or ticket offer, never a made-up answer and never a deflection.
- Out-of-scope cases (in the intent set): unrelated questions (weather, recipes, trivia, general coding help). Pass = polite deflection, no ticket offer (FR-D9).
- Compound set: 10+ multi-part questions across IT and HR, some with one gap part, some deliberately misroutable (FR-I4 to FR-I7).
- Intent set: 30+ utterances with expected intent, expected route, and expected tool (or no tool), including confidently ambiguous cases.
- Category set: utterances with expected category IT, HR, or MISC (FAQ Q15).
- Scenario scripts: KB answer, ticket create (yes, no, edit), status (0, 1, 2+), API down, degraded mode (FR-F7).
- Sample flow scripts reproducing document section 8. Flow A: password reset answered from the KB, then thanks and goodbye. Flow B: monitor flickering, no guide found, ticket offered, email asked, summary with title, description, category and email, confirmation, ticket ID returned. Flow C: status check, email asked, one open ticket listed with ID, title, status and team update.
- Seed tickets: created by `scripts/seed_tickets.py` for 0, 1, and 2+ tickets, plus one ticket created by hand to test the search fallback (FR-C8, FR-C11, FAQ Q17).
- Hallucination check (FAQ Q11): the evaluator asks KB and non-KB questions. For non-KB questions the agent must say it does not know or offer a ticket. Making up information loses marks.
- Retrieval demo queries: a short list of sample queries with their retrieved chunks and scores, recorded in the README.
- Safety set: 20+ adversarial inputs (prompt injection, PII, names, toxic text, off-topic, attempts to skip ticket confirmation) with the expected behaviour. Used for PII leakage rate and policy breach rate (FR-H3).
- Citation check: every KB answer in the eval run cites entry IDs, and citation accuracy is scored (FR-H12).
- Chunking ablation (FR-A17) and right-sizing comparison (FR-H17), both reported in the README.
- Metrics: Recall@k, grounded-answer rate, escalation precision and recall, latency, cost per query, per-tool failure rate and latency (FR-G9, FR-G10). Added: PII leakage rate, policy breach rate, task completion rate, task accuracy, steps to completion, end-to-end latency, cost per task, cost by tier, error rate, throughput, retrieval precision, recall and F1, hallucination rate, citation accuracy, factual precision (section 3H).
- Grounded-answer rate: the mean Ragas Faithfulness (supported claims divided by total claims, scored by an LLM judge) over the KB set. Target: 0.8 or higher.
- Gap-set pass rate: the share of gap questions that end in "I don't know" or a ticket offer, checked with pytest. Target: 100%, since made-up answers lose marks (FAQ Q11).
- Hallucination types and where each is caught: unsupported claims in a KB answer (Faithfulness), answering a question the KB does not cover (gap-set pass rate), answering from the wrong entry (Recall@k, Ragas Context Precision), invented ticket details (FR-B13 test).

## 8. Deliverables
- Public GitHub repo for the code (no keys, no `.env`), README, architecture diagram, KB files, and a 5-minute demo video covering RAG answer, ticket creation, ticket status (fits the document's 3 to 5 minutes).
- The video includes a short cross-session memory clip: create a ticket in one session, open a new session, give the same email, and show the agent recalling the ticket (bonus evidence, FR-H10).
- Public tickets repo (decided), so live issues are visible. Use fake emails in seed tickets, the README, and the demo.
- The repo includes all source code, KB files (Markdown policy files and JSON Q&A), and configuration (`.env.example`, `config.yaml`).

## 9. Open questions
- Decided: public code repo, public tickets repo, 5-minute demo video. Known risks of a public tickets repo are accepted.
- Decided: company name is SmartDesk AI. TBD: policy numbers.
- Decided: framework is LangGraph, vector store is ChromaDB, UI is Streamlit.
- Decided: deadline is 10/23 (October 23, 2026). Target: core (sections 3A to 3G) done by October 18. The safety and operations items that section 10 maps to Robustness (FR-H1, H2, H3, H8, H13) are due October 19; the core ships with no-op guard hooks in the chat pipeline so they plug in without graph changes.
- Resolved: the document says code and docs must be entirely your own (FAQ Q8). The user states that full use of Claude Code or any other platform is allowed.
- Decided: KB data source is Option C. KB format is Markdown per policy area (indexed) plus JSON Q&A (evals only). Chunking per FR-A3.
- Decided: tickets live in GitHub Issues, accessed through the GitHub REST API only; MCP is not used (decided 2026-10-20). Seed script plus search fallback for status lookups.
- Decided: status mapping from GitHub issue state and labels to Open, In Progress, Resolved, Closed (FR-C6). Categories IT, HR, MISC.
- Decided: escalation fires if any one signal fires, with one whole-KB retry first. TBD: the escalation threshold value and the routing confidence threshold (both tuned on the dev half of the eval sets once the KB exists, FR-A5).
- Decided: multi-agent architecture (section 3I) and cross-session memory (FR-H10) are in scope.
- Decided: routing is classifier first, model's tool pick as fallback (FR-D8). Scope boundary per FR-D9.
- Decided: names reduced to first name plus last initial for everyone. Reranker and Presidio degrade gracefully (FR-F7). Rate limits 100, 40, 40 per hour.
- Decided: model right-sizing with Main-Model and Light-Model (FR-H17).
- Decided: Main-Model is `gpt-5.4-mini` and Light-Model is `gpt-4o-mini` (avoid `gpt-3.5-turbo`, `o3-mini`, and `o4-mini`, which shut down on October 23, 2026).
- Decided: price table and context window sizes per the model reference table in section 2 (checked October 12, 2026; re-check before final eval runs, since prices change).

## 10. Traceability to the project document
| Document item | Spec |
|---|---|
| A. KB and RAG (25): KB content, chunking, retrieval quality, grounding, escalation | KB content: FR-A1, A2, A8, A9, A10. Chunking: FR-A3, A17. Retrieval quality: FR-A6, A7, A11, A12, A18. Grounding: FR-A4, A13, H2. Escalation: FR-A5, A14, A15, A16, A19, I6 |
| B. Ticket creation (20): API integration, mandatory fields, human confirmation, post-creation feedback | API: FR-B3, B10, B11, B12, B13. Fields: FR-B1, B5, B6, B7. Confirmation: FR-B2, B8, G5, G12. Feedback: FR-B4, B9 |
| C. Ticket status (15): API retrieval, multiple tickets, no tickets | Retrieval: FR-C1, C2, C5, C6, C7, C9, C10, C11. Multiple: FR-C3. None: FR-C4. Test data: FR-C8 |
| D. Orchestration (15): intent routing, multi-turn, out-of-scope and ambiguous | Routing: FR-D1, D7, D8, G1, G2, I1, I2. Multi-turn: FR-D2, D6, I3. Out-of-scope, ambiguous, tone, small talk: FR-D3, D4, D5, D9 |
| E. Code and docs (15): code, README, diagram | Section 3E |
| F. Robustness (10): API failure, retry or fallback, secrets, edge cases | FR-F1, F2, F3, F4, F5, F6, F7, G4, G5, G6 |
| F. Robustness (10), added operations: guardrails, budget, rate limits | FR-H1, H2, H3, H8, H13 |
| 6.1 Bonus (+10) | Section 4 items 1 to 6. Multi-agent: section 3I. Memory: FR-H10. Eval metrics: FR-H11, H12, H15. Caching: section 4 notes and cache AC |
| 7 Deliverables | Section 8 |
| 8 Sample flows | Section 7, sample flow scripts |
| 4 Non-functional, 10 FAQ, 11 self-assessment | Section 1 constraints, section 2 decisions, FR-A13, FR-D4 to D6, section 3E |
