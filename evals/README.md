# Evals

`python -m evals.run` runs the eval steps and writes a run record (`run_record.json`) with the prompt
versions, model IDs, thresholds and git commit next to the results, so numbers compare across versions
(FR-H11, FR-H14). Add `--mock` for a free offline run. Steps: `ingest`, `retrieval`, `chunking`,
`threshold`, `rag`, `agent`, `gap`.

## RAG metrics (`--only rag`, FR-H12)

Computed on the KB set (`kb_questions.jsonl`). The agent's own search and answer path produces each
answer, so what is scored is what an employee would see. A question the agent escalates (no answer shown)
counts in the retrieval metrics but not in the answer metrics.

| Metric | Definition | Source |
|---|---|---|
| Retrieval precision, recall, F1 @k (k = 1, 3, 5) | The distinct entries retrieved, cut at k, against the one expected entry. Precision divides by how many were retrieved (at most k). Averaged over questions. | computed here |
| Answer correctness (accuracy) | Agreement with the reference answer | Ragas `AnswerCorrectness` |
| Hallucination rate | Share of answers with any unsupported claim (Faithfulness below 1) | Ragas `Faithfulness` |
| Grounded-answer rate | Mean Faithfulness, target 0.8 or higher | Ragas `Faithfulness` |
| Citation accuracy | Cited entry IDs that were retrieved **and** support the answer (the entry the question is built from), divided by all cited IDs | computed here |
| Expected entry cited | Answers (out of those answered) that cite the entry the question is built from, whatever else they cite | computed here |
| Factual precision | Share of the answer's claims supported by the reference | Ragas `FactualCorrectness`, `mode="precision"` |

Substitutions and caveats:

- All three Ragas metrics exist in the pinned version (0.4.3), so none is replaced. They are the legacy
  classes in `ragas.metrics`; the newer `ragas.metrics.collections` versions need an `instructor` client
  and the pinned release marks the legacy path as deprecated (removal in 1.0).
- `import ragas` fails in this environment because Ragas imports a Vertex AI module that
  langchain-community 0.4.2 no longer ships. `evals/ragas_compat.py` registers an empty placeholder module
  first (`ragas_compat`). Remove it when the pins change.
- Mock runs use a lexical stand-in judge, so their correctness, faithfulness and precision numbers only
  exercise the code. They are not results.
- "Supports" in citation accuracy is decided by the ground-truth expected entry, not by a model. An answer that
  also cites a second entry that truly supports it is penalised; "expected entry cited" shows how many answers cite
  the right entry whatever else they cite.
- **The references are brief** (21 words on average, against 41 for the answers). Answer correctness and factual
  precision score the answer against that brief reference, so a correct answer that adds true detail (a deadline, a
  contact, an escalation rule) scores lower: correctness 0.695 and factual precision 0.508 understate answer
  quality. Read them as relative numbers between prompt versions, not as accuracy. Faithfulness, context recall and the
  gap set are the better quality signals. Key-fact references were considered and not adopted.

## Retrieval ablation (offline)

`python -m evals.retrieval_report [--mock] [--out-dir DIR]` gives hit@1/3/5 and MRR per strategy (dense, BM25, hybrid,
hybrid + rerank, dense + rerank) overall, per split, per question type and for the low and high lexical-overlap halves
of the questions, which shows paraphrase bias. A real run reads only `data/embedding_cache.json`; an uncached text raises,
so it makes no API call. `python -m evals.lexical_report` scores the same five strategies on the held-out exact-term set
`lexical_set.jsonl` (15 queries typed the way a user types a literal; no tuning on it). It also runs offline; add
`--embed-new` once to embed queries that are not cached (the only paid call, a few thousand tokens at most, recorded in the ledger).

**KB questions (53, paraphrased on purpose, FR-A18)**

| Strategy | hit@1 | hit@3 | hit@5 | MRR |
|---|---|---|---|---|
| dense | **0.925** | **1.000** | **1.000** | **0.956** |
| BM25 | 0.509 | 0.774 | 0.830 | 0.663 |
| hybrid (RRF) | 0.774 | 0.925 | 0.943 | 0.861 |
| hybrid+rerank (production) | 0.736 | 0.981 | **1.000** | 0.847 |
| dense+rerank | 0.755 | 0.981 | **1.000** | 0.862 |

**Exact-term queries (15, held out)**

| Strategy | hit@1 | hit@3 | hit@5 | MRR |
|---|---|---|---|---|
| dense | 0.800 | 0.867 | 0.867 | 0.843 |
| BM25 | **1.000** | **1.000** | **1.000** | **1.000** |
| hybrid (RRF) | 0.867 | 0.933 | 0.933 | 0.904 |
| hybrid+rerank (production) | **1.000** | **1.000** | **1.000** | **1.000** |
| dense+rerank | 0.933 | 0.933 | 0.933 | 0.933 |

The 15 queries (all with a literal that exists in only one KB entry):

1. `VPN error 691` → IT-VPN-02
2. `windows error 809 server unreachable` → IT-VPN-02
3. `what do I put in Server for vpn.smartdesk.example` → IT-VPN-01
4. `SmartDesk-Corp profile missing after update` → IT-VPN-01
5. `SmartDesk-Secure keeps asking for password` → IT-WIFI-02
6. `guest sign-in page not showing neverssl.com` → IT-WIFI-01
7. `outlook "device not compliant"` → IT-MAIL-01
8. `phishing@smartdesk.example screenshot` → IT-SEC-01
9. `travel@smartdesk.example flight booking` → HR-REIMB-02
10. `icc@smartdesk.example` → HR-HARASS-02
11. `gift register` → HR-COND-02
12. `comp-off 45 days lapse` → HR-LEAVE-08
13. `benefits@smartdesk.example enrol` → HR-ONB-02
14. `Download PDF payslip` → HR-PAY-02
15. `vesting grant date 1/48 per month` → HR-STOCK-01

**Paraphrase bias.** The KB questions share on average 44% of their content words with the expected entry. BM25 hit@1 is
0.385 on the lower-overlap half and 0.630 on the upper half; dense is flat (0.923 and 0.926); the correlation of overlap
with a rank-1 hit is 0.40 for BM25, 0.09 for dense, 0.28 for hybrid+rerank. So the paraphrased set handicaps BM25 and
favours dense, and the exact-term set does the opposite: dense drops to 0.800 hit@1 and misses `icc@smartdesk.example`
entirely, while BM25 and hybrid+rerank get 1.000. Read together: dense is better at paraphrases, keyword search at literals,
and hybrid+rerank is the only strategy with hit@5 = 1.000 on both sets. The decision rule was "dense, if it is at least as
good as hybrid+rerank on both sets"; it is not, so the production strategy stays. Details and the recommendation are in
PLAN.md section 14.5. Caveat: 15 queries, one author; one query is 6.7 points.

## Other steps

| Step | What it measures | Notes |
|---|---|---|
| `agent` (`--mock`) | Completion rate, task accuracy, steps, latency, cost per task, error rate over `evals/scenarios` | Definitions at the top of `agent_metrics.py`. Throughput comes from `scripts/load_test.py`. |
| `gap` (`--mock`) | Gap pass rate, KB answer rate, escalation precision and recall, end to end | A gap question passes only on a ticket offer: never an answer, never a deflection. The default mock answer model is lexical and answers some grey-zone gap questions, so the mock number is below 100%; the pytest uses a scripted model that reports insufficient context. |
| `duplicate` | Sweep of `thresholds.duplicate_title` over `title_pairs.jsonl` | Free and offline. Prints the current value and two proposals; never edits config. |
| `routing` | Sweep of `thresholds.routing_confidence` over `intents.jsonl` | Mock classifier by default; the live classifier needs `--confirm-spend`. Never edits config. |

`python -m evals.safety_eval --live --confirm-spend` repeats the safety set with the real classifier,
Moderation API and models (tickets stay mock).

## Live Ragas results (2026-10-18)

| Metric | Score | Note |
|---|---|---|
| Faithfulness (grounded-answer rate) | 0.971 | Ragas flagged 4 of 51 answers (hallucination rate 0.078); reading them, none is a real hallucination (14.3) |
| Answer relevancy | 0.636 | Embedding similarity of questions regenerated from a short, direct answer; low by construction |
| Context precision | 0.862 | Reference-based |
| Context recall | 0.965 | Reference-based |
| Answer correctness | 0.695 | Pulled down by reference scope, not by wrong facts (14.3) |
| Factual precision | 0.508 | Answers average 41 words against 21 in the references, so correct extra claims count against it |
| Citation accuracy | 0.879 (58 cited) | The expected entry is cited in 51 of 51 answers; 7 answers also cite a second entry (14.3) |
| Retrieval @1 / @3 / @5 | recall 0.736 / 0.981 / 1.000 | precision 0.736 / 0.327 / 0.200 |

Judge `gpt-5.4-mini` (effort none) over 53 KB questions, 51 answered, 310 judged items, 0 failed samples. Cost **$0.94 for the judge-only run** (726 chat calls, 411 embedding calls; the ledger says $0.9381, the token counter $0.9380). Total for the day with the one-sample smoke check: $0.9562 (`python -m smartdesk_ai.ops.spend_ledger --day 2026-10-18`). The answers came from the earlier attempt and were not rerun.

Reading them: see PLAN.md section 14.3 for the four flagged answers (none is a real hallucination), the seven
extra citations (four valid second sources, three over-citations) and five low-correctness samples (three are
reference scope, two are real omissions). Low answer relevancy and factual precision come from how the metrics
treat short, direct answers against short references, so judge the answer quality by faithfulness, context
recall and the gap set rather than by those two.

## Real runs cost money

A real judge run (`--judge ragas --confirm-spend`) makes many model calls. It is never run by the tests
and needs the user's explicit OK first (see TASKS.md, T6.2).

Every real run appends its spend to one fixed ledger, `data/spend_ledger.jsonl` (override: `SMARTDESK_LEDGER_PATH`),
whatever data dir the stack uses (`smartdesk_ai/ops/spend_ledger.py`). One line per API call, written as it happens,
so a crash loses nothing; numbers and names only. `python -m smartdesk_ai.ops.spend_ledger --day YYYY-MM-DD`
prints the spend per run and day (UTC, like the billing page). The app's $5/$10 cutoffs read `ops.db`, which holds
live app spend only; eval and Ragas spend is in the ledger, not in the cutoffs. Reasoning tokens are inside
`output_tokens` (billed once, at the output price).

The judge stage is built to survive failure (`judge_run.py`, FR-H12):

- `python -m evals.run --only rag --judge ragas --judge-only --confirm-spend --ceiling-usd N --out-dir evals/results`
  judges the saved `rag_outputs.json` and does not rerun the pipeline.
- Each model call has a 60s timeout and at most 3 attempts (Ragas defaults are 180s and 10; it also retried a
  "no credits" 429 for hours). Out-of-credits and bad-key errors stop the run on the first call.
- `judge_progress.log` gets a line per scored item (done/total, spend so far). `judge_checkpoint.json` is saved
  every 10 new items; rerunning the same command resumes, and the ceiling counts the earlier attempts' spend.
  A checkpoint is ignored if `rag_outputs.json` has changed. Delete it to start over.
- The spend ceiling stops the run even when the exception is swallowed in a worker thread.
- A watchdog aborts (exit code 3) after `--stall-seconds` (default 300) with no model call or item finished.

## Post-Phase-7 final runs (2026-10-19)

One resumable command runs them in order, with every stop rule: `python -m evals.post_phase7 --estimate` shows the table of estimates, `--confirm-spend` runs it, `--confirm-spend --resume` carries on after a stop (`--fresh` starts over; PLAN 19.15 has the plan and the rules). Results are in `evals/results/` (`orchestrator_plan_check.json`, `compound_eval_live.json`, `safety_eval.json`, `gap_eval.json`, `cache_cutoff.json`, `tier_comparison.json`, `rag_metrics_final.json`, `diagnostics/`); live traces are kept under `evals/results/traces/` and are not committed.

| Measure | Result |
|---|---|
| Gap pass / KB answer rate / answered with citations | 100% / 98.1% / 98.1% (kb-041 is the one miss) |
| Escalation precision / recall | 95.5% / 100% |
| Safety: leakage / breach | 0.00 / 0.00 (27 rows) |
| Planning check (orchestrator v8): intent / domain accuracy | 98.9% / 94.6%; 1 of 74 single-topic questions split after the guard (2 raw); 14 of 14 compound questions with the right number of parts |
| Live compound set | 13 of 14 rows (cmp-008 fails the retry check); routing 24 of 24 right |
| Ragas (one judge pass) | grounded 0.981, hallucination 0.118, correctness 0.616, relevancy 0.612, factual precision 0.348, context precision 0.852, context recall 0.965 (answer v4; the 2026-10-18 table used v3) |
| Drift baseline | faithfulness 0.948, recall@5 0.981, escalation rate 0.297 |
| Tier comparison | tier map $0.00193 per task, all-Main $0.00245; accuracy 0.917 on both |
| Semantic cache cut-off | 0.84 was below the highest different-answer cosine (0.864); raised to 0.97 (approved 2026-10-19, catches 5 of 23 paraphrase pairs) |

Known limits: the live compound set does not run the single-part controls (the gap and KB run covers them: 71 of 74 used one worker and no synthesizer call); the rewritten sub-questions come from the compound rows only, not from follow-up turns; a message that mixes a KB question with a ticket request stays one intent.

## Post-Phase-8 rerun with orchestrator v9 (2026-10-20)

`python -m evals.post_phase7 --profile phase8 --confirm-spend` (own state file; `--resume` carries on; PLAN 14.9 has the full table). It runs the planning check, compound, safety, gap + KB (drift point) and the T7.11 scenarios only (`python -m evals.tier_comparison --scenarios-only`: no KB set, no Ragas). A completed gap + KB checkpoint is replayed for free on `--resume`; to measure again, move `data/eval_checkpoints/golden.json` aside first.

| Measure | v8 | v9 (second gap + KB run) |
|---|---|---|
| Gap pass / KB answer rate / answered with citations | 100% / 98.1% / 98.1% | 100% / 98.1% / 98.1% (kb-041) |
| Escalation precision / recall | 95.5% / 100% | 95.5% / 100% |
| Safety leakage / breach | 0.00 / 0.00 | 0.00 / 0.00 |
| Planning check intent / domain | 98.9% / 94.6% | 97.7% / 90.5% |
| Live compound set | 13 of 14 | 14 of 14 (routing 24 of 24) |
| Drift point | 0.948 / 0.981 / 0.297 / 0.265 | 0.966 / 0.981 / 0.297 / 0.261 |
| Scenarios, tier map / all-Main per task | $0.00193 / $0.00245 | $0.00186 / $0.00192 (accuracy 0.917 both) |

The first v9 gap + KB run stopped on the 92% citation rule (90.6%: kb-010, kb-016, kb-018, kb-033, kb-041); its results are kept as `results/gap_eval_v9_run1.json`. Classified from the traces: kb-010, kb-016 and kb-033 variance (grounding judge or worker flag), kb-018 a borderline v9 routing flip, kb-041 already a miss under v8. The second run passed every rule. Drift history now has four live points (v5, v8, v9 twice). Batch cost $0.848 (estimate $0.84).

## Known failure cases

Live gap/KB runs (threshold 0.40, `evals/results/gap_sweep.json`): gap pass 100%, KB answer 98.1%. The one KB
question that is still escalated:

- **kb-041** ("A vendor wants to send me a festival hamper. Can I accept it?", expected HR-COND-02). The KB covers
  gifts, but nothing links "hamper" to "gift": the dense cosine is 0.342 (under 0.40) and answer v3 asks for the
  specific item named in the question, so the answerer marks it insufficient. The question is escalated to a ticket
  offer, the safe direction. Not fixed on purpose: adding "hamper" to the KB would tune the KB to the eval (FR-A18).
- **Dropped next steps** (kb-019, kb-037, from the Ragas run). The answer is grounded but leaves out an actionable step
  or deadline that sits in the retrieved entry (call Security on extension 4999 for a lost laptop; receipts over ₹500 and
  the 20th-of-the-month pay cut-off for an expense claim). Answer v3 asks for the item named in the question, which trims
  these. Proposed fix: answer v4 (not applied).
- **"The context" in user-facing text** (kb-007, kb-019, kb-050). Three answers say "the context says", "the policy also says"
  or "the context does not state". Proposed fix: answer v4 forbids it (not applied).
- **Extra citations** (kb-002, kb-007, kb-030 are over-citations; kb-001, 021, 025, 043 are valid second sources). The expected
  entry is cited in all 51 answers; citation accuracy 0.879 counts any other entry as wrong. Proposed fix: cite only entries
  used, report "expected entry cited", judge each cited chunk (not applied).
- **Expected entry not ranked first** in 13 of 51 answered questions (for example kb-043, where two unrelated entries rank above it).
  Look at it in the retrieval ablation before changing anything.
