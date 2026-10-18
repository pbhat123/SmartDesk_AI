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
- "Supports" in citation accuracy is decided by the ground-truth expected entry, not by a model.

## Other steps

| Step | What it measures | Notes |
|---|---|---|
| `agent` (`--mock`) | Completion rate, task accuracy, steps, latency, cost per task, error rate over `evals/scenarios` | Definitions at the top of `agent_metrics.py`. Throughput comes from `scripts/load_test.py`. |
| `gap` (`--mock`) | Gap pass rate, KB answer rate, escalation precision and recall, end to end | A gap question passes only on a ticket offer: never an answer, never a deflection. The default mock answer model is lexical and answers some grey-zone gap questions, so the mock number is below 100%; the pytest uses a scripted model that reports insufficient context. |
| `duplicate` | Sweep of `thresholds.duplicate_title` over `title_pairs.jsonl` | Free and offline. Prints the current value and two proposals; never edits config. |
| `routing` | Sweep of `thresholds.routing_confidence` over `intents.jsonl` | Mock classifier by default; the live classifier needs `--confirm-spend`. Never edits config. |

`python -m evals.safety_eval --live --confirm-spend` repeats the safety set with the real classifier,
Moderation API and models (tickets stay mock).

## Real runs cost money

A real judge run (`--judge ragas --confirm-spend`) makes many model calls. It is never run by the tests
and needs the user's explicit OK first (see TASKS.md, T6.2).

## Known failure cases

Live gap/KB runs (threshold 0.40, `evals/results/gap_sweep.json`): gap pass 100%, KB answer 98.1%. The one KB
question that is still escalated:

- **kb-041** ("A vendor wants to send me a festival hamper. Can I accept it?", expected HR-COND-02). The KB covers
  gifts, but nothing links "hamper" to "gift": the dense cosine is 0.342 (under 0.40) and answer v3 asks for the
  specific item named in the question, so the answerer marks it insufficient. The question is escalated to a ticket
  offer, the safe direction. Not fixed on purpose: adding "hamper" to the KB would tune the KB to the eval (FR-A18).
