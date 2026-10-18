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

## Real runs cost money

A real judge run (`--judge ragas --confirm-spend`) makes many model calls. It is never run by the tests
and needs the user's explicit OK first (see TASKS.md, T6.2).
