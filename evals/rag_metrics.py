"""RAG metrics on the KB set (FR-H12). See evals/README.md for each definition.

`run` answers every KB question through the agent's own search and answer path, then scores the answers.
Retrieval metrics and citation accuracy are computed here; answer correctness, faithfulness (so the
hallucination rate) and factual precision come from a judge. `MockJudge` is a free lexical stand-in;
`RagasJudge` wraps the pinned Ragas metrics for real runs.
"""
import json
import math
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any, Protocol

from pydantic import BaseModel, ConfigDict, Field

from evals.common import words
from evals.harness import Resources, kb_rows
from evals.schemas import KBQuestion
from smartdesk_ai.agent.answer import make_answerer, should_escalate
from smartdesk_ai.agent.grounding import make_grounding_judge
from smartdesk_ai.config.settings import Settings
from smartdesk_ai.llm.mock import MockAnswerModel, MockGroundingModel
from smartdesk_ai.ops.health import HealthRegistry
from smartdesk_ai.rag.index import STOPWORDS, KBIndex
from smartdesk_ai.tools.definitions import SearchKBArgs
from smartdesk_ai.tools.search_kb import make_search_kb

KS = (1, 3, 5)
MIN_WORD = 4


class RagOutput(BaseModel):
    """What the agent produced for one KB question."""

    question_id: str
    answer: str  # empty when the agent escalated (no answer shown)
    insufficient: bool
    citations: list[str] = Field(default_factory=list)
    retrieved: list[str] = Field(default_factory=list)  # entry IDs, best first
    contexts: list[str] = Field(default_factory=list)  # texts of the retrieved chunks


class Judge(Protocol):
    def faithfulness(self, question: str, answer: str, contexts: Sequence[str]) -> float | None: ...

    def answer_correctness(self, question: str, answer: str, reference: str) -> float | None: ...

    def factual_precision(self, answer: str, reference: str) -> float | None: ...


# --- computed metrics ---------------------------------------------------------------------------

def _distinct(ids: Sequence[str]) -> list[str]:
    return list(dict.fromkeys(ids))


def retrieval_scores(
    outputs: Sequence[RagOutput], rows: Sequence[KBQuestion], ks: Sequence[int] = KS
) -> dict[int, dict[str, float]]:
    """Mean precision, recall and F1 at each k against the one expected entry per question."""
    expected = {r.id: r.expected_entry_id for r in rows}
    result: dict[int, dict[str, float]] = {}
    for k in ks:
        precisions, recalls, f1s = [], [], []
        for output in outputs:
            top = _distinct(output.retrieved)[:k]
            hit = 1.0 if expected[output.question_id] in top else 0.0
            precision = hit / len(top) if top else 0.0
            f1 = 2 * precision * hit / (precision + hit) if precision + hit else 0.0
            precisions.append(precision), recalls.append(hit), f1s.append(f1)
        n = len(outputs) or 1
        result[k] = {"precision": sum(precisions) / n, "recall": sum(recalls) / n, "f1": sum(f1s) / n}
    return result


def citation_accuracy(outputs: Sequence[RagOutput], rows: Sequence[KBQuestion]) -> tuple[float | None, int]:
    """(share of cited IDs that were retrieved and support the answer, number of cited IDs).
    A cited ID supports the answer when it is the entry the question was built from."""
    expected = {r.id: r.expected_entry_id for r in rows}
    cited = correct = 0
    for output in outputs:
        if output.insufficient:
            continue
        for entry_id in _distinct(output.citations):
            cited += 1
            correct += entry_id in output.retrieved and entry_id == expected[output.question_id]
    return (correct / cited if cited else None), cited


# --- judges ---------------------------------------------------------------------------------------

def _content(text: str) -> set[str]:
    return {w for w in words(text) if len(w) >= MIN_WORD and w not in STOPWORDS}


def _sentences(text: str) -> list[str]:
    return [s for s in (part.strip() for part in text.replace("\n", " ").split(". ")) if s]


class MockJudge:
    """A free lexical stand-in: it makes the numbers deterministic and exercises the code; it measures nothing real."""

    name = "mock (lexical)"
    SUPPORTED_SHARE = 0.8

    def faithfulness(self, question: str, answer: str, contexts: Sequence[str]) -> float:
        known = _content(" ".join(contexts))
        sentences = _sentences(answer)
        if not sentences:
            return 1.0
        supported = 0
        for sentence in sentences:
            tokens = _content(sentence)
            supported += (not tokens) or len(tokens & known) / len(tokens) >= self.SUPPORTED_SHARE
        return supported / len(sentences)

    def answer_correctness(self, question: str, answer: str, reference: str) -> float:
        a, r = _content(answer), _content(reference)
        overlap = len(a & r)
        return 2 * overlap / (len(a) + len(r)) if a and r else 0.0

    def factual_precision(self, answer: str, reference: str) -> float:
        a, r = _content(answer), _content(reference)
        return len(a & r) / len(a) if a else 0.0


def build_ragas_metrics(llm: Any, embeddings: Any) -> dict[str, Any]:
    """The pinned Ragas metric objects (legacy `ragas.metrics` classes), judged by `llm`."""
    from evals.ragas_compat import import_ragas

    import_ragas()
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        from ragas.metrics import AnswerCorrectness, FactualCorrectness, Faithfulness

        return {
            "faithfulness": Faithfulness(llm=llm),
            "answer_correctness": AnswerCorrectness(llm=llm, embeddings=embeddings),
            "factual_precision": FactualCorrectness(llm=llm, mode="precision"),
        }


class RagasJudge:
    name = "ragas"

    def __init__(self, *, faithfulness: Any, answer_correctness: Any, factual_precision: Any) -> None:
        self._faithfulness, self._correctness, self._precision = faithfulness, answer_correctness, factual_precision

    @staticmethod
    def _sample(**fields: Any) -> Any:
        from evals.ragas_compat import import_ragas

        import_ragas()
        from ragas.dataset_schema import SingleTurnSample

        return SingleTurnSample(**fields)

    @staticmethod
    def _score(value: Any) -> float | None:
        return None if value is None or math.isnan(float(value)) else float(value)

    def faithfulness(self, question: str, answer: str, contexts: Sequence[str]) -> float | None:
        sample = self._sample(user_input=question, response=answer, retrieved_contexts=list(contexts))
        return self._score(self._faithfulness.single_turn_score(sample))

    def answer_correctness(self, question: str, answer: str, reference: str) -> float | None:
        sample = self._sample(user_input=question, response=answer, reference=reference)
        return self._score(self._correctness.single_turn_score(sample))

    def factual_precision(self, answer: str, reference: str) -> float | None:
        sample = self._sample(response=answer, reference=reference)
        return self._score(self._precision.single_turn_score(sample))


# --- report ----------------------------------------------------------------------------------------

class RagReport(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    judge: str = "unnamed"
    n_questions: int
    n_answered: int
    retrieval: dict[int, dict[str, float]]
    answer_correctness: float | None
    hallucination_rate: float | None
    grounded_answer_rate: float | None
    citation_accuracy: float | None
    n_cited: int
    factual_precision: float | None
    outputs: list[RagOutput] = Field(default_factory=list, exclude=True)


def _mean(values: list[float | None]) -> float | None:
    kept = [v for v in values if v is not None]
    return sum(kept) / len(kept) if kept else None


def compute_metrics(
    outputs: Sequence[RagOutput], rows: Sequence[KBQuestion], judge: Judge, ks: Sequence[int] = KS
) -> RagReport:
    by_id = {r.id: r for r in rows}
    answered = [o for o in outputs if not o.insufficient and o.answer.strip()]
    faith = [judge.faithfulness(by_id[o.question_id].question, o.answer, o.contexts) for o in answered]
    correct = [judge.answer_correctness(by_id[o.question_id].question, o.answer, by_id[o.question_id].reference_answer)
               for o in answered]
    precision = [judge.factual_precision(o.answer, by_id[o.question_id].reference_answer) for o in answered]
    scored = [f for f in faith if f is not None]
    accuracy, cited = citation_accuracy(answered, rows)
    return RagReport(
        judge=getattr(judge, "name", type(judge).__name__), n_questions=len(outputs), n_answered=len(answered),
        retrieval=retrieval_scores(outputs, rows, ks), answer_correctness=_mean(correct),
        hallucination_rate=(sum(f < 1.0 for f in scored) / len(scored)) if scored else None,
        grounded_answer_rate=_mean(faith), citation_accuracy=accuracy, n_cited=cited,
        factual_precision=_mean(precision), outputs=list(outputs),
    )


# --- running the agent's path over the KB set ----------------------------------------------------------

AnswerFn = Callable[[KBQuestion], RagOutput]


def make_answer_fn(
    settings: Settings, index: KBIndex, reranker: Any, answerer: Callable[..., Any]
) -> AnswerFn:
    """Search and answer exactly as the agent does: an escalated question gets no answer."""
    search = make_search_kb(index, settings, reranker=reranker, health=HealthRegistry())

    def answer(row: KBQuestion) -> RagOutput:
        found = search(SearchKBArgs(query=row.question))
        result = answerer(row.question, found.chunks) if found.chunks else None
        retrieved = [c.entry_id for c in found.chunks]
        contexts = [c.text for c in found.chunks]
        if result is None or should_escalate(found, result):
            return RagOutput(question_id=row.id, answer="", insufficient=True, retrieved=retrieved, contexts=contexts)
        return RagOutput(question_id=row.id, answer=result.answer, insufficient=False, citations=result.citations,
                         retrieved=retrieved, contexts=contexts)

    return answer


def mock_answerer(settings: Settings) -> Callable[..., Any]:
    """The mock-mode answer step: the mock answer model plus the mock grounding judge, as in the app."""
    judge = make_grounding_judge(settings, builder=lambda model_id, cfg: MockGroundingModel())
    return make_answerer(settings, builder=lambda model_id, cfg: MockAnswerModel(), judge=judge)


def run(
    res: Resources, index: KBIndex, out_dir: Path | None, *, judge: Judge | None = None,
    answer_fn: AnswerFn | None = None, split: str | None = None,
) -> RagReport:
    settings = res.settings
    if res.mock:  # hashing vectors score lower than real ones, so retrieval uses its mock threshold (as the app does)
        settings = settings.model_copy(update={"thresholds": settings.thresholds.model_copy(
            update={"escalation": settings.thresholds.escalation_mock})})
    rows = kb_rows(split)
    answer_fn = answer_fn or make_answer_fn(settings, index, res.reranker, mock_answerer(settings))
    judge = judge or MockJudge()
    report = compute_metrics([answer_fn(r) for r in rows], rows, judge)
    _print(report)
    if out_dir:
        out_dir.mkdir(parents=True, exist_ok=True)
        payload = {"mock": res.mock, **report.model_dump(mode="json"),
                   "per_question": [o.model_dump() for o in report.outputs]}
        (out_dir / "rag_metrics.json").write_text(json.dumps(payload, indent=2, default=str))
    return report


def _fmt(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.3f}"


def _print(report: RagReport) -> None:
    print(f"rag metrics ({report.judge}): {report.n_answered}/{report.n_questions} answered")
    for k, s in report.retrieval.items():
        print(f"  @{k}: precision {s['precision']:.3f}  recall {s['recall']:.3f}  f1 {s['f1']:.3f}")
    print(f"  answer correctness {_fmt(report.answer_correctness)}  hallucination rate {_fmt(report.hallucination_rate)}"
          f"  grounded-answer rate {_fmt(report.grounded_answer_rate)}")
    print(f"  citation accuracy {_fmt(report.citation_accuracy)} over {report.n_cited} cited  "
          f"factual precision {_fmt(report.factual_precision)}")


# --- the real run (paid models; only after the user approved the plan) -----------------------------------------------

class SpendCeilingExceeded(RuntimeError):
    """The judge's spend passed the ceiling the run was given; the run stops."""


class UsageCounter:
    """A LangChain callback that adds up the tokens of every model call it sees (the Ragas judge's calls).
    With `ceiling_usd` it raises once the spend passes it (`raise_error` makes LangChain let that through)."""

    raise_error = True

    def __init__(self, settings: Settings | None = None, ceiling_usd: float | None = None) -> None:
        self.calls = self.input_tokens = self.output_tokens = 0
        self._settings, self._ceiling = settings, ceiling_usd

    def on_llm_end(self, response: Any, **kwargs: Any) -> None:
        for generations in response.generations:
            for generation in generations:
                usage = getattr(getattr(generation, "message", None), "usage_metadata", None) or {}
                self.calls += 1
                self.input_tokens += usage.get("input_tokens", 0)
                self.output_tokens += usage.get("output_tokens", 0)
        if self._ceiling is not None and self._settings is not None and self.cost_usd(self._settings) > self._ceiling:
            raise SpendCeilingExceeded(f"judge spend ${self.cost_usd(self._settings):.2f} is over the ${self._ceiling:.2f} ceiling")

    def cost_usd(self, settings: Settings) -> float:
        price = settings.models.main.price
        return (self.input_tokens * price.input + self.output_tokens * price.output) / 1_000_000


def build_real_judge(settings: Settings, ceiling_usd: float | None = None) -> tuple[RagasJudge, UsageCounter]:
    """Ragas judged by the Main-Model with the configured embedding model. Calling it spends money."""
    from langchain_core.callbacks import BaseCallbackHandler
    from langchain_openai import ChatOpenAI

    from evals.ragas_compat import import_ragas
    from smartdesk_ai.config.env import load_env
    from smartdesk_ai.rag.embeddings import get_embeddings

    import_ragas()
    from ragas.embeddings import LangchainEmbeddingsWrapper
    from ragas.llms import LangchainLLMWrapper

    env = load_env()
    if env.openai_api_key is None:
        raise RuntimeError("OPENAI_API_KEY is not set (see .env.example)")
    counter = type("UsageHandler", (UsageCounter, BaseCallbackHandler), {})(settings, ceiling_usd)
    model = settings.models.main
    chat = ChatOpenAI(model=model.id, api_key=env.openai_api_key, reasoning_effort=model.reasoning_effort,
                      callbacks=[counter])
    metrics = build_ragas_metrics(LangchainLLMWrapper(chat), LangchainEmbeddingsWrapper(get_embeddings(settings, env)))
    return RagasJudge(**metrics), counter


def run_real(res: Resources, index: KBIndex, out_dir: Path | None, *, split: str | None = None,
             ceiling_usd: float | None = None) -> RagReport:
    """The KB set through the live answer step and grounding judge, scored by the Ragas judge. Every model
    call is made under the cost task "eval" (FR-H8), and the spend is printed at the end."""
    from smartdesk_ai.llm.factory import openai_builder
    from smartdesk_ai.ops.cost import CostedEmbeddings, CostTracker
    from smartdesk_ai.ops.runtime import RequestContext, Services, request_scope

    settings = res.settings
    judge, counter = build_real_judge(settings, ceiling_usd)
    cost = CostTracker(settings, priced=True)
    index.embeddings = CostedEmbeddings(index.embeddings, cost, settings)
    answerer = make_answerer(settings, builder=openai_builder, judge=make_grounding_judge(settings, builder=openai_builder))
    answer_fn = make_answer_fn(settings, index, res.reranker, answerer)
    with request_scope(RequestContext(session_id="eval", request_id="rag-eval", task="eval"), Services(cost=cost)):
        report = run(res, index, out_dir, judge=judge, answer_fn=answer_fn, split=split)
    print(f"spend: pipeline ${cost.total:.4f} ({len(cost.entries)} steps); Ragas judge ${counter.cost_usd(settings):.4f} "
          f"({counter.calls} calls, {counter.input_tokens} in / {counter.output_tokens} out tokens)")
    return report
