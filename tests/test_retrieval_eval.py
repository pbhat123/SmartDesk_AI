"""T2.10: Recall@1/3/5 for BM25, dense, hybrid and hybrid+rerank (FR-A7). Embeddings and reranker are mocked."""
import json

import pytest

from evals import retrieval_eval as re_
from evals.harness import make_resources
from evals.run import main
from evals.schemas import KBQuestion
from smartdesk_ai.rag.index import open_index
from smartdesk_ai.rag.ingest import ingest
from smartdesk_ai.rag.models import Chunk
from tests.kb_helpers import KB_DIR

def re_cfg():
    from smartdesk_ai.config.settings import load_settings

    return load_settings()


STRATEGIES = ["bm25", "dense", "hybrid", "hybrid+rerank"]


def row(i, expected, split="dev"):
    return KBQuestion(id=f"q{i}", question=f"question number {i} here?", expected_entry_id=expected,
                      expected_domain="IT", type="simple", reference_answer="reference answer text here", split=split)


def chunk(entry, piece=0):
    return Chunk(chunk_id=f"{entry}#{piece}", entry_id=entry, domain="IT", title=entry, header=f"Doc > {entry}",
                 text="t", source_file="it_support.md", piece=piece, pieces=2 if piece else 1)


@pytest.fixture(scope="module")
def resources():
    return make_resources(mock=True)


@pytest.fixture(scope="module")
def index(resources, tmp_path_factory, settings):
    d = tmp_path_factory.mktemp("eval_index")
    ingest(KB_DIR, d, resources.embeddings, settings.chunking)
    return open_index(d, resources.embeddings)


# --- the metric itself, with hand-made rankings ---------------------------------------

def test_recall_counts_a_hit_when_the_expected_entry_is_in_the_top_k(monkeypatch):
    rankings = {"q0": ["A", "B", "C", "D", "E"], "q1": ["X", "A", "B", "C", "D"], "q2": ["X", "Y", "Z", "A", "B"],
                "q3": ["X", "Y", "Z", "W", "V"]}
    monkeypatch.setattr(re_, "rank_results",
                        lambda strategy, ix, q, cfg, rr, **kw: [chunk(e) for e in rankings[f"q{q.split()[2]}"]])
    rows = [row(i, "A") for i in range(4)]
    out = re_.recall_at_k(None, rows, None, None, ["bm25"], ks=(1, 3, 5))
    assert out["bm25"] == {1: 0.25, 3: 0.5, 5: 0.75}


def test_pieces_of_one_entry_count_once_so_they_do_not_crowd_the_top_k(monkeypatch):
    from smartdesk_ai.rag.models import ScoredChunk

    stub = [ScoredChunk(chunk=chunk(e, p), score=1.0, stage="bm25") for e, p in (("X", 0), ("X", 1), ("A", 0))]
    monkeypatch.setattr(re_, "bm25_search", lambda *a, **kw: stub)  # the real rank_results collapse runs
    out = re_.recall_at_k(None, [row(0, "A")], re_cfg(), None, ["bm25"], ks=(1, 2, 3))
    assert out["bm25"] == {1: 0.0, 2: 1.0, 3: 1.0}
    kept = re_.recall_at_k(None, [row(0, "A")], re_cfg(), None, ["bm25"], ks=(2,), collapse=False)
    assert kept["bm25"] == {2: 0.0}  # without collapsing, the two X pieces fill the top 2


def test_a_custom_attribution_lets_one_result_count_for_several_entries(monkeypatch):
    monkeypatch.setattr(re_, "rank_results", lambda *a, **kw: [chunk("X")])
    out = re_.recall_at_k(None, [row(0, "A")], None, None, ["bm25"], ks=(1,), attribution=lambda c: {"X", "A"})
    assert out["bm25"][1] == 1.0


def test_split_filter_restricts_the_rows(monkeypatch):
    monkeypatch.setattr(re_, "rank_results", lambda *a, **kw: [chunk("A")])
    rows = [row(0, "A", "dev"), row(1, "B", "test")]
    assert re_.recall_at_k(None, rows, None, None, ["bm25"], ks=(1,), split="dev")["bm25"][1] == 1.0
    assert re_.recall_at_k(None, rows, None, None, ["bm25"], ks=(1,), split="test")["bm25"][1] == 0.0


# --- the real pipeline stages ----------------------------------------------------------

def test_all_four_strategies_give_monotonic_recall_between_zero_and_one(index, resources, settings):
    rows = re_.load_rows()
    out = re_.recall_at_k(index, rows, settings, resources.reranker, STRATEGIES)
    assert list(out) == STRATEGIES
    for strategy, by_k in out.items():
        assert set(by_k) == {1, 3, 5}
        assert 0.0 <= by_k[1] <= by_k[3] <= by_k[5] <= 1.0, strategy
    assert out["bm25"][5] > 0.5  # the mock setup is weak, but BM25 on exact wording is not random


def test_dev_and_test_halves_partition_the_set():
    rows = re_.load_rows()
    dev, test = re_.load_rows("dev"), re_.load_rows("test")
    assert len(dev) + len(test) == len(rows) and not {r.id for r in dev} & {r.id for r in test}


def test_hybrid_rerank_scores_the_fused_candidates_once_per_query(index, resources, settings):
    seen = []

    class Spy:
        def rerank(self, query, candidates):
            ids = [c.chunk.chunk_id for c in candidates]
            assert len(ids) == len(set(ids))
            seen.append(query)
            return resources.reranker.rerank(query, candidates)

    rows = re_.load_rows()[:5]
    re_.recall_at_k(index, rows, settings, Spy(), ["hybrid+rerank"])
    assert seen == [r.question for r in rows]


def test_hybrid_rerank_without_a_reranker_is_an_error(index, settings):
    with pytest.raises(ValueError, match="reranker"):
        re_.recall_at_k(index, re_.load_rows()[:1], settings, None, ["hybrid+rerank"])


# --- table and CLI ---------------------------------------------------------------------

def test_format_table_lists_every_strategy_and_k():
    text = re_.format_table({s: {1: 0.5, 3: 0.75, 5: 1.0} for s in STRATEGIES}, title="Recall@k", n=53)
    for s in STRATEGIES:
        assert s in text
    assert "R@1" in text and "R@3" in text and "R@5" in text and "0.75" in text and "53" in text


def test_cli_prints_the_four_strategy_table_in_mock_mode(capsys):
    assert main(["--only", "retrieval", "--mock"]) == 0
    out = capsys.readouterr().out
    assert "MOCK" in out
    for s in STRATEGIES:
        assert s in out
    assert "dev" in out and "test" in out


def test_cli_writes_a_results_file_only_when_asked(tmp_path, capsys):
    main(["--only", "retrieval", "--mock"])
    main(["--only", "retrieval", "--mock", "--out-dir", str(tmp_path)])
    data = json.loads((tmp_path / "retrieval.json").read_text())
    assert data["mock"] is True and set(data["overall"]) == set(STRATEGIES)
    assert data["n"] == len(re_.load_rows())


def test_real_mode_without_a_built_index_explains_what_to_run(tmp_path, capsys, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "not-a-real-key")
    assert main(["--only", "retrieval", "--index-dir", str(tmp_path / "none")]) == 1
    assert "--only ingest" in capsys.readouterr().err
