"""T2.11: heading-based vs fixed 512/50 chunking, Recall@k on the KB set (FR-A17). Embeddings are mocked."""
import json
import math

import pytest

from evals import chunking_ablation as ab
from evals.harness import make_resources
from evals.run import main
from smartdesk_ai.rag.chunking import MIN_SHARE, _encoding, chunk_fixed
from smartdesk_ai.rag.index import open_index
from smartdesk_ai.rag.ingest import build_index
from smartdesk_ai.rag.kb_schema import Entry
from tests.kb_helpers import load_all_entries

ENTRIES = load_all_entries()
enc = _encoding()


def entry(i, words, source="a.md", domain="IT"):
    body = " ".join(f"w{i}x{n}" for n in range(words))
    return Entry(id=f"IT-T{i}-01", domain=domain, topic="t", title=f"T{i}", owner="o", last_reviewed="2026-01-01",
                 heading=f"Heading {i}", body=body, source=source, document="Doc")


def overlap_chars(a, b):
    for n in range(min(len(a), len(b)), 0, -1):
        if a[-n:] == b[:n]:
            return n
    return 0


# --- the fixed chunker -----------------------------------------------------------------

def test_windows_are_at_most_the_window_size_and_numbered_per_file():
    chunks, _ = chunk_fixed([entry(1, 700), entry(2, 700)], window=128, overlap=16)
    assert len(chunks) >= 8
    assert all(len(enc.encode(c.text)) <= 128 + 2 for c in chunks)  # +2: re-encoding a slice can split a token
    assert [c.piece for c in chunks] == list(range(len(chunks)))
    assert {c.source_file for c in chunks} == {"a.md"} and all(c.pieces == len(chunks) for c in chunks)


def test_the_number_of_windows_follows_the_stride():
    e = entry(1, 900)
    n_tokens = len(enc.encode(f"{e.heading}\n{e.body}"))
    chunks, _ = chunk_fixed([e], window=100, overlap=10)
    assert len(chunks) == math.ceil((n_tokens - 10) / 90)


def test_consecutive_windows_overlap_by_about_the_configured_tokens():
    chunks, _ = chunk_fixed([entry(1, 900)], window=100, overlap=10)
    for a, b in zip(chunks, chunks[1:]):
        shared = len(enc.encode(a.text[-overlap_chars(a.text, b.text):]))
        assert 8 <= shared <= 12


def test_windows_cover_the_text_and_ignore_entry_boundaries():
    e1, e2 = entry(1, 150), entry(2, 150)
    chunks, attribution = chunk_fixed([e1, e2], window=128, overlap=16)
    assert any(len(attribution[c.chunk_id]) == 2 for c in chunks)  # a window straddles both entries
    assert e1.heading in chunks[0].text and e2.body.split()[-1] in chunks[-1].text


def test_a_window_only_counts_for_entries_that_make_up_enough_of_it():
    big, small = entry(1, 200), entry(2, 6)
    chunks, attribution = chunk_fixed([big, small], window=len(enc.encode(f"{big.heading}\n{big.body}")) + 12, overlap=4)
    first = attribution[chunks[0].chunk_id]
    assert big.id in first and small.id not in first  # a few trailing words are below the share threshold
    assert 0.0 < MIN_SHARE < 1.0


def test_the_primary_entry_is_the_one_covering_most_of_the_window():
    chunks, _ = chunk_fixed([entry(1, 200), entry(2, 6)], window=300, overlap=10)
    assert chunks[0].entry_id == "IT-T1-01"


def test_files_are_chunked_separately():
    chunks, attribution = chunk_fixed([entry(1, 50, "a.md"), entry(2, 50, "b.md")], window=512, overlap=50)
    assert {c.source_file for c in chunks} == {"a.md", "b.md"}
    assert all(len(attribution[c.chunk_id]) == 1 for c in chunks)


def test_every_real_entry_is_attributed_to_at_least_one_window():
    chunks, attribution = chunk_fixed(ENTRIES, window=512, overlap=50)
    assert {e for ids in attribution.values() for e in ids} == {e.id for e in ENTRIES}
    assert len(chunks) < len(ENTRIES)  # 512-token windows are bigger than one entry
    assert sum(len(v) > 1 for v in attribution.values()) >= 5  # many windows mix entries: the cost of naive chunking


# --- building an index from arbitrary chunks -------------------------------------------

def test_build_index_replaces_the_previous_contents(tmp_path, settings):
    res = make_resources(mock=True)
    chunks, _ = chunk_fixed(ENTRIES, window=512, overlap=50)
    build_index(tmp_path, chunks, ENTRIES, res.embeddings)
    build_index(tmp_path, chunks[:3], ENTRIES, res.embeddings)
    ix = open_index(tmp_path, res.embeddings)
    assert len(ix.chunks) == 3 and ix.collection.count() == 3
    assert set(ix.entries) == {e.id for e in ENTRIES}


# --- the ablation ------------------------------------------------------------------------

@pytest.fixture(scope="module")
def result(tmp_path_factory):
    res = make_resources(mock=True)
    return ab.run_ablation(res, tmp_path_factory.mktemp("abl"))


def test_both_chunkers_are_evaluated_on_all_four_strategies(result):
    assert set(result["recall"]) == {"heading", "fixed512", "fixed512_primary"}
    for chunker, table in result["recall"].items():
        assert list(table) == ["bm25", "dense", "hybrid", "hybrid+rerank"], chunker
        for by_k in table.values():
            assert 0.0 <= by_k[1] <= by_k[3] <= by_k[5] <= 1.0


def test_strict_attribution_never_beats_lenient(result):
    for strategy in result["recall"]["heading"]:
        for k in (1, 3, 5):
            assert result["recall"]["fixed512_primary"][strategy][k] <= result["recall"]["fixed512"][strategy][k]


def test_stats_describe_each_chunking(result):
    h, f = result["stats"]["heading"], result["stats"]["fixed512"]
    assert h["chunks"] == len(ENTRIES) and f["chunks"] < h["chunks"]
    assert f["avg_tokens"] > h["avg_tokens"]
    assert h["avg_entries_per_chunk"] == 1.0 and f["avg_entries_per_chunk"] > 1.0
    assert result["attribution_min_share"] == MIN_SHARE


def test_format_comparison_puts_the_two_chunkers_side_by_side(result):
    text = ab.format_comparison(result)
    assert "heading" in text and "fixed512 (lenient)" in text and "fixed512 (strict)" in text
    for s in ("bm25", "dense", "hybrid", "hybrid+rerank"):
        assert s in text
    assert "R@1" in text and "R@5" in text


def test_cli_prints_the_comparison_and_writes_json(tmp_path, capsys):
    assert main(["--only", "chunking", "--mock", "--out-dir", str(tmp_path)]) == 0
    out = capsys.readouterr().out
    assert "MOCK" in out and "heading" in out and "fixed512" in out
    data = json.loads((tmp_path / "chunking.json").read_text())
    assert data["mock"] is True and set(data["recall"]) == {"heading", "fixed512", "fixed512_primary"}
