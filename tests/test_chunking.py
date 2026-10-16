"""T2.1: heading-based chunker (FR-A3)."""
import re

import pytest

from smartdesk_ai.config.settings import Chunking, load_settings
from smartdesk_ai.rag.chunking import chunk_entry, count_tokens
from smartdesk_ai.rag.kb_schema import Entry
from tests.kb_helpers import load_all_entries

# Word-count "tokenizer" with small limits keeps the splitting tests readable.
SMALL = Chunking(target_min_tokens=20, target_max_tokens=50, split_above_tokens=60, overlap_pct=15)
words = lambda text: len(text.split())


def entry(body, **kw):
    base = dict(id="HR-LEAVE-01", domain="HR", topic="leave", title="Casual leave", owner="People Team",
                last_reviewed="2026-01-01", heading="Casual leave entitlement", body=body,
                source="hr_policies.md", document="HR Policies")
    return Entry(**{**base, **kw})


def sentences(n):
    return " ".join(f"Sentence number {i} says something about leave rules." for i in range(n))


def overlap_words(a, b):
    """Longest run of words that ends piece `a` and starts piece `b`."""
    wa, wb = a.split(), b.split()
    for n in range(min(len(wa), len(wb)), 0, -1):
        if wa[-n:] == wb[:n]:
            return n
    return 0


def test_short_entry_gives_one_chunk_with_parent_metadata():
    (c,) = chunk_entry(entry("Eight days per year."), SMALL, words)
    assert c.chunk_id == "HR-LEAVE-01#0" and c.entry_id == "HR-LEAVE-01"
    assert (c.domain, c.title, c.source_file) == ("HR", "Casual leave", "hr_policies.md")
    assert c.text == "Eight days per year."
    assert (c.piece, c.pieces) == (0, 1)


def test_header_is_document_arrow_section_and_prepended_to_embedded_text():
    (c,) = chunk_entry(entry("Eight days per year."), SMALL, words)
    assert c.header == "HR Policies > Casual leave entitlement"
    assert c.embed_text.startswith("HR Policies > Casual leave entitlement\n")
    assert c.embed_text.endswith("Eight days per year.")


def test_header_without_document_title_is_just_the_heading():
    (c,) = chunk_entry(entry("Text.", document=""), SMALL, words)
    assert c.header == "Casual leave entitlement"


def test_entry_at_the_split_threshold_stays_one_chunk_and_above_it_splits():
    at = " ".join(["w"] * 60)
    assert len(chunk_entry(entry(at), SMALL, words)) == 1
    assert len(chunk_entry(entry(at + " w"), SMALL, words)) >= 2


def test_long_entry_pieces_keep_parent_id_and_are_numbered():
    chunks = chunk_entry(entry(sentences(30)), SMALL, words)
    assert len(chunks) >= 3
    assert {c.entry_id for c in chunks} == {"HR-LEAVE-01"}
    assert [c.chunk_id for c in chunks] == [f"HR-LEAVE-01#{i}" for i in range(len(chunks))]
    assert all(c.pieces == len(chunks) for c in chunks)
    assert all(c.header == chunks[0].header for c in chunks)  # every piece is embedded with the header


def test_pieces_respect_the_target_size():
    for c in chunk_entry(entry(sentences(30)), SMALL, words):
        assert words(c.text) <= SMALL.target_max_tokens


def test_consecutive_pieces_overlap_by_about_15_percent():
    chunks = chunk_entry(entry(sentences(30)), SMALL, words)
    expected = SMALL.target_max_tokens * SMALL.overlap_pct / 100  # 7.5 tokens
    for a, b in zip(chunks, chunks[1:]):
        assert expected - 1 <= overlap_words(a.text, b.text) <= expected + 3


def test_pieces_cover_the_whole_entry_in_order():
    body = sentences(30)
    chunks = chunk_entry(entry(body), SMALL, words)
    assert body.startswith(chunks[0].text) and body.endswith(chunks[-1].text)
    merged = chunks[0].text.split()
    for c in chunks[1:]:
        merged += c.text.split()[overlap_words(" ".join(merged), c.text):]
    assert merged == body.split()


def test_unbroken_text_with_no_sentence_ends_still_splits():
    chunks = chunk_entry(entry(" ".join(f"w{i}" for i in range(200))), SMALL, words)
    assert len(chunks) >= 4 and all(words(c.text) <= 50 for c in chunks)


def test_zero_overlap_config_gives_disjoint_pieces():
    cfg = Chunking(target_min_tokens=20, target_max_tokens=50, split_above_tokens=60, overlap_pct=0)
    chunks = chunk_entry(entry(sentences(30)), cfg, words)
    assert all(overlap_words(a.text, b.text) == 0 for a, b in zip(chunks, chunks[1:]))


def test_every_real_kb_entry_is_one_chunk_under_the_config():
    cfg = load_settings().chunking
    entries = load_all_entries()
    for e in entries:
        chunks = chunk_entry(e, cfg)  # real tiktoken counter
        assert len(chunks) == 1, f"{e.id} split into {len(chunks)} pieces"
        assert count_tokens(e.body) <= cfg.split_above_tokens


def test_real_kb_entries_sit_inside_the_token_target_range():
    cfg = load_settings().chunking
    sizes = {e.id: count_tokens(e.body) for e in load_all_entries()}
    low = {i: n for i, n in sizes.items() if n < cfg.target_min_tokens}
    high = {i: n for i, n in sizes.items() if n > cfg.target_max_tokens}
    assert not high, f"above the target range: {high}"
    assert len(low) <= len(sizes) * 0.1, f"too many entries below the target range: {low}"


def test_count_tokens_counts_with_tiktoken():
    assert count_tokens("hello world") == 2
    assert count_tokens("") == 0
