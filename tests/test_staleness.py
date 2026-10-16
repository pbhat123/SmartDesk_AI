"""T2.9: per-file hashes, stale flags, selective re-embedding and the KB-changed event (FR-H6)."""
import json

import pytest

from smartdesk_ai.rag.embeddings import HashingEmbeddings
from smartdesk_ai.rag.index import BM25_FILE
from smartdesk_ai.rag.ingest import ingest
from smartdesk_ai.rag.staleness import STATE_FILE, check_staleness
from tests.kb_helpers import load_all_entries

ENTRIES = load_all_entries()
EMBEDDER_ID = HashingEmbeddings().name


def ids_in(filename):
    return sorted(f"{e.id}#0" for e in ENTRIES if e.source == filename)


class Spy(HashingEmbeddings):
    def __init__(self, dim=256):
        super().__init__(dim)
        self.embedded = []

    def embed_documents(self, texts):
        self.embedded.extend(texts)
        return super().embed_documents(texts)


@pytest.fixture
def built(kb_copy, fresh_index_dir, settings):
    ingest(kb_copy, fresh_index_dir, HashingEmbeddings(), settings.chunking)
    return kb_copy, fresh_index_dir


def check(kb, ix, settings):
    return check_staleness(kb, ix, settings.chunking, EMBEDDER_ID)


def edit(path, old, new):
    text = path.read_text()
    assert old in text
    path.write_text(text.replace(old, new))


def test_a_fresh_index_has_nothing_stale(built, settings):
    report = check(*built, settings)
    assert report.stale_files == [] and report.stale_chunk_ids == []
    assert report.new_files == [] and report.deleted_files == [] and report.reason is None


def test_the_state_file_holds_a_hash_per_policy_file_and_no_kb_text(built):
    state = json.loads((built[1] / STATE_FILE).read_text())
    assert sorted(state["files"]) == ["company.md", "hr_policies.md", "it_support.md", "onboarding.md", "payroll_comp.md"]
    assert all(len(h) == 64 for h in state["files"].values())
    assert "password" not in json.dumps(state).lower()


def test_editing_one_file_flags_only_that_files_chunks(built, settings):
    kb, ix = built
    edit(kb / "it_support.md", "at least **14 characters**", "at least **16 characters**")
    report = check(kb, ix, settings)
    assert report.stale_files == ["it_support.md"]
    assert report.stale_chunk_ids == ids_in("it_support.md")
    assert len(report.stale_chunk_ids) == 16


def test_editing_two_files_flags_exactly_their_chunks(built, settings):
    kb, ix = built
    edit(kb / "payroll_comp.md", "**40% of CTC**", "**45% of CTC**")
    edit(kb / "company.md", "founded in 2016", "founded in 2015")
    report = check(kb, ix, settings)
    assert report.stale_files == ["company.md", "payroll_comp.md"]
    assert report.stale_chunk_ids == sorted(ids_in("company.md") + ids_in("payroll_comp.md"))


def test_checking_is_read_only(built, settings):
    kb, ix = built
    edit(kb / "it_support.md", "180 days", "170 days")
    before = (ix / BM25_FILE).read_text(), (ix / STATE_FILE).read_text()
    check(kb, ix, settings)
    assert ((ix / BM25_FILE).read_text(), (ix / STATE_FILE).read_text()) == before


def test_reingest_embeds_only_the_flagged_chunks_and_clears_the_flags(built, settings):
    kb, ix = built
    edit(kb / "it_support.md", "at least **14 characters**", "at least **16 characters**")
    spy = Spy()
    report = ingest(kb, ix, spy, settings.chunking)
    assert sorted(report.embedded) == ids_in("it_support.md")
    assert len(spy.embedded) == 16  # not all 53
    after = check(kb, ix, settings)
    assert after.stale_files == [] and after.stale_chunk_ids == []


def test_unchanged_files_are_not_re_embedded(built, settings):
    kb, ix = built
    spy = Spy()
    report = ingest(kb, ix, spy, settings.chunking)
    assert report.embedded == [] and spy.embedded == []
    assert (report.added, report.removed) == ([], [])


def test_reingest_removes_chunks_of_deleted_entries_and_clears_flags(built, settings):
    kb, ix = built
    path = kb / "hr_policies.md"
    text = path.read_text()
    start, end = text.index("## Compensatory off for extra hours"), text.index("## Hybrid schedule and core office days")
    path.write_text(text[:start] + text[end:])
    assert check(kb, ix, settings).stale_files == ["hr_policies.md"]
    report = ingest(kb, ix, HashingEmbeddings(), settings.chunking)
    assert report.removed == ["HR-LEAVE-08#0"]
    assert check(kb, ix, settings).stale_chunk_ids == []


def test_a_deleted_file_flags_its_chunks_and_reingest_prunes_them(built, settings):
    kb, ix = built
    (kb / "payroll_comp.md").unlink()
    report = check(kb, ix, settings)
    assert report.deleted_files == ["payroll_comp.md"]
    assert report.stale_chunk_ids == ids_in("payroll_comp.md")
    ingest(kb, ix, HashingEmbeddings(), settings.chunking)
    after = check(kb, ix, settings)
    assert after.stale_chunk_ids == [] and after.deleted_files == []


def test_a_file_that_comes_back_is_new_and_gets_embedded(built, settings):
    kb, ix = built
    original = (kb / "payroll_comp.md").read_text()
    (kb / "payroll_comp.md").unlink()
    ingest(kb, ix, HashingEmbeddings(), settings.chunking)
    (kb / "payroll_comp.md").write_text(original)
    assert check(kb, ix, settings).new_files == ["payroll_comp.md"]
    spy = Spy()
    report = ingest(kb, ix, spy, settings.chunking)
    assert sorted(report.embedded) == ids_in("payroll_comp.md")


def test_the_kb_changed_event_fires_on_change_and_not_on_a_no_op(kb_copy, fresh_index_dir, settings):
    events = []
    run = lambda: ingest(kb_copy, fresh_index_dir, HashingEmbeddings(), settings.chunking, on_kb_changed=events.append)
    run()
    assert len(events) == 1 and events[0].kb_changed  # the first build
    run()
    assert len(events) == 1  # nothing changed, so the answer cache stays
    edit(kb_copy / "it_support.md", "180 days", "170 days")
    run()
    assert len(events) == 2 and events[1].embedded == ids_in("it_support.md")
    path = kb_copy / "hr_policies.md"
    text = path.read_text()
    path.write_text(text[: text.index("## Compensatory off for extra hours")] + text[text.index("## Hybrid schedule and core office days"):])
    run()
    assert len(events) == 3 and events[2].removed == ["HR-LEAVE-08#0"]


def test_a_failing_event_handler_does_not_break_ingest(kb_copy, fresh_index_dir, settings):
    def boom(report):
        raise RuntimeError("cache offline")

    report = ingest(kb_copy, fresh_index_dir, HashingEmbeddings(), settings.chunking, on_kb_changed=boom)
    assert report.chunks == len(ENTRIES)


def test_changing_the_chunking_config_re_embeds_everything(built, settings):
    kb, ix = built
    other = settings.chunking.model_copy(update={"split_above_tokens": 700})
    stale = check_staleness(kb, ix, other, EMBEDDER_ID)
    assert "chunking" in stale.reason and len(stale.stale_chunk_ids) == len(ENTRIES)
    spy = Spy()
    report = ingest(kb, ix, spy, other)
    assert len(report.embedded) == len(ENTRIES)


def test_switching_embedder_rebuilds_the_dense_index_even_with_a_new_vector_size(built, settings):
    kb, ix = built
    small = Spy(dim=128)
    assert small.name != EMBEDDER_ID
    stale = check_staleness(kb, ix, settings.chunking, small.name)
    assert "embedder" in stale.reason
    report = ingest(kb, ix, small, settings.chunking)
    assert len(report.embedded) == len(ENTRIES)
    assert check_staleness(kb, ix, settings.chunking, small.name).stale_chunk_ids == []


def test_an_index_without_state_is_treated_as_fully_stale(built, settings):
    kb, ix = built
    (ix / STATE_FILE).unlink()
    report = check(kb, ix, settings)
    assert len(report.stale_chunk_ids) == len(ENTRIES) and "no state" in report.reason


def test_checking_when_no_index_exists_reports_nothing_stale(kb_copy, fresh_index_dir, settings):
    report = check(kb_copy, fresh_index_dir, settings)
    assert report.stale_chunk_ids == [] and report.stale_files == []
