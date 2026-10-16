"""T2.2: ingest pipeline (FR-A11) with mocked embeddings. No network, no API spend."""
import json
import shutil
from pathlib import Path

import chromadb
import pytest
from chromadb.config import Settings as ChromaSettings

from smartdesk_ai.config.settings import load_settings
from smartdesk_ai.rag.embeddings import HashingEmbeddings
from smartdesk_ai.rag.index import BM25_FILE, CHROMA_DIR, COLLECTION
from smartdesk_ai.rag.ingest import POLICY_FILES, IngestError, ingest, main
from smartdesk_ai.rag.kb_schema import KBValidationError
from tests.kb_helpers import KB_DIR, load_all_entries

CFG = load_settings().chunking
N_ENTRIES = len(load_all_entries())
MARKER = "ZXQ-NOT-POLICY-MARKER"


@pytest.fixture
def kb(tmp_path):
    """A KB copy that also holds the three non-policy files, each carrying a marker, plus a stray .md."""
    d = tmp_path / "kb"
    shutil.copytree(KB_DIR, d)
    for name in ("GAPS.md", "SOURCES.md"):
        (d / name).write_text((d / name).read_text() + f"\n{MARKER}\n")
    qa = json.loads((d / "qa.json").read_text())
    qa[0]["answer"] += f" {MARKER}"
    (d / "qa.json").write_text(json.dumps(qa))
    (d / "stray.md").write_text(
        "# Stray\n\n## Stray entry\n\n```yaml\nid: HR-STRAY-01\ndomain: HR\ntopic: stray\ntitle: Stray entry\n"
        "owner: x\nlast_reviewed: 2026-01-01\n```\n\nThis file is not a policy file.\n"
    )
    return d


@pytest.fixture
def index_dir(tmp_path):
    return tmp_path / "index"


def run(kb, index_dir, embeddings=None):
    return ingest(kb, index_dir, embeddings or HashingEmbeddings(), CFG)


def collection(index_dir):
    client = chromadb.PersistentClient(path=str(index_dir / CHROMA_DIR), settings=ChromaSettings(anonymized_telemetry=False))
    return client.get_collection(COLLECTION)


def bm25_text(index_dir):
    return (index_dir / BM25_FILE).read_text()


def chroma_text(index_dir):
    got = collection(index_dir).get()
    return "\n".join(got["documents"] + [json.dumps(m) for m in got["metadatas"]] + got["ids"])


def test_policy_file_list_is_exactly_the_five_policy_files():
    assert POLICY_FILES == ("company.md", "it_support.md", "onboarding.md", "hr_policies.md", "payroll_comp.md")


def test_reads_only_the_five_policy_files(kb, index_dir):
    report = run(kb, index_dir)
    assert report.files_read == list(POLICY_FILES)
    sources = {m["source_file"] for m in collection(index_dir).get()["metadatas"]}
    assert sources == set(POLICY_FILES)


def test_gaps_sources_qa_and_stray_markdown_are_in_neither_index(kb, index_dir):
    run(kb, index_dir)
    qa_answers = [p["answer"] for p in json.loads((KB_DIR / "qa.json").read_text())]
    forbidden = [MARKER, "Deliberate KB Gaps", "Kreeda Labs", "HR-STRAY-01", "not a policy file", *qa_answers[:10]]
    for text in (chroma_text(index_dir), bm25_text(index_dir)):
        for needle in forbidden:
            assert needle not in text, f"'{needle[:40]}' leaked into an index"


def test_ingest_twice_gives_the_same_chunk_count_and_ids(kb, index_dir):
    first = run(kb, index_dir)
    ids_1 = sorted(collection(index_dir).get()["ids"])
    second = run(kb, index_dir)
    assert first.chunks == second.chunks == N_ENTRIES
    assert collection(index_dir).count() == N_ENTRIES
    assert sorted(collection(index_dir).get()["ids"]) == ids_1
    assert (second.added, second.removed) == ([], [])
    assert len(json.loads(bm25_text(index_dir))["chunks"]) == N_ENTRIES


def test_deleting_an_entry_removes_its_chunks_from_both_indexes(kb, index_dir):
    run(kb, index_dir)
    path = kb / "hr_policies.md"
    text = path.read_text()
    start = text.index("## Compensatory off for extra hours")
    end = text.index("## Hybrid schedule and core office days")
    path.write_text(text[:start] + text[end:])
    report = run(kb, index_dir)
    assert report.removed == ["HR-LEAVE-08#0"] and report.added == []
    assert collection(index_dir).count() == N_ENTRIES - 1
    assert "HR-LEAVE-08" not in chroma_text(index_dir) + bm25_text(index_dir)


def test_renaming_an_entry_id_prunes_the_old_chunk_and_adds_the_new(kb, index_dir):
    run(kb, index_dir)
    path = kb / "hr_policies.md"
    path.write_text(path.read_text().replace("id: HR-LEAVE-08", "id: HR-LEAVE-09"))
    report = run(kb, index_dir)
    assert report.removed == ["HR-LEAVE-08#0"] and report.added == ["HR-LEAVE-09#0"]
    assert collection(index_dir).count() == N_ENTRIES


def test_a_missing_policy_file_is_reported_and_its_chunks_pruned(kb, index_dir):
    run(kb, index_dir)
    (kb / "payroll_comp.md").unlink()
    report = run(kb, index_dir)
    assert report.missing_files == ["payroll_comp.md"]
    assert not any(i.startswith("HR-PAY") for i in collection(index_dir).get()["ids"])


def test_no_policy_files_at_all_is_an_error_and_leaves_no_index(tmp_path, index_dir):
    with pytest.raises(IngestError):
        run(tmp_path, index_dir)
    assert not (index_dir / BM25_FILE).exists()


def test_duplicate_ids_across_files_fail_before_the_index_changes(kb, index_dir):
    run(kb, index_dir)
    before = bm25_text(index_dir)
    path = kb / "company.md"
    path.write_text(path.read_text().replace("id: HR-ABOUT-01", "id: HR-LEAVE-01"))
    with pytest.raises(KBValidationError, match="HR-LEAVE-01"):
        run(kb, index_dir)
    assert bm25_text(index_dir) == before and collection(index_dir).count() == N_ENTRIES


def test_embeds_the_header_plus_text_and_stores_the_plain_text(kb, index_dir):
    seen = []

    class Spy(HashingEmbeddings):
        def embed_documents(self, texts):
            seen.extend(texts)
            return super().embed_documents(texts)

    run(kb, index_dir, Spy())
    assert len(seen) == N_ENTRIES
    assert all(" > " in t.split("\n", 1)[0] for t in seen)
    stored = collection(index_dir).get(ids=["IT-PWD-01#0"])
    assert stored["metadatas"][0]["domain"] == "IT"
    assert not stored["documents"][0].startswith("IT Support >")  # header is in the embedding, not the stored text


def test_changed_text_is_re_embedded_on_the_next_ingest(kb, index_dir):
    run(kb, index_dir)
    path = kb / "it_support.md"
    path.write_text(path.read_text().replace("at least **14 characters**", "at least **16 characters**"))
    run(kb, index_dir)
    assert "16 characters" in collection(index_dir).get(ids=["IT-PWD-02#0"])["documents"][0]


def test_cli_builds_an_index_with_mock_embeddings(kb, index_dir):
    assert main(["--mock", "--kb-dir", str(kb), "--index-dir", str(index_dir)]) == 0
    assert collection(index_dir).count() == N_ENTRIES
