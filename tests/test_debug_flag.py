"""T4.11: `--debug` lists the retrieved chunks with scores and the generated answer (FR-A12).
Mock LLM; retrieval runs for real over the shipped KB with the hashing embedder."""
from pathlib import Path

from smartdesk_ai.agent.__main__ import format_debug, main
from smartdesk_ai.agent.chat_service import ChatService, Turn
from smartdesk_ai.agent.graph import make_kb_responder
from tests.agent_helpers import build_test_graph
from tests.llm_fakes import FakeAnswerLLM

NO_FILE = Path("/nonexistent/.env")
QUESTION = "How do I reset my forgotten password?"


def run_cli(argv, responder, lines=(QUESTION, "quit")):
    feed = iter(lines)
    out: list[str] = []
    code = main(argv, input_fn=lambda _: next(feed), output=out.append, environ={}, dotenv_path=NO_FILE,
                responder=responder)
    assert code == 0
    return out


def responder_for(built_index_dir, settings, tmp_path, llm=None):
    graph, _ = build_test_graph(built_index_dir, settings, tmp_path, llm=llm)
    return make_kb_responder(graph)


def test_debug_lists_chunks_with_scores_and_the_generated_answer(built_index_dir, settings, tmp_path):
    out = run_cli(["--debug"], responder_for(built_index_dir, settings, tmp_path))
    debug = [line for line in out if line.startswith("[debug]")]
    chunk_lines = [line for line in debug if "score=" in line]
    assert 1 <= len(chunk_lines) <= settings.retrieval.top_n
    for line in chunk_lines:
        entry_id, score = line.split()[2], float(line.split("score=")[1].split()[0])
        assert "-" in entry_id and 0.0 <= score <= 1.0
    answers = [line for line in debug if line.startswith("[debug] answer:")]
    assert answers and "Here is what the policy says." in answers[0]


def test_chunk_lines_are_ranked_and_show_the_entry_title(built_index_dir, settings, tmp_path):
    out = run_cli(["--debug"], responder_for(built_index_dir, settings, tmp_path))
    ranked = [line for line in out if line.startswith("[debug]") and line.split()[1].rstrip(".").isdigit()]
    assert [int(line.split()[1].rstrip(".")) for line in ranked] == list(range(1, len(ranked) + 1))
    assert all(len(line.split("score=")[1].split(None, 1)) == 2 for line in ranked)  # a title follows


def test_without_the_flag_no_debug_lines_are_printed(built_index_dir, settings, tmp_path):
    out = run_cli([], responder_for(built_index_dir, settings, tmp_path))
    assert not any(line.startswith("[debug]") for line in out)
    assert any("Sources:" in line for line in out)


def test_a_gap_question_shows_the_chunks_and_the_i_dont_know_answer(built_index_dir, settings, tmp_path):
    responder = responder_for(built_index_dir, settings, tmp_path, llm=FakeAnswerLLM("insufficient"))
    out = run_cli(["--debug"], responder, lines=("Is there parking near the office?", "quit"))
    assert any(line.startswith("[debug] answer:") and "don't know" in line for line in out)
    assert any("score=" in line for line in out)


def test_the_default_stub_path_still_works_with_debug():
    out = run_cli(["--debug"], None)
    assert any("responder=" in line for line in out)


def test_service_merges_turn_debug_into_the_reply(settings):
    service = ChatService(settings, responder=lambda sid, text: Turn(text="hi", debug={"k": 1}))
    reply = service.handle("s1", "hello")
    assert reply.text == "hi" and reply.debug["k"] == 1


def test_format_debug_handles_missing_parts():
    assert format_debug({}) == []
    assert format_debug({"chunks": [], "answer": ""}) == ["[debug] no chunks retrieved", "[debug] answer: "]
