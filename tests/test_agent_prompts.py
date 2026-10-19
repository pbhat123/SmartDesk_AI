"""T7.1: each agent has its own versioned prompt, and its calls are labelled with its own name (FR-I1, FR-H14, FR-H9).

Mock models, no network."""
import json
from types import SimpleNamespace

from langchain_core.messages import HumanMessage

from smartdesk_ai.agent.agents import AGENT_PROMPTS, AGENTS
from smartdesk_ai.agent.answer import make_answerer
from smartdesk_ai.agent.prompt_loader import PROMPTS_DIR, load_prompt
from smartdesk_ai.agent.synthesizer import make_synthesizer
from smartdesk_ai.agent.workers import WORKER_AGENTS, make_worker_answerers
from smartdesk_ai.ops.runtime import RequestContext, Services, request_scope
from smartdesk_ai.ops.trace_log import TraceLog, make_masker
from smartdesk_ai.guardrails.pii import PiiRedactor
from smartdesk_ai.tools.search_kb import KBChunk


def test_the_five_agents_of_fr_i1_each_name_a_prompt_that_loads(settings):
    assert set(AGENTS) == {"orchestrator", "it_worker", "hr_worker", "synthesizer", "ticket_agent"}
    names = {agent: AGENT_PROMPTS[agent] for agent in AGENTS}
    assert len(set(names.values())) == 5, "every agent has its own prompt file"
    for agent, name in names.items():
        loaded = load_prompt(name, settings)
        assert loaded.name == name and loaded.version == settings.prompts[name], agent


def test_worker_prompts_carry_the_answer_v4_body_verbatim(settings):
    """The live KB and gap numbers were measured on answer v4; the workers must send the same words."""
    answer = (PROMPTS_DIR / "answer_v4.md").read_text()
    assert load_prompt("answer", settings).version == 4
    for name in ("worker_it", "worker_hr"):
        assert load_prompt(name, settings).text == answer


def test_synthesizer_prompt_only_merges_what_the_workers_found(settings):
    text = load_prompt("synthesizer", settings).text
    assert '"answer"' in text and "Never add" in text and "Sources" in text


class RecordingModel:
    def __init__(self):
        self.calls = []

    def invoke(self, messages):
        self.calls.append(messages)
        return SimpleNamespace(content=json.dumps({"answer": "Reset it at the portal.", "insufficient": False,
                                                    "citations": ["IT-PWD-01"]}),
                               usage_metadata={"input_tokens": 10, "output_tokens": 5}, response_metadata={})


CHUNK = KBChunk(entry_id="IT-PWD-01", title="Password reset", header="IT > Password", text="Use the portal.", domain="IT", score=0.9)


def run_worker(settings, tmp_path, agent):
    model = RecordingModel()
    trace = TraceLog(tmp_path / "t.jsonl", make_masker(PiiRedactor(None, 0.5)), prompt_versions=settings.prompts)
    answerers = make_worker_answerers(settings, builder=lambda m, c: model)
    with request_scope(RequestContext(session_id="s", request_id="r"), Services(trace=trace)):
        answerers[agent]("How do I reset my password?", [CHUNK])
    [record] = [r for r in trace.events if r["kind"] == "llm_call"]
    return model, record


def test_each_worker_sends_its_own_prompt_and_logs_its_own_agent_and_version(settings, tmp_path):
    assert set(WORKER_AGENTS) == {"it_worker", "hr_worker", "misc_search"}
    for agent, prompt in (("it_worker", "worker_it"), ("hr_worker", "worker_hr")):
        model, record = run_worker(settings, tmp_path, agent)
        assert model.calls[0][0].content == load_prompt(prompt, settings).text
        assert record["agent"] == agent and record["prompt_version"] == f"{prompt}_v{settings.prompts[prompt]}"
        assert record["task"] == "WORKER_ANSWER" and record["tier"] == "main"


def test_the_orchestrator_answer_step_for_misc_and_retries_uses_answer_v4(settings, tmp_path):
    model, record = run_worker(settings, tmp_path, "misc_search")
    assert model.calls[0][0].content == load_prompt("answer", settings).text
    assert record["agent"] == "orchestrator" and record["prompt_version"] == "answer_v4"


def test_a_worker_input_holds_only_the_sub_question_and_its_chunks(settings, tmp_path):
    model, _ = run_worker(settings, tmp_path, "it_worker")
    human = model.calls[0][1]
    assert isinstance(human, HumanMessage) and len(model.calls[0]) == 2
    assert human.content.startswith("Question: How do I reset my password?\n\nContext:\n[IT-PWD-01]")


def test_make_answerer_defaults_are_unchanged(settings):
    """Callers that do not name an agent still get the task-named agent and the answer prompt."""
    model = RecordingModel()
    make_answerer(settings, builder=lambda m, c: model)("q", [CHUNK])
    assert model.calls[0][0].content == load_prompt("answer", settings).text


def test_the_synthesizer_loads_its_prompt_and_is_labelled(settings, tmp_path):
    model = RecordingModel()
    trace = TraceLog(tmp_path / "s.jsonl", make_masker(PiiRedactor(None, 0.5)), prompt_versions=settings.prompts)
    synth = make_synthesizer(settings, builder=lambda m, c: model)
    with request_scope(RequestContext(session_id="s", request_id="r"), Services(trace=trace)):
        synth.llm.invoke([HumanMessage("x")])
    [record] = [r for r in trace.events if r["kind"] == "llm_call"]
    assert record["agent"] == "synthesizer" and record["tier"] == "main"
    assert record["prompt_version"] == f"synthesizer_v{settings.prompts['synthesizer']}"
    assert synth.prompt == load_prompt("synthesizer", settings).text


def test_orchestrator_v6_is_v5_with_the_split_for_different_entries_rule(settings):
    """New versions are new files: every v5 line stays except the one bullet v6 rewrites (v5 is v4 with one bullet rewritten, v4
    is v3 plus the parts section). The prompt's "at most four" matches config `orchestrator.max_parts`."""
    v3 = (PROMPTS_DIR / "orchestrator_v3.md").read_text().splitlines()
    v4 = (PROMPTS_DIR / "orchestrator_v4.md").read_text().splitlines()
    v5 = (PROMPTS_DIR / "orchestrator_v5.md").read_text().splitlines()
    v6 = type("P", (), {"text": (PROMPTS_DIR / "orchestrator_v6.md").read_text()})()
    assert settings.orchestrator.max_parts == 4
    assert all(line in v4 for line in v3 if not line.startswith('{"intent"'))
    assert len([line for line in v4 if line not in v5]) == 1
    changed = [line for line in v5 if line not in v6.text.splitlines()]
    assert len(changed) == 1 and changed[0].startswith('- "parts" lists each separate question')
    assert "needs its own knowledge base entry, even when both are in the same domain" in v6.text
    assert "When unsure, keep one part" not in v6.text and "Clauses about one topic stay one part" not in v6.text
    assert "At most four parts" in v6.text and "Conversation so far" in v6.text and "Never put an email address" in v6.text


def test_orchestrator_v6_has_one_example_of_each_decision(settings):
    text = (PROMPTS_DIR / "orchestrator_v6.md").read_text()
    assert '"What is the probation period, and does the company offer a sabbatical?" is two parts' in text
    assert '"How many sick days do I get, and do I need a certificate?" is one part' in text
    assert text.count("is two parts") == 1 and text.count("is one part") == 1


def test_orchestrator_v7_is_v6_plus_the_work_policy_clarification(settings):
    """v7 changes one line of v6, the out_of_scope definition: rules for using tools, code, data or AI services at work are
    kb_query (kb-011), while asking the assistant for general coding help stays out_of_scope."""
    v6 = (PROMPTS_DIR / "orchestrator_v6.md").read_text().splitlines()
    v7 = type("P", (), {"text": (PROMPTS_DIR / "orchestrator_v7.md").read_text()})()
    assert [line for line in v6 if line not in v7.text.splitlines()] == [v6[10]] and v6[10].startswith("- out_of_scope:")
    [new] = [line for line in v7.text.splitlines() if line not in v6]
    assert new.startswith(v6[10]) and len(v7.text.splitlines()) == len(v6)
    assert "Is it fine to paste our source code into a public AI chatbot" in new and "kb_query" in new
    assert "Can I install this software?" in new and "share a file with someone outside the company" in new
    assert '"write me a Python sort function"' in new and "non-work topics" in new
    assert "general coding help" in new


def test_orchestrator_v8_is_v7_plus_the_fragment_rule(settings):
    """v8 changes one line of v7, the parts bullet: a part is a complete standalone question, and a clause that only asks about the
    same thing's timing, count, place or duration stays with it. Its examples have the shape of the two over-split golden questions
    in other words, so the golden set does not leak into the prompt."""
    v7 = (PROMPTS_DIR / "orchestrator_v7.md").read_text().splitlines()
    v8 = load_prompt("orchestrator", settings)
    assert v8.version == 8
    changed = [line for line in v7 if line not in v8.text.splitlines()]
    assert len(changed) == 1 and changed[0].startswith('- "parts" lists each separate question') and len(v8.text.splitlines()) == len(v7)
    [new] = [line for line in v8.text.splitlines() if line not in v7]
    assert 'never list a fragment such as "for how long?" as a part' in new and "timing, count, place or duration" in new
    assert "Who is assigned to look after a new joiner, and for how long?" in new and "when is the bonus decided?" in new
    assert "needs its own knowledge base entry" in new and "is two parts" in new and "At most four parts" in new
    assert "Who is the person assigned to help new starters" not in v8.text and "possible performance scores" not in v8.text
    out_of_scope = [line for line in v8.text.splitlines() if line.startswith("- out_of_scope:")][0]
    assert "write me a Python sort function" in out_of_scope  # v7's clarification is kept
