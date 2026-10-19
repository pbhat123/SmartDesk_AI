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


def test_orchestrator_v5_is_v4_with_the_split_only_for_different_entries_rule(settings):
    """New versions are new files: every v4 line stays except the one bullet v5 rewrites. The prompt's "at most four"
    matches config `orchestrator.max_parts`; v4 is itself v3 plus the parts section."""
    v3 = (PROMPTS_DIR / "orchestrator_v3.md").read_text().splitlines()
    v4 = (PROMPTS_DIR / "orchestrator_v4.md").read_text().splitlines()
    v5 = load_prompt("orchestrator", settings)
    assert v5.version == 5 and settings.orchestrator.max_parts == 4
    assert all(line in v4 for line in v3 if not line.startswith('{"intent"'))
    changed = [line for line in v4 if line not in v5.text.splitlines()]
    assert len(changed) == 1 and changed[0].startswith('- "parts" lists each separate question')
    assert "only when the questions need different knowledge base entries or belong to different domains" in v5.text
    assert "Clauses about one topic stay one part" in v5.text and "When unsure, keep one part" in v5.text
    assert "At most four parts" in v5.text and "Conversation so far" in v5.text and "Never put an email address" in v5.text
