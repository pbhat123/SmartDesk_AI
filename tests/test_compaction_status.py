"""FR-C9 and FR-H5: a support-team comment is untrusted text no model may see. A ticket status reply quotes it, so
the compaction summarizer and the orchestrator (through the summary) see a placeholder instead.
Mock models, no network."""
import json
from types import SimpleNamespace

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from smartdesk_ai.agent.graph import run_turn
from smartdesk_ai.llm.mock import MockRoutingModel
from smartdesk_ai.agent.state import STATUS_PLACEHOLDER, STATUS_REPLY, new_thread_id
from smartdesk_ai.ops.compaction import SUMMARY_PREFIX, Compactor, count_tokens_in
from tests.agent_helpers import assemble
from tests.llm_fakes import FakeAnswerLLM
from tests.seed_helper import ONE_EMAIL
from tests.test_compaction import small

SECRET = "SECRET-TEAM-TEXT-12345"


class Echo:
    """A summarizer that records what it was shown and puts it, whole, into the summary."""

    def __init__(self):
        self.seen = []

    def __call__(self, prior, transcript, tier):
        self.seen.append(transcript)
        return f"{prior} {transcript}"


def test_the_transcript_the_summarizer_gets_has_a_placeholder_for_a_status_reply(settings):
    cfg = small(settings, main_window=300, light_window=4000, keep=1)
    echo = Echo()
    messages = [HumanMessage("status?", id="h0"),
                AIMessage(f"Ticket #4\n> {SECRET}", id="a0", name=STATUS_REPLY),
                HumanMessage("thanks " + "x " * 120, id="h1"), AIMessage("you're welcome " + "y " * 120, id="a1"),
                HumanMessage("another question", id="h2")]
    result = Compactor(cfg, echo).compact(messages)
    assert result is not None
    assert SECRET not in echo.seen[0] and STATUS_PLACEHOLDER in echo.seen[0]
    assert SECRET not in result.summary


def test_a_status_reply_counts_as_its_placeholder_in_the_window(settings):
    reply = AIMessage("Ticket #4\n> " + SECRET * 40, name=STATUS_REPLY)
    plain = AIMessage(STATUS_PLACEHOLDER)
    model = settings.models.main.id
    assert count_tokens_in([reply], model) == count_tokens_in([plain], model)


class Orchestrator:
    """The mock orchestrator, recording every input it is shown."""

    def __init__(self):
        self.inputs, self.inner = [], MockRoutingModel()

    def invoke(self, messages):
        self.inputs.append(" ".join(str(m.content) for m in messages))
        return self.inner.invoke(messages)


def test_no_support_comment_reaches_the_summarizer_or_the_orchestrator_via_the_summary(built_index_dir, settings, tmp_path):
    cfg = small(settings, main_window=200, light_window=4000, keep=1)
    echo, orchestrator = Echo(), Orchestrator()
    stack = assemble(built_index_dir, cfg, tmp_path, llm=FakeAnswerLLM(), routed=True, seeded=True,
                     compactor=Compactor(cfg, echo), route_model=orchestrator)
    [number] = stack.store.numbers_for(ONE_EMAIL)
    stack.client.add_comment(number, SECRET)
    tid = new_thread_id()
    status = run_turn(stack.graph, tid, "Any updates on my monitor ticket?", email=ONE_EMAIL)
    assert SECRET in status.reply, "the employee does see the comment"
    for question in ("How do I reset my forgotten password?", "How do I set up MFA on a new phone?",
                     "How do I connect to the VPN from home?", "How do I install new software?"):
        state = run_turn(stack.graph, tid, question)
    assert echo.seen, "the history reached the trigger, so it was summarized"
    assert all(SECRET not in text for text in echo.seen)
    assert len(orchestrator.inputs) > 3 and all(SECRET not in text for text in orchestrator.inputs)
    assert isinstance(state.messages[0], SystemMessage) and state.messages[0].content.startswith(SUMMARY_PREFIX)
    assert SECRET not in state.messages[0].content
