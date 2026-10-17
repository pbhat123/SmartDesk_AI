"""Builds the KB graph over the real KB and a spy search tool, for agent tests (mock LLM)."""
from smartdesk_ai.agent.answer import make_answerer
from smartdesk_ai.agent.graph import KBDeps, build_kb_graph
from smartdesk_ai.agent.router import make_router
from smartdesk_ai.agent.ticket_agent import TicketDeps, make_drafter
from smartdesk_ai.llm.mock import MockRoutingModel
from smartdesk_ai.memory.checkpointer import make_checkpointer
from smartdesk_ai.agent.state import STATE_MODELS
from smartdesk_ai.ops.health import HealthRegistry
from smartdesk_ai.rag.embeddings import HashingEmbeddings
from smartdesk_ai.rag.index import open_index
from smartdesk_ai.tools.executor import ToolExecutor
from smartdesk_ai.tools.registry import ToolRegistry
from smartdesk_ai.tools.search_kb import make_search_kb
from tests.llm_fakes import FakeAnswerLLM


class SearchSpy:
    """Wraps the real search_kb function and records each call's query and domain."""

    def __init__(self, fn):
        self.fn, self.calls = fn, []

    def __call__(self, args):
        self.calls.append((args.query, args.domain))
        return self.fn(args)


def build_test_graph(built_index_dir, settings, tmp_path, llm=None, name="chat.db", routed=False, drafter_llm=None):
    index = open_index(built_index_dir, HashingEmbeddings())
    spy = SearchSpy(make_search_kb(index, settings, reranker=None, health=HealthRegistry()))
    registry = ToolRegistry()
    registry.register("search_kb", spy)
    executor = ToolExecutor(registry, settings)
    answer = make_answerer(settings, builder=lambda model_id, cfg: llm or FakeAnswerLLM())
    saver = make_checkpointer(tmp_path / name, STATE_MODELS)
    route = make_router(settings, builder=lambda model_id, cfg: MockRoutingModel()) if routed else None
    tickets = TicketDeps(drafter=make_drafter(settings, builder=lambda model_id, cfg: drafter_llm)) if drafter_llm else None
    return build_kb_graph(KBDeps(executor=executor, answer=answer), saver, route=route, tickets=tickets), spy
