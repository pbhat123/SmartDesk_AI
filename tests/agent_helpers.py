"""Builds the agent graph over the real KB, a spy search tool and a mock tickets client (mock LLMs)."""
from types import SimpleNamespace

from smartdesk_ai.agent.answer import make_answerer
from smartdesk_ai.agent.confirm import rule_parser
from smartdesk_ai.agent.graph import KBDeps, build_kb_graph
from smartdesk_ai.agent.router import make_router
from smartdesk_ai.agent.state import STATE_MODELS
from smartdesk_ai.agent.ticket_agent import TicketDeps, make_drafter
from smartdesk_ai.llm.mock import MockRoutingModel
from smartdesk_ai.memory.checkpointer import make_checkpointer
from smartdesk_ai.ops.health import HealthRegistry
from smartdesk_ai.rag.embeddings import HashingEmbeddings
from smartdesk_ai.rag.index import open_index
from smartdesk_ai.tickets import MockTicketClient
from smartdesk_ai.tickets.sqlite_map import TicketStore
from smartdesk_ai.tools.executor import ToolExecutor
from smartdesk_ai.tools.registry import ToolRegistry
from smartdesk_ai.tools.search_kb import make_search_kb
from smartdesk_ai.tools.ticket_tools import register_ticket_tools
from tests.llm_fakes import FakeAnswerLLM
from tests.seed_helper import seed


class SearchSpy:
    """Wraps the real search_kb function and records each call's query and domain."""

    def __init__(self, fn):
        self.fn, self.calls = fn, []

    def __call__(self, args):
        self.calls.append((args.query, args.domain))
        return self.fn(args)


def assemble(built_index_dir, settings, tmp_path, *, llm=None, name="chat.db", routed=False,
             drafter_llm=None, parse_confirm=None, client=None, seeded=False, chitchat=None, judge=None, compactor=None, hitl=None):
    """Everything the tests need: the compiled graph plus the spy, the tickets client and the store."""
    index = open_index(built_index_dir, HashingEmbeddings())
    spy = SearchSpy(make_search_kb(index, settings, reranker=None, health=HealthRegistry()))
    client = client or MockTicketClient()
    store = TicketStore(tmp_path / "tickets.db")
    registry = ToolRegistry()
    registry.register("search_kb", spy)
    register_ticket_tools(registry, client, store)
    executor = ToolExecutor(registry, settings, request_store=store, sleep=lambda _: None)
    if seeded:  # the demo tickets of scripts/seed_tickets.py, through the same client and map
        seed(client, store)
    answer = make_answerer(settings, builder=lambda model_id, cfg: llm or FakeAnswerLLM(), judge=judge)
    saver = make_checkpointer(tmp_path / name, STATE_MODELS)
    route = make_router(settings, builder=lambda model_id, cfg: MockRoutingModel()) if routed else None
    tickets = None
    if drafter_llm is not None:
        tickets = TicketDeps(
            drafter=make_drafter(settings, builder=lambda model_id, cfg: drafter_llm),
            executor=executor,
            parse_confirm=parse_confirm or rule_parser,
            duplicate_threshold=settings.thresholds.duplicate_title,
            hitl=hitl,
        )
    graph = build_kb_graph(KBDeps(executor=executor, answer=answer), saver, route=route, tickets=tickets,
                             status_max_listed=settings.status.max_listed, chitchat=chitchat, compactor=compactor)
    return SimpleNamespace(graph=graph, spy=spy, client=client, store=store, executor=executor, saver=saver)


def build_test_graph(built_index_dir, settings, tmp_path, llm=None, name="chat.db", routed=False, drafter_llm=None,
                     parse_confirm=None):
    stack = assemble(built_index_dir, settings, tmp_path, llm=llm, name=name, routed=routed,
                     drafter_llm=drafter_llm, parse_confirm=parse_confirm)
    return stack.graph, stack.spy
