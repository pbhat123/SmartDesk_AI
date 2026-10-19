"""The workers (FR-I1, FR-I2): who answers which kind of question, and the answer step each one runs.

The IT worker handles IT Support and IT equipment; the HR worker handles HR policy, onboarding (except equipment)
and payroll. A MISC question (no clear IT or HR fit, such as facilities) goes to no worker: the orchestrator runs the
same answering step with no domain filter (`misc_search`). Each worker has its own versioned prompt; MISC and the
whole-KB retry use the answer prompt (FR-H14). Workers do not pick a tool: the search is a fixed `search_kb` call
with their domain, made through the executor (FR-G2).
"""
import time
from collections.abc import Callable, Sequence
from typing import Any

from smartdesk_ai.agent.agents import HR_WORKER, IT_WORKER, ORCHESTRATOR
from smartdesk_ai.agent.answer import AnswerResult, Judge, make_answerer
from smartdesk_ai.config.settings import Settings
from smartdesk_ai.llm.factory import ModelBuilder, openai_builder
from smartdesk_ai.tools.search_kb import KBChunk

MISC_SEARCH = "misc_search"  # the graph node that answers a MISC part, run by the orchestrator
WORKER_AGENTS = (IT_WORKER, HR_WORKER, MISC_SEARCH)

# Graph node per question domain, and the search filter it uses (None is the whole KB).
NODE_FOR_DOMAIN = {"IT": IT_WORKER, "HR": HR_WORKER, "MISC": MISC_SEARCH}
FILTER_FOR_NODE: dict[str, str | None] = {IT_WORKER: "IT", HR_WORKER: "HR", MISC_SEARCH: None}

Answerer = Callable[[str, Sequence[KBChunk]], AnswerResult]


def make_worker_answerers(
    settings: Settings,
    *,
    builder: ModelBuilder = openai_builder,
    budget_level: int = 0,
    structured: bool | None = None,
    sleep: Callable[[float], None] = time.sleep,
    judge: Judge | None = None,
) -> dict[str, Answerer]:
    """One answering step per worker node. All run the WORKER_ANSWER task; they differ in prompt and in the agent name
    their calls are recorded under. The grounding `judge` runs inside each, once per part (FR-H2)."""
    common: dict[str, Any] = {"builder": builder, "budget_level": budget_level, "structured": structured,
                              "sleep": sleep, "judge": judge}
    return {
        IT_WORKER: make_answerer(settings, agent=IT_WORKER, prompt_name="worker_it", **common),
        HR_WORKER: make_answerer(settings, agent=HR_WORKER, prompt_name="worker_hr", **common),
        MISC_SEARCH: make_answerer(settings, agent=ORCHESTRATOR, prompt_name="answer", **common),
    }
