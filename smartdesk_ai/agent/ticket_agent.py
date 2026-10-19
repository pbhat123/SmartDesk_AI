"""Ticket agent (flow B). Escalation handoff first (FR-A16); collection, drafting and the
approval gate arrive in T4.12 onward.

The handoff merges the "I don't know" notice, the ticket offer and the email request into one
message. Giving an email counts as agreeing. If the email is already known the offer is a plain
yes or no. A refusal ends the flow. Nothing here creates a ticket (FR-B2).
"""
import difflib
import re
import time
from collections.abc import Callable, Sequence
from typing import Any, Literal

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langgraph.types import interrupt
from pydantic import BaseModel, ConfigDict, Field, field_validator

from smartdesk_ai.agent.answer import INSUFFICIENT_REPLY
from smartdesk_ai.agent.chitchat import ACKNOWLEDGEMENT_PHRASES
from smartdesk_ai.agent.confirm import (
    CANCELLED,
    GAVE_UP,
    MAX_UNCLEAR,
    REASK,
    ConfirmParser,
    apply_edit,
)
from smartdesk_ai.agent.prompt_loader import load_prompt
from smartdesk_ai.agent.state import (
    AgentState,
    Category,
    ConfirmDecision,
    PendingOffer,
    Priority,
    TicketDraft,
    TicketRef,
)
from smartdesk_ai.config.settings import Settings
from smartdesk_ai.llm.fallback import GRACEFUL_MESSAGE, FallbackChat
from smartdesk_ai.llm.factory import ModelBuilder, openai_builder
from smartdesk_ai.llm.structured import ReplyError, parse_model_reply
from smartdesk_ai.ops.hitl_counters import HitlCounters
from smartdesk_ai.ops.runtime import current_scope
from smartdesk_ai.tickets.schemas import EMAIL_PATTERN
from smartdesk_ai.tools.executor import Outcome

_EDGE_PUNCTUATION = "<>()[]{}\"',.;:!?"
_YES = {"yes", "y", "yeah", "yep", "yup", "sure", "ok", "okay", "please", "yes please", "sure please",
        "please do", "go ahead", "do it"}
_NO = {"no", "n", "nope", "no thanks", "no thank you", "not now", "never mind", "nevermind", "no need"}

OFFER_WITH_EMAIL_REQUEST = (
    "I can open a support ticket for you. If you'd like that, please share your work email address."
)
OFFER_YES_NO = "Would you like me to open a support ticket for this? Please answer yes or no."
ASK_EMAIL = "Thanks. What's your work email address?"
DECLINED = "No problem, I won't open a ticket. Is there anything else I can help with?"
AGREED_KNOWN_EMAIL = "Thanks, let's get the ticket details together."
AGREED_NEW_EMAIL = "Thanks, I have your email. Let's get the ticket details together."
INVALID_EMAIL = "That doesn't look like a valid email address. Could you check it and send it again?"
ASK_EMAIL_FOR_TICKET = "To open a ticket I need your work email address. What is it?"
ASK_ISSUE = "What would you like the ticket to be about?"
READY_REPLY = "Thanks, let's get the ticket details together."

# Words that carry no issue by themselves: "I want to raise a ticket" says nothing about the problem.
_FILLER = frozenset(
    "a an the to for of my me i we you please can could would will want like need ticket tickets raise "
    "create open log file make submit new support request issue help with this that it".split()
)


class OfferOutcome(BaseModel):
    model_config = ConfigDict(frozen=True)

    kind: Literal["agreed", "ask_email", "invalid_email", "declined", "unclear"]
    reply: str = ""
    update: dict[str, Any] = {}  # state changes to apply


def escalation_offer(email_known: bool) -> str:
    """The one message after a KB gap: an acknowledgement (FR-D4), "I don't know", the offer (FR-A16)."""
    offer = OFFER_YES_NO if email_known else OFFER_WITH_EMAIL_REQUEST
    return f"{ACKNOWLEDGEMENT_PHRASES[0]} {INSUFFICIENT_REPLY} {offer}"


def find_email(text: str) -> str | None:
    """The first token that is a valid email address, if any. Never guessed (FR-B13)."""
    for token in text.split():
        candidate = token.strip(_EDGE_PUNCTUATION)
        if EMAIL_PATTERN.fullmatch(candidate):
            return candidate
    return None


def is_refusal(text: str) -> bool:
    """A plain "no", "no thanks", "never mind" and the like."""
    words = _words(text)
    return words in _NO or words.startswith("no ")


def _words(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().strip(_EDGE_PUNCTUATION).lower())


def handle_offer_reply(state: AgentState, text: str) -> OfferOutcome:
    """Read the reply to a ticket offer or email request. Anything else is `unclear`, and the
    router decides (a new topic clears the offer, FR-D2)."""
    if state.awaiting not in ("offer_reply", "email") or state.pending_offer is None:
        return OfferOutcome(kind="unclear")
    email = find_email(text)
    if email is not None:
        return OfferOutcome(kind="agreed", reply=AGREED_NEW_EMAIL, update={"email": email, "awaiting": "none"})
    if state.awaiting == "email" and "@" in text:  # tried to give an email and got it wrong
        return OfferOutcome(kind="invalid_email", reply=INVALID_EMAIL)
    words = _words(text)
    if words in _NO or words.startswith("no "):
        return OfferOutcome(
            kind="declined", reply=DECLINED,
            update={"awaiting": "none", "pending_offer": None, "ticket": None},
        )
    if words in _YES:
        if state.email:
            return OfferOutcome(kind="agreed", reply=AGREED_KNOWN_EMAIL, update={"awaiting": "none"})
        return OfferOutcome(kind="ask_email", reply=ASK_EMAIL, update={"awaiting": "email"})
    return OfferOutcome(kind="unclear")


def _has_substance(text: str) -> bool:
    return any(word not in _FILLER for word in re.findall(r"[a-z0-9']+", text.lower()))


def employee_statements(state: AgentState) -> list[str]:
    """What the employee said about the current ticket, in order: their own messages since the
    ticket's context began, without yes/no and email-only replies and without email addresses.
    A bare "raise a ticket" says nothing about the issue and is left out (FR-B6, FR-B13)."""
    statements = []
    for message in state.messages[state.ticket_context_start:]:
        if not isinstance(message, HumanMessage):
            continue
        tokens = [t for t in str(message.content).split() if find_email(t) is None]
        text = " ".join(tokens).strip()
        if not text or _words(text) in _YES or _words(text) in _NO or not _has_substance(text):
            continue
        statements.append(text)
    return statements


class CollectOutcome(BaseModel):
    model_config = ConfigDict(frozen=True)

    kind: Literal["need_issue", "need_email", "ready"]
    reply: str = ""
    update: dict[str, Any] = {}


def collect_step(state: AgentState) -> CollectOutcome:
    """What the ticket still needs from the employee (FR-B1, FR-B5). The email is asked for only
    when the session does not have it, and is never guessed (FR-B13)."""
    statements = employee_statements(state)
    if not statements:
        return CollectOutcome(kind="need_issue", reply=ASK_ISSUE)
    if not state.email:
        marker = state.pending_offer or PendingOffer(unanswered_text=statements[-1])
        return CollectOutcome(
            kind="need_email", reply=ASK_EMAIL_FOR_TICKET,
            update={"awaiting": "email", "pending_offer": marker},
        )
    return CollectOutcome(kind="ready")


def make_t_collect_node(*, drafting: bool = False) -> Callable[[AgentState], dict[str, Any]]:
    """Collect what the ticket needs. With `drafting`, a ready ticket continues to `t_draft` and
    this node says nothing; otherwise it ends the turn with a short acknowledgement."""

    def t_collect(state: AgentState) -> dict[str, Any]:
        out = collect_step(state)
        steps = state.steps + ["t_collect"]
        if out.kind == "ready" and drafting:
            return {"steps": steps}
        reply = READY_REPLY if out.kind == "ready" else out.reply
        return {**out.update, "steps": steps, "reply": reply, "messages": [AIMessage(reply)]}

    return t_collect


class TicketDeps(BaseModel):
    """What the ticket path needs from outside the graph."""

    model_config = ConfigDict(arbitrary_types_allowed=True, frozen=True)

    drafter: Callable[[Sequence[str]], TicketDraft]
    executor: Any  # ToolExecutor with create_ticket and get_tickets registered; every tool call goes through it
    parse_confirm: ConfirmParser
    duplicate_threshold: float  # config: thresholds.duplicate_title
    hitl: HitlCounters | None = None  # approvals, rejections and edits at the gate (FR-H4)


CONFIRM_QUESTION = "Shall I go ahead and create this?"
_PRIORITY_WORDS = {"low": "Low", "medium": "Medium", "high": "High", "critical": "Critical"}


def confirmation_summary(draft: TicketDraft, email: str) -> str:
    """The summary shown at the approval gate: every field, then an explicit question (FR-B8)."""
    lines = [
        "Here is the ticket I'm ready to create:",
        "",
        f"Title: {draft.title}",
        f"Description: {draft.description}",
        f"Category: {draft.category}",
    ]
    if draft.priority is not None:
        lines.append(f"Priority: {_PRIORITY_WORDS[draft.priority]}")
    lines += [f"Email: {email}", "", CONFIRM_QUESTION]
    return "\n".join(lines)


def make_t_present_node() -> Callable[[AgentState], dict[str, Any]]:
    """Show the summary of the current draft. Reached after drafting, after the duplicate question
    and after every edit. Each approval question has its own allowance of unclear replies."""

    def t_present(state: AgentState) -> dict[str, Any]:
        reply = confirmation_summary(state.ticket, state.email or "")
        return {"steps": state.steps + ["t_present"], "reply": reply, "messages": [AIMessage(reply)],
                "gate_unclear": 0, "duplicate": None}

    return t_present


# --- duplicate request check (FR-F4) -------------------------------------------------------------------

DUPLICATE_ASK = "Do you want me to open another one?"
DUPLICATE_REASK = "Sorry, I didn't catch that. Please reply yes to open another ticket, or no to keep the one you have."


def _normal_title(title: str) -> str:
    return " ".join(re.findall(r"[a-z0-9]+", title.lower()))


def title_similarity(a: str, b: str) -> float:
    """How alike two ticket titles are, 0 to 1, ignoring case and punctuation."""
    return difflib.SequenceMatcher(None, _normal_title(a), _normal_title(b)).ratio()


def find_duplicate(title: str, tickets: Sequence[Any], threshold: float) -> Any | None:
    """The open ticket whose title is closest to `title`, if it is at least `threshold` alike."""
    best, best_score = None, 0.0
    for ticket in tickets:
        if ticket.state != "open":
            continue
        score = title_similarity(title, ticket.title)
        if score >= threshold and score > best_score:
            best, best_score = ticket, score
    return best


def duplicate_question(ref: TicketRef) -> str:
    if ref.url:
        named = f"#{ref.number} \"{ref.title}\" ({ref.status}), {ref.url}"
    else:  # a mock ticket has no link
        named = f"#{ref.number} (mock) \"{ref.title}\" ({ref.status})"
    return f"You already have an open ticket that looks the same: {named}. {DUPLICATE_ASK}"


def make_t_dup_check_node(deps: TicketDeps) -> Callable[[AgentState], dict[str, Any]]:
    def t_dup_check(state: AgentState) -> dict[str, Any]:
        steps = state.steps + ["t_dup_check"]
        result = deps.executor.execute("get_tickets", {"email": state.email}, agent="ticket_agent")
        if result.outcome is not Outcome.SUCCESS:  # the request-ID protection still holds (FR-G6)
            return {"steps": steps + ["dup_check_skipped"], "duplicate": None}
        match = find_duplicate(state.ticket.title, result.data.tickets, deps.duplicate_threshold)
        if match is None:
            return {"steps": steps, "duplicate": None}
        ref = TicketRef(number=match.number, title=match.title, status=match.status.value, url=match.url)
        reply = duplicate_question(ref)
        return {"steps": steps, "duplicate": ref, "reply": reply, "messages": [AIMessage(reply)]}

    return t_dup_check


def make_t_dup_ask_node(deps: TicketDeps) -> Callable[[AgentState], dict[str, Any]]:
    """The second interrupt: yes or no to opening another ticket. Same rules as the gate."""

    def t_dup_ask(state: AgentState) -> dict[str, Any]:
        reply = str(interrupt({"question": duplicate_question(state.duplicate)}))
        decision = deps.parse_confirm(reply, state.ticket)
        if decision.decision not in ("yes", "no"):  # an edit makes no sense here
            decision = ConfirmDecision(decision="unclear")
        unclear = state.gate_unclear + 1 if decision.decision == "unclear" else state.gate_unclear
        return {"steps": state.steps + ["t_dup_ask"], "gate": decision, "gate_unclear": unclear}

    return t_dup_ask


def after_t_dup_ask(state: AgentState) -> str:
    decision = state.gate.decision if state.gate else "unclear"
    if decision == "yes":
        return "t_present"
    if decision == "no":
        return "t_cancel"
    return "t_cancel" if state.gate_unclear >= MAX_UNCLEAR else "t_dup_reask"


def t_dup_reask(state: AgentState) -> dict[str, Any]:
    reply = f"{DUPLICATE_REASK}\n\n{duplicate_question(state.duplicate)}"
    return {"steps": state.steps + ["t_dup_reask"], "reply": reply, "messages": [AIMessage(reply)]}


def after_t_dup_check(state: AgentState) -> str:
    return "t_dup_ask" if state.duplicate is not None else "t_present"


def make_t_draft_node(deps: TicketDeps) -> Callable[[AgentState], dict[str, Any]]:
    def t_draft(state: AgentState) -> dict[str, Any]:
        steps = state.steps + ["t_draft"]
        try:
            draft = deps.drafter(employee_statements(state))
        except DraftUnavailable as outage:  # nothing is created; the employee can try again later
            return {"steps": steps + ["llm_unavailable"], "reply": outage.message,
                    "messages": [AIMessage(outage.message)], "awaiting": "none", "pending_offer": None}
        return {"steps": steps, "ticket": draft, "awaiting": "none", "gate": None, "gate_unclear": 0}

    return t_draft


# --- drafting the ticket (FR-B6, FR-B13) --------------------------------------------------------------

TITLE_MAX = 80
# Facts that must come from the employee: numbers and codes, URLs, emails, IDs (#123, req-abc), names.
_SPECIFIC = re.compile(r"\S*\d\S*|https?://\S+|\S+@\S+|#\w+|\breq-\w+", re.I)
_TOKEN_EDGE = "\"'()[]{}<>,;:.!?"


class DraftUnavailable(Exception):
    """The drafting model is unreachable on every tier (FR-F2). `message` is for the employee."""

    def __init__(self, message: str = GRACEFUL_MESSAGE) -> None:
        super().__init__(message)
        self.message = message


class DraftReply(BaseModel):
    """The exact shape the drafting model must return (FR-H2)."""

    model_config = ConfigDict(extra="forbid")

    title: str = Field(min_length=1, max_length=120)
    description: str = Field(min_length=1)
    category: Category
    priority: Priority | None = None

    @field_validator("category", mode="before")
    @classmethod
    def _category_case(cls, value: Any) -> Any:
        return value.strip().upper() if isinstance(value, str) else value

    @field_validator("priority", mode="before")
    @classmethod
    def _priority_case(cls, value: Any) -> Any:
        return value.strip().lower() if isinstance(value, str) else value

    @field_validator("title")
    @classmethod
    def _one_line(cls, value: str) -> str:
        if "\n" in value or not value.strip():
            raise ValueError("title must be one non-empty line")
        return value.strip()


def _strip_emails(text: str) -> str:
    return " ".join(t for t in text.split() if find_email(t) is None).strip()


def is_grounded(text: str, said: Sequence[str]) -> bool:
    """True when `text` invents no fact (FR-B13). Numbers and codes, URLs, emails, ticket or request
    IDs, and capitalised names (a capital that does not start a sentence) must appear in what the
    employee said. Ordinary paraphrase is allowed: a summary may use its own words."""
    said_text = " ".join(said).lower()
    for token in _SPECIFIC.findall(text):
        if token.strip(_TOKEN_EDGE).lower() not in said_text:
            return False
    said_words = set(re.findall(r"[a-z0-9]+", said_text))
    sentence_start = True
    for match in re.finditer(r"\S+", text):
        raw = match.group()
        word = re.sub(r"['’]s$", "", raw.strip(_TOKEN_EDGE))
        starts, sentence_start = sentence_start, bool(re.search(r"[.!?:]$", raw))
        if match.start() > 0 and text[match.start() - 1] == "\n":
            starts = True
        if starts or len(word) < 2 or not word[0].isupper() or any(c.isdigit() for c in word):
            continue
        if word.lower() not in said_words:
            return False
    return True


def _short_title(text: str) -> str:
    line = " ".join(text.split())
    if len(line) <= TITLE_MAX:
        return line
    return line[: TITLE_MAX - 1].rsplit(" ", 1)[0].rstrip(".,;:") + "…"


def fallback_draft(statements: Sequence[str]) -> TicketDraft:
    """A draft made only of the employee's own words, for when the model's draft cannot be used.
    The category is MISC and there is no priority; the employee can change both at the gate."""
    return TicketDraft(
        title=_short_title(statements[0]), description="\n".join(statements), category="MISC", priority=None
    )


def draft_from_statements(llm: Any, prompt: str, statements: Sequence[str]) -> TicketDraft:
    """One TICKET_DRAFT call. A malformed reply, or one that adds details the employee did not give,
    falls back to their own words; an outage raises `DraftUnavailable` (FR-F2)."""
    human = "Employee statements:\n" + "\n".join(f"- {s}" for s in statements)
    try:
        response = llm.invoke([SystemMessage(prompt), HumanMessage(human)])
    except Exception as error:  # the message is not kept: it could echo the statements (FR-G11)
        raise DraftUnavailable(getattr(error, "message", GRACEFUL_MESSAGE)) from error
    plain = fallback_draft(statements)
    try:
        reply = parse_model_reply(DraftReply, getattr(response, "content", response))
    except ReplyError:
        return plain
    # Judge the raw text: the statements carry no email, so any email in a draft is an invented fact.
    return TicketDraft(
        title=reply.title if is_grounded(reply.title, statements) else plain.title,
        description=reply.description if is_grounded(reply.description, statements) else plain.description,
        category=reply.category,
        priority=reply.priority,
    )


def add_details(draft: TicketDraft, details: str) -> TicketDraft:
    """Add what the employee said to the description, in their words (FR-B6)."""
    extra = _strip_emails(details)
    if not extra:
        return draft
    return draft.model_copy(update={"description": f"{draft.description}\n{extra}"})


def make_drafter(
    settings: Settings,
    *,
    builder: ModelBuilder = openai_builder,
    budget_level: int = 0,
    structured: bool | None = None,
    sleep: Callable[[float], None] = time.sleep,
) -> Callable[[Sequence[str]], TicketDraft]:
    """A drafter for the TICKET_DRAFT task and the active ticket_summary prompt, behind the outage
    fallback. The live client uses structured-output mode; the mock path gets plain JSON."""
    use_schema = structured if structured is not None else builder is openai_builder
    llm = FallbackChat(
        "TICKET_DRAFT", settings, builder=builder, budget_level=budget_level,
        schema=DraftReply if use_schema else None, sleep=sleep,
    )
    prompt = load_prompt("ticket_summary", settings).text

    def drafter(statements: Sequence[str]) -> TicketDraft:
        return draft_from_statements(llm, prompt, statements)

    return drafter


# --- the approval gate (FR-B2, FR-G12) ----------------------------------------------------------------

CREATE_FAILED = "Sorry, something went wrong while creating the ticket, so I haven't created it. Please try again."


def _summary(state: AgentState) -> str:
    return confirmation_summary(state.ticket, state.email or "")


def _end_flow(state: AgentState) -> dict[str, Any]:
    """Leave the ticket flow: the draft and the pending offer are done with, the email stays."""
    return {
        "ticket": None, "pending_offer": None, "awaiting": "none", "gate": None, "gate_unclear": 0, "duplicate": None,
        "ticket_context_start": len(state.messages) + 1,  # past the reply this node is about to add
    }


def make_t_confirm_node(deps: TicketDeps) -> Callable[[AgentState], dict[str, Any]]:
    """The only interrupt: it pauses for the employee's reply and parses it. Nodes restart from the
    top on resume, so nothing before `interrupt` may have a side effect (PLAN 12.1)."""

    def t_confirm(state: AgentState) -> dict[str, Any]:
        reply = str(interrupt({"summary": _summary(state)}))
        decision = deps.parse_confirm(reply, state.ticket)
        if decision.decision == "edit" and not apply_edit(state.ticket, decision)[1]:
            decision = ConfirmDecision(decision="unclear")  # an edit that changes nothing is no answer
        unclear = state.gate_unclear + 1 if decision.decision == "unclear" else state.gate_unclear
        if deps.hitl is not None and decision.decision != "unclear":  # after the interrupt, so once per decision
            deps.hitl.record(decision.decision, session=_session_id())
        return {"steps": state.steps + ["t_confirm"], "gate": decision, "gate_unclear": unclear}

    return t_confirm


def after_t_confirm(state: AgentState) -> str:
    decision = state.gate.decision if state.gate else "unclear"
    if decision == "yes":
        return "t_create"
    if decision == "no":
        return "t_cancel"
    if decision == "edit":
        return "t_apply_edit"
    return "t_cancel" if state.gate_unclear >= MAX_UNCLEAR else "t_reask"


def t_apply_edit(state: AgentState) -> dict[str, Any]:
    ticket, _ = apply_edit(state.ticket, state.gate)
    return {"steps": state.steps + ["t_apply_edit"], "ticket": ticket, "gate_unclear": 0}


def t_reask(state: AgentState) -> dict[str, Any]:
    reply = f"{REASK}\n\n{_summary(state)}"
    return {"steps": state.steps + ["t_reask"], "reply": reply, "messages": [AIMessage(reply)]}


def t_cancel(state: AgentState) -> dict[str, Any]:
    gave_up = state.gate is not None and state.gate.decision == "unclear"
    reply = GAVE_UP if gave_up else CANCELLED
    return {**_end_flow(state), "steps": state.steps + ["t_cancel"], "reply": reply,
            "messages": [AIMessage(reply)]}


UPDATES_NOTE = "You can ask me for updates on it anytime."


def created_reply(ticket: Any) -> str:
    """The reply after creation. The ID, link and status are copied from the response, never composed
    (FR-B4, FR-B13)."""
    return f"Your ticket has been created. {ticket.receipt()}\n\n{UPDATES_NOTE}"


def already_created_reply(ticket: Any) -> str:
    """The reply when the create found its ticket already made (a lost response): same facts, said as what happened."""
    return f"Your ticket was already created: {ticket.receipt()}\n\n{UPDATES_NOTE}"


def make_t_create_node(deps: TicketDeps) -> Callable[[AgentState], dict[str, Any]]:
    def t_create(state: AgentState) -> dict[str, Any]:
        if state.gate is None or state.gate.decision != "yes" or state.ticket is None:
            raise RuntimeError("create_ticket needs the employee's confirmation first (FR-B2)")
        draft = state.ticket
        try:
            result = _create(deps, draft, state)
        except BaseException:
            if deps.hitl is not None:
                deps.hitl.record_outcome("failed", session=_session_id())  # approved, and then it broke
            raise
        if deps.hitl is not None:
            deps.hitl.record_outcome("created" if result.outcome is Outcome.SUCCESS else "failed", session=_session_id())
        steps = state.steps + ["t_create"]
        if result.outcome is Outcome.SUCCESS:
            reply = already_created_reply(result.data) if result.recovered else created_reply(result.data)
        else:  # the executor's reply is polite and names no service (FR-F1)
            reply = result.reply or CREATE_FAILED
            steps = steps + ["create_failed"]
        return {**_end_flow(state), "steps": steps, "reply": reply, "messages": [AIMessage(reply)]}

    return t_create


def _session_id() -> str | None:
    """The conversation's random thread ID, from the request scope (never an email)."""
    scope = current_scope()
    return scope[0].session_id if scope else None


def _create(deps: "TicketDeps", draft: Any, state: AgentState) -> Any:
    """The one create_ticket call, through the executor (FR-G5)."""
    return deps.executor.execute(
        "create_ticket",
        {"title": draft.title, "description": draft.description, "category": draft.category,
         "priority": draft.priority, "email": state.email},
        agent="ticket_agent",
        confirmed=True,
    )
