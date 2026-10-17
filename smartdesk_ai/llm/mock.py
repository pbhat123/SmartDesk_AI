"""Offline stand-ins for the chat models, used when `modes.llm` is `mock` (FR-B10 style).

They are deterministic and read only what the prompt gives them, so the CLI and the
Streamlit app run end to end with no keys and no network. Each task adds its own mock here.
"""
import json
import re
from types import SimpleNamespace

_ENTRY = re.compile(r"^\[([A-Za-z0-9-]+)\] ([^\n]*)\n(.*?)(?=\n\n\[|\Z)", re.M | re.S)


_STOPWORDS = frozenset(
    "what when where which that this with have from your about does there their they will would could should "
    "into than then them these those been being were more most some such only also just like want need can "
    "how why who whom whose".split()
)
MIN_OVERLAP = 3  # content words the question must share with the top entry for the mock to answer


def _content_words(text: str) -> set[str]:
    return {w for w in re.findall(r"[a-z]{4,}", text.lower()) if w not in _STOPWORDS}


class MockAnswerModel:
    """Answers with the first sentence of the top context entry and cites that entry, but only when the
    question shares enough content words with it; otherwise it reports insufficient context. A
    stand-in for a real model's judgement, so the mock-mode demo offers a ticket for gaps."""

    def invoke(self, messages):
        human = str(messages[-1].content)
        entry = _ENTRY.search(human)
        question = re.search(r"^Question: (.*)$", human, re.M)
        reply = {"answer": "", "insufficient": True, "citations": []}
        if entry and question:
            entry_id, title, text = entry.groups()
            if len(_content_words(question.group(1)) & _content_words(f"{title} {text}")) >= MIN_OVERLAP:
                first_sentence = re.split(r"(?<=[.!?])\s", text.strip(), maxsplit=1)[0]
                reply = {"answer": first_sentence, "insufficient": False, "citations": [entry_id]}
        return SimpleNamespace(content=json.dumps(reply))


_TICKET_REQUEST = re.compile(r"\b(raise|create|open|log|file|submit)\b.*\b(ticket|request|issue)\b|\bticket for\b", re.I)
_STATUS = re.compile(
    r"\b(status|updates?)\b.*\b(ticket|issue|request)\b|\bdid anyone look\b|\bmy tickets?\b", re.I
)
_SMALLTALK = re.compile(r"^(hi|hello|hey|thanks|thank you|bye|goodbye|good morning|how are you)\b", re.I)
_OUT_OF_SCOPE = re.compile(r"\b(weather|recipe|cricket|world cup|poem|joke|python function|python script|mutual funds?|prime minister)\b", re.I)
_AMBIGUOUS = re.compile(r"^(it'?s not working|i need (some )?help|i have a question|can you sort that out)", re.I)
_ORDINAL = re.compile(r"\b(first|second|third|fourth|last|one|\d+)\b", re.I)
_YES_WORDS = re.compile(r"^(yes|yeah|yep|sure|ok|okay|please do|go ahead)\b", re.I)


class _NoToolPick:
    def invoke(self, messages):
        return SimpleNamespace(content="", tool_calls=[])


class MockRoutingModel:
    """Keyword-based stand-in for the intent classifier and the tool pick."""

    def bind_tools(self, tools):
        return _NoToolPick()

    def invoke(self, messages):
        human = str(messages[-1].content)
        awaiting = re.search(r"^Awaiting: (\w+)$", human, re.M).group(1)
        text = re.search(r"^Message: (.*)$", human, re.M | re.S).group(1).strip()
        intent, domain = self._classify(text, awaiting)
        return SimpleNamespace(content=json.dumps({"intent": intent, "confidence": 0.9, "domain": domain}))

    @staticmethod
    def _classify(text: str, awaiting: str) -> tuple[str, str | None]:
        if awaiting in ("offer_reply", "email") and ("@" in text or _YES_WORDS.match(text)):
            return "create_ticket", None
        if awaiting == "ticket_pick" and _ORDINAL.search(text):
            return "ticket_status", None
        if _TICKET_REQUEST.search(text):
            return "create_ticket", None
        if _STATUS.search(text):
            return "ticket_status", None
        if _AMBIGUOUS.match(text):
            return "ambiguous", None
        if _SMALLTALK.match(text):
            return "smalltalk", None
        if awaiting == "offer_reply" and re.match(r"^no\b", text, re.I):
            return "smalltalk", None
        if _OUT_OF_SCOPE.search(text):
            return "out_of_scope", None
        return "kb_query", infer_mock_category(text)  # the same rules as the drafter: one domain view (FR-D7)


_DRAFT_IT = re.compile(
    r"\b(password|vpn|laptop|mfa|tableau|screen|outlook|wi-?fi|phishing|dock|headset|software|install\w*|"
    r"monitor|keyboard|printer|mouse|slack|login|account|computer|machine|phone)\b", re.I)
_DRAFT_HR = re.compile(
    r"\b(leave|payslip|payroll|salary|policy|tax|first day|buddy|insurance|probation|colleague|stock|"
    r"performance review|claim|conflict of interest|onboarding|paternity|maternity|appraisal|expenses?|benefits?|"
    r"sick|pay|paid|reimburse\w*|wfh|work from home|review|resign\w*|sabbatical|holiday|travel|office on)\b",
    re.I)
_CRITICAL = re.compile(r"\b(nobody can|no one can|everyone|all of us|security|stolen|phishing|breach)\b", re.I)
_HIGH = re.compile(r"\b(cannot|can't|can not|unable to|locked out|blocked)\b.*\b(work|login|log in|access)\b|locked out",
                   re.I)
_MEDIUM = re.compile(r"\b(slow|keeps|dropping|flicker\w*|intermittent|sometimes|broken|failing|not working)\b", re.I)


def infer_mock_category(text: str) -> str:
    if _DRAFT_IT.search(text):
        return "IT"
    return "HR" if _DRAFT_HR.search(text) else "MISC"


def infer_mock_priority(text: str) -> str | None:
    """Impact words only, as the prompt asks: never a question, and None when there is no sign."""
    for pattern, level in ((_CRITICAL, "critical"), (_HIGH, "high"), (_MEDIUM, "medium")):
        if pattern.search(text):
            return level
    return None


class MockDraftModel:
    """Drafts a ticket from the employee's statements using only their own words."""

    def invoke(self, messages):
        human = str(messages[-1].content)
        statements = [line[2:] for line in human.splitlines() if line.startswith("- ")]
        joined = " ".join(statements)
        title = " ".join(statements[0].split()).rstrip("?.!") if statements else "Support request"
        reply = {
            "title": title[:80],
            "description": "\n".join(statements) or title,
            "category": infer_mock_category(joined),
            "priority": infer_mock_priority(joined),
        }
        return SimpleNamespace(content=json.dumps(reply))


_MOCK_CHITCHAT = {
    "greeting": "Hello! I can answer IT and HR questions, open a support ticket, or check on a ticket. How can I help?",
    "thanks": "Glad I could help! Is there anything else you need?",
    "goodbye": "Goodbye, have a great day! Come back any time.",
    "other": "Happy to chat. I can help with IT and HR questions, tickets and ticket status.",
    "deflect": "Sorry, I can't help with that one. I'm here for IT, HR and workplace questions.",
    "clarify": "Could you tell me a bit more about what you need?",
}


class MockChitchatModel:
    """Stand-in for the CHITCHAT task: a fixed reply per situation, read from the prompt."""

    def invoke(self, messages):
        situation = re.search(r"^Situation: (\w+)$", str(messages[-1].content), re.M).group(1)
        return SimpleNamespace(content=_MOCK_CHITCHAT[situation])


_CONFIRM_YES = re.compile(r"^(yes|yeah|yep|sure|ok(ay)?|go ahead|please do|looks good|that'?s right)\b", re.I)
_CONFIRM_NO = re.compile(r"^(no|nope|cancel|never mind|don'?t|do not)\b", re.I)
_CONFIRM_PRIORITY = re.compile(
    r"\b(?:priority(?: to| is)?|make it|set it to)\s+(low|medium|high|critical)\b|\b(low|medium|high|critical) priority\b", re.I)
_CONFIRM_TITLE = re.compile(r"\b(?:change|set|rename)\b.*\btitle (?:to|as)\s+(.+)$", re.I)
_CONFIRM_CATEGORY = re.compile(r"\b(?:category (?:to |is )?|this is for |it'?s an? )(it|hr|misc)\b", re.I)
_CONFIRM_DETAILS = re.compile(r"^(?:also|additionally|and also|add)[:,]?\s+(.+)$", re.I)


class MockConfirmModel:
    """Keyword stand-in for the CONFIRM_PARSE task: yes, no, a few kinds of edit, else unclear."""

    def invoke(self, messages):
        human = str(messages[-1].content)
        text = human.split("Employee reply:", 1)[1].strip()
        reply: dict[str, str | None] = {"decision": "unclear"}
        edit: dict[str, str] = {}
        if (match := _CONFIRM_PRIORITY.search(text)):
            edit["priority"] = (match.group(1) or match.group(2)).lower()
        if (match := _CONFIRM_TITLE.search(text)):
            edit["title"] = match.group(1).strip()
        if (match := _CONFIRM_CATEGORY.search(text)):
            edit["category"] = match.group(1).upper()
        if (match := _CONFIRM_DETAILS.match(text)):
            edit["details"] = match.group(1).strip()
        if edit:
            reply = {"decision": "edit", **edit}
        elif _CONFIRM_NO.match(text):
            reply = {"decision": "no"}
        elif _CONFIRM_YES.match(text):
            reply = {"decision": "yes"}
        return SimpleNamespace(content=json.dumps(reply))


_MOCK_INJECTION = re.compile(
    r"\b(?:ignore|disregard|forget)\b.*\b(?:instructions|rules|policies|knowledge base)\b|system prompt|developer mode|"
    r"above this message|^\s*\[system\]", re.I | re.S)
_MOCK_INJECTION_BENIGN = re.compile(r"\b(?:laptop|screen|boot|error)\b", re.I)


class MockInjectionModel:
    """Stand-in for the INJECTION_CLASSIFIER task: a text that tells the assistant to drop its rules is an
    injection, unless it only quotes such a phrase while describing a device problem."""

    def invoke(self, messages):
        text = str(messages[-1].content).split("<<<TEXT\n", 1)[-1].rsplit("\nTEXT>>>", 1)[0]
        hit = bool(_MOCK_INJECTION.search(text)) and not _MOCK_INJECTION_BENIGN.search(text)
        return SimpleNamespace(content=json.dumps({"injection": hit, "reason": "mock rule" if hit else None}))


class MockGroundingModel:
    """Stand-in for the GROUNDING_JUDGE task: grounded when every content word and every number of the
    draft answer also appears in the context."""

    def invoke(self, messages):
        human = str(messages[-1].content)
        answer = human.split("Draft answer:\n", 1)[1].split("\n\nContext:\n", 1)[0]
        context = human.split("\n\nContext:\n", 1)[1]
        tokens = lambda text: _content_words(text) | set(re.findall(r"\d+", text))  # noqa: E731
        missing = sorted(tokens(answer) - tokens(context))
        return SimpleNamespace(content=json.dumps({"grounded": not missing, "unsupported": missing[:5]}))
