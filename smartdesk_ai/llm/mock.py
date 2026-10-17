"""Offline stand-ins for the chat models, used when `modes.llm` is `mock` (FR-B10 style).

They are deterministic and read only what the prompt gives them, so the CLI and the
Streamlit app run end to end with no keys and no network. Each task adds its own mock here.
"""
import json
import re
from types import SimpleNamespace

_ENTRY = re.compile(r"^\[([A-Za-z0-9-]+)\] (.*)\n(.*)$", re.M)


class MockAnswerModel:
    """Answers with the first sentence of the top context entry and cites that entry."""

    def invoke(self, messages):
        human = str(messages[-1].content)
        match = _ENTRY.search(human)
        if match is None:
            reply = {"answer": "", "insufficient": True, "citations": []}
        else:
            entry_id, _title, text = match.groups()
            first_sentence = re.split(r"(?<=[.!?])\s", text.strip(), maxsplit=1)[0]
            reply = {"answer": first_sentence, "insufficient": False, "citations": [entry_id]}
        return SimpleNamespace(content=json.dumps(reply))


_TICKET_REQUEST = re.compile(r"\b(raise|create|open|log|file|submit)\b.*\b(ticket|request|issue)\b|\bticket for\b", re.I)
_STATUS = re.compile(
    r"\b(status|updates?)\b.*\b(ticket|issue|request)\b|\bdid anyone look\b|\bmy tickets?\b", re.I
)
_SMALLTALK = re.compile(r"^(hi|hello|hey|thanks|thank you|bye|goodbye|good morning|how are you)\b", re.I)
_OUT_OF_SCOPE = re.compile(r"\b(weather|recipe|cricket|world cup|poem|joke|python function)\b", re.I)
_AMBIGUOUS = re.compile(r"^(it'?s not working|i need (some )?help|i have a question|can you sort that out)", re.I)
_ORDINAL = re.compile(r"\b(first|second|third|fourth|last|one|\d+)\b", re.I)
_IT = re.compile(
    r"\b(password|vpn|laptop|monitor|wi-?fi|mfa|software|install|keyboard|printer|slack|screen|outlook|"
    r"computer|machine|login|account locked|tableau)\b", re.I)
_HR = re.compile(
    r"\b(leave|sick|payroll|salary|payslip|pay|paid|expenses?|reimburse\w*|benefits?|stock|probation|"
    r"wfh|work from home|appraisal|review|resign\w*|sabbatical|holiday|office on|travel)\b", re.I)
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
        if _IT.search(text):
            return "kb_query", "IT"
        if _HR.search(text):
            return "kb_query", "HR"
        return "kb_query", "MISC"


_DRAFT_IT = re.compile(
    r"\b(password|vpn|laptop|mfa|tableau|screen|outlook|wi-?fi|phishing|dock|headset|software|install\w*|"
    r"monitor|keyboard|printer|mouse|slack|login|account|computer|phone)\b", re.I)
_DRAFT_HR = re.compile(
    r"\b(leave|payslip|payroll|salary|policy|tax|first day|buddy|insurance|probation|colleague|stock|"
    r"performance review|claim|conflict of interest|onboarding|paternity|maternity|appraisal|expenses?|benefits?)\b",
    re.I)
_CRITICAL = re.compile(r"\b(nobody can|no one can|everyone|all of us|security|stolen|phishing|breach)\b", re.I)
_HIGH = re.compile(r"\b(cannot|can't|can not|unable to|locked out|blocked)\b.*\b(work|login|log in|access)\b|locked out",
                   re.I)
_MEDIUM = re.compile(r"\b(slow|keeps|dropping|flicker\w*|intermittent|sometimes)\b", re.I)


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
