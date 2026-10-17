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
