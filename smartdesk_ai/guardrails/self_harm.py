"""A deterministic check for first-person self-harm statements (FR-H1).

Runs first in the input guard, before Moderation, so it needs no API call and cannot be missed by an outage.
It matches only the speaker's own intent or feelings ("I want to kill myself", "I'm suicidal"), not technical
wording ("kill this process"), reports about someone else or policy questions. An idiom with no intent word
("I could kill myself for forgetting my password") is not matched. The reply is fixed: it points to the
manager, the People Team and local emergency services, and offers no ticket.
"""
import re

SELF_HARM_REPLY = (
    "I'm really sorry you're going through this, and I'm glad you said something. You don't have to handle it "
    "alone. Please talk to someone today: your manager or the People Team, or someone you trust. If you might act "
    "on these thoughts or feel in danger, please call your local emergency services right away. I'm only a help "
    "desk assistant, so I can't give you the support you deserve, but the people above can."
)
WITHHELD = "[message withheld]"  # what the trace log keeps in place of the statement

_ME = r"(?:myself|my\s+(?:own\s+)?life)"
_INTENT = r"(?:i(?:'m|\s+am|'ve\s+been|\s+have\s+been)?\s+(?:going\s+to|gonna|planning\s+to|about\s+to|want\s+to|wanna|thinking\s+(?:about|of)|trying\s+to)|i'll|i\s+will)"
_ACT = r"(?:kill|hurt|harm|cut|end|take)"
_PATTERNS = [
    re.compile(rf"\b{_INTENT}\s+{_ACT}(?:ing)?\s+{_ME}", re.I),
    re.compile(rf"\bi(?:'ve\s+been|\s+have\s+been|'m|\s+am)?\s+thinking\s+(?:about|of)\s+{_ACT}ing\s+{_ME}", re.I),
    re.compile(rf"\bi(?:'m|\s+am)\s+(?:thinking\s+about\s+)?suicid(?:al|e)\b", re.I),
    re.compile(r"\bi(?:'m|\s+am|'ve\s+been)?\s+thinking\s+(?:about|of)\s+suicide\b", re.I),
    re.compile(r"\bi\s+(?:just\s+)?(?:want|wanna)\s+to\s+die\b(?!\s*\.{0,3}\s*of\b)", re.I),
    re.compile(r"\bi\s+don'?t\s+want\s+to\s+(?:live|be\s+alive|be\s+here)\s*(?:any\s*more|anymore)?", re.I),
    re.compile(r"\bi\s+feel\s+like\s+(?:ending\s+it\s+all|killing\s+myself|dying)\b", re.I),
    re.compile(r"\bi(?:\s+(?:feel|felt)|\s+am\s+feeling|'m\s+feeling|(?:'ve|\s+have)\s+been\s+feeling)\s+suicidal\b", re.I),
    re.compile(r"\bi\s+wish\s+i\s+(?:were|was)\s+dead\b", re.I),
    re.compile(r"\b(?:i\s+want\s+to|i'm\s+going\s+to)\s+end\s+it\s+all\b", re.I),
]


def is_self_harm(text: str) -> bool:
    return any(p.search(text) for p in _PATTERNS)
