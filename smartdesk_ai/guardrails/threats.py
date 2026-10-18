"""A local rule for explicit threats of violence (FR-H1).

Runs in the input guard before Moderation, so a threat is refused even when the API does not flag it
and costs no call. It only matches a first-person intent to hurt a person ("I'm going to hurt someone"),
not rudeness, reports about someone else, or technical wording ("kill a process"). Self-harm is not
handled here.
"""
import re

_INTENT = r"(?:i(?:'m| am)?\s+(?:going\s+to|gonna|about\s+to|want\s+to|wanna|will)|i'll|im\s+(?:going\s+to|gonna))"
_VIOLENCE = r"(?:hurt|harm|kill|murder|stab|shoot|punch|beat(?:\s+up)?|attack|strangle|choke|assault)"
_TARGET = (r"(?:someone|somebody|anyone|everyone|people|them(?:\s+all)?|him|her|you|"
           r"my\s+(?:manager|boss|colleague|coworker|co-worker|teammate|team\s*lead)|the\s+(?:manager|boss))")
_THREAT = re.compile(rf"\b{_INTENT}\s+{_VIOLENCE}\s+{_TARGET}\b", re.I)
_WANT_TO = re.compile(rf"\bi\s+want\s+to\s+{_VIOLENCE}\s+{_TARGET}\b", re.I)


def is_violent_threat(text: str) -> bool:
    return _THREAT.search(text) is not None or _WANT_TO.search(text) is not None
