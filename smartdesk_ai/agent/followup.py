"""Follow-up questions (FR-D6, FR-D2): "what about sick leave?" is answered with the earlier topic.

`last_topic` is the employee's last standalone KB question. A message that opens with a follow-up
phrase is searched and answered as "<message> (following up on: <topic>)", so the retrieval and
the answer model both see the topic. The topic changes only when a new standalone question comes.
Rule-based on purpose: no extra model call, and a fresh question is never rewritten.
"""
import re

TOPIC_MAX_CHARS = 200

_FOLLOW_UP = re.compile(r"^\W*(what|how) about\b|^\W*(and|also|plus)\b|^\W*what if\b|^\W*same (for|question)\b", re.I)


def is_follow_up(text: str) -> bool:
    return _FOLLOW_UP.match(text) is not None


def contextual_question(text: str, last_topic: str | None) -> str:
    """The question to search and answer: the text itself, or a follow-up joined to its topic."""
    if last_topic and is_follow_up(text):
        return f"{text.strip()} (following up on: {last_topic})"
    return text


def topic_after(text: str, last_topic: str | None) -> str | None:
    """The topic once `text` has been answered: kept for a follow-up, replaced by a new question."""
    if last_topic and is_follow_up(text):
        return last_topic
    return " ".join(text.split())[:TOPIC_MAX_CHARS] or last_topic
