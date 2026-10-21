"""Keeps a calm reply from repeating an insult (FR-H1).

When Moderation flags a message only for plain harassment, the message is answered rather than refused. The models
are not asked to echo it, but a reply that copies a run of the employee's own words back is replaced by a fixed
calm reply, so the insult is never repeated.
"""
import re

ECHO_WORDS = 4  # a reply sharing this many words in a row with the message repeats it
CALM_REPLY = ("I'm sorry this has been frustrating. I'm here to help with IT or HR questions or to look at one of "
              "your tickets whenever you're ready.")


def _words(text: str) -> list[str]:
    return re.findall(r"[a-z0-9']+", text.lower())


def repeats_message(reply: str, message: str, run: int = ECHO_WORDS) -> bool:
    """True when `reply` contains `run` or more consecutive words of `message`."""
    said = _words(message)
    shown = " " + " ".join(_words(reply)) + " "
    return any(" " + " ".join(said[i:i + run]) + " " in shown for i in range(len(said) - run + 1))
