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
