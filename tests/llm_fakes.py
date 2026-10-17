"""Mock chat models shared by the agent tests. No network, no keys."""
import json
import re
from types import SimpleNamespace


class FakeAnswerLLM:
    """Answers from the context in the prompt, like a worker model would.

    mode "cite_top": answers and cites the first context entry.
    mode "insufficient": reports insufficient context.
    mode "raw": returns `raw` as is (for malformed-output tests).
    """

    def __init__(self, mode="cite_top", raw=None, citations=None):
        self.mode, self.raw, self.citations, self.calls = mode, raw, citations, []

    def invoke(self, messages):
        self.calls.append(messages)
        if self.mode == "raw":
            return SimpleNamespace(content=self.raw)
        if self.mode == "insufficient":
            return SimpleNamespace(content=json.dumps({"answer": "", "insufficient": True, "citations": []}))
        ids = re.findall(r"^\[([A-Z]+-[A-Z0-9-]+)\]", messages[-1].content, re.M)
        cited = self.citations if self.citations is not None else ids[:1]
        return SimpleNamespace(
            content=json.dumps({"answer": "Here is what the policy says.", "insufficient": False, "citations": cited})
        )
