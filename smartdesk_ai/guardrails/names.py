"""Name reduction (FR-H1, FR-G11): every person name becomes first name plus last initial.

"John Doe" -> "John D."; a single name ("Priya") stays as it is. Applied to user input only, never to
retrieved KB text. Names come from Presidio's PERSON entity; without Presidio nothing is reduced and the
degraded mode is already recorded by `make_presidio_engine` (FR-F7).
"""
import logging

from pydantic import BaseModel, ConfigDict

from smartdesk_ai.guardrails.pii import PresidioEngine

logger = logging.getLogger(__name__)


class Reduction(BaseModel):
    model_config = ConfigDict(frozen=True)

    text: str
    count: int = 0  # how many names were shortened


def reduce_name(name: str) -> str:
    parts = name.replace(".", " ").split()
    if len(parts) < 2:
        return name
    return f"{parts[0]} {parts[-1][0].upper()}."


class NameReducer:
    def __init__(self, engine: PresidioEngine | None, min_score: float) -> None:
        self._engine = engine
        self._min_score = min_score

    def reduce(self, text: str) -> Reduction:
        if self._engine is None:
            return Reduction(text=text)
        try:
            hits = self._engine.find(text, ["PERSON"], self._min_score)
        except Exception as error:  # a failing engine only skips name reduction (FR-F7)
            logger.warning("name reduction skipped, presidio failed (%s)", type(error).__name__)
            return Reduction(text=text)
        out, cursor, count = [], 0, 0
        for _, start, end in sorted(hits, key=lambda h: h[1]):
            if start < cursor:
                continue
            name = text[start:end]
            short = reduce_name(name)
            if short.endswith(".") and text[end:end + 1] == ".":
                end += 1  # the reduced name brings its own dot: do not double it
                name += "."
            if short != name:
                count += 1
            out += [text[cursor:start], short]
            cursor = end
        out.append(text[cursor:])
        logger.debug("names reduced: %d", count)
        return Reduction(text="".join(out), count=count)
