"""KB entry schema and validator (FR-A1).

An entry is one `##` section of a Markdown file with a fenced YAML metadata block
directly under the heading. Domain is read per entry, never per file.
"""
import re
from collections import Counter
from datetime import date
from typing import Literal

import yaml
from pydantic import BaseModel, ValidationError


class KBValidationError(ValueError):
    """Raised when a KB section or the entry set is invalid."""


class Entry(BaseModel):
    id: str
    domain: Literal["IT", "HR"]
    topic: str
    title: str
    owner: str
    last_reviewed: date
    heading: str = ""
    body: str = ""
    source: str = ""
    document: str = ""


_ID_RE = re.compile(r"^[A-Z]+-[A-Z0-9]+-\d{2,}$")
_H1_RE = re.compile(r"^# (.+)$", re.MULTILINE)
_SECTION_SPLIT = re.compile(r"^## ", re.MULTILINE)
_META_RE = re.compile(r"\A\s*```ya?ml\s*\n(.*?)\n```[ \t]*\n?", re.DOTALL)


def parse_entries(text: str, source: str) -> list[Entry]:
    """Parse every `##` section of one Markdown file into validated entries."""
    h1 = _H1_RE.search(text)
    document = h1.group(1).strip() if h1 else ""
    entries: list[Entry] = []
    for chunk in _SECTION_SPLIT.split(text)[1:]:
        heading, _, rest = chunk.partition("\n")
        heading = heading.strip()
        where = f"{source} > {heading}"
        match = _META_RE.match(rest)
        if not match:
            raise KBValidationError(f"{where}: missing YAML metadata block under heading")
        try:
            meta = yaml.safe_load(match.group(1))
        except yaml.YAMLError as exc:
            raise KBValidationError(f"{where}: metadata is not valid YAML ({exc})") from exc
        if not isinstance(meta, dict):
            raise KBValidationError(f"{where}: metadata block must be a mapping")
        body = rest[match.end():].strip()
        try:
            entry = Entry(**meta, heading=heading, body=body, source=source, document=document)
        except ValidationError as exc:
            fields = ", ".join(str(e["loc"][0]) for e in exc.errors() if e["loc"])
            raise KBValidationError(f"{where}: invalid or missing field(s): {fields}") from exc
        if not _ID_RE.match(entry.id):
            raise KBValidationError(f"{where}: id '{entry.id}' does not match e.g. HR-LEAVE-03")
        entries.append(entry)
    return entries


def validate_entries(entries: list[Entry]) -> list[Entry]:
    """Check set-level rules (unique ids across all files)."""
    dupes = sorted(i for i, n in Counter(e.id for e in entries).items() if n > 1)
    if dupes:
        raise KBValidationError(f"duplicate entry id(s): {', '.join(dupes)}")
    return entries
