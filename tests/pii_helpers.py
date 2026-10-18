"""Checks that the fake phone number used across the tests does not appear in a text.

`"98765" not in blob` looks safe but is not: traces, spans and cost logs carry random hex ids (uuid4 run ids, request
ids), and any five-digit run such as 98765 turns up in one of them by chance (about 1 conversation in 150). That made
`test_with_keys_spans_and_runs_carry_masked_data_only` fail now and then. The whole number is matched instead, in the
forms it can take after redaction or reformatting; the number must stand alone as a token (no letter or digit on either side), so it cannot match inside a hex id
of any length; an id is one unbroken token, and a uuid's groups are 8, 4, 4, 4 and 12 characters, never 5 or 10.
"""
import re

PHONE_RE = re.compile(r"(?<![0-9A-Za-z])98765[\s\-.]?43210(?![0-9A-Za-z])")


def has_phone(text: str) -> bool:
    return PHONE_RE.search(text) is not None
