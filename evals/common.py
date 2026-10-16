"""Shared helpers for eval data sets: JSONL loading and the FR-A18 leakage measure."""
import json
import re
from pathlib import Path

EVALS_DIR = Path(__file__).resolve().parent
_WORD = re.compile(r"[a-z0-9₹%.+@/-]+(?:'[a-z]+)?")


def load_jsonl(path: Path) -> list[dict]:
    """Read one JSON object per non-empty line."""
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def words(text: str) -> list[str]:
    """Lower-case word tokens with surrounding punctuation stripped."""
    return [w.strip(".,;:!?()\"'*-") for w in _WORD.findall(text.lower()) if w.strip(".,;:!?()\"'*-")]


def ngrams(tokens: list[str], n: int) -> set[tuple[str, ...]]:
    return {tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1)}


def longest_shared_span(text: str, corpus: list[str]) -> int:
    """Length in words of the longest contiguous word run that `text` shares with any corpus text."""
    tokens = words(text)
    corpus_tokens = [words(c) for c in corpus]
    best = 0
    for n in range(1, len(tokens) + 1):
        grams = ngrams(tokens, n)
        if any(grams & ngrams(c, n) for c in corpus_tokens):
            best = n
        else:
            break
    return best
