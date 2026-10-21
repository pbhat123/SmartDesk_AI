"""FR-H1: a message flagged only for `illicit`, with an illicit score below `guardrails.illicit_block_score`
(0.50), is not refused. The scores are the ones measured on the live Moderation API on 2026-10-21
(kb-006, safe-020, a firewall question and four clearly illicit phrases). Moderation is stubbed; no network."""
import logging
from types import SimpleNamespace

import pytest

from smartdesk_ai.guardrails.input_guard import InputGuard
from smartdesk_ai.guardrails.moderation import ModerationResult, OpenAIModerator
from smartdesk_ai.guardrails.names import NameReducer
from smartdesk_ai.guardrails.pii import PiiRedactor

THRESHOLD = 0.50


class StubModerator:
    def __init__(self, result):
        self.result = result

    def check(self, text):
        return self.result


def decide(result, threshold=THRESHOLD):
    guard = InputGuard(
        pii=PiiRedactor(None, 0.5), names=NameReducer(None, 0.5), moderator=StubModerator(result),
        illicit_block_score=threshold,
    )
    return guard("some message")


def illicit(score, *others):
    return ModerationResult(flagged=True, categories=sorted(["illicit", *others]), illicit_score=score)


@pytest.mark.parametrize("score", [0.2808, 0.2810, 0.3349])  # kb-006, safe-020, firewall-to-update-server question
def test_illicit_alone_below_the_threshold_goes_through(score, caplog):
    with caplog.at_level(logging.INFO):
        decision = decide(illicit(score))
    assert not decision.blocked and not decision.calm
    assert "allowed" in caplog.text and "moderation:illicit" in caplog.text


@pytest.mark.parametrize("score", [0.7491, 0.8482, 0.9463, 0.9549])  # the four clearly illicit phrases
def test_illicit_well_above_the_threshold_is_refused(score):
    decision = decide(illicit(score))
    assert decision.blocked and decision.reason == "moderation:illicit"


def test_illicit_at_exactly_the_threshold_is_refused():
    assert decide(illicit(0.50)).blocked


def test_just_below_the_threshold_goes_through():
    assert not decide(illicit(0.4999)).blocked


@pytest.mark.parametrize("other", ["harassment", "hate", "sexual", "violence", "self-harm", "illicit/violent"])
def test_illicit_with_any_other_category_is_refused_even_at_a_low_score(other):
    decision = decide(illicit(0.30, other))
    assert decision.blocked and decision.reason == f"moderation:{','.join(sorted(['illicit', other]))}"


def test_illicit_violent_alone_is_never_relaxed():
    result = ModerationResult(flagged=True, categories=["illicit/violent"], illicit_score=0.10)
    assert decide(result).blocked


def test_a_missing_score_is_refused():
    assert decide(ModerationResult(flagged=True, categories=["illicit"])).blocked


def test_without_a_configured_threshold_nothing_is_relaxed():
    assert decide(illicit(0.28), threshold=None).blocked


def test_the_live_moderator_carries_the_illicit_score():
    categories = SimpleNamespace(model_dump=lambda: {"illicit": True, "illicit/violent": False, "harassment": False})
    scores = SimpleNamespace(model_dump=lambda: {"illicit": 0.2808, "illicit/violent": 0.0001})
    result = SimpleNamespace(flagged=True, categories=categories, category_scores=scores)
    client = SimpleNamespace(moderations=SimpleNamespace(create=lambda **kwargs: SimpleNamespace(results=[result])))
    checked = OpenAIModerator(client, "omni-moderation-latest").check("text")
    assert checked.categories == ["illicit"] and checked.illicit_score == pytest.approx(0.2808)


def test_the_threshold_comes_from_config(settings):
    assert settings.guardrails.illicit_block_score == THRESHOLD
