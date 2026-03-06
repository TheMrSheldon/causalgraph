"""Unit tests for regex causality detector."""
import pytest

from pipeline.protocols import Post
from pipeline.step1_detection.regex_detector import RegexDetector


@pytest.fixture
def detector():
    return RegexDetector()


CAUSAL_TITLES = [
    "Smoking causes lung cancer",
    "Exercise leads to better mental health",
    "Sugar consumption linked to obesity",
    "Air pollution reduces life expectancy",
    "Study shows coffee improves memory",
    "Stress increases risk of heart disease",
    "Vitamin D deficiency associated with depression",
    "Antibiotics trigger gut microbiome disruption",
    "Sleep deprivation results in cognitive decline",
    "Meditation prevents anxiety disorders",
    "Scientists find that loneliness accelerates aging",
    "Obesity due to poor dietary habits, researchers say",
    "New drug inhibits tumor growth in mice",
    "Researchers reveal that exercise boosts brain function",
    "Climate change contributes to species extinction",
]

NON_CAUSAL_TITLES = [
    "Scientists discover new species in Amazon rainforest",
    "Mars rover sends images from the red planet",
    "New telescope launched into orbit",
    "Researchers describe fossils found in Antarctica",
    "Water on the moon: new observations",
]


def _make_post(title: str, post_id: str = "test") -> Post:
    return Post(id=post_id, title=title, score=10, num_comments=5, created_utc=1700000000)


def test_detects_explicit_causal_titles(detector):
    posts = [_make_post(t, str(i)) for i, t in enumerate(CAUSAL_TITLES)]
    result = detector.detect(posts)
    result_titles = {p.title for p in result}
    missed = [t for t in CAUSAL_TITLES if t not in result_titles]
    # Allow at most 2 misses (regex won't catch everything)
    assert len(missed) <= 2, f"Too many missed causal titles: {missed}"


def test_does_not_false_positive_on_non_causal(detector):
    posts = [_make_post(t, str(i)) for i, t in enumerate(NON_CAUSAL_TITLES)]
    result = detector.detect(posts)
    # Allow at most 1 false positive
    assert len(result) <= 1, f"Too many false positives: {[p.title for p in result]}"


def test_empty_input(detector):
    assert detector.detect([]) == []


def test_name_property(detector):
    assert detector.name == "regex"


def test_returns_subset_of_input(detector):
    posts = [_make_post(t, str(i)) for i, t in enumerate(CAUSAL_TITLES + NON_CAUSAL_TITLES)]
    result = detector.detect(posts)
    result_ids = {p.id for p in result}
    all_ids = {p.id for p in posts}
    assert result_ids.issubset(all_ids)
