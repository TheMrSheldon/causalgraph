"""Unit tests for regex+spaCy causal extractor."""
import pytest

from pipeline.protocols import Post
from pipeline.step2_extraction.regex_spacy_extractor import RegexSpacyExtractor


@pytest.fixture
def extractor():
    return RegexSpacyExtractor(spacy_model="en_core_web_sm")


def _post(title: str, pid: str = "t1") -> Post:
    return Post(id=pid, title=title, score=10, num_comments=5, created_utc=1700000000)


EXTRACTION_CASES = [
    # (title, expected_cause_fragment, expected_effect_fragment)
    ("Smoking causes lung cancer", "smoking", "lung cancer"),
    ("Exercise leads to better mental health", "exercise", "mental health"),
    ("Obesity linked to higher diabetes risk", "obesity", "diabetes"),
    ("Sleep deprivation results in cognitive decline", "sleep deprivation", "cognitive"),
    ("Air pollution reduces life expectancy", "air pollution", "life expectancy"),
    ("Stress increases risk of heart disease", "stress", "heart"),
    ("Antibiotics trigger gut microbiome disruption", "antibiotics", "gut"),
    ("Scientists find that loneliness accelerates aging", "loneliness", "aging"),
]


@pytest.mark.parametrize("title,cause_frag,effect_frag", EXTRACTION_CASES)
def test_extracts_cause_and_effect(extractor, title, cause_frag, effect_frag):
    post = _post(title)
    relations = extractor.extract(post)
    assert len(relations) >= 1, f"No relations extracted from: {title}"
    rel = relations[0]
    assert cause_frag.lower() in rel.cause_text.lower() or cause_frag.lower() in rel.cause_norm, (
        f"Expected cause '{cause_frag}' in '{rel.cause_text}'"
    )
    assert effect_frag.lower() in rel.effect_text.lower() or effect_frag.lower() in rel.effect_norm, (
        f"Expected effect '{effect_frag}' in '{rel.effect_text}'"
    )


def test_post_id_preserved(extractor):
    post = _post("Stress causes heart disease", "my_id")
    relations = extractor.extract(post)
    assert all(r.post_id == "my_id" for r in relations)


def test_normalized_fields_are_lowercase(extractor):
    post = _post("Exercise leads to improved mental health")
    relations = extractor.extract(post)
    for r in relations:
        assert r.cause_norm == r.cause_norm.lower()
        assert r.effect_norm == r.effect_norm.lower()


def test_extractor_name(extractor):
    assert extractor.name == "regex_spacy"


def test_empty_extraction_returns_list(extractor):
    # A title with no causal signal should return empty list, not raise
    post = _post("The quick brown fox jumps over the lazy dog")
    relations = extractor.extract(post)
    assert isinstance(relations, list)
