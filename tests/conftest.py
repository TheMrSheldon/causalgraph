"""Shared pytest fixtures."""
from __future__ import annotations

import pytest

from pipeline.db import Database
from pipeline.protocols import CausalRelation, Post


@pytest.fixture
def tmp_db(tmp_path):
    db = Database(str(tmp_path / "test.db"))
    db.initialize_schema()
    return db


@pytest.fixture
def sample_causal_posts() -> list[Post]:
    return [
        Post(id="abc1", title="Smoking causes lung cancer, new study confirms", score=100, num_comments=50, created_utc=1700000000),
        Post(id="abc2", title="Exercise leads to improved mental health outcomes", score=200, num_comments=80, created_utc=1700000001),
        Post(id="abc3", title="Sugar consumption linked to increased obesity risk", score=150, num_comments=60, created_utc=1700000002),
        Post(id="abc4", title="Air pollution reduces life expectancy by 2 years", score=300, num_comments=120, created_utc=1700000003),
    ]


@pytest.fixture
def sample_non_causal_posts() -> list[Post]:
    return [
        Post(id="xyz1", title="Scientists discover new species of deep-sea fish", score=50, num_comments=20, created_utc=1700000010),
        Post(id="xyz2", title="Mars rover sends back stunning images of red planet", score=80, num_comments=30, created_utc=1700000011),
        Post(id="xyz3", title="What is the largest animal that ever lived?", score=40, num_comments=15, created_utc=1700000012),
    ]


@pytest.fixture
def sample_relations(sample_causal_posts) -> list[CausalRelation]:
    return [
        CausalRelation(
            post_id="abc1",
            cause_text="Smoking",
            effect_text="lung cancer",
            cause_norm="smoking",
            effect_norm="lung cancer",
            cause_canonical="Smoking",
            effect_canonical="lung cancer",
            confidence=1.0,
            extractor="regex_spacy",
        ),
        CausalRelation(
            post_id="abc2",
            cause_text="Exercise",
            effect_text="improved mental health outcomes",
            cause_norm="exercise",
            effect_norm="improved mental health outcomes",
            cause_canonical="Exercise",
            effect_canonical="improved mental health outcomes",
            confidence=1.0,
            extractor="regex_spacy",
        ),
        CausalRelation(
            post_id="abc3",
            cause_text="Sugar consumption",
            effect_text="increased obesity risk",
            cause_norm="sugar consumption",
            effect_norm="increased obesity risk",
            cause_canonical="Sugar consumption",
            effect_canonical="increased obesity risk",
            confidence=1.0,
            extractor="regex_spacy",
        ),
    ]
