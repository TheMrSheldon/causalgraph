"""Unit tests for Step 3 canonization implementations."""
import dataclasses

import pytest

from pipeline.protocols import CausalRelation, EventCanonizer
from pipeline.step3_canonization.passthrough_canonizer import PassthroughCanonizer


@pytest.fixture
def canonizer():
    return PassthroughCanonizer()


@pytest.fixture
def relations():
    return [
        CausalRelation(
            post_id="p1",
            cause_text="Smoking",
            effect_text="lung cancer",
            cause_norm="smoking",
            effect_norm="lung cancer",
            confidence=1.0,
            extractor="regex_spacy",
        ),
        CausalRelation(
            post_id="p2",
            cause_text="Regular exercise",
            effect_text="improved mental health",
            cause_norm="regular exercise",
            effect_norm="improved mental health",
            confidence=0.9,
            extractor="regex_spacy",
        ),
    ]


# --- Protocol conformance ---

def test_implements_protocol(canonizer):
    assert isinstance(canonizer, EventCanonizer)


def test_name(canonizer):
    assert canonizer.name == "passthrough"


# --- Return type ---

def test_returns_list(canonizer, relations):
    result = canonizer.canonize(relations)
    assert isinstance(result, list)


def test_returns_causal_relations(canonizer, relations):
    result = canonizer.canonize(relations)
    assert all(isinstance(r, CausalRelation) for r in result)


def test_same_length(canonizer, relations):
    result = canonizer.canonize(relations)
    assert len(result) == len(relations)


# --- Empty input ---

def test_empty_input(canonizer):
    result = canonizer.canonize([])
    assert result == []


# --- Canonical field population ---

def test_cause_canonical_set_to_cause_text(canonizer, relations):
    result = canonizer.canonize(relations)
    for orig, can in zip(relations, result):
        assert can.cause_canonical == orig.cause_text


def test_effect_canonical_set_to_effect_text(canonizer, relations):
    result = canonizer.canonize(relations)
    for orig, can in zip(relations, result):
        assert can.effect_canonical == orig.effect_text


# --- Immutability: original relations are not mutated ---

def test_originals_not_mutated(canonizer, relations):
    originals = [dataclasses.replace(r) for r in relations]
    canonizer.canonize(relations)
    for orig, saved in zip(relations, originals):
        assert orig == saved


# --- Other fields preserved ---

def test_post_id_preserved(canonizer, relations):
    result = canonizer.canonize(relations)
    for orig, can in zip(relations, result):
        assert can.post_id == orig.post_id


def test_cause_norm_preserved(canonizer, relations):
    result = canonizer.canonize(relations)
    for orig, can in zip(relations, result):
        assert can.cause_norm == orig.cause_norm


def test_effect_norm_preserved(canonizer, relations):
    result = canonizer.canonize(relations)
    for orig, can in zip(relations, result):
        assert can.effect_norm == orig.effect_norm


def test_confidence_preserved(canonizer, relations):
    result = canonizer.canonize(relations)
    for orig, can in zip(relations, result):
        assert can.confidence == orig.confidence


def test_extractor_preserved(canonizer, relations):
    result = canonizer.canonize(relations)
    for orig, can in zip(relations, result):
        assert can.extractor == orig.extractor


# --- Single relation ---

def test_single_relation(canonizer):
    rel = CausalRelation(
        post_id="x",
        cause_text="Air pollution",
        effect_text="lower life expectancy",
        cause_norm="air pollution",
        effect_norm="lower life expectancy",
        confidence=1.0,
        extractor="regex_spacy",
    )
    result = canonizer.canonize([rel])
    assert len(result) == 1
    assert result[0].cause_canonical == "Air pollution"
    assert result[0].effect_canonical == "lower life expectancy"


# --- Registry integration ---

def test_registry_builds_passthrough():
    from pipeline.registry import build_canonizer
    cfg = {"pipeline": {"step3_canonization": {"implementation": "passthrough"}}}
    canon = build_canonizer(cfg)
    assert isinstance(canon, PassthroughCanonizer)
    assert canon.name == "passthrough"
