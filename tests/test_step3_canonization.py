"""Unit tests for Step 3 canonization implementations."""
import pytest

from pipeline.protocols import EventCanonizer
from pipeline.step3_canonization.passthrough_canonizer import PassthroughCanonizer


@pytest.fixture
def canonizer():
    return PassthroughCanonizer()


# Spans as (text, (start, end)) tuples
_SPANS = [
    ("Smoking increases lung cancer risk.", (0, 7)),           # "Smoking"
    ("Regular exercise improves mental health.", (0, 16)),     # "Regular exercise"
]


# --- Protocol conformance ---

def test_implements_protocol(canonizer):
    assert isinstance(canonizer, EventCanonizer)


def test_name(canonizer):
    assert canonizer.name == "passthrough"


# --- Return type ---

def test_returns_list(canonizer):
    result = canonizer.canonize(_SPANS)
    assert isinstance(result, list)


def test_returns_strings(canonizer):
    result = canonizer.canonize(_SPANS)
    assert all(isinstance(s, str) for s in result)


def test_same_length(canonizer):
    result = canonizer.canonize(_SPANS)
    assert len(result) == len(_SPANS)


# --- Empty input ---

def test_empty_input(canonizer):
    assert canonizer.canonize([]) == []


# --- Passthrough: returns the raw span text ---

def test_passthrough_returns_span_text(canonizer):
    result = canonizer.canonize(_SPANS)
    for (text, (start, end)), canonical in zip(_SPANS, result):
        assert canonical == text[start:end]


def test_single_span(canonizer):
    text = "Air pollution reduces life expectancy."
    span = (text, (0, 13))   # "Air pollution"
    result = canonizer.canonize([span])
    assert result == ["Air pollution"]


def test_mid_span(canonizer):
    text = "Air pollution reduces life expectancy."
    span = (text, (22, 37))  # "life expectancy"
    result = canonizer.canonize([span])
    assert result == ["life expectancy"]


# --- Instantiation via qualified name ---

def test_qualified_name_instantiation():
    import importlib
    module = importlib.import_module("pipeline.step3_canonization.passthrough_canonizer")
    cls = getattr(module, "PassthroughCanonizer")
    canon = cls()
    assert isinstance(canon, PassthroughCanonizer)
    assert canon.name == "passthrough"
