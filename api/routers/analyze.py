"""Text analysis endpoint: extract causal relations from free text."""
from __future__ import annotations

import re
from functools import lru_cache

from fastapi import APIRouter

from api.models import AnalysisEvent, AnalysisRelationItem, AnalysisRequest, AnalysisResult
from pipeline.protocols import Post
from pipeline.step2_extraction.regex_spacy_extractor import RegexSpacyExtractor

router = APIRouter(prefix="/api/analyze", tags=["analyze"])

# Sentence boundary: after .!? optionally followed by quote/paren, then whitespace or end
_SENT_SPLIT_RE = re.compile(r'(?<=[.!?])["\')]?\s+')


@lru_cache(maxsize=1)
def _get_extractor() -> RegexSpacyExtractor:
    return RegexSpacyExtractor()


def _split_sentences(text: str) -> list[tuple[str, int]]:
    """
    Split text into (sentence, start_offset) pairs.
    Newlines are also treated as sentence boundaries.
    """
    # Normalise newlines to a sentinel that looks like a sentence end
    normalised = text.replace("\n", ". ")
    parts = _SENT_SPLIT_RE.split(normalised)
    sentences: list[tuple[str, int]] = []
    search_from = 0
    for part in parts:
        stripped = part.strip().rstrip(".")
        if not stripped:
            continue
        # Locate this part in the *original* text
        idx = text.find(stripped, search_from)
        if idx == -1:
            # Fallback: try case-insensitive search
            lower = text.lower()
            idx = lower.find(stripped.lower(), search_from)
        if idx != -1:
            sentences.append((stripped, idx))
            search_from = idx + len(stripped)
    return sentences


def _find_span(sentence: str, phrase: str, sent_offset: int) -> tuple[int, int] | None:
    """
    Find phrase in sentence (case-insensitive). Returns absolute offsets into
    the original text, or None if not found.
    """
    idx = sentence.lower().find(phrase.lower())
    if idx == -1:
        return None
    return (sent_offset + idx, sent_offset + idx + len(phrase))


@router.post("", response_model=AnalysisResult)
def analyze_text(body: AnalysisRequest) -> AnalysisResult:
    """
    Split input text into sentences, run the configured causal extractor on
    each sentence, and return:
      - events: unique events with their span positions in the original text
      - relations: cause→effect pairs with per-label prediction certainty
    """
    text = body.text.strip()
    if not text:
        return AnalysisResult(text=text, events=[], relations=[])

    extractor = _get_extractor()
    sentences = _split_sentences(text)

    # Unique event text → index (for palette coloring)
    event_index_map: dict[str, int] = {}
    events: list[AnalysisEvent] = []
    relations: list[AnalysisRelationItem] = []

    def _get_or_add_event(phrase: str, sent_text: str, sent_offset: int) -> int | None:
        """
        Register an event; return its index. Records the span position on
        first occurrence. Returns None if the span cannot be located.
        """
        key = phrase.lower()
        if key in event_index_map:
            return event_index_map[key]
        span = _find_span(sent_text, phrase, sent_offset)
        if span is None:
            return None
        idx = len(events)
        event_index_map[key] = idx
        span_text = text[span[0]:span[1]]
        events.append(AnalysisEvent(
            index=idx,
            span_text=span_text,
            description=phrase,        # extractor's cleaned version
            start=span[0],
            end=span[1],
        ))
        return idx

    for sent_text, sent_offset in sentences:
        # Create a minimal Post from the sentence
        post = Post(
            id=f"_analyze_{sent_offset}",
            title=sent_text,
            score=0,
            num_comments=0,
            created_utc=0,
        )
        extracted = extractor.extract(post)
        for rel in extracted:
            cause_idx = _get_or_add_event(rel.cause_text, sent_text, sent_offset)
            effect_idx = _get_or_add_event(rel.effect_text, sent_text, sent_offset)
            if cause_idx is None or effect_idx is None:
                continue
            relations.append(AnalysisRelationItem(
                cause_event_index=cause_idx,
                effect_event_index=effect_idx,
                cause_text=rel.cause_text,
                effect_text=rel.effect_text,
                is_countercausal=rel.is_countercausal,
                p_none=rel.p_none,
                p_causal=rel.p_causal,
                p_countercausal=rel.p_countercausal,
            ))

    return AnalysisResult(text=text, events=events, relations=relations)
