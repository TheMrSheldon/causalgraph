"""
Pipeline step server.

Exposes each pipeline step as a REST endpoint so external clients (e.g. the
frontend text analyzer) can call individual steps without running the full
pipeline.

Start with:
    uvicorn pipeline.server:app --host 0.0.0.0 --port 8001

Endpoints
---------
POST /identify   — Step 1: decide whether text contains causal language
POST /extract    — Step 2: extract (cause, effect) pairs with certainty scores
GET  /health     — liveness check
"""
from __future__ import annotations

import re
from functools import lru_cache

import yaml
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from pipeline.protocols import Post
from pipeline.registry import build_identifier, build_extractor


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@lru_cache(maxsize=1)
def _get_config() -> dict:
    with open("config.yaml") as f:
        return yaml.safe_load(f)


@lru_cache(maxsize=1)
def _get_identifier():
    return build_identifier(_get_config())


@lru_cache(maxsize=1)
def _get_extractor():
    return build_extractor(_get_config())


# ---------------------------------------------------------------------------
# Shared sentence splitter
# ---------------------------------------------------------------------------

_SENT_SPLIT_RE = re.compile(r'(?<=[.!?])["\']?\s+')


def _split_sentences(text: str) -> list[tuple[str, int]]:
    """Return list of (sentence, start_offset_in_original_text)."""
    normalised = text.replace("\n", ". ")
    parts = _SENT_SPLIT_RE.split(normalised)
    sentences: list[tuple[str, int]] = []
    search_from = 0
    for part in parts:
        stripped = part.strip().rstrip(".")
        if not stripped:
            continue
        idx = text.find(stripped, search_from)
        if idx == -1:
            idx = text.lower().find(stripped.lower(), search_from)
        if idx != -1:
            sentences.append((stripped, idx))
            search_from = idx + len(stripped)
    return sentences


def _find_span(sentence: str, phrase: str, sent_offset: int) -> tuple[int, int] | None:
    idx = sentence.lower().find(phrase.lower())
    if idx == -1:
        return None
    return (sent_offset + idx, sent_offset + idx + len(phrase))


# ---------------------------------------------------------------------------
# Request / response models
# ---------------------------------------------------------------------------

class IdentifyRequest(BaseModel):
    text: str


class IdentifyResponse(BaseModel):
    is_causal: bool


class ExtractRequest(BaseModel):
    text: str


class EventItem(BaseModel):
    index: int
    span_text: str
    description: str
    start: int
    end: int


class RelationItem(BaseModel):
    cause_event_index: int
    effect_event_index: int
    cause_text: str
    effect_text: str
    is_countercausal: bool
    p_none: float
    p_causal: float
    p_countercausal: float


class ExtractResponse(BaseModel):
    text: str
    events: list[EventItem]
    relations: list[RelationItem]


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

def create_app() -> FastAPI:
    config = _get_config()
    cors_origins = config.get("pipeline_server", {}).get("cors_origins", ["*"])

    app = FastAPI(
        title="Pipeline Step API",
        description="Per-step REST interface for the r/science causal pipeline.",
        version="0.1.0",
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=cors_origins,
        allow_credentials=True,
        allow_methods=["GET", "POST"],
        allow_headers=["*"],
    )

    @app.get("/health")
    def health() -> dict:
        return {"status": "ok"}

    @app.post("/identify", response_model=IdentifyResponse)
    def identify(body: IdentifyRequest) -> IdentifyResponse:
        """
        Step 1 — Decide whether the given text contains causal language.
        Returns ``is_causal: true`` if the configured identifier classifies
        the text as causal.
        """
        text = body.text.strip()
        if not text:
            return IdentifyResponse(is_causal=False)
        post = Post(id="_identify_0", title=text, score=0, num_comments=0, created_utc=0)
        result = _get_identifier().identify([post])
        return IdentifyResponse(is_causal=len(result) > 0)

    @app.post("/extract", response_model=ExtractResponse)
    def extract(body: ExtractRequest) -> ExtractResponse:
        """
        Step 2 — Extract causal relations from free text.

        Splits the input into sentences, runs the configured extractor on
        each, and returns:
        - ``events``: unique event spans with character offsets in the original text
        - ``relations``: cause→effect pairs with per-label certainty scores
        """
        text = body.text.strip()
        if not text:
            return ExtractResponse(text=text, events=[], relations=[])

        extractor = _get_extractor()
        sentences = _split_sentences(text)

        event_index_map: dict[str, int] = {}
        events: list[EventItem] = []
        relations: list[RelationItem] = []

        def _get_or_add_event(phrase: str, sent_text: str, sent_offset: int) -> int | None:
            key = phrase.lower()
            if key in event_index_map:
                return event_index_map[key]
            span = _find_span(sent_text, phrase, sent_offset)
            if span is None:
                return None
            idx = len(events)
            event_index_map[key] = idx
            events.append(EventItem(
                index=idx,
                span_text=text[span[0]:span[1]],
                description=phrase,
                start=span[0],
                end=span[1],
            ))
            return idx

        for sent_text, sent_offset in sentences:
            post = Post(
                id=f"_extract_{sent_offset}",
                title=sent_text,
                score=0,
                num_comments=0,
                created_utc=0,
            )
            for rel in extractor.extract(post):
                cause_idx = _get_or_add_event(rel.cause_text, sent_text, sent_offset)
                effect_idx = _get_or_add_event(rel.effect_text, sent_text, sent_offset)
                if cause_idx is None or effect_idx is None:
                    continue
                relations.append(RelationItem(
                    cause_event_index=cause_idx,
                    effect_event_index=effect_idx,
                    cause_text=rel.cause_text,
                    effect_text=rel.effect_text,
                    is_countercausal=rel.is_countercausal,
                    p_none=rel.p_none,
                    p_causal=rel.p_causal,
                    p_countercausal=rel.p_countercausal,
                ))

        return ExtractResponse(text=text, events=events, relations=relations)

    return app


app = create_app()
