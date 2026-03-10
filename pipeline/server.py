"""
Pipeline step server.

Exposes each pipeline step as a REST endpoint so external clients (e.g. the
frontend text analyzer) can call individual steps without running the full
pipeline.

Start with:
    uvicorn pipeline.server:app --host 0.0.0.0 --port 8001

Endpoints
---------
POST /detect    — Step 1: decide whether text contains causal language
POST /extract   — Step 2+3: extract (cause, effect) pairs and canonize them
GET  /health    — liveness check
"""
from __future__ import annotations

import copy
import logging
import os
import re
from contextlib import asynccontextmanager
from functools import lru_cache
from typing import AsyncGenerator

# When running inside the Docker container, routes are mounted under /pipeline
# so that the lighttpd reverse proxy can forward /pipeline/* without any path
# rewriting.  In the dev environment (Vite proxy strips the prefix) this stays
# empty so all routes are served at their plain paths (/detect, /extract, …).
_PIPELINE_ROOT = os.environ.get("PIPELINE_ROOT", "").rstrip("/")

import yaml
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

import importlib

from .protocols import CausalityDetector, CausalityExtractor, EventCanonizer, Post


def _build(step_cfg: dict, protocol_cls):
    """Instantiate a pipeline step from a qualified class name in config."""
    import copy
    cfg = copy.deepcopy(step_cfg)
    qualified = cfg.pop("implementation")
    module_path, _, class_name = qualified.rpartition(".")
    cls = getattr(importlib.import_module(module_path), class_name)
    obj = cls(**cfg)
    if not isinstance(obj, protocol_cls):
        raise TypeError(f"{type(obj).__name__} does not satisfy {protocol_cls.__name__}")
    return obj

# Use uvicorn's own error logger so messages appear in the uvicorn console
# output without needing to configure a separate handler.
logger = logging.getLogger("uvicorn.error")

# ── GPU detection (done once at import time) ─────────────────────────────────
try:
    import torch as _torch
    _DEVICE: int = 0 if _torch.cuda.is_available() else -1
    _GPU_NAME: str | None = _torch.cuda.get_device_name(0) if _DEVICE == 0 else None
except Exception:
    _DEVICE = -1
    _GPU_NAME = None


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@lru_cache(maxsize=1)
def _get_config() -> dict:
    with open("pipeline.yaml") as f:
        return yaml.safe_load(f)


@lru_cache(maxsize=1)
def _get_detector():
    return _build(_get_config()["step1_detection"], CausalityDetector)


@lru_cache(maxsize=1)
def _get_extractor():
    return _build(_get_config()["step2_extraction"], CausalityExtractor)


@lru_cache(maxsize=1)
def _get_canonizer():
    step_cfg = copy.deepcopy(_get_config()["step3_canonization"])
    # Inject GPU device for TransformerCanonizer unless already specified
    if step_cfg.get("implementation", "").endswith("TransformerCanonizer") and "device" not in step_cfg:
        step_cfg["device"] = _DEVICE
    return _build(step_cfg, EventCanonizer)


# ── Startup: preload all models ───────────────────────────────────────────────

@asynccontextmanager
async def lifespan(_app: FastAPI) -> AsyncGenerator[None, None]:
    if _DEVICE == 0:
        logger.info("GPU detected: %s — pipeline will use CUDA device 0", _GPU_NAME)
    else:
        logger.info("No GPU detected — pipeline will run on CPU")

    logger.info("Preloading pipeline components…")
    for name, loader in [("detector", _get_detector), ("extractor", _get_extractor), ("canonizer", _get_canonizer)]:
        try:
            loader()
            logger.info("  %s ready", name)
        except Exception as exc:  # noqa: BLE001
            logger.warning("  %s failed to preload: %s", name, exc)
    logger.info("Pipeline server ready")
    yield


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

class DetectRequest(BaseModel):
    text: str


class DetectResponse(BaseModel):
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
    cause_canonical: str
    effect_canonical: str
    relation_type: str
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
    app = FastAPI(
        title="Pipeline Step API",
        description="Per-step REST interface for the r/science causal pipeline.",
        version="0.2.0",
        lifespan=lifespan,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["GET", "POST"],
        allow_headers=["*"],
    )

    @app.get(f"{_PIPELINE_ROOT}/health")
    def health() -> dict:
        return {"status": "ok"}

    @app.post(f"{_PIPELINE_ROOT}/detect", response_model=DetectResponse)
    def detect(body: DetectRequest) -> DetectResponse:
        """
        Step 1 — Decide whether the given text contains causal language.
        Returns ``is_causal: true`` if the configured detector classifies
        the text as causal.
        """
        text = body.text.strip()
        if not text:
            return DetectResponse(is_causal=False)
        post = Post(id="_detect_0", title=text, score=0, num_comments=0, created_utc=0)
        result = _get_detector().detect([post])
        return DetectResponse(is_causal=len(result) > 0)

    @app.post(f"{_PIPELINE_ROOT}/extract", response_model=ExtractResponse)
    def extract(body: ExtractRequest) -> ExtractResponse:
        """
        Steps 2+3 — Extract causal relations from free text, then canonize
        the event descriptions.

        Splits the input into sentences, runs the configured extractor on
        each, canonizes the resulting spans, and returns:
        - ``events``: unique event spans with character offsets in the original text
        - ``relations``: cause→effect pairs with canonical descriptions and
          per-label certainty scores
        """
        text = body.text.strip()
        if not text:
            return ExtractResponse(text=text, events=[], relations=[])

        extractor = _get_extractor()
        canonizer = _get_canonizer()
        sentences = _split_sentences(text)

        # Collect all raw relations across sentences first so we can batch-canonize
        from pipeline.protocols import CausalRelation as CR
        raw_relations: list[tuple[CR, str, int]] = []  # (relation, sent_text, sent_offset)

        for sent_text, sent_offset in sentences:
            post = Post(
                id=f"_extract_{sent_offset}",
                title=sent_text,
                score=0,
                num_comments=0,
                created_utc=0,
            )
            for rel in extractor.extract(post):
                rel.post_title = text  # full input text as canonization context
                raw_relations.append((rel, sent_text, sent_offset))

        # Build (text, (start, end)) span inputs for the canonizer —
        # cause then effect for each relation, using the full input as context.
        def _span_idx(ctx: str, phrase: str) -> tuple[int, int]:
            idx = ctx.lower().find(phrase.lower())
            return (idx, idx + len(phrase)) if idx != -1 else (0, len(phrase))

        if raw_relations:
            span_inputs: list[tuple[str, tuple[int, int]]] = []
            for rel, _, _ in raw_relations:
                ctx = rel.post_title  # full input text
                span_inputs.append((ctx, _span_idx(ctx, rel.cause_text)))
                span_inputs.append((ctx, _span_idx(ctx, rel.effect_text)))
            canonical_strings = canonizer.canonize(span_inputs)
        else:
            canonical_strings = []

        # Build events + response relations
        event_index_map: dict[str, int] = {}
        events: list[EventItem] = []
        relations: list[RelationItem] = []

        def _get_or_add_event(phrase: str, sent_text: str, sent_offset: int, description: str) -> int | None:
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
                description=description,
                start=span[0],
                end=span[1],
            ))
            return idx

        for i, (rel, sent_text, sent_offset) in enumerate(raw_relations):
            cause_canonical = canonical_strings[i * 2] if canonical_strings else rel.cause_text
            effect_canonical = canonical_strings[i * 2 + 1] if canonical_strings else rel.effect_text
            cause_idx = _get_or_add_event(
                rel.cause_text, sent_text, sent_offset,
                description=cause_canonical,
            )
            effect_idx = _get_or_add_event(
                rel.effect_text, sent_text, sent_offset,
                description=effect_canonical,
            )
            if cause_idx is None or effect_idx is None:
                continue
            relations.append(RelationItem(
                cause_event_index=cause_idx,
                effect_event_index=effect_idx,
                cause_text=rel.cause_text,
                effect_text=rel.effect_text,
                cause_canonical=cause_canonical,
                effect_canonical=effect_canonical,
                relation_type=rel.relation_type.value,
                p_none=rel.p_none,
                p_causal=rel.p_causal,
                p_countercausal=rel.p_countercausal,
            ))

        return ExtractResponse(text=text, events=events, relations=relations)

    return app


app = create_app()
