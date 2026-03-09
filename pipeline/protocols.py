"""
Protocol definitions for the four pluggable pipeline steps.

Each concrete implementation must satisfy structural subtyping (duck typing).
@runtime_checkable enables isinstance() checks in the loader.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable


# ---------------------------------------------------------------------------
# Shared data structures
# ---------------------------------------------------------------------------

@dataclass
class Post:
    """A single r/science submission row."""
    id: str
    title: str
    score: int
    num_comments: int
    created_utc: int
    author: str | None = None
    url: str | None = None
    permalink: str | None = None


@dataclass
class CausalRelation:
    """
    One extracted (cause, effect) pair tied to its source post.

    cause_text / effect_text   — raw phrase as extracted from the title
    cause_norm / effect_norm   — lowercased/stripped form used as a stable
                                 deduplication key across the pipeline
    cause_canonical / effect_canonical — self-contained event description
                                 produced by Step 3 (canonization). Empty
                                 until that step runs; hierarchy falls back
                                 to cause_text when this is empty.

    confidence=1.0 for deterministic (rule-based) extractors.
    is_countercausal=True when the title explicitly negates the causal claim
    (e.g. "X does not cause Y", "no link between X and Y").

    p_none / p_causal / p_countercausal: per-label probabilities (or
    pseudo-probabilities for rule-based extractors). These are NOT persisted
    to the database; they are populated at inference / analysis time only.
    """
    post_id: str
    cause_text: str
    effect_text: str
    cause_norm: str       # lowercased / stripped for dedup
    effect_norm: str
    confidence: float = 1.0
    extractor: str = ""
    is_countercausal: bool = False
    cause_canonical: str = ""   # self-contained description (from Step 3)
    effect_canonical: str = ""  # self-contained description (from Step 3)
    post_title: str = ""        # title of the source post — context for canonization
                                # populated at runtime; NOT persisted in DB
    # Per-label prediction certainty (not stored in DB)
    p_none: float = 0.0
    p_causal: float = 0.0
    p_countercausal: float = 0.0


@dataclass
class EventCluster:
    """
    A node in the hierarchy graph.
    parent_id=None → top-level cluster.
    level: 0=leaf, 1=mid, 2=top (or higher for deeper hierarchies).
    """
    label: str
    level: int
    parent_id: int | None
    member_count: int = 0
    clusterer: str = ""


# ---------------------------------------------------------------------------
# Step 1: Causality Detection
# ---------------------------------------------------------------------------

@runtime_checkable
class CausalityDetector(Protocol):
    """
    Scans a batch of Post objects and returns those that express
    a causal relationship (explicit or implicit).
    """

    def detect(self, posts: list[Post]) -> list[Post]:
        """
        Args:
            posts: A batch of Post objects (typically 1000–5000 at a time).

        Returns:
            Subset of posts that express a causal relationship.
            Favor precision over recall for the default implementation.
        """
        ...

    @property
    def name(self) -> str:
        """Short identifier for logging, e.g. 'regex', 'llm_openai', 'zero_shot'."""
        ...


# ---------------------------------------------------------------------------
# Step 2: Causal Extraction
# ---------------------------------------------------------------------------

@runtime_checkable
class CausalExtractor(Protocol):
    """
    Given a Post confirmed to express causality, extracts one or more
    (cause, effect) pairs as CausalRelation objects.
    """

    def extract(self, post: Post) -> list[CausalRelation]:
        """
        Args:
            post: A Post confirmed to contain a causal claim.

        Returns:
            One or more CausalRelation objects. Returns empty list only
            if extraction genuinely fails (do not raise).
            cause_canonical / effect_canonical are left empty here;
            they are filled by Step 3 (canonization).
        """
        ...

    @property
    def name(self) -> str:
        """Short identifier for logging, e.g. 'regex_spacy', 'llm_openai'."""
        ...


# ---------------------------------------------------------------------------
# Step 3: Event Canonization
# ---------------------------------------------------------------------------

@runtime_checkable
class EventCanonizer(Protocol):
    """
    Takes a list of (text, span) pairs and returns one canonical string per input.

    Each input pair:
        text:  the surrounding text (e.g. a post title or sentence)
        span:  (start, end) — character indices into ``text`` identifying the
               event span to canonize.  ``text[start:end]`` is the raw span.

    The canonical string should be a self-contained phrase that makes sense
    without the surrounding sentence — e.g. resolving pronouns, expanding
    abbreviations, or reformulating truncated spans using context from ``text``.
    """

    def canonize(self, spans: list[tuple[str, tuple[int, int]]]) -> list[str]:
        """
        Args:
            spans: list of (text, (start, end)) pairs.
                   ``text[start:end]`` is the raw event span to canonize.

        Returns:
            A list of canonical strings, one per input span, in the same order.
        """
        ...

    @property
    def name(self) -> str:
        """Short identifier for logging, e.g. 'passthrough', 'transformer', 'llm_anthropic'."""
        ...


# ---------------------------------------------------------------------------
# Step 4: Hierarchy Inference
# ---------------------------------------------------------------------------

@runtime_checkable
class HierarchyInferrer(Protocol):
    """
    Takes all CausalRelation objects and groups cause/effect event texts
    into a multi-level cluster hierarchy.
    """

    def infer(
        self,
        relations: list[CausalRelation],
    ) -> tuple[list[EventCluster], list[tuple[int, int, str, str]]]:
        """
        Args:
            relations: All extracted causal relations (with canonical fields set).

        Returns:
            A tuple of:
              clusters    – list of EventCluster objects (no DB IDs yet).
              memberships – list of (relation_index, cluster_index, role, event_text)
                            where role is 'cause' or 'effect'.
        """
        ...

    @property
    def name(self) -> str:
        """Short identifier for logging, e.g. 'embedding_hdbscan', 'tfidf_ward', 'llm'."""
        ...
