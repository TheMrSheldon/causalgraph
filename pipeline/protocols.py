"""
Protocol definitions for the three pluggable pipeline steps.

Each concrete implementation must satisfy structural subtyping (duck typing).
Use @runtime_checkable so the registry can validate conformance with isinstance().
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
    cause_norm: str       # lowercased / lemmatized for dedup
    effect_norm: str
    confidence: float = 1.0
    extractor: str = ""
    is_countercausal: bool = False
    # Per-label prediction certainty (not stored in DB)
    p_none: float = 0.0
    p_causal: float = 0.0
    p_countercausal: float = 0.0


@dataclass
class EventCluster:
    """
    A node in the hierarchy graph.
    parent_id=None → top-level cluster.
    level: 0=leaf, 1=mid, 2=top.
    """
    label: str
    level: int
    parent_id: int | None
    member_count: int = 0
    clusterer: str = ""


# ---------------------------------------------------------------------------
# Step 1: Causality Identification
# ---------------------------------------------------------------------------

@runtime_checkable
class CausalityIdentifier(Protocol):
    """
    Scans a batch of Post objects and returns those that express
    a causal relationship (explicit or implicit).

    Implementations:
      - RegexIdentifier      (pipeline.step1_identification.regex_identifier)
      - LLMIdentifier        (pipeline.step1_identification.llm_identifier)
      - ZeroShotIdentifier   (pipeline.step1_identification.zero_shot_identifier)
    """

    def identify(self, posts: list[Post]) -> list[Post]:
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
        """Unique registry key, e.g. 'regex', 'llm_openai', 'zero_shot'."""
        ...


# ---------------------------------------------------------------------------
# Step 2: Causal Extraction
# ---------------------------------------------------------------------------

@runtime_checkable
class CausalExtractor(Protocol):
    """
    Given a Post confirmed to express causality, extracts one or more
    (cause, effect) pairs as CausalRelation objects.

    Implementations:
      - RegexSpacyExtractor  (pipeline.step2_extraction.regex_spacy_extractor)
      - LLMExtractor         (pipeline.step2_extraction.llm_extractor)
    """

    def extract(self, post: Post) -> list[CausalRelation]:
        """
        Args:
            post: A Post confirmed to contain a causal claim.

        Returns:
            One or more CausalRelation objects. Returns empty list only
            if extraction genuinely fails (do not raise).
        """
        ...

    @property
    def name(self) -> str:
        """Unique registry key, e.g. 'regex_spacy', 'llm_openai'."""
        ...


# ---------------------------------------------------------------------------
# Step 3: Hierarchy Inference
# ---------------------------------------------------------------------------

@runtime_checkable
class HierarchyInferrer(Protocol):
    """
    Takes all CausalRelation objects and groups cause/effect event texts
    into a multi-level cluster hierarchy.

    Implementations:
      - EmbeddingClusterer  (pipeline.step3_hierarchy.embedding_clusterer)
      - TFIDFClusterer      (pipeline.step3_hierarchy.tfidf_clusterer)
      - LLMTopicGrouper     (pipeline.step3_hierarchy.llm_topic_grouper)
    """

    def infer(
        self,
        relations: list[CausalRelation],
    ) -> tuple[list[EventCluster], list[tuple[int, int, str, str]]]:
        """
        Args:
            relations: All extracted causal relations.

        Returns:
            A tuple of:
              clusters    – list of EventCluster objects (no DB IDs yet).
              memberships – list of (relation_index, cluster_index, role, event_text)
                            where role is 'cause' or 'effect'.
        """
        ...

    @property
    def name(self) -> str:
        """Unique registry key, e.g. 'embedding_hdbscan', 'tfidf_ward', 'llm'."""
        ...
