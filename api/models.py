"""Pydantic response models for the FastAPI application."""
from __future__ import annotations

from pydantic import BaseModel


class ClusterNode(BaseModel):
    id: int
    label: str
    level: int          # 0=leaf, 1=mid, 2=top
    parent_id: int | None
    member_count: int


class GraphEdge(BaseModel):
    source_cluster_id: int
    target_cluster_id: int
    relation_count: int
    post_count: int
    avg_score: float
    countercausal_count: int = 0


class GraphResponse(BaseModel):
    nodes: list[ClusterNode]
    edges: list[GraphEdge]


class LevelCounts(BaseModel):
    levels: list[int]
    counts: dict[str, int]


class PostSummary(BaseModel):
    id: str
    title: str
    score: int
    num_comments: int
    created_utc: int
    permalink: str | None


class PostDetail(PostSummary):
    cause_text: str | None = None
    effect_text: str | None = None
    confidence: float | None = None
    is_countercausal: bool = False


class ClusterDetail(BaseModel):
    cluster: ClusterNode
    children: list[ClusterNode]
    top_events: list[str]
    posts: list[EdgePostSummary]


class PaginatedPosts(BaseModel):
    posts: list[PostSummary]
    total: int
    limit: int
    offset: int


class EdgePostSummary(PostSummary):
    cause_text: str | None = None
    effect_text: str | None = None
    is_countercausal: bool = False


class PaginatedEdgePosts(BaseModel):
    posts: list[EdgePostSummary]
    total: int
    limit: int
    offset: int


# ---------------------------------------------------------------------------
# Text Analysis endpoint models (POST /api/analyze)
# ---------------------------------------------------------------------------

class AnalysisRequest(BaseModel):
    text: str


class AnalysisEvent(BaseModel):
    """A unique event identified in the text, with its span position."""
    index: int          # palette index; same event text → same index
    span_text: str      # text as it appears in the original document
    description: str    # cleaned/extracted event phrase (may differ for LLM extractors)
    start: int          # character offset in original text
    end: int            # exclusive


class AnalysisRelationItem(BaseModel):
    """One extracted causal/countercausal relationship between two events."""
    cause_event_index: int
    effect_event_index: int
    cause_text: str
    effect_text: str
    is_countercausal: bool
    p_none: float
    p_causal: float
    p_countercausal: float


class AnalysisResult(BaseModel):
    text: str
    events: list[AnalysisEvent]
    relations: list[AnalysisRelationItem]
