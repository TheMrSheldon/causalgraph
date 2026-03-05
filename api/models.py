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


class ClusterDetail(BaseModel):
    cluster: ClusterNode
    children: list[ClusterNode]
    top_events: list[str]
    posts: list[PostSummary]


class PaginatedPosts(BaseModel):
    posts: list[PostSummary]
    total: int
    limit: int
    offset: int


class EdgePostSummary(PostSummary):
    cause_text: str | None = None
    effect_text: str | None = None


class PaginatedEdgePosts(BaseModel):
    posts: list[EdgePostSummary]
    total: int
    limit: int
    offset: int
