"""Cluster expand/collapse and detail endpoints."""
from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Query

from api.dependencies import get_db
from api.models import (
    ClusterDetail,
    ClusterNode,
    EdgePostSummary,
    GraphEdge,
    GraphResponse,
    PaginatedEdgePosts,
    PaginatedPosts,
    PostSummary,
    RelationSpan,
)
from api.db import GraphDatabase

router = APIRouter(prefix="/api/clusters", tags=["clusters"])


def _to_node(raw: dict) -> ClusterNode:
    return ClusterNode(
        id=raw["id"],
        label=raw["label"],
        level=raw["level"],
        parent_id=raw["parent_id"],
        member_count=raw["member_count"],
    )


@router.get("/{cluster_id}", response_model=ClusterDetail)
def get_cluster(cluster_id: int, db: GraphDatabase = Depends(get_db)) -> ClusterDetail:
    """Return cluster metadata, its children, top event phrases, and sample posts."""
    raw = db.get_cluster_by_id(cluster_id)
    if raw is None:
        raise HTTPException(status_code=404, detail=f"Cluster {cluster_id} not found")

    children_raw = db.get_children(cluster_id)
    top_events = db.get_top_events_for_cluster(cluster_id, n=20)
    posts_raw, _ = db.get_posts_for_cluster(cluster_id, limit=10, sort="score")

    post_ids = [p["id"] for p in posts_raw]
    all_relations = db.get_all_relations_for_posts(post_ids)

    return ClusterDetail(
        cluster=_to_node(raw),
        children=[_to_node(c) for c in children_raw],
        top_events=top_events,
        posts=[
            EdgePostSummary(
                id=p["id"],
                title=p["title"],
                score=p["score"],
                num_comments=p["num_comments"],
                created_utc=p["created_utc"],
                permalink=p["permalink"],
                relations=[
                    RelationSpan(
                        cause_text=r["cause_text"],
                        effect_text=r["effect_text"],
                        cause_canonical=r.get("cause_canonical"),
                        effect_canonical=r.get("effect_canonical"),
                        cause_cluster_id=r.get("cause_cluster_id"),
                        effect_cluster_id=r.get("effect_cluster_id"),
                        is_countercausal=bool(r.get("is_countercausal", False)),
                    )
                    for r in all_relations.get(p["id"], [])
                ],
            )
            for p in posts_raw
        ],
    )


@router.get("/{cluster_id}/expand", response_model=GraphResponse)
def expand_cluster(
    cluster_id: int,
    min_post_count: int = Query(default=1, ge=1),
    context_ids: str = Query(default="", description="Comma-separated IDs of other visible clusters"),
    db: GraphDatabase = Depends(get_db),
) -> GraphResponse:
    """
    Return child nodes of this cluster plus intra-cluster edges and border edges
    to any clusters listed in context_ids.
    """
    parent = db.get_cluster_by_id(cluster_id)
    if parent is None:
        raise HTTPException(status_code=404, detail=f"Cluster {cluster_id} not found")

    children_raw = db.get_children(cluster_id)
    if not children_raw:
        # Leaf node: return the node itself with its posts as a hint
        return GraphResponse(nodes=[_to_node(parent)], edges=[])

    child_ids = [c["id"] for c in children_raw]
    ext_ids = [int(x) for x in context_ids.split(",") if x.strip().lstrip("-").isdigit()]
    all_ids = child_ids + [eid for eid in ext_ids if eid not in child_ids]
    raw_edges = db.get_edges(cluster_ids=all_ids, min_post_count=min_post_count)

    return GraphResponse(
        nodes=[_to_node(c) for c in children_raw],
        edges=[
            GraphEdge(
                source_cluster_id=e["source_cluster_id"],
                target_cluster_id=e["target_cluster_id"],
                relation_count=e["relation_count"],
                post_count=e["post_count"],
                avg_score=round(e["avg_score"] or 0.0, 1),
                countercausal_count=e.get("countercausal_count", 0),
            )
            for e in raw_edges
            if e["source_cluster_id"] != e["target_cluster_id"]
        ],
    )


@router.get("/{cluster_id}/posts", response_model=PaginatedEdgePosts)
def get_cluster_posts(
    cluster_id: int,
    limit: int = Query(default=50, ge=1, le=200),
    offset: int = Query(default=0, ge=0),
    sort: str = Query(default="score", pattern="^(score|date|comments)$"),
    db: GraphDatabase = Depends(get_db),
) -> PaginatedEdgePosts:
    """Return paginated posts associated with this cluster."""
    if db.get_cluster_by_id(cluster_id) is None:
        raise HTTPException(status_code=404, detail=f"Cluster {cluster_id} not found")

    posts_raw, total = db.get_posts_for_cluster(cluster_id, limit=limit, offset=offset, sort=sort)
    post_ids = [p["id"] for p in posts_raw]
    all_relations = db.get_all_relations_for_posts(post_ids)
    return PaginatedEdgePosts(
        posts=[
            EdgePostSummary(
                id=p["id"],
                title=p["title"],
                score=p["score"],
                num_comments=p["num_comments"],
                created_utc=p["created_utc"],
                permalink=p["permalink"],
                relations=[
                    RelationSpan(
                        cause_text=r["cause_text"],
                        effect_text=r["effect_text"],
                        cause_canonical=r.get("cause_canonical"),
                        effect_canonical=r.get("effect_canonical"),
                        cause_cluster_id=r.get("cause_cluster_id"),
                        effect_cluster_id=r.get("effect_cluster_id"),
                        is_countercausal=bool(r.get("is_countercausal", False)),
                    )
                    for r in all_relations.get(p["id"], [])
                ],
            )
            for p in posts_raw
        ],
        total=total,
        limit=limit,
        offset=offset,
    )
