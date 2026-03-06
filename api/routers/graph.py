"""Graph topology endpoints."""
from __future__ import annotations

from fastapi import APIRouter, Depends, Query

from api.dependencies import get_db
from api.models import ClusterNode, GraphEdge, GraphResponse, LevelCounts
from api.db import GraphDatabase

router = APIRouter(prefix="/api/graph", tags=["graph"])


@router.get("", response_model=GraphResponse)
def get_graph(
    level: int = Query(default=2, ge=0, le=10, description="Hierarchy level to show as top nodes"),
    min_post_count: int = Query(default=1, ge=1, description="Filter edges with fewer posts"),
    db: GraphDatabase = Depends(get_db),
) -> GraphResponse:
    """
    Return top-level graph: all cluster nodes at the requested level,
    plus all edges between them.
    """
    raw_nodes = db.get_clusters_at_level(level)
    if not raw_nodes:
        return GraphResponse(nodes=[], edges=[])

    cluster_ids = [n["id"] for n in raw_nodes]
    raw_edges = db.get_edges(cluster_ids=cluster_ids, min_post_count=min_post_count)

    nodes = [ClusterNode(**n) for n in raw_nodes]
    edges = [
        GraphEdge(
            source_cluster_id=e["source_cluster_id"],
            target_cluster_id=e["target_cluster_id"],
            relation_count=e["relation_count"],
            post_count=e["post_count"],
            avg_score=round(e["avg_score"] or 0.0, 1),
            countercausal_count=e.get("countercausal_count", 0),
        )
        for e in raw_edges
        if e["source_cluster_id"] != e["target_cluster_id"]  # no self-loops
    ]

    return GraphResponse(nodes=nodes, edges=edges)


@router.get("/levels", response_model=LevelCounts)
def get_levels(db: GraphDatabase = Depends(get_db)) -> LevelCounts:
    """Return available hierarchy levels and the number of clusters at each."""
    counts = db.get_level_counts()
    levels = sorted(counts.keys())
    return LevelCounts(
        levels=levels,
        counts={str(k): v for k, v in counts.items()},
    )
