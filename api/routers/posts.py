"""Post-level endpoints."""
from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Query

from api.dependencies import get_db
from api.models import EdgePostSummary, PaginatedEdgePosts, PaginatedPosts, PostDetail, PostSummary, RelationSpan
from api.db import GraphDatabase

router = APIRouter(prefix="/api/posts", tags=["posts"])


@router.get("", response_model=PaginatedEdgePosts)
def get_posts_for_edge(
    source_cluster_id: int = Query(..., description="Cause-side cluster ID"),
    target_cluster_id: int = Query(..., description="Effect-side cluster ID"),
    limit: int = Query(default=50, ge=1, le=200),
    offset: int = Query(default=0, ge=0),
    db: GraphDatabase = Depends(get_db),
) -> PaginatedEdgePosts:
    """
    Return posts where the cause event belongs to source_cluster and the
    effect event belongs to target_cluster. Triggered by clicking an edge.
    """
    posts_raw, total = db.get_posts_for_edge(
        source_cluster_id, target_cluster_id, limit=limit, offset=offset
    )
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
                        is_countercausal=bool(r.get("is_countercausal", 0)),
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


@router.get("/{post_id}", response_model=PostDetail)
def get_post(post_id: str, db: GraphDatabase = Depends(get_db)) -> PostDetail:
    """Return a single post with its extracted causal pair."""
    row = db.get_post_by_id(post_id)
    if row is None:
        raise HTTPException(status_code=404, detail=f"Post {post_id} not found")
    return PostDetail(
        id=row["id"],
        title=row["title"],
        score=row["score"],
        num_comments=row["num_comments"],
        created_utc=row["created_utc"],
        permalink=row["permalink"],
        cause_text=row.get("cause_text"),
        effect_text=row.get("effect_text"),
        confidence=row.get("confidence"),
        is_countercausal=bool(row.get("is_countercausal", 0)),
    )
