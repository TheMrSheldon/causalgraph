"""
Read-only SQLite access layer for the backend API.

This module has no dependency on the pipeline package. It reads the database
that the pipeline writes; the expected schema is documented in docs/graphformat.md.
"""
from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from typing import Generator


class GraphDatabase:
    """Read-only view of a pipeline-produced SQLite database."""

    def __init__(self, db_path: str) -> None:
        self.db_path = db_path

    @contextmanager
    def _connect(self) -> Generator[sqlite3.Connection, None, None]:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
        finally:
            conn.close()

    # ------------------------------------------------------------------
    # Cluster hierarchy
    # ------------------------------------------------------------------

    def get_clusters_at_level(self, level: int) -> list[dict]:
        sql = """
            SELECT id, label, level, parent_id, member_count
            FROM clusters WHERE level = ?
            ORDER BY member_count DESC
        """
        with self._connect() as conn:
            return [dict(r) for r in conn.execute(sql, (level,)).fetchall()]

    def get_cluster_by_id(self, cluster_id: int) -> dict | None:
        sql = "SELECT id, label, level, parent_id, member_count FROM clusters WHERE id = ?"
        with self._connect() as conn:
            row = conn.execute(sql, (cluster_id,)).fetchone()
            return dict(row) if row else None

    def get_children(self, cluster_id: int) -> list[dict]:
        sql = """
            SELECT id, label, level, parent_id, member_count
            FROM clusters WHERE parent_id = ?
            ORDER BY member_count DESC
        """
        with self._connect() as conn:
            return [dict(r) for r in conn.execute(sql, (cluster_id,)).fetchall()]

    def get_level_counts(self) -> dict[int, int]:
        sql = "SELECT level, COUNT(*) AS cnt FROM clusters GROUP BY level"
        with self._connect() as conn:
            rows = conn.execute(sql).fetchall()
        return {r["level"]: r["cnt"] for r in rows}

    def get_top_events_for_cluster(self, cluster_id: int, n: int = 10) -> list[str]:
        sql = """
            SELECT event_text, COUNT(*) AS cnt
            FROM cluster_members WHERE cluster_id = ?
            GROUP BY event_text ORDER BY cnt DESC LIMIT ?
        """
        with self._connect() as conn:
            return [r["event_text"] for r in conn.execute(sql, (cluster_id, n)).fetchall()]

    # ------------------------------------------------------------------
    # Edges
    # ------------------------------------------------------------------

    def _get_descendant_leaf_ids(self, conn: sqlite3.Connection, ancestor_ids: list[int]) -> dict[int, int]:
        """Return {leaf_cluster_id: ancestor_id} for all descendants of the given IDs."""
        all_clusters = conn.execute("SELECT id, parent_id FROM clusters").fetchall()
        parent_of: dict[int, int | None] = {r[0]: r[1] for r in all_clusters}
        ancestor_set = set(ancestor_ids)
        leaf_to_ancestor: dict[int, int] = {}
        for cid in parent_of:
            cur: int | None = cid
            visited: set[int] = set()
            while cur is not None:
                if cur in visited:
                    break
                visited.add(cur)
                if cur in ancestor_set:
                    leaf_to_ancestor[cid] = cur
                    break
                cur = parent_of.get(cur)
        return leaf_to_ancestor

    def get_edges(self, cluster_ids: list[int] | None = None, min_post_count: int = 1) -> list[dict]:
        with self._connect() as conn:
            if cluster_ids is None:
                rows = conn.execute(
                    """SELECT source_cluster_id, target_cluster_id, relation_count,
                              post_count, avg_score, countercausal_count
                       FROM leaf_edges WHERE post_count >= ?""",
                    [min_post_count],
                ).fetchall()
                return [dict(r) for r in rows]

            leaf_to_ancestor = self._get_descendant_leaf_ids(conn, cluster_ids)
            if not leaf_to_ancestor:
                return []

            leaf_ids = list(leaf_to_ancestor.keys())
            ph = ",".join("?" * len(leaf_ids))
            rows = conn.execute(
                f"""SELECT source_cluster_id, target_cluster_id, relation_count,
                           post_count, avg_score, countercausal_count
                    FROM leaf_edges
                    WHERE source_cluster_id IN ({ph}) AND target_cluster_id IN ({ph})""",
                leaf_ids + leaf_ids,
            ).fetchall()

            agg: dict[tuple[int, int], dict] = {}
            for r in rows:
                src = leaf_to_ancestor.get(r["source_cluster_id"])
                tgt = leaf_to_ancestor.get(r["target_cluster_id"])
                if src is None or tgt is None or src == tgt:
                    continue
                key = (src, tgt)
                if key not in agg:
                    agg[key] = {"source_cluster_id": src, "target_cluster_id": tgt,
                                "relation_count": 0, "post_count": 0,
                                "avg_score_sum": 0.0, "n": 0, "countercausal_count": 0}
                agg[key]["relation_count"] += r["relation_count"]
                agg[key]["post_count"] += r["post_count"]
                agg[key]["avg_score_sum"] += (r["avg_score"] or 0.0) * r["post_count"]
                agg[key]["n"] += r["post_count"]
                agg[key]["countercausal_count"] += r["countercausal_count"]

            return [
                {"source_cluster_id": v["source_cluster_id"],
                 "target_cluster_id": v["target_cluster_id"],
                 "relation_count": v["relation_count"],
                 "post_count": v["post_count"],
                 "avg_score": v["avg_score_sum"] / v["n"] if v["n"] > 0 else 0.0,
                 "countercausal_count": v["countercausal_count"]}
                for v in agg.values()
                if v["post_count"] >= min_post_count
            ]

    # ------------------------------------------------------------------
    # Posts
    # ------------------------------------------------------------------

    def get_posts_for_cluster(
        self,
        cluster_id: int,
        limit: int = 50,
        offset: int = 0,
        sort: str = "score",
    ) -> tuple[list[dict], int]:
        sort_col = {"score": "p.score", "date": "p.created_utc", "comments": "p.num_comments"}.get(
            sort, "p.score"
        )
        sql = f"""
            SELECT p.id, p.title, p.score, p.num_comments, p.created_utc, p.permalink,
                   cr.cause_text, cr.effect_text, cr.is_countercausal
            FROM posts p
            JOIN (
                SELECT cr2.post_id, MIN(cr2.id) AS min_cr_id
                FROM causal_relations cr2
                JOIN cluster_members cm2 ON cm2.relation_id = cr2.id
                WHERE cm2.cluster_id = ?
                GROUP BY cr2.post_id
            ) best ON best.post_id = p.id
            JOIN causal_relations cr ON cr.id = best.min_cr_id
            ORDER BY {sort_col} DESC
            LIMIT ? OFFSET ?
        """
        count_sql = """
            SELECT COUNT(DISTINCT p.id) FROM posts p
            JOIN causal_relations cr ON cr.post_id = p.id
            JOIN cluster_members cm ON cm.relation_id = cr.id
            WHERE cm.cluster_id = ?
        """
        with self._connect() as conn:
            total = conn.execute(count_sql, (cluster_id,)).fetchone()[0]
            rows = [dict(r) for r in conn.execute(sql, (cluster_id, limit, offset)).fetchall()]
        return rows, total

    def get_posts_for_edge(
        self,
        source_cluster_id: int,
        target_cluster_id: int,
        limit: int = 50,
        offset: int = 0,
    ) -> tuple[list[dict], int]:
        with self._connect() as conn:
            leaf_map = self._get_descendant_leaf_ids(conn, [source_cluster_id, target_cluster_id])
            src_leaves = [cid for cid, anc in leaf_map.items() if anc == source_cluster_id] or [source_cluster_id]
            tgt_leaves = [cid for cid, anc in leaf_map.items() if anc == target_cluster_id] or [target_cluster_id]

            src_ph = ",".join("?" * len(src_leaves))
            tgt_ph = ",".join("?" * len(tgt_leaves))
            params = src_leaves + tgt_leaves

            sql = f"""
                SELECT p.id, p.title, p.score, p.num_comments, p.created_utc, p.permalink,
                       cr.cause_text, cr.effect_text, cr.is_countercausal
                FROM posts p
                JOIN (
                    SELECT cr2.post_id, MIN(cr2.id) AS min_cr_id
                    FROM causal_relations cr2
                    JOIN cluster_members cm_cause
                      ON cm_cause.relation_id = cr2.id AND cm_cause.role = 'cause'
                    JOIN cluster_members cm_effect
                      ON cm_effect.relation_id = cr2.id AND cm_effect.role = 'effect'
                    WHERE cm_cause.cluster_id IN ({src_ph})
                      AND cm_effect.cluster_id IN ({tgt_ph})
                    GROUP BY cr2.post_id
                ) best ON best.post_id = p.id
                JOIN causal_relations cr ON cr.id = best.min_cr_id
                ORDER BY p.score DESC
                LIMIT ? OFFSET ?
            """
            count_sql = f"""
                SELECT COUNT(DISTINCT p.id)
                FROM posts p
                JOIN causal_relations cr ON cr.post_id = p.id
                JOIN cluster_members cm_cause
                  ON cm_cause.relation_id = cr.id AND cm_cause.role = 'cause'
                JOIN cluster_members cm_effect
                  ON cm_effect.relation_id = cr.id AND cm_effect.role = 'effect'
                WHERE cm_cause.cluster_id IN ({src_ph}) AND cm_effect.cluster_id IN ({tgt_ph})
            """
            total = conn.execute(count_sql, params).fetchone()[0]
            rows = [dict(r) for r in conn.execute(sql, params + [limit, offset]).fetchall()]
        return rows, total

    def get_post_by_id(self, post_id: str) -> dict | None:
        sql = """
            SELECT p.id, p.title, p.score, p.num_comments, p.created_utc, p.permalink,
                   cr.cause_text, cr.effect_text, cr.confidence, cr.is_countercausal
            FROM posts p
            LEFT JOIN causal_relations cr ON cr.post_id = p.id
            WHERE p.id = ?
            LIMIT 1
        """
        with self._connect() as conn:
            row = conn.execute(sql, (post_id,)).fetchone()
            return dict(row) if row else None
