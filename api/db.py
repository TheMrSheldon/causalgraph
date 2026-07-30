"""
Read-only SQLite access layer for the backend API.

This module has no dependency on the pipeline package. It reads the database
that the pipeline writes; the expected schema is documented in docs/graphformat.md.
"""
from __future__ import annotations

import sqlite3
import time
from contextlib import contextmanager
from typing import Generator


class GraphDatabase:
    """Read-only view of a pipeline-produced SQLite database."""

    _CACHE_TTL = 300  # seconds

    def __init__(self, db_path: str) -> None:
        self.db_path = db_path
        self._cache: dict[str, tuple[float, object]] = {}

    def _cached(self, key: str, fn):
        entry = self._cache.get(key)
        if entry and time.time() - entry[0] < self._CACHE_TTL:
            return entry[1]
        result = fn()
        self._cache[key] = (time.time(), result)
        return result

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

    _CLUSTER_COLS = """
        c.id, c.label, c.level, c.parent_id, c.member_count,
        EXISTS(SELECT 1 FROM clusters c2 WHERE c2.parent_id = c.id) AS has_children
    """

    def get_root_clusters(self) -> list[dict]:
        def _query():
            sql = f"""
                SELECT {self._CLUSTER_COLS}
                FROM clusters c WHERE c.parent_id IS NULL
                ORDER BY c.member_count DESC
            """
            with self._connect() as conn:
                return [dict(r) for r in conn.execute(sql).fetchall()]
        return self._cached("root_clusters", _query)

    def get_clusters_at_level(self, level: int) -> list[dict]:
        sql = f"""
            SELECT {self._CLUSTER_COLS}
            FROM clusters c WHERE c.level = ?
            ORDER BY c.member_count DESC
        """
        with self._connect() as conn:
            return [dict(r) for r in conn.execute(sql, (level,)).fetchall()]

    def get_cluster_by_id(self, cluster_id: int) -> dict | None:
        sql = f"""
            SELECT {self._CLUSTER_COLS}
            FROM clusters c WHERE c.id = ?
        """
        with self._connect() as conn:
            row = conn.execute(sql, (cluster_id,)).fetchone()
            return dict(row) if row else None

    def get_children(self, cluster_id: int) -> list[dict]:
        sql = f"""
            SELECT {self._CLUSTER_COLS}
            FROM clusters c WHERE c.parent_id = ?
            ORDER BY c.member_count DESC
        """
        with self._connect() as conn:
            return [dict(r) for r in conn.execute(sql, (cluster_id,)).fetchall()]

    def get_level_counts(self) -> dict[int, int]:
        sql = "SELECT level, COUNT(*) AS cnt FROM clusters GROUP BY level"
        with self._connect() as conn:
            rows = conn.execute(sql).fetchall()
        return {r["level"]: r["cnt"] for r in rows}

    def get_top_events_for_cluster(self, cluster_id: int, n: int = 10) -> list[str]:
        with self._connect() as conn:
            leaf_ids = self._get_leaf_cluster_ids(conn, cluster_id)
            if not leaf_ids:
                return []
            ph = ",".join("?" * len(leaf_ids))
            sql = f"""
                SELECT event_text, COUNT(*) AS cnt
                FROM cluster_members WHERE cluster_id IN ({ph})
                GROUP BY event_text ORDER BY cnt DESC LIMIT ?
            """
            return [r["event_text"] for r in conn.execute(sql, leaf_ids + [n]).fetchall()]

    # ------------------------------------------------------------------
    # Edges
    # ------------------------------------------------------------------

    def _get_leaf_cluster_ids(self, conn: sqlite3.Connection, cluster_id: int) -> list[int]:
        """Return all level-0 leaf cluster IDs that are descendants of cluster_id (or itself if leaf)."""
        rows = conn.execute(
            """
            WITH RECURSIVE desc(id) AS (
                SELECT ? UNION ALL
                SELECT c.id FROM clusters c JOIN desc d ON c.parent_id = d.id
            )
            SELECT id FROM clusters WHERE id IN (SELECT id FROM desc) AND level = 0
            """,
            (cluster_id,),
        ).fetchall()
        return [r[0] for r in rows]

    def get_edges(self, cluster_ids: list[int] | None = None, min_post_count: int = 1) -> list[dict]:
        cache_key = f"edges:{','.join(map(str, sorted(cluster_ids) if cluster_ids else []))}:{min_post_count}"

        def _query():
            with self._connect() as conn:
                if cluster_ids is None:
                    rows = conn.execute(
                        """SELECT source_cluster_id, target_cluster_id, relation_count,
                                  post_count, avg_score, countercausal_count
                           FROM leaf_edges WHERE post_count >= ?""",
                        [min_post_count],
                    ).fetchall()
                    return [dict(r) for r in rows]

                if not cluster_ids:
                    return []

                ph = ",".join("?" * len(cluster_ids))

                # Use precomputed expand_edges when available (fast indexed lookup).
                # Falls back to the recursive CTE only when expand_edges is empty
                # (e.g. right after schema migration before rebuild_expand_edges is run).
                has_precomputed = conn.execute(
                    "SELECT 1 FROM expand_edges LIMIT 1"
                ).fetchone()

                if has_precomputed:
                    rows = conn.execute(
                        f"""SELECT source_cluster_id, target_cluster_id,
                                   relation_count, post_count, avg_score, countercausal_count
                            FROM expand_edges
                            WHERE source_cluster_id IN ({ph})
                              AND target_cluster_id IN ({ph})
                              AND post_count >= ?""",
                        cluster_ids + cluster_ids + [min_post_count],
                    ).fetchall()
                    return [dict(r) for r in rows]

                # Fallback: recursive CTE (used before first rebuild_expand_edges run)
                rows = conn.execute(
                    f"""
                    WITH RECURSIVE desc(id, ancestor_id) AS (
                        SELECT id, id FROM clusters WHERE id IN ({ph})
                        UNION ALL
                        SELECT c.id, d.ancestor_id FROM clusters c JOIN desc d ON c.parent_id = d.id
                    ),
                    leaf_map(leaf_id, ancestor_id) AS (
                        SELECT d.id, d.ancestor_id FROM desc d JOIN clusters c ON c.id = d.id
                        WHERE c.level = 0
                    )
                    SELECT
                        src.ancestor_id AS source_cluster_id,
                        tgt.ancestor_id AS target_cluster_id,
                        SUM(le.relation_count)    AS relation_count,
                        SUM(le.post_count)        AS post_count,
                        SUM(le.avg_score * le.post_count) / SUM(le.post_count) AS avg_score,
                        SUM(le.countercausal_count) AS countercausal_count
                    FROM leaf_edges le
                    JOIN leaf_map src ON src.leaf_id = le.source_cluster_id
                    JOIN leaf_map tgt ON tgt.leaf_id = le.target_cluster_id
                    WHERE src.ancestor_id != tgt.ancestor_id
                    GROUP BY src.ancestor_id, tgt.ancestor_id
                    HAVING SUM(le.post_count) >= ?
                    """,
                    cluster_ids + [min_post_count],
                ).fetchall()
                return [dict(r) for r in rows]

        return self._cached(cache_key, _query)

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
        _SORT_COLS = {"score": "p.score", "date": "p.created_utc", "comments": "p.num_comments"}
        sort_col = _SORT_COLS.get(sort, "p.score")
        assert sort_col in _SORT_COLS.values(), f"invalid sort: {sort!r}"

        with self._connect() as conn:
            sql = f"""
                WITH RECURSIVE desc(id) AS (
                    SELECT ? UNION ALL
                    SELECT c.id FROM clusters c JOIN desc d ON c.parent_id = d.id
                ),
                leaves(cluster_id) AS (
                    SELECT id FROM clusters WHERE id IN (SELECT id FROM desc) AND level = 0
                )
                SELECT p.id, p.title, p.score, p.num_comments, p.created_utc, p.permalink,
                       cr.cause_text, cr.effect_text,
                       (cr.relation_type = 'countercausal') AS is_countercausal
                FROM posts p
                JOIN (
                    SELECT cr2.post_id, MIN(cr2.id) AS min_cr_id
                    FROM causal_relations cr2
                    JOIN cluster_members cm2 ON cm2.relation_id = cr2.id
                    JOIN leaves ON leaves.cluster_id = cm2.cluster_id
                    GROUP BY cr2.post_id
                ) best ON best.post_id = p.id
                JOIN causal_relations cr ON cr.id = best.min_cr_id
                ORDER BY {sort_col} DESC
                LIMIT ? OFFSET ?
            """
            count_sql = """
                WITH RECURSIVE desc(id) AS (
                    SELECT ? UNION ALL
                    SELECT c.id FROM clusters c JOIN desc d ON c.parent_id = d.id
                ),
                leaves(cluster_id) AS (
                    SELECT id FROM clusters WHERE id IN (SELECT id FROM desc) AND level = 0
                )
                SELECT COUNT(DISTINCT p.id) FROM posts p
                JOIN causal_relations cr ON cr.post_id = p.id
                JOIN cluster_members cm ON cm.relation_id = cr.id
                JOIN leaves ON leaves.cluster_id = cm.cluster_id
            """
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
            src_leaves = self._get_leaf_cluster_ids(conn, source_cluster_id) or [source_cluster_id]
            tgt_leaves = self._get_leaf_cluster_ids(conn, target_cluster_id) or [target_cluster_id]

            src_ph = ",".join("?" * len(src_leaves))
            tgt_ph = ",".join("?" * len(tgt_leaves))
            params = src_leaves + tgt_leaves

            sql = f"""
                SELECT p.id, p.title, p.score, p.num_comments, p.created_utc, p.permalink,
                       cr.cause_text, cr.effect_text,
                       (cr.relation_type = 'countercausal') AS is_countercausal
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

    def get_all_relations_for_posts(self, post_ids: list[str]) -> dict[str, list[dict]]:
        """Return all causal relations for each post, keyed by post_id.

        Each relation includes cause/effect canonical descriptions and the
        leaf-level cluster IDs from cluster_members (or None when unclustered).
        """
        if not post_ids:
            return {}
        ph = ",".join("?" * len(post_ids))
        sql = f"""
            SELECT
                cr.post_id,
                cr.cause_text, cr.effect_text,
                cr.cause_canonical, cr.effect_canonical,
                (cr.relation_type = 'countercausal') AS is_countercausal,
                cm_cause.cluster_id  AS cause_cluster_id,
                cm_effect.cluster_id AS effect_cluster_id
            FROM causal_relations cr
            LEFT JOIN cluster_members cm_cause
                ON cm_cause.relation_id = cr.id AND cm_cause.role = 'cause'
            LEFT JOIN cluster_members cm_effect
                ON cm_effect.relation_id = cr.id AND cm_effect.role = 'effect'
            WHERE cr.post_id IN ({ph})
            ORDER BY cr.post_id, cr.id
        """
        result: dict[str, list[dict]] = {}
        with self._connect() as conn:
            for row in conn.execute(sql, post_ids).fetchall():
                d = dict(row)
                result.setdefault(d["post_id"], []).append(d)
        return result

    def get_post_by_id(self, post_id: str) -> dict | None:
        sql = """
            SELECT p.id, p.title, p.score, p.num_comments, p.created_utc, p.permalink,
                   cr.cause_text, cr.effect_text, cr.confidence,
                   (cr.relation_type = 'countercausal') AS is_countercausal
            FROM posts p
            LEFT JOIN causal_relations cr ON cr.post_id = p.id
            WHERE p.id = ?
            LIMIT 1
        """
        with self._connect() as conn:
            row = conn.execute(sql, (post_id,)).fetchone()
            return dict(row) if row else None
