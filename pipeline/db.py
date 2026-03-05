"""
SQLite Data Access Layer.

Schema:
  posts            – causal r/science posts (from Step 1)
  causal_relations – extracted (cause, effect) pairs (from Step 2)
  clusters         – hierarchy nodes (from Step 3)
  cluster_members  – event→cluster assignments (from Step 3)
  graph_edges      – VIEW: cluster-level edge aggregation
  pipeline_runs    – execution log for idempotency
"""
from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Generator

from pipeline.protocols import CausalRelation, EventCluster, Post

SCHEMA_SQL = """
PRAGMA journal_mode=WAL;
PRAGMA foreign_keys=ON;

CREATE TABLE IF NOT EXISTS posts (
    id            TEXT PRIMARY KEY,
    title         TEXT NOT NULL,
    score         INTEGER NOT NULL DEFAULT 0,
    num_comments  INTEGER NOT NULL DEFAULT 0,
    created_utc   INTEGER NOT NULL,
    author        TEXT,
    url           TEXT,
    permalink     TEXT,
    subreddit     TEXT NOT NULL DEFAULT 'science'
);

CREATE INDEX IF NOT EXISTS posts_score_idx   ON posts(score DESC);
CREATE INDEX IF NOT EXISTS posts_created_idx ON posts(created_utc);

CREATE TABLE IF NOT EXISTS causal_relations (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    post_id       TEXT NOT NULL REFERENCES posts(id) ON DELETE CASCADE,
    cause_text    TEXT NOT NULL,
    effect_text   TEXT NOT NULL,
    cause_norm    TEXT NOT NULL,
    effect_norm   TEXT NOT NULL,
    confidence    REAL NOT NULL DEFAULT 1.0,
    extractor     TEXT NOT NULL DEFAULT '',
    extracted_at  TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS relations_post_idx    ON causal_relations(post_id);
CREATE INDEX IF NOT EXISTS relations_cause_idx   ON causal_relations(cause_norm);
CREATE INDEX IF NOT EXISTS relations_effect_idx  ON causal_relations(effect_norm);

CREATE TABLE IF NOT EXISTS clusters (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    label         TEXT NOT NULL,
    level         INTEGER NOT NULL,
    parent_id     INTEGER REFERENCES clusters(id) ON DELETE SET NULL,
    member_count  INTEGER NOT NULL DEFAULT 0,
    clusterer     TEXT NOT NULL DEFAULT '',
    created_at    TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS clusters_level_idx  ON clusters(level);
CREATE INDEX IF NOT EXISTS clusters_parent_idx ON clusters(parent_id);

CREATE TABLE IF NOT EXISTS cluster_members (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    cluster_id  INTEGER NOT NULL REFERENCES clusters(id) ON DELETE CASCADE,
    relation_id INTEGER NOT NULL REFERENCES causal_relations(id) ON DELETE CASCADE,
    role        TEXT NOT NULL CHECK(role IN ('cause', 'effect')),
    event_text  TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS members_cluster_idx  ON cluster_members(cluster_id);
CREATE INDEX IF NOT EXISTS members_relation_idx ON cluster_members(relation_id);

CREATE TABLE IF NOT EXISTS pipeline_runs (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    step        TEXT NOT NULL,
    implementer TEXT NOT NULL,
    started_at  TEXT NOT NULL DEFAULT (datetime('now')),
    finished_at TEXT,
    rows_in     INTEGER,
    rows_out    INTEGER,
    status      TEXT NOT NULL DEFAULT 'running'
              CHECK(status IN ('running', 'done', 'failed')),
    error_msg   TEXT
);

-- Materialized leaf-level edges (populated by rebuild_leaf_edges())
CREATE TABLE IF NOT EXISTS leaf_edges (
    source_cluster_id  INTEGER NOT NULL,
    target_cluster_id  INTEGER NOT NULL,
    relation_count     INTEGER NOT NULL DEFAULT 0,
    post_count         INTEGER NOT NULL DEFAULT 0,
    avg_score          REAL NOT NULL DEFAULT 0,
    PRIMARY KEY (source_cluster_id, target_cluster_id)
);
CREATE INDEX IF NOT EXISTS leaf_edges_src ON leaf_edges(source_cluster_id);
CREATE INDEX IF NOT EXISTS leaf_edges_tgt ON leaf_edges(target_cluster_id);
"""

# SQL to (re)build the leaf_edges table — run once after Step 3
_REBUILD_LEAF_EDGES_SQL = """
DELETE FROM leaf_edges;
INSERT INTO leaf_edges (source_cluster_id, target_cluster_id, relation_count, post_count, avg_score)
SELECT
    cm_cause.cluster_id,
    cm_effect.cluster_id,
    COUNT(DISTINCT cr.id),
    COUNT(DISTINCT cr.post_id),
    AVG(p.score)
FROM causal_relations cr
JOIN cluster_members cm_cause
  ON cm_cause.relation_id = cr.id AND cm_cause.role = 'cause'
JOIN cluster_members cm_effect
  ON cm_effect.relation_id = cr.id AND cm_effect.role = 'effect'
JOIN posts p ON p.id = cr.post_id
WHERE cm_cause.cluster_id != cm_effect.cluster_id
GROUP BY cm_cause.cluster_id, cm_effect.cluster_id;
"""


class Database:
    def __init__(self, db_path: str) -> None:
        self.db_path = db_path
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)

    @contextmanager
    def _connect(self) -> Generator[sqlite3.Connection, None, None]:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def initialize_schema(self) -> None:
        with self._connect() as conn:
            conn.executescript(SCHEMA_SQL)

    # ------------------------------------------------------------------
    # Posts
    # ------------------------------------------------------------------

    def upsert_posts(self, posts: list[Post]) -> int:
        if not posts:
            return 0
        sql = """
            INSERT OR REPLACE INTO posts
                (id, title, score, num_comments, created_utc, author, url, permalink)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """
        rows = [
            (p.id, p.title, p.score, p.num_comments, p.created_utc, p.author, p.url, p.permalink)
            for p in posts
        ]
        with self._connect() as conn:
            conn.executemany(sql, rows)
        return len(rows)

    def count_posts(self) -> int:
        with self._connect() as conn:
            return conn.execute("SELECT COUNT(*) FROM posts").fetchone()[0]

    # ------------------------------------------------------------------
    # Causal relations
    # ------------------------------------------------------------------

    def insert_relations(self, relations: list[CausalRelation]) -> list[int]:
        if not relations:
            return []
        sql = """
            INSERT INTO causal_relations
                (post_id, cause_text, effect_text, cause_norm, effect_norm, confidence, extractor)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """
        ids: list[int] = []
        with self._connect() as conn:
            for r in relations:
                cur = conn.execute(sql, (
                    r.post_id, r.cause_text, r.effect_text,
                    r.cause_norm, r.effect_norm, r.confidence, r.extractor,
                ))
                ids.append(cur.lastrowid)
        return ids

    def count_relations(self) -> int:
        with self._connect() as conn:
            return conn.execute("SELECT COUNT(*) FROM causal_relations").fetchone()[0]

    def get_all_relations(self) -> list[CausalRelation]:
        sql = """
            SELECT post_id, cause_text, effect_text, cause_norm, effect_norm, confidence, extractor
            FROM causal_relations
        """
        with self._connect() as conn:
            rows = conn.execute(sql).fetchall()
        return [
            CausalRelation(
                post_id=r["post_id"],
                cause_text=r["cause_text"],
                effect_text=r["effect_text"],
                cause_norm=r["cause_norm"],
                effect_norm=r["effect_norm"],
                confidence=r["confidence"],
                extractor=r["extractor"],
            )
            for r in rows
        ]

    # ------------------------------------------------------------------
    # Clusters
    # ------------------------------------------------------------------

    def clear_clusters(self) -> None:
        """Drop and recreate cluster data (for re-running Step 3)."""
        with self._connect() as conn:
            conn.execute("DELETE FROM leaf_edges")
            conn.execute("DELETE FROM cluster_members")
            conn.execute("DELETE FROM clusters")

    def rebuild_leaf_edges(self) -> int:
        """Materialize the leaf-level edge table. Call once after Step 3."""
        with self._connect() as conn:
            conn.executescript(_REBUILD_LEAF_EDGES_SQL)
            count = conn.execute("SELECT COUNT(*) FROM leaf_edges").fetchone()[0]
        return count

    def insert_clusters(self, clusters: list[EventCluster]) -> list[int]:
        """Insert clusters in level order (top → leaf) for FK integrity."""
        if not clusters:
            return []
        sql = """
            INSERT INTO clusters (label, level, parent_id, member_count, clusterer)
            VALUES (?, ?, ?, ?, ?)
        """
        ids: list[int] = []
        with self._connect() as conn:
            for c in clusters:
                cur = conn.execute(sql, (c.label, c.level, c.parent_id, c.member_count, c.clusterer))
                ids.append(cur.lastrowid)
        return ids

    def insert_memberships(
        self,
        memberships: list[tuple[int, int, str, str]],
        relation_ids: list[int],
        cluster_ids: list[int],
    ) -> None:
        """
        Args:
            memberships:  [(relation_index, cluster_index, role, event_text), ...]
            relation_ids: DB IDs returned by insert_relations, indexed by relation_index
            cluster_ids:  DB IDs returned by insert_clusters, indexed by cluster_index
        """
        if not memberships:
            return
        sql = """
            INSERT INTO cluster_members (cluster_id, relation_id, role, event_text)
            VALUES (?, ?, ?, ?)
        """
        rows = [
            (cluster_ids[cluster_idx], relation_ids[relation_idx], role, event_text)
            for relation_idx, cluster_idx, role, event_text in memberships
        ]
        with self._connect() as conn:
            conn.executemany(sql, rows)

    def update_cluster_parent_ids(self, pairs: list[tuple[int, int]]) -> None:
        """Set parent_id for clusters. pairs = [(cluster_db_id, parent_db_id), ...]"""
        with self._connect() as conn:
            conn.executemany(
                "UPDATE clusters SET parent_id = ? WHERE id = ?",
                [(parent_id, cluster_id) for cluster_id, parent_id in pairs],
            )

    def update_cluster_member_counts(self) -> None:
        """
        Update member_count for all clusters bottom-up:
        1. Leaf counts = direct cluster_members rows.
        2. Each parent accumulates its children's counts.
        All done in Python to avoid recursive SQL performance issues.
        """
        with self._connect() as conn:
            # Direct member counts for leaf clusters
            rows = conn.execute(
                "SELECT cluster_id, COUNT(*) FROM cluster_members GROUP BY cluster_id"
            ).fetchall()
            counts: dict[int, int] = {r[0]: r[1] for r in rows}

            # Get all clusters sorted leaf → top (by level ascending)
            clusters = conn.execute(
                "SELECT id, parent_id, level FROM clusters ORDER BY level ASC"
            ).fetchall()

            # Bottom-up propagation
            for cluster_id, parent_id, _ in clusters:
                c = counts.get(cluster_id, 0)
                if parent_id is not None:
                    counts[parent_id] = counts.get(parent_id, 0) + c

            # Batch update
            conn.executemany(
                "UPDATE clusters SET member_count = ? WHERE id = ?",
                [(cnt, cid) for cid, cnt in counts.items()],
            )

    # ------------------------------------------------------------------
    # Graph queries (used by API)
    # ------------------------------------------------------------------

    def get_clusters_at_level(self, level: int) -> list[dict]:
        sql = """
            SELECT id, label, level, parent_id, member_count, clusterer
            FROM clusters WHERE level = ?
            ORDER BY member_count DESC
        """
        with self._connect() as conn:
            return [dict(r) for r in conn.execute(sql, (level,)).fetchall()]

    def get_cluster_by_id(self, cluster_id: int) -> dict | None:
        sql = """
            SELECT id, label, level, parent_id, member_count, clusterer
            FROM clusters WHERE id = ?
        """
        with self._connect() as conn:
            row = conn.execute(sql, (cluster_id,)).fetchone()
            return dict(row) if row else None

    def get_children(self, cluster_id: int) -> list[dict]:
        sql = """
            SELECT id, label, level, parent_id, member_count, clusterer
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

    def _get_descendant_leaf_ids(self, conn, ancestor_ids: list[int]) -> dict[int, int]:
        """
        Returns {leaf_cluster_id: ancestor_id} mapping for all descendants
        of the given ancestor cluster IDs. Uses Python-level BFS (avoids
        recursive SQL issues on Python 3.14).
        """
        # Load all clusters once
        all_clusters = conn.execute(
            "SELECT id, parent_id FROM clusters"
        ).fetchall()
        parent_of: dict[int, int | None] = {r[0]: r[1] for r in all_clusters}

        # Build ancestor set for quick lookup
        ancestor_set = set(ancestor_ids)

        # BFS downward: for each cluster find its root ancestor among ancestor_set
        leaf_to_ancestor: dict[int, int] = {}
        for cid, parent_id in parent_of.items():
            # Walk up to find if this cluster is under one of our ancestors
            cur = cid
            visited: set[int] = set()
            while cur is not None:
                if cur in visited:
                    break  # cycle guard
                visited.add(cur)
                if cur in ancestor_set:
                    leaf_to_ancestor[cid] = cur
                    break
                cur = parent_of.get(cur)

        return leaf_to_ancestor

    def get_edges(
        self,
        cluster_ids: list[int] | None = None,
        min_post_count: int = 1,
    ) -> list[dict]:
        """
        Return edges between clusters. When cluster_ids are provided,
        traverses the hierarchy to find leaf-level members and aggregates
        edges back up to the requested cluster level.
        """
        with self._connect() as conn:
            if cluster_ids is None:
                rows = conn.execute(
                    "SELECT source_cluster_id, target_cluster_id, relation_count, post_count, avg_score FROM leaf_edges WHERE post_count >= ?",
                    [min_post_count],
                ).fetchall()
                return [dict(r) for r in rows]

            # Map leaf clusters → their ancestor in cluster_ids
            leaf_to_ancestor = self._get_descendant_leaf_ids(conn, cluster_ids)
            if not leaf_to_ancestor:
                return []

            # Get all leaf-level edges from the materialized table (fast)
            leaf_ids = list(leaf_to_ancestor.keys())
            placeholders = ",".join("?" * len(leaf_ids))
            rows = conn.execute(
                f"""SELECT source_cluster_id, target_cluster_id, relation_count, post_count, avg_score
                    FROM leaf_edges
                    WHERE source_cluster_id IN ({placeholders})
                      AND target_cluster_id IN ({placeholders})""",
                leaf_ids + leaf_ids,
            ).fetchall()

            # Aggregate up to ancestor level
            agg: dict[tuple[int, int], dict] = {}
            for r in rows:
                src_anc = leaf_to_ancestor.get(r[0])
                tgt_anc = leaf_to_ancestor.get(r[1])
                if src_anc is None or tgt_anc is None or src_anc == tgt_anc:
                    continue
                key = (src_anc, tgt_anc)
                if key not in agg:
                    agg[key] = {"source_cluster_id": src_anc, "target_cluster_id": tgt_anc,
                                "relation_count": 0, "post_count": 0, "avg_score_sum": 0.0, "n": 0}
                agg[key]["relation_count"] += r[2]
                agg[key]["post_count"] += r[3]
                agg[key]["avg_score_sum"] += (r[4] or 0.0) * r[3]
                agg[key]["n"] += r[3]

            result = []
            for v in agg.values():
                if v["post_count"] >= min_post_count:
                    result.append({
                        "source_cluster_id": v["source_cluster_id"],
                        "target_cluster_id": v["target_cluster_id"],
                        "relation_count": v["relation_count"],
                        "post_count": v["post_count"],
                        "avg_score": v["avg_score_sum"] / v["n"] if v["n"] > 0 else 0.0,
                    })
            return result

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
            SELECT DISTINCT p.id, p.title, p.score, p.num_comments, p.created_utc, p.permalink
            FROM posts p
            JOIN causal_relations cr ON cr.post_id = p.id
            JOIN cluster_members cm ON cm.relation_id = cr.id
            WHERE cm.cluster_id = ?
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
            # Resolve to leaf cluster IDs (handles both leaf and higher-level clusters)
            leaf_map = self._get_descendant_leaf_ids(conn, [source_cluster_id, target_cluster_id])
            src_leaves = [cid for cid, anc in leaf_map.items() if anc == source_cluster_id]
            tgt_leaves = [cid for cid, anc in leaf_map.items() if anc == target_cluster_id]

            # If the cluster IS already a leaf, include itself
            if not src_leaves:
                src_leaves = [source_cluster_id]
            if not tgt_leaves:
                tgt_leaves = [target_cluster_id]

            src_ph = ",".join("?" * len(src_leaves))
            tgt_ph = ",".join("?" * len(tgt_leaves))
            params = src_leaves + tgt_leaves

            sql = f"""
                SELECT DISTINCT p.id, p.title, p.score, p.num_comments, p.created_utc, p.permalink
                FROM posts p
                JOIN causal_relations cr ON cr.post_id = p.id
                JOIN cluster_members cm_cause
                  ON cm_cause.relation_id = cr.id AND cm_cause.role = 'cause'
                JOIN cluster_members cm_effect
                  ON cm_effect.relation_id = cr.id AND cm_effect.role = 'effect'
                WHERE cm_cause.cluster_id IN ({src_ph}) AND cm_effect.cluster_id IN ({tgt_ph})
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
                   cr.cause_text, cr.effect_text, cr.confidence
            FROM posts p
            LEFT JOIN causal_relations cr ON cr.post_id = p.id
            WHERE p.id = ?
            LIMIT 1
        """
        with self._connect() as conn:
            row = conn.execute(sql, (post_id,)).fetchone()
            return dict(row) if row else None

    def get_top_events_for_cluster(self, cluster_id: int, n: int = 10) -> list[str]:
        sql = """
            SELECT event_text, COUNT(*) AS cnt
            FROM cluster_members WHERE cluster_id = ?
            GROUP BY event_text ORDER BY cnt DESC LIMIT ?
        """
        with self._connect() as conn:
            return [r["event_text"] for r in conn.execute(sql, (cluster_id, n)).fetchall()]

    # ------------------------------------------------------------------
    # Pipeline run tracking
    # ------------------------------------------------------------------

    def start_run(self, step: str, implementer: str, rows_in: int) -> int:
        sql = """
            INSERT INTO pipeline_runs (step, implementer, rows_in)
            VALUES (?, ?, ?)
        """
        with self._connect() as conn:
            cur = conn.execute(sql, (step, implementer, rows_in))
            return cur.lastrowid

    def finish_run(self, run_id: int, rows_out: int, status: str = "done", error: str = "") -> None:
        sql = """
            UPDATE pipeline_runs
            SET finished_at = datetime('now'), rows_out = ?, status = ?, error_msg = ?
            WHERE id = ?
        """
        with self._connect() as conn:
            conn.execute(sql, (rows_out, status, error or None, run_id))
