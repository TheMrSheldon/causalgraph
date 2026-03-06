# Graph Database Format

The backend API (`api/`) reads from a SQLite database produced by the pipeline (`pipeline/`). The two components share **no code** — only this file format. The default path is `data/pipeline.db` (configured in `config.yaml` under `api.db_path`).

---

## Tables

### `posts`

One row per r/science submission that was classified as causal by Step 1.

| Column | Type | Description |
|--------|------|-------------|
| `id` | TEXT PK | Reddit post ID (e.g. `9tn4i5`) |
| `title` | TEXT | Post title |
| `score` | INTEGER | Reddit upvote score |
| `num_comments` | INTEGER | Comment count at collection time |
| `created_utc` | INTEGER | Unix timestamp |
| `author` | TEXT | Reddit username (may be null) |
| `url` | TEXT | External link URL (may be null) |
| `permalink` | TEXT | Reddit-relative URL, e.g. `/r/science/comments/…` |
| `subreddit` | TEXT | Always `science` in this corpus |

---

### `causal_relations`

One row per extracted (cause, effect) pair. A single post may contribute multiple rows.

| Column | Type | Description |
|--------|------|-------------|
| `id` | INTEGER PK | Auto-increment |
| `post_id` | TEXT FK→posts | Source post |
| `cause_text` | TEXT | Raw cause phrase as extracted |
| `effect_text` | TEXT | Raw effect phrase as extracted |
| `cause_norm` | TEXT | Lowercased/lemmatised cause (for deduplication) |
| `effect_norm` | TEXT | Lowercased/lemmatised effect |
| `confidence` | REAL | Extractor confidence in [0, 1] |
| `extractor` | TEXT | Extractor implementation name |
| `is_countercausal` | INTEGER | `1` if the relation is refuted/countercausal, `0` otherwise |
| `extracted_at` | TEXT | ISO-8601 timestamp |

---

### `clusters`

Hierarchy of event clusters produced by Step 3. Self-referencing tree.

| Column | Type | Description |
|--------|------|-------------|
| `id` | INTEGER PK | Auto-increment |
| `label` | TEXT | Human-readable cluster label (top keywords) |
| `level` | INTEGER | `0` = leaf, `1` = mid, `2` = top (highest) |
| `parent_id` | INTEGER FK→clusters | Parent cluster; `NULL` for top-level clusters |
| `member_count` | INTEGER | Total events in this cluster (including descendants) |
| `clusterer` | TEXT | Clusterer implementation name |
| `created_at` | TEXT | ISO-8601 timestamp |

**Invariants expected by the API:**
- Exactly one level must be the "top" level (highest `level` value). The API infers this with `MAX(level)`.
- Top-level clusters must have `parent_id = NULL`.
- `member_count` must be propagated bottom-up before the API is started (the pipeline runner does this).
- At least one cluster level must exist; the API returns an empty graph otherwise.

---

### `cluster_members`

Maps individual causal relation events to leaf clusters.

| Column | Type | Description |
|--------|------|-------------|
| `id` | INTEGER PK | Auto-increment |
| `cluster_id` | INTEGER FK→clusters | Must reference a **leaf** cluster (`level = 0`) |
| `relation_id` | INTEGER FK→causal_relations | Source relation |
| `role` | TEXT | `cause` or `effect` |
| `event_text` | TEXT | The event phrase (may differ from cause/effect text after normalisation) |

**Constraint:** only leaf (`level = 0`) clusters appear in this table. The API walks the `clusters` hierarchy to aggregate edges for higher levels.

---

### `leaf_edges`

Materialised edge table for fast graph queries. Must be rebuilt by the pipeline after Step 3 (see `pipeline/db.py → rebuild_leaf_edges()`).

| Column | Type | Description |
|--------|------|-------------|
| `source_cluster_id` | INTEGER PK (composite) | Cause-side **leaf** cluster |
| `target_cluster_id` | INTEGER PK (composite) | Effect-side **leaf** cluster |
| `relation_count` | INTEGER | Number of distinct causal relations |
| `post_count` | INTEGER | Number of distinct posts |
| `avg_score` | REAL | Average Reddit score across supporting posts |
| `countercausal_count` | INTEGER | Relations with `is_countercausal = 1` |

The API aggregates these rows up to higher cluster levels at query time.

---

### `pipeline_runs` (not read by the API)

Execution log used by the pipeline for idempotency. Ignored by the backend.

---

## Indexes

The pipeline creates the following indexes for query performance:

```sql
posts(score DESC), posts(created_utc)
causal_relations(post_id), causal_relations(cause_norm), causal_relations(effect_norm)
clusters(level), clusters(parent_id)
cluster_members(cluster_id), cluster_members(relation_id)
leaf_edges(source_cluster_id), leaf_edges(target_cluster_id)
```

---

## Producing a compatible database

Any process (not just this pipeline) can produce a database that the backend will serve, as long as it satisfies this schema. The minimum viable dataset is:

1. Populate `posts` with at least one row.
2. Populate `causal_relations` with at least one (cause, effect) pair referencing a post.
3. Create at least two `clusters` rows — one leaf (`level=0`) and one top-level (`level=N`, `parent_id=NULL`).
4. Populate `cluster_members` linking leaf clusters to relations.
5. Run the leaf-edge materialisation:

```sql
INSERT INTO leaf_edges (source_cluster_id, target_cluster_id,
                        relation_count, post_count, avg_score, countercausal_count)
SELECT cm_c.cluster_id, cm_e.cluster_id,
       COUNT(DISTINCT cr.id), COUNT(DISTINCT cr.post_id),
       AVG(p.score),
       SUM(CASE WHEN cr.is_countercausal = 1 THEN 1 ELSE 0 END)
FROM causal_relations cr
JOIN cluster_members cm_c ON cm_c.relation_id = cr.id AND cm_c.role = 'cause'
JOIN cluster_members cm_e ON cm_e.relation_id = cr.id AND cm_e.role = 'effect'
JOIN posts p ON p.id = cr.post_id
WHERE cm_c.cluster_id != cm_e.cluster_id
GROUP BY cm_c.cluster_id, cm_e.cluster_id;
```
