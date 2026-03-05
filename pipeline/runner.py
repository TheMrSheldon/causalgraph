"""
Pipeline orchestrator: runs all three steps end-to-end.

Usage (via CLI):
    python scripts/run_pipeline.py [--config config.yaml] [--step 1|2|3]
"""
from __future__ import annotations

import time
from typing import Literal

from pipeline.db import Database
from pipeline.parquet_reader import ParquetReader
from pipeline.protocols import CausalExtractor, CausalityIdentifier, HierarchyInferrer
from pipeline.registry import build_extractor, build_identifier, build_inferrer, load_config


def run_step1(
    identifier: CausalityIdentifier,
    reader: ParquetReader,
    db: Database,
    batch_size: int,
) -> int:
    """Identify causal posts and persist them. Returns count of causal posts."""
    print(f"[Step 1] Using '{identifier.name}' identifier")
    total_scanned = 0
    total_causal = 0
    t0 = time.perf_counter()

    run_id = db.start_run("identification", identifier.name, rows_in=0)
    try:
        for batch in reader.iter_batches(batch_size):
            causal = identifier.identify(batch)
            if causal:
                db.upsert_posts(causal)
            total_scanned += len(batch)
            total_causal += len(causal)
            if total_scanned % 50_000 == 0:
                elapsed = time.perf_counter() - t0
                pct = total_causal / total_scanned * 100 if total_scanned else 0
                print(
                    f"  Scanned {total_scanned:,}  |  Causal: {total_causal:,} ({pct:.1f}%)"
                    f"  |  {elapsed:.0f}s elapsed"
                )
        db.finish_run(run_id, rows_out=total_causal)
    except Exception as e:
        db.finish_run(run_id, rows_out=total_causal, status="failed", error=str(e))
        raise

    elapsed = time.perf_counter() - t0
    pct = total_causal / total_scanned * 100 if total_scanned else 0
    print(
        f"[Step 1] Done. Scanned {total_scanned:,} posts, "
        f"identified {total_causal:,} causal ({pct:.1f}%) in {elapsed:.1f}s"
    )
    return total_causal


def run_step2(
    extractor: CausalExtractor,
    db: Database,
) -> int:
    """Extract (cause, effect) pairs from causal posts. Returns relation count."""
    print(f"[Step 2] Using '{extractor.name}' extractor")
    causal_posts = db.upsert_posts([])  # just get count
    post_count = db.count_posts()
    print(f"[Step 2] Extracting from {post_count:,} causal posts...")

    # We need to iterate posts from DB; use a simple sqlite query
    import sqlite3
    conn = sqlite3.connect(db.db_path)
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        "SELECT id, title, score, num_comments, created_utc, author, url, permalink FROM posts"
    ).fetchall()
    conn.close()

    from pipeline.protocols import Post
    posts = [
        Post(
            id=r["id"],
            title=r["title"],
            score=r["score"],
            num_comments=r["num_comments"],
            created_utc=r["created_utc"],
            author=r["author"],
            url=r["url"],
            permalink=r["permalink"],
        )
        for r in rows
    ]

    run_id = db.start_run("extraction", extractor.name, rows_in=len(posts))
    all_relations = []
    try:
        t0 = time.perf_counter()
        for i, post in enumerate(posts):
            relations = extractor.extract(post)
            all_relations.extend(relations)
            if (i + 1) % 10_000 == 0:
                elapsed = time.perf_counter() - t0
                print(f"  Processed {i+1:,}/{len(posts):,} posts | {len(all_relations):,} relations | {elapsed:.0f}s")

        db.insert_relations(all_relations)
        db.finish_run(run_id, rows_out=len(all_relations))
    except Exception as e:
        db.finish_run(run_id, rows_out=len(all_relations), status="failed", error=str(e))
        raise

    elapsed = time.perf_counter() - t0
    print(f"[Step 2] Done. Extracted {len(all_relations):,} relations in {elapsed:.1f}s")
    return len(all_relations)


def run_step3(
    inferrer: HierarchyInferrer,
    db: Database,
) -> int:
    """Infer cluster hierarchy over all extracted events. Returns cluster count."""
    print(f"[Step 3] Using '{inferrer.name}' inferrer")
    relations = db.get_all_relations()
    relation_ids_raw = []

    import sqlite3
    conn = sqlite3.connect(db.db_path)
    relation_ids_raw = [r[0] for r in conn.execute("SELECT id FROM causal_relations ORDER BY id").fetchall()]
    conn.close()

    print(f"[Step 3] Building hierarchy over {len(relations):,} relations...")
    run_id = db.start_run("hierarchy", inferrer.name, rows_in=len(relations))

    try:
        t0 = time.perf_counter()
        db.clear_clusters()
        clusters, memberships = inferrer.infer(relations)

        # --- Insert clusters in two passes to resolve parent_id correctly ---
        # Pass 1: Insert all clusters with parent_id=None; collect DB ids.
        # Pass 2: Update parent_id from list-index to actual DB id.

        # Insert all with parent_id=None first (avoids FK ordering issues)
        import copy
        flat_clusters = [copy.copy(c) for c in clusters]
        original_parent_ids = [c.parent_id for c in flat_clusters]
        for c in flat_clusters:
            c.parent_id = None
        cluster_ids = db.insert_clusters(flat_clusters)

        # Build list-index → db-id mapping and update parent_ids
        idx_to_db_id = {i: db_id for i, db_id in enumerate(cluster_ids)}
        # Only update parent_ids for non-top-level clusters; skip any that would
        # create a self-reference or point to themselves (guards against clusterer bugs)
        parent_updates = [
            (cluster_ids[i], idx_to_db_id[orig])
            for i, orig in enumerate(original_parent_ids)
            if orig is not None and idx_to_db_id.get(orig) != cluster_ids[i]
        ]
        db.update_cluster_parent_ids(parent_updates)

        # Insert memberships (cluster_idx maps directly into cluster_ids list)
        db.insert_memberships(memberships, relation_ids_raw, cluster_ids)
        db.update_cluster_member_counts()
        n_edges = db.rebuild_leaf_edges()
        print(f"[Step 3] Materialized {n_edges:,} leaf-level edges")

        elapsed = time.perf_counter() - t0
        db.finish_run(run_id, rows_out=len(clusters))
        print(f"[Step 3] Done. Created {len(clusters):,} clusters in {elapsed:.1f}s")
    except Exception as e:
        db.finish_run(run_id, rows_out=0, status="failed", error=str(e))
        raise

    return len(clusters)


def run_all(config_path: str = "config.yaml", step: int | None = None) -> None:
    config = load_config(config_path)
    pipeline_cfg = config["pipeline"]
    db_path = pipeline_cfg["db_path"]
    batch_size = pipeline_cfg.get("batch_size", 5000)
    min_score = pipeline_cfg.get("min_score", 1)

    db = Database(db_path)
    db.initialize_schema()

    reader = ParquetReader(pipeline_cfg["parquet_path"], min_score=min_score)

    if step is None or step == 1:
        identifier = build_identifier(config)
        run_step1(identifier, reader, db, batch_size)

    if step is None or step == 2:
        extractor = build_extractor(config)
        run_step2(extractor, db)

    if step is None or step == 3:
        inferrer = build_inferrer(config)
        run_step3(inferrer, db)

    print("[Pipeline] All steps complete.")
