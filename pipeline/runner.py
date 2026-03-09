"""
Pipeline orchestrator: runs all four steps end-to-end.

Usage:
    python -m pipeline.runner [--config config.yaml] [--step 1|2|3|4]
"""
from __future__ import annotations

import time
from typing import Literal

from pipeline.db import Database
from pipeline.parquet_reader import ParquetReader
import copy
import importlib

import yaml

from pipeline.protocols import (
    CausalityDetector,
    CausalExtractor,
    EventCanonizer,
    HierarchyInferrer,
)


def _load_config(path: str = "pipeline.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def _build(step_cfg: dict, protocol_cls):
    """Instantiate a pipeline step from a qualified class name in config."""
    cfg = copy.deepcopy(step_cfg)
    qualified = cfg.pop("implementation")
    module_path, _, class_name = qualified.rpartition(".")
    cls = getattr(importlib.import_module(module_path), class_name)
    obj = cls(**cfg)
    if not isinstance(obj, protocol_cls):
        raise TypeError(f"{type(obj).__name__} does not satisfy {protocol_cls.__name__}")
    return obj


def run_step1(
    detector: CausalityDetector,
    reader: ParquetReader,
    db: Database,
    batch_size: int,
) -> int:
    """Detect causal posts and persist them. Returns count of causal posts."""
    print(f"[Step 1] Using '{detector.name}' detector")
    total_scanned = 0
    total_causal = 0
    t0 = time.perf_counter()

    run_id = db.start_run("detection", detector.name, rows_in=0)
    try:
        for batch in reader.iter_batches(batch_size):
            causal = detector.detect(batch)
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
        f"detected {total_causal:,} causal ({pct:.1f}%) in {elapsed:.1f}s"
    )
    return total_causal


def run_step2(
    extractor: CausalExtractor,
    db: Database,
) -> int:
    """Extract (cause, effect) pairs from causal posts. Returns relation count."""
    print(f"[Step 2] Using '{extractor.name}' extractor")
    post_count = db.count_posts()
    print(f"[Step 2] Extracting from {post_count:,} causal posts...")

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
            for r in relations:
                r.post_title = post.title
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


def _span_indices(text: str, span: str) -> tuple[int, int]:
    """Return (start, end) of ``span`` inside ``text`` (case-insensitive).
    Falls back to (0, len(span)) when not found so callers always get a valid range."""
    idx = text.lower().find(span.lower())
    if idx == -1:
        return (0, len(span))
    return (idx, idx + len(span))


def run_step3(
    canonizer: EventCanonizer,
    db: Database,
) -> int:
    """Canonize event descriptions. Returns count of canonized relations."""
    import dataclasses
    import sqlite3

    print(f"[Step 3] Using '{canonizer.name}' canonizer")
    relations = db.get_all_relations()

    conn = sqlite3.connect(db.db_path)
    relation_ids = [r[0] for r in conn.execute("SELECT id FROM causal_relations ORDER BY id").fetchall()]
    conn.close()

    # Build flat (text, (start, end)) list: cause then effect for each relation.
    span_inputs: list[tuple[str, tuple[int, int]]] = []
    for r in relations:
        ctx = r.post_title or r.cause_text  # use title as context when available
        span_inputs.append((ctx, _span_indices(ctx, r.cause_text)))
        ctx = r.post_title or r.effect_text
        span_inputs.append((ctx, _span_indices(ctx, r.effect_text)))

    print(f"[Step 3] Canonizing {len(relations):,} relations ({len(span_inputs):,} spans)…")
    run_id = db.start_run("canonization", canonizer.name, rows_in=len(relations))
    try:
        t0 = time.perf_counter()
        canonical_strings = canonizer.canonize(span_inputs)
        # Map flat results back: index i*2 = cause, i*2+1 = effect
        canonized = [
            dataclasses.replace(
                r,
                cause_canonical=canonical_strings[i * 2],
                effect_canonical=canonical_strings[i * 2 + 1],
            )
            for i, r in enumerate(relations)
        ]
        db.update_canonical_fields(canonized, relation_ids)
        db.finish_run(run_id, rows_out=len(canonized))
        elapsed = time.perf_counter() - t0
        print(f"[Step 3] Done. Canonized {len(canonized):,} relations in {elapsed:.1f}s")
    except Exception as e:
        db.finish_run(run_id, rows_out=0, status="failed", error=str(e))
        raise

    return len(relations)


def run_step4(
    inferrer: HierarchyInferrer,
    db: Database,
) -> int:
    """Infer cluster hierarchy over all extracted events. Returns cluster count."""
    print(f"[Step 4] Using '{inferrer.name}' inferrer")
    relations = db.get_all_relations()

    import sqlite3
    conn = sqlite3.connect(db.db_path)
    relation_ids_raw = [r[0] for r in conn.execute("SELECT id FROM causal_relations ORDER BY id").fetchall()]
    conn.close()

    print(f"[Step 4] Building hierarchy over {len(relations):,} relations...")
    run_id = db.start_run("hierarchy", inferrer.name, rows_in=len(relations))

    try:
        t0 = time.perf_counter()
        db.clear_clusters()
        clusters, memberships = inferrer.infer(relations)

        # Insert clusters in two passes to resolve parent_id correctly
        import copy
        flat_clusters = [copy.copy(c) for c in clusters]
        original_parent_ids = [c.parent_id for c in flat_clusters]
        for c in flat_clusters:
            c.parent_id = None
        cluster_ids = db.insert_clusters(flat_clusters)

        idx_to_db_id = {i: db_id for i, db_id in enumerate(cluster_ids)}
        parent_updates = [
            (cluster_ids[i], idx_to_db_id[orig])
            for i, orig in enumerate(original_parent_ids)
            if orig is not None and idx_to_db_id.get(orig) != cluster_ids[i]
        ]
        db.update_cluster_parent_ids(parent_updates)

        db.insert_memberships(memberships, relation_ids_raw, cluster_ids)
        db.update_cluster_member_counts()
        n_edges = db.rebuild_leaf_edges()
        print(f"[Step 4] Materialized {n_edges:,} leaf-level edges")

        elapsed = time.perf_counter() - t0
        db.finish_run(run_id, rows_out=len(clusters))
        print(f"[Step 4] Done. Created {len(clusters):,} clusters in {elapsed:.1f}s")
    except Exception as e:
        db.finish_run(run_id, rows_out=0, status="failed", error=str(e))
        raise

    return len(clusters)


def run_all(config_path: str = "pipeline.yaml", step: int | None = None) -> None:
    config = _load_config(config_path)
    db_path = config["db_path"]
    batch_size = config.get("batch_size", 5000)
    min_score = config.get("min_score", 1)

    db = Database(db_path)
    db.initialize_schema()

    reader = ParquetReader(config["parquet_path"], min_score=min_score)

    if step is None or step == 1:
        detector = _build(config["step1_detection"], CausalityDetector)
        run_step1(detector, reader, db, batch_size)

    if step is None or step == 2:
        extractor = _build(config["step2_extraction"], CausalExtractor)
        run_step2(extractor, db)

    if step is None or step == 3:
        canonizer = _build(config["step3_canonization"], EventCanonizer)
        run_step3(canonizer, db)

    if step is None or step == 4:
        inferrer = _build(config["step4_hierarchy"], HierarchyInferrer)
        run_step4(inferrer, db)

    print("[Pipeline] All steps complete.")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="r/science causal relationship extraction pipeline"
    )
    parser.add_argument("--config", default="pipeline.yaml", help="Path to pipeline.yaml")
    parser.add_argument(
        "--step",
        type=int,
        choices=[1, 2, 3, 4],
        default=None,
        help="Run only a specific step (1=detect, 2=extract, 3=canonize, 4=cluster). Omit to run all.",
    )
    args = parser.parse_args()
    run_all(config_path=args.config, step=args.step)
