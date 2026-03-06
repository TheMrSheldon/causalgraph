"""Round-trip tests for the SQLite DAL."""
import pytest

from pipeline.protocols import CausalRelation, EventCluster, Post


def test_initialize_schema(tmp_db):
    # Should not raise; calling twice is idempotent
    tmp_db.initialize_schema()


def test_upsert_and_count_posts(tmp_db, sample_causal_posts):
    n = tmp_db.upsert_posts(sample_causal_posts)
    assert n == len(sample_causal_posts)
    assert tmp_db.count_posts() == len(sample_causal_posts)


def test_upsert_posts_idempotent(tmp_db, sample_causal_posts):
    tmp_db.upsert_posts(sample_causal_posts)
    tmp_db.upsert_posts(sample_causal_posts)  # INSERT OR REPLACE
    assert tmp_db.count_posts() == len(sample_causal_posts)


def test_insert_and_count_relations(tmp_db, sample_causal_posts, sample_relations):
    tmp_db.upsert_posts(sample_causal_posts)
    ids = tmp_db.insert_relations(sample_relations)
    assert len(ids) == len(sample_relations)
    assert tmp_db.count_relations() == len(sample_relations)


def test_insert_clusters(tmp_db):
    clusters = [
        EventCluster(label="health", level=2, parent_id=None, member_count=100, clusterer="test"),
        EventCluster(label="smoking", level=1, parent_id=None, member_count=30, clusterer="test"),
        EventCluster(label="cancer", level=0, parent_id=None, member_count=10, clusterer="test"),
    ]
    ids = tmp_db.insert_clusters(clusters)
    assert len(ids) == 3
    assert all(isinstance(i, int) for i in ids)


def test_get_clusters_at_level(tmp_db):
    clusters = [
        EventCluster(label="top", level=2, parent_id=None, member_count=50, clusterer="test"),
        EventCluster(label="mid", level=1, parent_id=None, member_count=20, clusterer="test"),
    ]
    tmp_db.insert_clusters(clusters)
    level2 = tmp_db.get_clusters_at_level(2)
    assert len(level2) == 1
    assert level2[0]["label"] == "top"


def test_get_cluster_by_id(tmp_db):
    ids = tmp_db.insert_clusters([
        EventCluster(label="mynode", level=0, parent_id=None, member_count=5, clusterer="test")
    ])
    result = tmp_db.get_cluster_by_id(ids[0])
    assert result is not None
    assert result["label"] == "mynode"


def test_get_cluster_by_id_missing(tmp_db):
    assert tmp_db.get_cluster_by_id(99999) is None


def test_get_children(tmp_db):
    parent_ids = tmp_db.insert_clusters([
        EventCluster(label="parent", level=1, parent_id=None, member_count=10, clusterer="test")
    ])
    parent_db_id = parent_ids[0]
    # We need to simulate parent_id resolution (in real pipeline, cluster list indices are resolved)
    import sqlite3
    conn = sqlite3.connect(tmp_db.db_path)
    conn.execute(
        "INSERT INTO clusters (label, level, parent_id, member_count, clusterer) VALUES (?,?,?,?,?)",
        ("child1", 0, parent_db_id, 5, "test"),
    )
    conn.commit()
    conn.close()

    children = tmp_db.get_children(parent_db_id)
    assert len(children) == 1
    assert children[0]["label"] == "child1"


def test_get_post_by_id(tmp_db, sample_causal_posts, sample_relations):
    tmp_db.upsert_posts(sample_causal_posts)
    tmp_db.insert_relations(sample_relations)
    result = tmp_db.get_post_by_id("abc1")
    assert result is not None
    assert result["title"] == sample_causal_posts[0].title


def test_get_post_by_id_missing(tmp_db):
    assert tmp_db.get_post_by_id("nonexistent") is None


def test_pipeline_run_tracking(tmp_db):
    run_id = tmp_db.start_run("detection", "regex", rows_in=1000)
    assert isinstance(run_id, int)
    tmp_db.finish_run(run_id, rows_out=150)
