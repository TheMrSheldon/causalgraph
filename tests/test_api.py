"""
Integration tests for the FastAPI application.
Uses an in-memory SQLite DB injected via dependency override.
"""
import pytest
from fastapi.testclient import TestClient

from api.dependencies import get_db
from api.main import app
from pipeline.db import Database
from pipeline.protocols import CausalRelation, EventCluster, Post


@pytest.fixture
def db_with_data(tmp_path):
    db = Database(str(tmp_path / "api_test.db"))
    db.initialize_schema()

    posts = [
        Post(id="p1", title="Smoking causes lung cancer", score=100, num_comments=50, created_utc=1700000000),
        Post(id="p2", title="Exercise leads to better mental health", score=200, num_comments=80, created_utc=1700000001),
    ]
    db.upsert_posts(posts)

    relations = [
        CausalRelation(post_id="p1", cause_text="Smoking", effect_text="lung cancer",
                       cause_norm="smoking", effect_norm="lung cancer", confidence=1.0, extractor="test"),
        CausalRelation(post_id="p2", cause_text="Exercise", effect_text="mental health",
                       cause_norm="exercise", effect_norm="mental health", confidence=1.0, extractor="test"),
    ]
    relation_ids = db.insert_relations(relations)

    clusters = [
        EventCluster(label="health effects", level=2, parent_id=None, member_count=4, clusterer="test"),
        EventCluster(label="lung disease", level=1, parent_id=None, member_count=2, clusterer="test"),
        EventCluster(label="fitness", level=1, parent_id=None, member_count=2, clusterer="test"),
    ]
    cluster_ids = db.insert_clusters(clusters)

    # Set parent_id for level-1 clusters → level-2 cluster
    import sqlite3
    conn = sqlite3.connect(db.db_path)
    conn.execute(f"UPDATE clusters SET parent_id = {cluster_ids[0]} WHERE id IN ({cluster_ids[1]}, {cluster_ids[2]})")
    conn.commit()
    conn.close()

    memberships = [
        (0, 1, "cause", "smoking"),
        (0, 0, "effect", "lung cancer"),
        (1, 2, "cause", "exercise"),
        (1, 0, "effect", "mental health"),
    ]
    db.insert_memberships(memberships, relation_ids, cluster_ids)
    db.update_cluster_member_counts()

    return db


@pytest.fixture
def client(db_with_data):
    app.dependency_overrides[get_db] = lambda: db_with_data
    with TestClient(app) as c:
        yield c
    app.dependency_overrides.clear()


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok"}


def test_graph_returns_nodes_and_edges(client):
    r = client.get("/api/graph?level=2")
    assert r.status_code == 200
    data = r.json()
    assert "nodes" in data
    assert "edges" in data
    assert len(data["nodes"]) >= 1


def test_graph_levels(client):
    r = client.get("/api/graph/levels")
    assert r.status_code == 200
    data = r.json()
    assert "levels" in data
    assert "counts" in data


def test_cluster_detail(client, db_with_data):
    clusters = db_with_data.get_clusters_at_level(2)
    cluster_id = clusters[0]["id"]
    r = client.get(f"/api/clusters/{cluster_id}")
    assert r.status_code == 200
    data = r.json()
    assert data["cluster"]["id"] == cluster_id
    assert "children" in data
    assert "top_events" in data
    assert "posts" in data


def test_cluster_not_found(client):
    r = client.get("/api/clusters/99999")
    assert r.status_code == 404


def test_cluster_expand(client, db_with_data):
    clusters = db_with_data.get_clusters_at_level(2)
    cluster_id = clusters[0]["id"]
    r = client.get(f"/api/clusters/{cluster_id}/expand")
    assert r.status_code == 200
    data = r.json()
    assert "nodes" in data
    assert "edges" in data


def test_cluster_posts(client, db_with_data):
    clusters = db_with_data.get_clusters_at_level(2)
    cluster_id = clusters[0]["id"]
    r = client.get(f"/api/clusters/{cluster_id}/posts")
    assert r.status_code == 200
    data = r.json()
    assert "posts" in data
    assert "total" in data


def test_posts_for_edge(client, db_with_data):
    clusters = db_with_data.get_clusters_at_level(1)
    if len(clusters) >= 2:
        r = client.get(
            f"/api/posts?source_cluster_id={clusters[0]['id']}&target_cluster_id={clusters[1]['id']}"
        )
        assert r.status_code == 200
        assert "posts" in r.json()


def test_post_detail(client):
    r = client.get("/api/posts/p1")
    assert r.status_code == 200
    data = r.json()
    assert data["id"] == "p1"
    assert "cause_text" in data


def test_post_not_found(client):
    r = client.get("/api/posts/nonexistent")
    assert r.status_code == 404
