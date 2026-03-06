"""
Tests for EmbeddingWardClusterer (Step 3 implementation).

These tests load the real sentence-transformer model (all-MiniLM-L6-v2) to
avoid mocking away the logic under test. The model is ~80 MB and is cached
after the first download, so the suite runs fast on repeat invocations.

Fixture sizes are kept small (≤20 relations) so the scipy linkage step
completes in milliseconds.
"""
from __future__ import annotations

import pytest

from pipeline.protocols import CausalRelation, HierarchyInferrer
from pipeline.step3_hierarchy.embedding_ward_clusterer import EmbeddingWardClusterer


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

def _rel(pid: str, cause: str, effect: str) -> CausalRelation:
    return CausalRelation(
        post_id=pid,
        cause_text=cause,
        effect_text=effect,
        cause_norm=cause.lower(),
        effect_norm=effect.lower(),
    )


@pytest.fixture(scope="module")
def relations_12() -> list[CausalRelation]:
    """12 relations covering three loose semantic domains."""
    return [
        _rel("p1",  "Smoking",             "lung cancer"),
        _rel("p2",  "Smoking",             "heart disease"),
        _rel("p3",  "Air pollution",       "asthma"),
        _rel("p4",  "Air pollution",       "respiratory illness"),
        _rel("p5",  "Exercise",            "weight loss"),
        _rel("p6",  "Exercise",            "improved cardiovascular health"),
        _rel("p7",  "Healthy diet",        "lower cholesterol"),
        _rel("p8",  "Stress",              "depression"),
        _rel("p9",  "Stress",              "anxiety"),
        _rel("p10", "Sleep deprivation",   "cognitive decline"),
        _rel("p11", "Alcohol",             "liver damage"),
        _rel("p12", "Sedentary lifestyle", "obesity"),
    ]


@pytest.fixture(scope="module")
def result_3level(relations_12):
    """Pre-computed 3-level result; shared across tests in this module."""
    c = EmbeddingWardClusterer(n_clusters_per_level=[5, 3, 2])
    return c.infer(relations_12)


@pytest.fixture(scope="module")
def result_4level(relations_12):
    """Pre-computed 4-level result for N-level tests."""
    c = EmbeddingWardClusterer(n_clusters_per_level=[6, 4, 2, 1])
    return c.infer(relations_12)


# ---------------------------------------------------------------------------
# Protocol conformance
# ---------------------------------------------------------------------------

def test_protocol_conformance():
    assert isinstance(EmbeddingWardClusterer(), HierarchyInferrer)


def test_name():
    assert EmbeddingWardClusterer().name == "embedding_ward"


# ---------------------------------------------------------------------------
# Empty input
# ---------------------------------------------------------------------------

def test_empty_input_returns_empty():
    clusters, memberships = EmbeddingWardClusterer().infer([])
    assert clusters == []
    assert memberships == []


# ---------------------------------------------------------------------------
# Cluster structure — 3-level fixture
# ---------------------------------------------------------------------------

def test_cluster_levels_present(result_3level):
    clusters, _ = result_3level
    levels = {cl.level for cl in clusters}
    assert levels == {0, 1, 2}


def test_top_level_clusters_have_no_parent(result_3level):
    clusters, _ = result_3level
    top_level = max(cl.level for cl in clusters)
    for cl in clusters:
        if cl.level == top_level:
            assert cl.parent_id is None, f"Top cluster has parent_id={cl.parent_id}"


def test_non_top_clusters_have_valid_parent(result_3level):
    clusters, _ = result_3level
    top_level = max(cl.level for cl in clusters)
    n = len(clusters)
    for i, cl in enumerate(clusters):
        if cl.level < top_level:
            assert cl.parent_id is not None, f"Cluster {i} (level {cl.level}) has no parent"
            assert 0 <= cl.parent_id < n


def test_parent_points_to_next_level(result_3level):
    clusters, _ = result_3level
    for i, cl in enumerate(clusters):
        if cl.parent_id is not None:
            parent = clusters[cl.parent_id]
            assert parent.level == cl.level + 1, (
                f"Cluster {i} level={cl.level} has parent level={parent.level}"
            )


def test_no_self_referencing_parent(result_3level):
    clusters, _ = result_3level
    for i, cl in enumerate(clusters):
        assert cl.parent_id != i


def test_cluster_labels_are_non_empty(result_3level):
    clusters, _ = result_3level
    for cl in clusters:
        assert isinstance(cl.label, str) and cl.label.strip()


def test_cluster_member_counts_positive(result_3level):
    clusters, _ = result_3level
    for cl in clusters:
        assert cl.member_count > 0


def test_clusterer_field(result_3level):
    clusters, _ = result_3level
    for cl in clusters:
        assert cl.clusterer == "embedding_ward"


# ---------------------------------------------------------------------------
# Membership structure
# ---------------------------------------------------------------------------

def test_membership_count(relations_12, result_3level):
    _, memberships = result_3level
    assert len(memberships) == 2 * len(relations_12)


def test_membership_relation_indices_in_range(relations_12, result_3level):
    _, memberships = result_3level
    n = len(relations_12)
    for rel_idx, _, _, _ in memberships:
        assert 0 <= rel_idx < n


def test_membership_cluster_indices_point_to_leaves(result_3level):
    clusters, memberships = result_3level
    n = len(clusters)
    for _, cluster_pos, _, _ in memberships:
        assert 0 <= cluster_pos < n
        assert clusters[cluster_pos].level == 0, (
            f"Membership points to non-leaf cluster (level={clusters[cluster_pos].level})"
        )


def test_membership_roles_are_valid(result_3level):
    _, memberships = result_3level
    for _, _, role, _ in memberships:
        assert role in ("cause", "effect")


def test_every_relation_has_cause_and_effect_membership(relations_12, result_3level):
    _, memberships = result_3level
    n = len(relations_12)
    cause_seen = {rel_idx for rel_idx, _, role, _ in memberships if role == "cause"}
    effect_seen = {rel_idx for rel_idx, _, role, _ in memberships if role == "effect"}
    assert cause_seen == set(range(n))
    assert effect_seen == set(range(n))


def test_membership_event_text_matches_norm(relations_12, result_3level):
    _, memberships = result_3level
    for rel_idx, _, role, event_text in memberships:
        rel = relations_12[rel_idx]
        expected = rel.cause_norm if role == "cause" else rel.effect_norm
        assert event_text == expected


# ---------------------------------------------------------------------------
# Deduplication: identical norm → same leaf cluster
# ---------------------------------------------------------------------------

def test_identical_norm_texts_share_cluster():
    rels = [
        _rel("p1", "Smoking",  "lung cancer"),
        _rel("p2", "smoking",  "heart disease"),   # same norm after .lower()
        _rel("p3", "Exercise", "weight loss"),
        _rel("p4", "Exercise", "cardiovascular improvement"),  # same norm
    ]
    c = EmbeddingWardClusterer(n_clusters_per_level=[3, 2, 1])
    _, memberships = c.infer(rels)

    def leaf_for(rel_idx, role):
        for ri, cp, r, _ in memberships:
            if ri == rel_idx and r == role:
                return cp

    assert leaf_for(0, "cause") == leaf_for(1, "cause"), \
        "Identical cause norms mapped to different leaf clusters"
    assert leaf_for(2, "cause") == leaf_for(3, "cause"), \
        "Identical cause norms mapped to different leaf clusters"


# ---------------------------------------------------------------------------
# N-level hierarchy (the key new feature)
# ---------------------------------------------------------------------------

def test_4level_levels_present(result_4level):
    clusters, _ = result_4level
    levels = {cl.level for cl in clusters}
    assert levels == {0, 1, 2, 3}


def test_4level_parent_chain_is_valid(result_4level):
    clusters, _ = result_4level
    top_level = max(cl.level for cl in clusters)
    for i, cl in enumerate(clusters):
        if cl.parent_id is not None:
            parent = clusters[cl.parent_id]
            assert parent.level == cl.level + 1
        else:
            assert cl.level == top_level


def test_4level_memberships_complete(relations_12, result_4level):
    _, memberships = result_4level
    assert len(memberships) == 2 * len(relations_12)


def test_arbitrary_single_level():
    """n_clusters_per_level of length 1 → only leaf clusters, no parents."""
    rels = [_rel(f"p{i}", f"cause {i}", f"effect {i}") for i in range(6)]
    c = EmbeddingWardClusterer(n_clusters_per_level=[3])
    clusters, memberships = c.infer(rels)
    assert all(cl.level == 0 for cl in clusters)
    assert all(cl.parent_id is None for cl in clusters)
    assert len(memberships) == 2 * len(rels)


def test_5_levels():
    """Five-level hierarchy on a moderate relation set."""
    rels = [_rel(f"p{i}", f"cause phrase {i}", f"effect phrase {i}") for i in range(30)]
    c = EmbeddingWardClusterer(n_clusters_per_level=[20, 10, 5, 3, 1])
    clusters, memberships = c.infer(rels)
    levels = {cl.level for cl in clusters}
    assert levels == {0, 1, 2, 3, 4}
    assert len(memberships) == 2 * len(rels)
    # Root clusters have no parent
    for cl in clusters:
        if cl.level == 4:
            assert cl.parent_id is None


# ---------------------------------------------------------------------------
# max_texts cap and out-of-sample assignment
# ---------------------------------------------------------------------------

def test_max_texts_cap():
    """With max_texts below unique-text count, OOS assignment runs."""
    rels = [_rel(f"p{i}", f"cause phrase {i}", f"effect phrase {i}") for i in range(20)]
    c = EmbeddingWardClusterer(n_clusters_per_level=[3, 2, 1], max_texts=10)
    clusters, memberships = c.infer(rels)
    assert len(memberships) == 2 * len(rels)
    top_level = max(cl.level for cl in clusters)
    for cl in clusters:
        if cl.level < top_level:
            assert cl.parent_id is not None


# ---------------------------------------------------------------------------
# use_norm flag
# ---------------------------------------------------------------------------

def test_use_norm_produces_valid_output(relations_12):
    c = EmbeddingWardClusterer(n_clusters_per_level=[4, 2, 1], use_norm=True)
    clusters, memberships = c.infer(relations_12)
    assert len(memberships) == 2 * len(relations_12)
    assert {cl.level for cl in clusters} == {0, 1, 2}


# ---------------------------------------------------------------------------
# Linkage method variants
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("method", ["complete", "average"])
def test_linkage_methods(relations_12, method):
    c = EmbeddingWardClusterer(n_clusters_per_level=[4, 2, 1], linkage_method=method)
    clusters, memberships = c.infer(relations_12)
    assert len(memberships) == 2 * len(relations_12)
    assert any(cl.level == 0 for cl in clusters)


# ---------------------------------------------------------------------------
# Registry integration
# ---------------------------------------------------------------------------

def test_registry_builds_from_config():
    from pipeline.registry import build_inferrer
    config = {
        "pipeline": {
            "step3_hierarchy": {
                "implementation": "embedding_ward",
                "n_clusters_per_level": [4, 2, 1],
            }
        }
    }
    inferrer = build_inferrer(config)
    assert isinstance(inferrer, EmbeddingWardClusterer)
    assert isinstance(inferrer, HierarchyInferrer)
    assert inferrer.n_clusters_per_level == [4, 2, 1]
