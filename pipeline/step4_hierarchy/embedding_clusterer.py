"""
Step 3 default implementation: sentence-transformer embeddings + HDBSCAN clustering.

Strategy:
  1. Collect all unique normalized event texts (cause_norm + effect_norm).
  2. Embed with sentence-transformers (all-MiniLM-L6-v2, 384 dims).
  3. Run HDBSCAN → leaf clusters (level 0).
  4. Compute cluster centroids → embed → re-cluster → mid-level (level 1).
  5. Repeat for top-level (level 2), targeting 20–50 top clusters.
  6. Label each cluster with its most frequent content words.
"""
from __future__ import annotations

import re
from collections import Counter

import numpy as np

from ..protocols import CausalRelation, EventCluster, HierarchyInferrer

_STOPWORDS = frozenset({
    "a", "an", "the", "of", "in", "to", "and", "or", "for", "with",
    "on", "at", "by", "from", "as", "is", "are", "was", "were", "be",
    "been", "being", "have", "has", "had", "do", "does", "did", "will",
    "would", "could", "should", "may", "might", "that", "this", "it",
    "its", "they", "their", "not", "no", "more", "than", "which", "who",
    "new", "study", "research", "scientists", "researchers", "people",
    "human", "humans", "patient", "patients",
})


def _label_from_texts(texts: list[str], n_words: int = 3) -> str:
    """Generate a cluster label from the most frequent content words."""
    words: list[str] = []
    for t in texts:
        words.extend(re.findall(r"[a-z]{3,}", t.lower()))
    counts = Counter(w for w in words if w not in _STOPWORDS)
    top = [w for w, _ in counts.most_common(n_words)]
    return " / ".join(top) if top else "unlabeled"


class EmbeddingClusterer(HierarchyInferrer):
    """
    Implements HierarchyInferrer using sentence-transformer embeddings
    and HDBSCAN hierarchical density clustering.
    """

    def __init__(
        self,
        embedding_model: str = "all-MiniLM-L6-v2",
        min_cluster_size: int = 10,
        min_samples: int = 5,
        n_levels: int = 3,
        **kwargs,
    ) -> None:
        self.embedding_model = embedding_model
        self.min_cluster_size = min_cluster_size
        self.min_samples = min_samples
        self.n_levels = n_levels
        self._model = None

    def _get_model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self.embedding_model)
        return self._model

    @property
    def name(self) -> str:
        return "embedding_hdbscan"

    def infer(
        self,
        relations: list[CausalRelation],
    ) -> tuple[list[EventCluster], list[tuple[int, int, str, str]]]:
        import hdbscan

        if not relations:
            return [], []

        # ------------------------------------------------------------------
        # 1. Collect unique event texts per role
        # ------------------------------------------------------------------
        # For each relation, we have a cause_norm and effect_norm.
        # We cluster the full set of event texts (deduplicated for embedding,
        # but we still record all (relation_idx, role) pairs).

        all_texts: list[str] = []
        text_set: set[str] = set()
        text_to_idx: dict[str, int] = {}

        for r in relations:
            for norm, canonical in (
                (r.cause_norm, r.cause_canonical or r.cause_text or r.cause_norm),
                (r.effect_norm, r.effect_canonical or r.effect_text or r.effect_norm),
            ):
                if norm not in text_set:
                    text_to_idx[norm] = len(all_texts)
                    all_texts.append(canonical)
                    text_set.add(norm)

        print(f"[EmbeddingClusterer] Embedding {len(all_texts)} unique event texts...")
        model = self._get_model()
        embeddings = model.encode(
            all_texts,
            batch_size=256,
            show_progress_bar=True,
            normalize_embeddings=True,
        )

        # ------------------------------------------------------------------
        # 2. Cluster into leaf nodes (level 0)
        # ------------------------------------------------------------------
        print("[EmbeddingClusterer] Running HDBSCAN for leaf clusters...")
        leaf_clusterer = hdbscan.HDBSCAN(
            min_cluster_size=self.min_cluster_size,
            min_samples=self.min_samples,
            metric="euclidean",
            cluster_selection_method="eom",
        )
        leaf_labels = leaf_clusterer.fit_predict(embeddings)

        # Assign noise points (-1) to a special "Other" cluster
        n_leaf_clusters = int(leaf_labels.max()) + 1
        noise_cluster_idx = n_leaf_clusters  # index for noise

        clusters: list[EventCluster] = []
        memberships: list[tuple[int, int, str, str]] = []

        # Build leaf cluster objects
        leaf_cluster_start = 0
        leaf_texts_by_cluster: dict[int, list[str]] = {}
        for text_idx, label in enumerate(leaf_labels):
            key = label if label >= 0 else noise_cluster_idx
            leaf_texts_by_cluster.setdefault(key, []).append(all_texts[text_idx])

        # Map from HDBSCAN label → position in clusters list
        label_to_cluster_pos: dict[int, int] = {}

        for label in sorted(leaf_texts_by_cluster.keys()):
            cluster_pos = len(clusters)
            label_to_cluster_pos[label] = cluster_pos
            texts = leaf_texts_by_cluster[label]
            clusters.append(EventCluster(
                label=_label_from_texts(texts),
                level=0,
                parent_id=None,  # filled in after mid-level clustering
                member_count=len(texts),
                clusterer=self.name,
            ))

        # Map text_idx → cluster_pos
        text_to_cluster_pos: dict[int, int] = {}
        for text_idx, label in enumerate(leaf_labels):
            key = label if label >= 0 else noise_cluster_idx
            text_to_cluster_pos[text_idx] = label_to_cluster_pos[key]

        # Build memberships
        for rel_idx, relation in enumerate(relations):
            for role, norm_text in (("cause", relation.cause_norm), ("effect", relation.effect_norm)):
                text_idx = text_to_idx[norm_text]
                cluster_pos = text_to_cluster_pos[text_idx]
                memberships.append((rel_idx, cluster_pos, role, norm_text))

        if self.n_levels < 2:
            return clusters, memberships

        # ------------------------------------------------------------------
        # 3. Mid-level clustering (level 1) over leaf cluster centroids
        # ------------------------------------------------------------------
        print("[EmbeddingClusterer] Running mid-level clustering...")
        n_leaf = len(clusters)
        leaf_centroids = np.zeros((n_leaf, embeddings.shape[1]), dtype=np.float32)
        leaf_text_lookup: dict[int, list[str]] = {}

        for text_idx, cluster_pos in text_to_cluster_pos.items():
            leaf_text_lookup.setdefault(cluster_pos, []).append(all_texts[text_idx])

        for i in range(n_leaf):
            idxs = [ti for ti, cp in text_to_cluster_pos.items() if cp == i]
            if idxs:
                leaf_centroids[i] = embeddings[idxs].mean(axis=0)

        mid_min_size = max(3, self.min_cluster_size // 3)
        mid_clusterer = hdbscan.HDBSCAN(
            min_cluster_size=mid_min_size,
            min_samples=max(2, self.min_samples // 2),
            metric="euclidean",
        )
        mid_labels = mid_clusterer.fit_predict(leaf_centroids)
        n_mid = int(mid_labels.max()) + 1
        noise_mid = n_mid

        mid_texts_by_label: dict[int, list[str]] = {}
        for leaf_pos, mid_label in enumerate(mid_labels):
            key = mid_label if mid_label >= 0 else noise_mid
            mid_texts_by_label.setdefault(key, []).extend(
                leaf_text_lookup.get(leaf_pos, [])
            )

        mid_label_to_cluster_pos: dict[int, int] = {}
        for mid_label in sorted(mid_texts_by_label.keys()):
            cluster_pos = len(clusters)
            mid_label_to_cluster_pos[mid_label] = cluster_pos
            texts = mid_texts_by_label[mid_label]
            clusters.append(EventCluster(
                label=_label_from_texts(texts),
                level=1,
                parent_id=None,  # filled after top-level
                member_count=len(texts),
                clusterer=self.name,
            ))

        # Set parent_id (as cluster list index) for leaf clusters
        for leaf_pos, mid_label in enumerate(mid_labels):
            key = mid_label if mid_label >= 0 else noise_mid
            clusters[leaf_pos].parent_id = mid_label_to_cluster_pos[key]

        if self.n_levels < 3:
            return clusters, memberships

        # ------------------------------------------------------------------
        # 4. Top-level clustering (level 2) over mid-level centroids
        # ------------------------------------------------------------------
        print("[EmbeddingClusterer] Running top-level clustering...")
        n_mid_clusters = len(clusters) - n_leaf
        mid_centroids = np.zeros((n_mid_clusters, embeddings.shape[1]), dtype=np.float32)
        mid_start = n_leaf

        for rel_idx in range(n_mid_clusters):
            cluster_pos = mid_start + rel_idx
            # Average of leaf centroids that map to this mid cluster
            leaf_idxs = [
                li for li, ml in enumerate(mid_labels)
                if (ml if ml >= 0 else noise_mid) == list(mid_label_to_cluster_pos.keys())[rel_idx]
            ]
            if leaf_idxs:
                mid_centroids[rel_idx] = leaf_centroids[leaf_idxs].mean(axis=0)

        top_min_size = max(2, mid_min_size // 2)
        top_clusterer = hdbscan.HDBSCAN(
            min_cluster_size=top_min_size,
            min_samples=max(1, self.min_samples // 3),
            metric="euclidean",
        )
        top_labels = top_clusterer.fit_predict(mid_centroids)
        n_top = int(top_labels.max()) + 1
        noise_top = n_top

        top_texts_by_label: dict[int, list[str]] = {}
        for mid_rel_idx, top_label in enumerate(top_labels):
            mid_cluster_pos = mid_start + mid_rel_idx
            key = top_label if top_label >= 0 else noise_top
            texts = [
                all_texts[ti]
                for ti, cp in text_to_cluster_pos.items()
                if clusters[cp].parent_id == mid_cluster_pos
            ]
            top_texts_by_label.setdefault(key, []).extend(texts)

        top_label_to_cluster_pos: dict[int, int] = {}
        for top_label in sorted(top_texts_by_label.keys()):
            cluster_pos = len(clusters)
            top_label_to_cluster_pos[top_label] = cluster_pos
            texts = top_texts_by_label[top_label]
            clusters.append(EventCluster(
                label=_label_from_texts(texts),
                level=2,
                parent_id=None,
                member_count=len(texts),
                clusterer=self.name,
            ))

        # Set parent_id for mid-level clusters
        for mid_rel_idx, top_label in enumerate(top_labels):
            mid_cluster_pos = mid_start + mid_rel_idx
            key = top_label if top_label >= 0 else noise_top
            clusters[mid_cluster_pos].parent_id = top_label_to_cluster_pos[key]

        return clusters, memberships
