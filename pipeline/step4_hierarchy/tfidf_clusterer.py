"""
Step 3 alternative: TF-IDF + MiniBatchKMeans hierarchical clustering.

Uses scikit-learn exclusively. Keeps sparse matrices throughout to avoid
the O(n × features) dense-array memory blow-up that kills Ward linkage.
Produces a keyword-based 3-level hierarchy via successive K-Means rounds.
"""
from __future__ import annotations

import re
from collections import Counter

import numpy as np

from pipeline.protocols import CausalRelation, EventCluster, HierarchyInferrer

_STOPWORDS = frozenset({
    "a", "an", "the", "of", "in", "to", "and", "or", "for", "with",
    "on", "at", "by", "from", "as", "is", "are", "was", "were", "be",
    "been", "being", "have", "has", "had", "do", "does", "did", "will",
    "would", "could", "should", "may", "might", "that", "this", "it",
    "its", "they", "their", "not", "no", "more", "than", "which", "who",
    "new", "study", "research", "scientists", "researchers", "people",
    "human", "humans", "patient", "patients",
})


def _label_from_texts(texts: list[str], n: int = 3) -> str:
    words: list[str] = []
    for t in texts:
        words.extend(re.findall(r"[a-z]{3,}", t.lower()))
    counts = Counter(w for w in words if w not in _STOPWORDS)
    top = [w for w, _ in counts.most_common(n)]
    return " / ".join(top) if top else "unlabeled"


class TFIDFClusterer:
    """
    Implements HierarchyInferrer using TF-IDF vectors and MiniBatchKMeans.
    All operations use sparse matrices — no dense blow-up.
    """

    def __init__(
        self,
        tfidf_max_features: int = 3000,
        tfidf_n_top_clusters: int = 150,
        n_levels: int = 3,
        **kwargs,
    ) -> None:
        self.max_features = tfidf_max_features
        self.n_top_clusters = min(tfidf_n_top_clusters, 300)
        self.n_levels = n_levels

    @property
    def name(self) -> str:
        return "tfidf_ward"

    def infer(
        self,
        relations: list[CausalRelation],
    ) -> tuple[list[EventCluster], list[tuple[int, int, str, str]]]:
        from sklearn.cluster import MiniBatchKMeans
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.preprocessing import normalize

        if not relations:
            return [], []

        # ------------------------------------------------------------------
        # Collect unique event texts
        # ------------------------------------------------------------------
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

        print(f"[TFIDFClusterer] Vectorizing {len(all_texts)} unique texts…")
        vectorizer = TfidfVectorizer(
            max_features=self.max_features,
            ngram_range=(1, 2),
            stop_words=list(_STOPWORDS),
            sublinear_tf=True,
        )
        # X is sparse: shape (n_texts, n_features)
        X = vectorizer.fit_transform(all_texts)
        X = normalize(X)  # L2-normalize in place (still sparse)

        # ------------------------------------------------------------------
        # Leaf clusters (level 0): fine-grained via MiniBatchKMeans
        # ------------------------------------------------------------------
        n_leaf = min(max(10, len(all_texts) // 10), 500)
        print(f"[TFIDFClusterer] Leaf clustering into {n_leaf} clusters…")
        leaf_km = MiniBatchKMeans(
            n_clusters=n_leaf,
            batch_size=min(4096, len(all_texts)),
            max_iter=100,
            random_state=42,
            n_init=3,
        )
        leaf_labels = leaf_km.fit_predict(X)

        clusters: list[EventCluster] = []
        leaf_texts_by_label: dict[int, list[str]] = {}
        for ti, label in enumerate(leaf_labels):
            leaf_texts_by_label.setdefault(label, []).append(all_texts[ti])

        label_to_cluster_pos: dict[int, int] = {}
        for label in sorted(leaf_texts_by_label.keys()):
            pos = len(clusters)
            label_to_cluster_pos[label] = pos
            texts = leaf_texts_by_label[label]
            clusters.append(EventCluster(
                label=_label_from_texts(texts),
                level=0,
                parent_id=None,
                member_count=len(texts),
                clusterer=self.name,
            ))

        text_to_cluster_pos: dict[int, int] = {
            ti: label_to_cluster_pos[lb] for ti, lb in enumerate(leaf_labels)
        }

        # Build memberships
        memberships: list[tuple[int, int, str, str]] = []
        for rel_idx, relation in enumerate(relations):
            for role, norm in (("cause", relation.cause_norm), ("effect", relation.effect_norm)):
                ti = text_to_idx[norm]
                memberships.append((rel_idx, text_to_cluster_pos[ti], role, norm))

        if self.n_levels < 2:
            return clusters, memberships

        # ------------------------------------------------------------------
        # Mid-level clusters (level 1): cluster the leaf centroids (dense,
        # but only n_leaf × n_features which is small)
        # ------------------------------------------------------------------
        print(f"[TFIDFClusterer] Mid-level clustering…")
        n_leaf_actual = len(clusters)
        # Leaf centroids from KMeans are already dense and small
        leaf_centroids = leaf_km.cluster_centers_  # shape (n_leaf, n_features)
        leaf_centroids_norm = leaf_centroids / (
            np.linalg.norm(leaf_centroids, axis=1, keepdims=True).clip(min=1e-9)
        )

        n_mid = min(self.n_top_clusters, n_leaf_actual // 3)
        n_mid = max(5, n_mid)
        mid_km = MiniBatchKMeans(
            n_clusters=n_mid,
            batch_size=min(4096, n_leaf_actual),
            max_iter=100,
            random_state=42,
            n_init=3,
        )
        mid_labels = mid_km.fit_predict(leaf_centroids_norm)

        mid_texts_by_label: dict[int, list[str]] = {}
        for leaf_pos, mid_label in enumerate(mid_labels):
            mid_texts_by_label.setdefault(mid_label, []).extend(
                leaf_texts_by_label.get(leaf_pos, [])
            )

        mid_label_to_pos: dict[int, int] = {}
        for mid_label in sorted(mid_texts_by_label.keys()):
            pos = len(clusters)
            mid_label_to_pos[mid_label] = pos
            texts = mid_texts_by_label[mid_label]
            clusters.append(EventCluster(
                label=_label_from_texts(texts),
                level=1,
                parent_id=None,
                member_count=len(texts),
                clusterer=self.name,
            ))

        for leaf_pos, mid_label in enumerate(mid_labels):
            clusters[leaf_pos].parent_id = mid_label_to_pos[mid_label]

        if self.n_levels < 3:
            return clusters, memberships

        # ------------------------------------------------------------------
        # Top-level clusters (level 2): cluster the mid centroids
        # ------------------------------------------------------------------
        print(f"[TFIDFClusterer] Top-level clustering…")
        n_top = max(5, n_mid // 5)
        mid_centroids = mid_km.cluster_centers_
        mid_centroids_norm = mid_centroids / (
            np.linalg.norm(mid_centroids, axis=1, keepdims=True).clip(min=1e-9)
        )

        top_km = MiniBatchKMeans(
            n_clusters=n_top,
            batch_size=min(4096, n_mid),
            max_iter=100,
            random_state=42,
            n_init=3,
        )
        top_labels = top_km.fit_predict(mid_centroids_norm)

        top_texts_by_label: dict[int, list[str]] = {}
        for mid_rel, top_label in enumerate(top_labels):
            top_texts_by_label.setdefault(top_label, []).extend(
                mid_texts_by_label.get(mid_rel, [])
            )

        mid_start = n_leaf_actual
        top_label_to_pos: dict[int, int] = {}
        for top_label in sorted(top_texts_by_label.keys()):
            pos = len(clusters)
            top_label_to_pos[top_label] = pos
            texts = top_texts_by_label[top_label]
            clusters.append(EventCluster(
                label=_label_from_texts(texts),
                level=2,
                parent_id=None,
                member_count=len(texts),
                clusterer=self.name,
            ))

        for mid_rel, top_label in enumerate(top_labels):
            clusters[mid_start + mid_rel].parent_id = top_label_to_pos[top_label]

        return clusters, memberships
