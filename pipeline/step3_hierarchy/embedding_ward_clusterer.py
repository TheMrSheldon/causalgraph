"""
Step 3 alternative: sentence-transformer embeddings + scipy Ward linkage.

Unlike the HDBSCAN implementation, Ward linkage produces a proper dendrogram
that can be cut at any height.  An arbitrary number of levels is supported by
specifying ``n_clusters_per_level`` — a list whose length determines how many
levels the hierarchy will have and whose values specify how many clusters to
produce at each level (finest → coarsest).

Because every cut is a partition of the same dendrogram, the containment
property is guaranteed: every cluster at level k is a strict subset of exactly
one cluster at level k+1.  Parent pointers are therefore uniquely determined
by the dendrogram and require no heuristics.

Memory constraint
-----------------
scipy Ward linkage on dense embeddings requires an O(n²) condensed distance
matrix.  For n > ~8 000 unique event texts this becomes prohibitive.  The
implementation therefore:
  1. Clusters the first ``max_texts`` unique texts via linkage.
  2. Assigns remaining texts to their nearest leaf-cluster centroid (level 0).
     Their path up the hierarchy follows the leaf cluster's own parent chain,
     so out-of-sample texts never break the tree structure.

Event text choice
-----------------
Set ``use_norm=False`` (the default) to embed the raw extracted phrase
(``cause_text`` / ``effect_text``) rather than the lowercased/lemmatised norm.
The raw phrase preserves capitalisation and phrasing that the sentence
transformer can use.  Set ``use_norm=True`` to match the other implementations.
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


class EmbeddingWardClusterer:
    """
    Implements HierarchyInferrer using sentence-transformer embeddings and
    scipy Ward linkage with dendrogram cuts at an arbitrary number of levels.

    Parameters
    ----------
    embedding_model : str
        Any sentence-transformers model name.
    linkage_method : str
        Linkage criterion: 'ward' (default), 'complete', or 'average'.
        'ward' minimises within-cluster variance and usually gives the most
        interpretable clusters.
    n_clusters_per_level : list[int]
        Number of clusters at each level, ordered from **finest** (level 0,
        most clusters) to **coarsest** (top level, fewest clusters).
        The length of the list is the number of hierarchy levels produced.
        Example: ``[500, 100, 30, 8]`` → 4 levels.
        Default: ``[300, 60, 15]`` (3 levels).
    max_texts : int
        Hard cap on unique texts fed to the linkage algorithm.  Texts beyond
        this limit are assigned to their nearest leaf-cluster centroid after
        the fact.  Keep below ~8 000 to avoid OOM (O(n²) distance matrix).
    use_norm : bool
        If True, embed ``cause_norm`` / ``effect_norm`` (lowercased/lemmatised).
        If False (default), embed the richer raw ``cause_text`` / ``effect_text``.
    """

    def __init__(
        self,
        embedding_model: str = "all-MiniLM-L6-v2",
        linkage_method: str = "ward",
        n_clusters_per_level: list[int] | None = None,
        max_texts: int = 8000,
        use_norm: bool = False,
        **kwargs,
    ) -> None:
        self.embedding_model = embedding_model
        self.linkage_method = linkage_method
        self.n_clusters_per_level: list[int] = n_clusters_per_level or [300, 60, 15]
        self.max_texts = max_texts
        self.use_norm = use_norm
        self._model = None

    @property
    def name(self) -> str:
        return "embedding_ward"

    def _get_model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self.embedding_model)
        return self._model

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def infer(
        self,
        relations: list[CausalRelation],
    ) -> tuple[list[EventCluster], list[tuple[int, int, str, str]]]:
        from scipy.cluster.hierarchy import fcluster, linkage
        from scipy.spatial.distance import pdist

        if not relations:
            return [], []

        n_levels = len(self.n_clusters_per_level)

        # ------------------------------------------------------------------
        # 1. Collect unique event (norm, raw) pairs — norm is the dedup key
        # ------------------------------------------------------------------
        norm_to_raw: dict[str, str] = {}
        for r in relations:
            for raw, norm in ((r.cause_text, r.cause_norm), (r.effect_text, r.effect_norm)):
                if norm not in norm_to_raw:
                    norm_to_raw[norm] = raw

        all_norms: list[str] = list(norm_to_raw.keys())
        all_raws: list[str] = [norm_to_raw[n] for n in all_norms]
        embed_texts = all_norms if self.use_norm else all_raws

        n_total = len(all_norms)
        norm_to_idx: dict[str, int] = {n: i for i, n in enumerate(all_norms)}

        print(
            f"[EmbeddingWardClusterer] Embedding {n_total} unique event texts "
            f"({'norm' if self.use_norm else 'raw'} text, {n_levels} levels)…"
        )
        model = self._get_model()
        embeddings = model.encode(
            embed_texts,
            batch_size=256,
            show_progress_bar=True,
            normalize_embeddings=True,
        )  # shape (n_total, dim), float32

        # ------------------------------------------------------------------
        # 2. Fit linkage on at most max_texts samples
        # ------------------------------------------------------------------
        if n_total > self.max_texts:
            print(
                f"[EmbeddingWardClusterer] {n_total} texts > max_texts={self.max_texts}. "
                f"Fitting linkage on first {self.max_texts}; rest assigned by nearest centroid."
            )
            fit_idx = np.arange(self.max_texts)
            oos_idx = np.arange(self.max_texts, n_total)
        else:
            fit_idx = np.arange(n_total)
            oos_idx = np.array([], dtype=int)

        X_fit = embeddings[fit_idx]
        n_fit = len(fit_idx)

        # Clamp all cluster counts to what the data supports
        clamped = [min(n, n_fit - 1) for n in self.n_clusters_per_level]
        # Ensure strict decrease from fine → coarse (avoid duplicate cut points)
        for i in range(1, len(clamped)):
            clamped[i] = min(clamped[i], clamped[i - 1] - 1)
        clamped = [max(1, c) for c in clamped]

        print(
            f"[EmbeddingWardClusterer] Running {self.linkage_method} linkage on {n_fit} texts; "
            f"cutting at {clamped} clusters per level…"
        )
        if self.linkage_method == "ward":
            Z = linkage(X_fit, method="ward", metric="euclidean")
        else:
            dist = pdist(X_fit, metric="cosine")
            Z = linkage(dist, method=self.linkage_method)

        # Cut dendrogram once per level; shape (n_fit,) per cut, 0-based labels
        all_fit_labels: list[np.ndarray] = [
            fcluster(Z, c, criterion="maxclust") - 1
            for c in clamped
        ]

        # ------------------------------------------------------------------
        # 3. Handle out-of-sample texts: assign to nearest leaf centroid
        #    then follow the existing hierarchy upward (no independent cuts
        #    per level — avoids tree inconsistency for OOS points)
        # ------------------------------------------------------------------
        n_leaf_actual = int(all_fit_labels[0].max()) + 1
        leaf_centroids = _centroids(all_fit_labels[0], X_fit, n_leaf_actual)

        # Full-length label arrays (indices into fit-set cluster IDs)
        all_labels: list[np.ndarray] = [np.empty(n_total, dtype=int) for _ in range(n_levels)]
        for lvl, fit_lbs in enumerate(all_fit_labels):
            all_labels[lvl][fit_idx] = fit_lbs

        if len(oos_idx):
            X_oos = embeddings[oos_idx]
            # Assign OOS to nearest leaf cluster, then propagate upward via
            # the fit-based leaf→parent map (built below after clusters are created).
            oos_leaf_labels = _assign_oos(X_oos, leaf_centroids)
            all_labels[0][oos_idx] = oos_leaf_labels
            # Levels > 0 are filled after parent pointers are known (see below).

        # ------------------------------------------------------------------
        # 4. Build EventCluster objects for every level
        # ------------------------------------------------------------------
        clusters: list[EventCluster] = []
        # level_label_to_pos[lvl][label] = position in `clusters` list
        level_label_to_pos: list[dict[int, int]] = []

        for lvl in range(n_levels):
            fit_lbs = all_fit_labels[lvl]
            texts_by_label: dict[int, list[str]] = {}
            for ti, lbl in enumerate(fit_lbs):
                texts_by_label.setdefault(int(lbl), []).append(all_norms[fit_idx[ti]])

            label_to_pos: dict[int, int] = {}
            for lbl in sorted(texts_by_label):
                pos = len(clusters)
                label_to_pos[lbl] = pos
                clusters.append(EventCluster(
                    label=_label_from_texts(texts_by_label[lbl]),
                    level=lvl,
                    parent_id=None,
                    member_count=len(texts_by_label[lbl]),
                    clusterer=self.name,
                ))
            level_label_to_pos.append(label_to_pos)

        # ------------------------------------------------------------------
        # 5. Set parent pointers using the dendrogram's containment property.
        #    For Ward linkage, every cluster at level k is a strict subset of
        #    a unique cluster at level k+1 — so a single majority scan suffices.
        # ------------------------------------------------------------------
        for lvl in range(n_levels - 1):
            fine_labels   = all_fit_labels[lvl]      # shape (n_fit,)
            coarse_labels = all_fit_labels[lvl + 1]  # shape (n_fit,)

            fine_to_coarse: dict[int, int] = {}
            for fi in range(n_fit):
                fine_lbl   = int(fine_labels[fi])
                coarse_lbl = int(coarse_labels[fi])
                fine_to_coarse[fine_lbl] = coarse_lbl  # unique by Ward property

            for fine_lbl, coarse_lbl in fine_to_coarse.items():
                fine_pos   = level_label_to_pos[lvl][fine_lbl]
                coarse_pos = level_label_to_pos[lvl + 1][coarse_lbl]
                clusters[fine_pos].parent_id = coarse_pos

        # ------------------------------------------------------------------
        # 6. Propagate OOS labels upward via parent pointers
        # ------------------------------------------------------------------
        if len(oos_idx):
            for lvl in range(1, n_levels):
                for i, oos_i in enumerate(oos_idx):
                    leaf_lbl  = int(all_labels[0][oos_i])
                    leaf_pos  = level_label_to_pos[0][leaf_lbl]
                    # Walk up (lvl) steps from the leaf cluster
                    pos = leaf_pos
                    for _ in range(lvl):
                        if clusters[pos].parent_id is None:
                            break
                        pos = clusters[pos].parent_id
                    # Reverse-look up the label at this level
                    lvl_label_to_pos = level_label_to_pos[lvl]
                    for lbl, p in lvl_label_to_pos.items():
                        if p == pos:
                            all_labels[lvl][oos_i] = lbl
                            break

        # ------------------------------------------------------------------
        # 7. Build memberships (always reference level-0 / leaf clusters)
        # ------------------------------------------------------------------
        memberships = _build_memberships(
            relations, norm_to_idx, all_labels[0], level_label_to_pos[0]
        )
        return clusters, memberships


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------

def _centroids(labels: np.ndarray, X: np.ndarray, n_clusters: int) -> np.ndarray:
    dim = X.shape[1]
    C = np.zeros((n_clusters, dim), dtype=np.float32)
    counts = np.zeros(n_clusters, dtype=int)
    for i, lbl in enumerate(labels):
        C[lbl] += X[i]
        counts[lbl] += 1
    nz = counts > 0
    C[nz] /= counts[nz, np.newaxis]
    norms = np.linalg.norm(C, axis=1, keepdims=True).clip(min=1e-9)
    return C / norms


def _assign_oos(X_oos: np.ndarray, centroids: np.ndarray) -> np.ndarray:
    X_norm = X_oos / np.linalg.norm(X_oos, axis=1, keepdims=True).clip(min=1e-9)
    sims = X_norm @ centroids.T
    return sims.argmax(axis=1)


def _build_memberships(
    relations: list[CausalRelation],
    norm_to_idx: dict[str, int],
    leaf_labels: np.ndarray,
    leaf_label_to_pos: dict[int, int],
) -> list[tuple[int, int, str, str]]:
    memberships: list[tuple[int, int, str, str]] = []
    for rel_idx, rel in enumerate(relations):
        for role, norm in (("cause", rel.cause_norm), ("effect", rel.effect_norm)):
            ti = norm_to_idx[norm]
            cluster_pos = leaf_label_to_pos[int(leaf_labels[ti])]
            memberships.append((rel_idx, cluster_pos, role, norm))
    return memberships
