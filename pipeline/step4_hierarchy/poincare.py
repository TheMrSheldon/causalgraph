"""
Step 4 alternative: Poincaré ball embeddings + cone-based hierarchical clustering.

Uses the ugao25w3 EncoderHat model (all-mpnet-base-v2 → 2-layer Möbius MLP → 50-d
Poincaré ball) to embed event texts, then builds the hierarchy directly from the
entailment cone structure learned by the model (Ganea et al. 2018).

Cone geometry recap
-------------------
Each embedding u ∈ ℍ⁵⁰ defines an entailment cone:
  ψ(u) = arcsin(K (1−‖u‖²) / ‖u‖)       half-aperture
  ξ(u,v) = angle between v and u's outward axis
  E(u,v)  = ReLU(ξ(u,v) − ψ(u))          cone energy

E(u,v) = 0  ⟹  v lies inside u's cone  ⟹  u entails v (u is more general than v).
Radial depth: small ‖u‖ ≈ abstract/general; large ‖u‖ ≈ specific.

Clustering algorithm
--------------------
Top-down recursive cone subdivision — depth adapts to data:

1. Optionally inject WordNet anchor nodes (inject_knowledge="wordnet").  WordNet
   lemma names are embedded alongside event texts.  Anchors are embedded by their
   primary lemma name (e.g. "physical entity", "causal agent") rather than their
   definition, because the ugao25w3 model was trained on lemma names and maps them
   closer to the Poincaré origin.  Whether they become top-level cluster heads
   depends on their norm relative to the event texts; in practice they tend to
   appear in the mid-levels of the hierarchy.

2. From the current node's member set, select K = min(branching_factor,
   |members| / min_cluster_size) min-norm texts as sub-cluster heads; assign all
   members to the nearest head by cone energy.  The cluster label is the text of
   the min-norm head, which by the entailment-cone geometry is the most general
   concept in the cluster (smallest aperture angle ψ → largest cone → contains
   the most other embeddings).

3. Recurse on each child until |members| ≤ min_cluster_size (leaf) or depth
   exceeds max_depth (safety cap).  Depth varies per branch.

4. Level = post-order height: leaves = 0, each parent = max(children) + 1.
   Root clusters have the highest level and are the initial API view.

5. Cluster label = the primary lemma name of the WordNet synset if the min-norm
   member is an anchor; otherwise the canonical text of the most abstract event
   in the cluster (the head selected in step 2).
"""
from __future__ import annotations

import importlib.util
import pathlib
from collections import deque
from dataclasses import dataclass, field

import numpy as np
import torch

from ..protocols import CausalRelation, EventCluster, HierarchyInferrer

_CONF_PROJECT = pathlib.Path(
    "/mnt/ceph/storage/data-tmp/2026/thagen/conf-causal-hierarchical-embeddings"
)
_TRAINING_DIR = _CONF_PROJECT / "training"

_DEFAULT_CHECKPOINT = str(
    _CONF_PROJECT
    / "hierarchical-embeddings-paper"
    / "ugao25w3"
    / "checkpoints"
    / "best.ckpt"
)


# ---------------------------------------------------------------------------
# Model loading & embedding
# ---------------------------------------------------------------------------

def _load_model(checkpoint_path: str, device: str):
    """Load EncoderHat from a Lightning checkpoint via importlib (not pip-installed)."""
    spec = importlib.util.spec_from_file_location(
        "poincare_encoder_hat", _TRAINING_DIR / "model.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    torch.serialization.add_safe_globals([pathlib.PosixPath])
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    hparams = {
        k: v
        for k, v in ckpt.get("hyper_parameters", {}).items()
        if not k.startswith("_")
    }
    hparams["pretrained_ckpt"]    = ""
    hparams["distill_teacher_p1"] = ""
    model = mod.EncoderHat(**hparams)
    model.load_state_dict(ckpt["state_dict"], strict=False)
    return model.eval().to(device)


def _embed_poincare(
    model, texts: list[str], batch_size: int, device: str, desc: str = "embedding"
) -> np.ndarray:
    """Embed texts in batches → (N, D) float32 array in the open unit ball."""
    try:
        from tqdm import tqdm
        batches = range(0, len(texts), batch_size)
        batches = tqdm(batches, desc=desc, unit="batch", leave=False)
    except ImportError:
        batches = range(0, len(texts), batch_size)

    parts = []
    with torch.no_grad():
        for i in batches:
            parts.append(model(texts[i : i + batch_size]).cpu().float().numpy())
    return np.concatenate(parts, axis=0)


# ---------------------------------------------------------------------------
# Anchor concept loading
# ---------------------------------------------------------------------------

def _is_single_verb(word: str) -> bool:
    """Return True if a single word is primarily a verb in WordNet.

    Uses majority vote over all synsets: filtered if verb count strictly
    exceeds noun count.  Ties (e.g. "change": 10n/10v) are kept as nouns.
    Words absent from WordNet are kept (assumed to be nouns/domain terms).
    """
    try:
        from nltk.corpus import wordnet as wn
        synsets = wn.synsets(word)
        if not synsets:
            return False
        n_noun = sum(1 for s in synsets if s.pos() == "n")
        n_verb = sum(1 for s in synsets if s.pos() == "v")
        return n_verb > n_noun
    except Exception:
        return False


def _load_abspyramid_concepts(path: str) -> list[tuple[str, str]]:
    """Return (display_label, embed_text) pairs from AbsPyramid concept phrases.

    Loads the unique ``positive`` concept phrases from the AbsPyramid parquet
    file.  ugao25w3 was trained on these phrases as abstraction targets, so they
    are reliably positioned in the Poincaré ball near the origin (general) end.

    Filtering removes:
    - Single-word verbs (e.g. "get", "stimulate") — poor cluster labels.
      Single-word nouns ("war", "death") are kept.
    - Phrases containing PersonX/PersonY template markers
    - Phrases containing '<' or '>' (annotation artifacts)
    - Empty / whitespace-only strings
    """
    try:
        import pyarrow.parquet as pq
    except ImportError:
        print("[PoincareClusterer] pyarrow not installed — skipping AbsPyramid injection")
        return []

    table = pq.read_table(path, columns=["positive"])
    phrases: list[str] = table.column("positive").to_pylist()

    seen: set[str] = set()
    anchors: list[tuple[str, str]] = []
    for phrase in phrases:
        if not isinstance(phrase, str):
            continue
        phrase = phrase.strip()
        if not phrase or phrase in seen:
            continue
        if "PersonX" in phrase or "PersonY" in phrase:
            continue
        if "<" in phrase or ">" in phrase:
            continue
        tokens = phrase.split()
        if len(tokens) == 1 and _is_single_verb(phrase):
            continue
        seen.add(phrase)
        anchors.append((phrase, phrase))

    return anchors


# ---------------------------------------------------------------------------
# Cone geometry  (Ganea et al. 2018, Eqs. 26–33)
# ---------------------------------------------------------------------------

def _cone_energy_rect(
    u_embs: np.ndarray,
    v_embs: np.ndarray,
    cone_K: float,
    device: str = "cpu",
) -> np.ndarray:
    """Compute E(u_i, v_j) = max(0, ξ(u_i, v_j) − ψ(u_i)) for all (i, j).

    u_embs: (K, D)  —  candidate parent / cluster-head embeddings
    v_embs: (N, D)  —  all embeddings to be assigned
    Returns: (K, N) float32 cone-energy matrix on CPU.
    """
    u = torch.tensor(u_embs, device=device, dtype=torch.float32)
    v = torch.tensor(v_embs, device=device, dtype=torch.float32)

    eps = 1e-6

    norms_u     = u.norm(dim=-1)
    norms_u_sq  = norms_u.pow(2)
    norms_v_sq  = v.pow(2).sum(dim=-1)
    dots        = u @ v.T

    sin_psi  = (cone_K * (1.0 - norms_u_sq) / norms_u.clamp(min=eps)).clamp(0.0, 1.0 - eps)
    apertures = sin_psi.arcsin()

    diff_sq    = (norms_u_sq[:, None] + norms_v_sq[None, :] - 2.0 * dots).clamp(min=0.0)
    diff_norms = diff_sq.sqrt().clamp(min=eps)
    numer      = dots * (1.0 + norms_u_sq[:, None]) \
                 - norms_u_sq[:, None] * (1.0 + norms_v_sq[None, :])
    inner      = (1.0 + norms_u_sq[:, None] * norms_v_sq[None, :] - 2.0 * dots).clamp(min=eps)
    denom      = (norms_u[:, None] * diff_norms * inner.sqrt()).clamp(min=eps)
    cos_xi     = (numer / denom).clamp(-1.0 + eps, 1.0 - eps)
    xi         = cos_xi.arccos()

    return torch.relu(xi - apertures[:, None]).cpu().float().numpy()


def _assign_by_cone(
    head_embs: np.ndarray,
    member_embs: np.ndarray,
    cone_K: float,
    device: str,
    head_member_positions: np.ndarray | None = None,
) -> np.ndarray:
    """Assign each member to the head with minimum cone energy → 0-based labels.

    head_member_positions: for each head k, the index j into member_embs where
    head k appears.  When provided, E[k, j] is forced to 0 because Ξ(u, u) is
    numerically π/2 (the formula is undefined at u=v and the denominator is
    clamped to eps), giving spuriously high energy for a head against itself.
    """
    E = _cone_energy_rect(head_embs, member_embs, cone_K, device)
    if head_member_positions is not None:
        for k, j in enumerate(head_member_positions):
            if j >= 0:
                E[k, j] = 0.0
    return E.argmin(axis=0).astype(np.int32)


# ---------------------------------------------------------------------------
# Adaptive recursive tree
# ---------------------------------------------------------------------------

@dataclass
class _TreeNode:
    indices: np.ndarray              # indices into the combined embedding array
    children: list[_TreeNode] = field(default_factory=list)
    level: int = 0                   # filled post-order: leaf=0, root=max
    cluster_pos: int = -1            # position in the flat EventCluster list


def _recursive_subdivide(
    node: _TreeNode,
    embeddings: np.ndarray,
    norms: np.ndarray,
    min_cluster_size: int,
    branching_factor: int,
    cone_K: float,
    device: str,
    depth: int,
    max_depth: int,
    banned_heads: frozenset[int] = frozenset(),
) -> None:
    """Recursively split `node` until each leaf has ≤ min_cluster_size members.

    `banned_heads` accumulates the global indices of ALL ancestor head texts.
    Excluding them from head selection at each level prevents ultra-abstract
    texts near the Poincaré origin from absorbing 95%+ of members at every
    successive depth, which was the cause of 130K+ member leaf clusters.
    """
    N = len(node.indices)
    if N <= min_cluster_size or depth >= max_depth:
        return  # leaf

    K = max(2, min(branching_factor, N // min_cluster_size))

    local_norms  = norms[node.indices]
    sorted_local = node.indices[np.argsort(local_norms)]

    # Exclude all ancestor heads so they cannot dominate their own subtrees.
    if banned_heads:
        mask         = ~np.isin(sorted_local, np.fromiter(banned_heads, dtype=np.int64))
        sorted_local = sorted_local[mask]
        if len(sorted_local) < 2:
            return

    head_indices = sorted_local[:K]

    # For each head k, find its position in node.indices so self-energy can be zeroed.
    head_member_positions = np.array([
        int(np.where(node.indices == h)[0][0]) for h in head_indices
    ], dtype=np.int64)

    # assignments[j] ∈ [0, K) — which head member node.indices[j] belongs to
    assignments = _assign_by_cone(
        embeddings[head_indices], embeddings[node.indices], cone_K, device,
        head_member_positions=head_member_positions,
    )
    # Remap raw head indices (0..K-1) to contiguous child labels (handles empty clusters)
    unique_orig, assignments = np.unique(assignments, return_inverse=True)

    new_banned = banned_heads | frozenset(int(h) for h in head_indices)

    for lbl_local in range(len(unique_orig)):
        child_mask    = assignments == lbl_local
        child_indices = node.indices[child_mask]
        if len(child_indices) == 0:
            continue
        child = _TreeNode(indices=child_indices)
        node.children.append(child)
        _recursive_subdivide(
            child, embeddings, norms,
            min_cluster_size, branching_factor, cone_K, device,
            depth + 1, max_depth,
            banned_heads=new_banned,
        )


def _assign_levels(node: _TreeNode) -> int:
    """Post-order: set node.level = height of subtree (leaf=0)."""
    if not node.children:
        node.level = 0
        return 0
    max_child = max(_assign_levels(c) for c in node.children)
    node.level = max_child + 1
    return node.level


# ---------------------------------------------------------------------------
# Tree → EventCluster conversion
# ---------------------------------------------------------------------------

def _get_label(
    idx: int,
    n_wn: int,
    wn_labels: list[str],
    all_canonicals: list[str],
) -> str:
    if idx < n_wn:
        return wn_labels[idx]
    return all_canonicals[idx - n_wn]


def _tree_to_clusters(
    roots: list[_TreeNode],
    n_wn: int,
    wn_labels: list[str],
    all_canonicals: list[str],
    emb_norms: np.ndarray,
    clusterer_name: str,
) -> tuple[list[EventCluster], dict[int, int]]:
    """BFS traversal → flat EventCluster list + {embedding_idx: leaf_cluster_pos}."""
    clusters: list[EventCluster] = []
    leaf_assignments: dict[int, int] = {}   # embedding index → leaf cluster_pos

    queue: deque[tuple[_TreeNode, int | None]] = deque(
        (r, None) for r in roots
    )
    while queue:
        node, parent_pos = queue.popleft()

        # Label: min-norm member (WordNet anchors preferred — they sit near origin)
        min_local = int(emb_norms[node.indices].argmin())
        label_idx = int(node.indices[min_local])
        label     = _get_label(label_idx, n_wn, wn_labels, all_canonicals)

        # member_count: only event texts (not WordNet anchors)
        n_event = int((node.indices >= n_wn).sum())

        pos = len(clusters)
        node.cluster_pos = pos
        clusters.append(EventCluster(
            label        = label,
            level        = node.level,
            parent_id    = parent_pos,
            member_count = n_event,
            clusterer    = clusterer_name,
        ))

        if node.children:
            for child in node.children:
                queue.append((child, pos))
        else:
            # Leaf: record embedding→cluster mapping for all event members
            for idx in node.indices:
                if idx >= n_wn:
                    leaf_assignments[int(idx)] = pos

    return clusters, leaf_assignments


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class PoincareClusterer(HierarchyInferrer):
    """
    Implements HierarchyInferrer using 50-dimensional Poincaré ball embeddings.

    Hierarchy depth is adaptive: the algorithm keeps subdividing clusters until
    each leaf contains ≤ min_cluster_size unique event texts.  Depth varies per
    branch; there is no fixed number of levels.

    When inject_knowledge="wordnet", WordNet noun synsets in [wn_depth_min,
    wn_depth_max] are embedded alongside event texts.  Because the model was
    trained on WordNet, these synsets land near the origin (small norm, large
    cone aperture) and naturally become top-level cluster heads.  The resulting
    cluster labels are human-readable synset definitions rather than specific
    event phrases.

    Parameters
    ----------
    checkpoint_path : str
        Path to the ugao25w3 Lightning checkpoint.
    min_cluster_size : int
        Stop subdividing when a cluster has ≤ this many unique event texts.
    branching_factor : int
        Target number of children per subdivision step.
    max_depth : int
        Hard recursion cap (safety; natural depth is usually much less).
    cone_K : float
        Aperture scaling — must match model training (ugao25w3: 0.5).
    batch_size : int
        Texts per forward-pass batch.
    device : str
        Torch device string.
    use_norm : bool
        Embed lowercased norm (True) or canonical description (False).
    inject_knowledge : str
        "wordnet" — embed WordNet synset definitions as top-level anchors.
        "none"    — use event texts only.
    wn_depth_min, wn_depth_max : int
        WordNet synset depth range to include (min_depth() in the IS-A tree).
    wn_pos : str
        WordNet part-of-speech filter: "n" (noun), "v" (verb), "a" (adj),
        or None for all.
    """

    def __init__(
        self,
        checkpoint_path: str = _DEFAULT_CHECKPOINT,
        min_cluster_size: int = 20,
        branching_factor: int = 10,
        max_depth: int = 10,
        cone_K: float = 0.5,
        batch_size: int = 256,
        device: str = "cpu",
        use_norm: bool = False,
        inject_knowledge: str = "abspyramid",
        abspyramid_path: str = "",
        **kwargs,
    ) -> None:
        self.checkpoint_path  = checkpoint_path
        self.min_cluster_size = min_cluster_size
        self.branching_factor = branching_factor
        self.max_depth        = max_depth
        self.cone_K           = cone_K
        self.batch_size       = batch_size
        self.device           = device
        self.use_norm         = use_norm
        self.inject_knowledge = inject_knowledge
        self.abspyramid_path  = abspyramid_path
        self._model           = None

    @property
    def name(self) -> str:
        return "poincare_cone"

    def _get_model(self):
        if self._model is None:
            print(f"[PoincareClusterer] Loading model from {self.checkpoint_path} …")
            self._model = _load_model(self.checkpoint_path, self.device)
        return self._model

    def infer(
        self,
        relations: list[CausalRelation],
    ) -> tuple[list[EventCluster], list[tuple[int, int, str, str]]]:

        if not relations:
            return [], []

        # ------------------------------------------------------------------
        # 1. Collect unique event texts
        # ------------------------------------------------------------------
        norm_to_canonical: dict[str, str] = {}
        for r in relations:
            for norm, canonical in (
                (r.cause_norm,   r.cause_canonical   or r.cause_text   or r.cause_norm),
                (r.effect_norm,  r.effect_canonical  or r.effect_text  or r.effect_norm),
            ):
                if norm not in norm_to_canonical:
                    norm_to_canonical[norm] = canonical

        all_norms:      list[str] = list(norm_to_canonical.keys())
        all_canonicals: list[str] = [norm_to_canonical[n] for n in all_norms]
        embed_texts               = all_norms if self.use_norm else all_canonicals
        n_total                   = len(all_norms)
        norm_to_idx               = {n: i for i, n in enumerate(all_norms)}

        # ------------------------------------------------------------------
        # 2. Load concept anchors (optional)
        # ------------------------------------------------------------------
        wn_labels:      list[str]         = []
        wn_embeddings:  np.ndarray | None = None

        if self.inject_knowledge == "abspyramid":
            if not self.abspyramid_path:
                print("[PoincareClusterer] inject_knowledge='abspyramid' but abspyramid_path not set — skipping")
            else:
                anchors = _load_abspyramid_concepts(self.abspyramid_path)
                if anchors:
                    wn_labels         = [a[0] for a in anchors]
                    wn_embed_texts    = [a[1] for a in anchors]
                    print(
                        f"[PoincareClusterer] Embedding {len(wn_labels)} AbsPyramid concept anchors …"
                    )
                    model         = self._get_model()
                    wn_embeddings = _embed_poincare(
                        model, wn_embed_texts, self.batch_size, self.device,
                        desc="AbsPyramid anchors",
                    )

        n_wn = len(wn_labels)

        # ------------------------------------------------------------------
        # 3. Embed event texts
        # ------------------------------------------------------------------
        print(
            f"[PoincareClusterer] Embedding {n_total} unique event texts "
            f"({'norm' if self.use_norm else 'canonical'}) …"
        )
        model            = self._get_model()
        event_embeddings = _embed_poincare(
            model, embed_texts, self.batch_size, self.device,
            desc="event texts",
        )

        # Combined array: [wn_anchors | event_texts]
        # WordNet indices: 0 … n_wn-1
        # Event indices:   n_wn … n_wn+n_total-1
        if n_wn > 0:
            all_embeddings = np.concatenate([wn_embeddings, event_embeddings], axis=0)
        else:
            all_embeddings = event_embeddings

        all_norms_arr = np.linalg.norm(all_embeddings, axis=1)   # (n_wn + n_total,)

        # Shift event norm_to_idx by n_wn so they point into all_embeddings
        norm_to_idx_shifted = {n: i + n_wn for n, i in norm_to_idx.items()}

        # ------------------------------------------------------------------
        # 4. Recursive cone clustering (adaptive depth)
        # ------------------------------------------------------------------
        print(
            f"[PoincareClusterer] Building adaptive-depth cone hierarchy "
            f"(branching={self.branching_factor}, min_leaf={self.min_cluster_size}, "
            f"max_depth={self.max_depth}) …"
        )

        # Virtual root holds ALL indices (WN anchors + event texts)
        all_indices = np.arange(len(all_embeddings), dtype=np.int64)
        virtual_root = _TreeNode(indices=all_indices)
        _recursive_subdivide(
            virtual_root, all_embeddings, all_norms_arr,
            self.min_cluster_size, self.branching_factor,
            self.cone_K, self.device,
            depth=0, max_depth=self.max_depth,
        )

        # The virtual root's children are the actual root clusters.
        # If no subdivision happened (all texts in one leaf), treat root itself.
        actual_roots = virtual_root.children if virtual_root.children else [virtual_root]

        # ------------------------------------------------------------------
        # 5. Assign post-order levels (leaf=0, root=max)
        # ------------------------------------------------------------------
        for root in actual_roots:
            _assign_levels(root)

        max_level = max(r.level for r in actual_roots)
        n_roots   = len(actual_roots)
        print(
            f"[PoincareClusterer] Tree: {n_roots} root clusters, "
            f"max depth {max_level + 1} levels."
        )

        # ------------------------------------------------------------------
        # 6. BFS → flat EventCluster list + leaf assignments
        # ------------------------------------------------------------------
        clusters, leaf_assignments = _tree_to_clusters(
            actual_roots, n_wn, wn_labels, all_canonicals,
            all_norms_arr, self.name,
        )

        # ------------------------------------------------------------------
        # 7. Build memberships (leaf level only, event texts only)
        # ------------------------------------------------------------------
        memberships: list[tuple[int, int, str, str]] = []
        for rel_idx, rel in enumerate(relations):
            for role, norm in (
                ("cause",  rel.cause_norm),
                ("effect", rel.effect_norm),
            ):
                event_idx   = norm_to_idx_shifted[norm]  # index into all_embeddings
                cluster_pos = leaf_assignments.get(event_idx)
                if cluster_pos is None:
                    # Should not happen; event is somewhere in the tree
                    continue
                memberships.append((rel_idx, cluster_pos, role, norm))

        return clusters, memberships
