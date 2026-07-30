"""Step 4 alternative: precision-first hierarchy via a fixed WordNet anchor scaffold.

Why this exists (see notes.txt, 2026-07-02 discussion)
-------------------------------------------------------
PoincareClusterer (poincare.py) discovers structure by picking cluster heads
from the event pool itself (min-norm member). That works only if the pool has
a genuine abstraction gradient. Reddit causal events mostly don't: "climate
change", "cancer", "depression" are all roughly equally specific, so min-norm
head selection amounts to routing by embedding-norm noise, not real hierarchy.
Two concrete failures were observed on the live pipeline.db output:
  - ~47% of sampled events had E=0 for *no* WordNet anchor at all, yet the
    old algorithm still force-assigned them to the least-bad (argmin-energy)
    head — silently discarding the cone's E=0 containment guarantee.
  - Events with multiple valid ancestors (e.g. "PTSD" sits inside 7 nested
    cones: process, state, condition, disorder, attribute, feeling, emotion)
    got tie-broken by min energy rather than by which cone is tightest,
    picking "process" over the correct "disorder".

AnchorConeClusterer fixes both by never discovering structure among events:

  1. Anchors are FIXED, external, and carry real WordNet IS-A edges (built by
     build_wordnet_anchors.py — top-of-hierarchy nouns plus a curated
     r/science-relevant seed list with its ancestor closure). Events only
     ever attach to anchors, never to each other.
  2. An event attaches to an anchor only when E(anchor, event) == 0. If no
     anchor's cone contains the event, it is left UNPARENTED in a dedicated
     "(unclustered)" bucket rather than force-assigned — precision over
     coverage, matching the project's stated goal of avoiding wrong parents.
  3. Among all anchors that DO contain the event (its full valid ancestor
     chain), the immediate parent is the one with the LARGEST norm — the
     most specific (tightest) cone — not the minimum-energy one. This fixes
     the PTSD-style tie-break error.

Coverage (fraction of events successfully parented) is reported explicitly;
a sparse-but-correct tree is preferred over a dense-but-wrong one.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from ..protocols import CausalRelation, EventCluster, HierarchyInferrer
from .poincare import _cone_energy_rect, _embed_poincare, _load_model, _DEFAULT_CHECKPOINT

_SCAFFOLD_DIR = Path(__file__).parent / "resources"

UNCLUSTERED_LABEL = "(unclustered)"


def _normalize_scaffolds(scaffold, anchors_path: str = "") -> list[tuple[str, str]]:
    """Return [(source_name, json_path), ...] for one or several scaffolds.

    ``scaffold`` may be a single name ("wikidata"), a list (["wikidata", "go"]),
    or a comma-separated string ("wikidata,go"). Each name selects
    ``resources/<name>_anchors.json`` (built by build_<name>_anchors.py). An
    explicit ``anchors_path`` overrides to a single scaffold at that path.
    """
    if anchors_path:
        return [("custom", anchors_path)]
    if isinstance(scaffold, str):
        names = [s.strip() for s in scaffold.split(",") if s.strip()]
    else:
        names = [str(s).strip() for s in scaffold if str(s).strip()]
    return [(n.lower(), str(_SCAFFOLD_DIR / f"{n.lower()}_anchors.json")) for n in names]


# ---------------------------------------------------------------------------
# Anchor scaffold loading
# ---------------------------------------------------------------------------

def _load_anchor_scaffold(
    sources: list[tuple[str, str]],
) -> tuple[list[str], list[str], dict[str, str | None]]:
    """Load and MERGE one or more scaffold JSON files.

    Scaffold format (produced by any build_<source>_anchors.py):
        {"nodes": [{"name": <stable id>, "label": <display/embed text>,
                    "hypernyms": [<parent name>, ...]}, ...]}
    (the legacy WordNet builder used the key "synsets" — accepted too.)

    When several scaffolds are given they are merged into one forest. Node names
    are namespaced with their source ("wikidata:Q198") so ids never collide
    across sources, and each scaffold's internal hypernym references stay intact.
    The scaffolds remain disjoint trees that coexist — an event attaches to
    whichever anchor's cone contains it most specifically, regardless of source.

    hypernym_of[name] = the node's parent restricted to its own scaffold (None
    for roots). A node may list >1 hypernym (the source is a DAG); only the
    first in-scaffold one is kept so each scaffold renders as a tree.
    """
    names: list[str] = []
    labels: list[str] = []
    hypernym_of: dict[str, str | None] = {}

    for source, path in sources:
        data = json.loads(Path(path).read_text())
        entries = data.get("nodes", data.get("synsets", []))
        present = {e["name"] for e in entries}
        pref = f"{source}:"
        for entry in entries:
            nid = pref + entry["name"]
            names.append(nid)
            labels.append(entry["label"])
            parents = [h for h in entry.get("hypernyms", []) if h in present]
            hypernym_of[nid] = (pref + parents[0]) if parents else None

    return names, labels, hypernym_of


# ---------------------------------------------------------------------------
# Assignment
# ---------------------------------------------------------------------------

def _assign_events_to_anchors(
    anchor_embs: np.ndarray,
    event_embs: np.ndarray,
    cone_K: float,
    device: str,
    eps: float = 1e-4,
    max_cells: int = 8e7,
) -> np.ndarray:
    """Return assigned_anchor_idx (N,) int array; -1 = unparented (no E=0 anchor).

    Among anchors with E(anchor, event) <= eps, picks the one with the largest
    Euclidean norm (tightest / most specific containing cone).

    The per-chunk cone-energy matrix is (n_anchors, chunk_size); chunk_size is
    derived from ``max_cells`` so that matrix stays bounded regardless of how
    many anchors the (possibly merged, multi-source) scaffold contributes —
    e.g. wikidata+go is ~27k anchors, which would otherwise make a 20k-event
    chunk a 540M-cell (>2GB) array.
    """
    anchor_norms = np.linalg.norm(anchor_embs, axis=1)
    n_events = len(event_embs)
    n_anchors = max(1, len(anchor_embs))
    chunk_size = max(1000, int(max_cells // n_anchors))
    assigned = np.full(n_events, -1, dtype=np.int64)

    for start in range(0, n_events, chunk_size):
        chunk = event_embs[start:start + chunk_size]
        E = _cone_energy_rect(anchor_embs, chunk, cone_K, device)  # (n_anchors, chunk_n)
        valid = E <= eps                                           # (n_anchors, chunk_n)
        # Mask invalid anchors to -inf norm so argmax skips them.
        masked_norms = np.where(valid, anchor_norms[:, None], -np.inf)
        best = masked_norms.argmax(axis=0)
        has_any = valid.any(axis=0)
        chunk_assigned = np.where(has_any, best, -1)
        assigned[start:start + chunk_size] = chunk_assigned

    return assigned


def _collapse_small_leaves(
    assigned_anchor_idx: np.ndarray,
    names: list[str],
    hypernym_of: dict[str, str | None],
    min_leaf_size: int,
) -> np.ndarray:
    """Reassign events from anchors with too few direct members to their
    nearest ancestor with enough accumulated members.

    Without this, a leaf cluster is whatever anchor an event's tightest
    containing cone happened to be — with ~27K fixed anchors available, many
    end up hit by only one or two corpus events each (empirically 21-43% of
    leaves were singletons before this fix). Leaves are meant to be the most
    specific grouping that still generalizes over multiple posts, not a
    one-off match to an overly specific anchor. The catch-all root is always
    kept regardless of size (hypernym_of[root] is None, so the climb stops).
    """
    if min_leaf_size <= 1:
        return assigned_anchor_idx

    name_to_idx = {n: i for i, n in enumerate(names)}
    direct_counts: dict[str, int] = {}
    for idx in assigned_anchor_idx:
        if idx >= 0:
            direct_counts[names[idx]] = direct_counts.get(names[idx], 0) + 1

    # Used anchors = every anchor with a direct event, plus all its ancestors.
    used: set[str] = set()
    for name in direct_counts:
        cur = name
        while cur is not None and cur not in used:
            used.add(cur)
            cur = hypernym_of.get(cur)

    children_of: dict[str, list[str]] = {n: [] for n in used}
    roots: list[str] = []
    for n in used:
        parent = hypernym_of.get(n)
        if parent is not None and parent in used:
            children_of[parent].append(n)
        else:
            roots.append(n)

    # Bottom-up tree DP: each used anchor either "keeps" its own pooled count
    # (resolved[node] = node) if it meets min_leaf_size (or has no parent),
    # or defers its whole pool upward (resolved[node] = None) for the parent
    # to accumulate — critically, a node's decision only happens AFTER all of
    # its children are resolved, so sibling leaves under the same parent get
    # to pool together before that parent decides whether it, in turn, has
    # enough. Processing one child at a time and climbing eagerly (an earlier,
    # buggy version of this function) could skip past a parent that would
    # have met the threshold once ALL its children's counts were combined.
    resolved: dict[str, str | None] = {}

    def resolve(node: str) -> int:
        total = direct_counts.get(node, 0)
        for child in children_of.get(node, []):
            child_total = resolve(child)
            if resolved.get(child) is None:
                total += child_total
        if total >= min_leaf_size or hypernym_of.get(node) is None:
            resolved[node] = node
        else:
            resolved[node] = None
        return total

    for r in roots:
        resolve(r)

    def final_of(name: str) -> str:
        cur = name
        while resolved.get(cur) is None:
            cur = hypernym_of[cur]
        return cur

    redirect = {n: final_of(n) for n in used if final_of(n) != n}
    if not redirect:
        return assigned_anchor_idx

    new_assigned = assigned_anchor_idx.copy()
    for i, idx in enumerate(assigned_anchor_idx):
        if idx < 0:
            continue
        name = names[idx]
        if name in redirect:
            new_assigned[i] = name_to_idx[redirect[name]]
    return new_assigned


# ---------------------------------------------------------------------------
# Tree construction
# ---------------------------------------------------------------------------

@dataclass
class _Node:
    name: str
    label: str
    children: list["_Node"] = field(default_factory=list)
    direct_count: int = 0     # events assigned directly to this anchor
    level: int = 0
    cluster_pos: int = -1


def _build_used_tree(
    names: list[str],
    labels: list[str],
    hypernym_of: dict[str, str | None],
    assigned_anchor_idx: np.ndarray,
) -> tuple[list[_Node], _Node | None]:
    """Build the tree restricted to anchors on the path from a triggered anchor
    to the root, plus a synthetic "(unclustered)" root for unparented events.

    Returns (root_list, unclustered_node_or_None). root_list has 0 or 1 real
    root (entity.n.01, generally) since the scaffold is a single connected tree.
    """
    direct_counts: dict[str, int] = {}
    for idx in assigned_anchor_idx:
        if idx >= 0:
            name = names[idx]
            direct_counts[name] = direct_counts.get(name, 0) + 1

    # Every anchor on the path from a triggered anchor up to the root is "used".
    used: set[str] = set()
    for name in direct_counts:
        cur = name
        while cur is not None and cur not in used:
            used.add(cur)
            cur = hypernym_of.get(cur)

    nodes: dict[str, _Node] = {
        n: _Node(name=n, label=labels[names.index(n)], direct_count=direct_counts.get(n, 0))
        for n in used
    }
    roots: list[_Node] = []
    for n in used:
        parent_name = hypernym_of.get(n)
        if parent_name is not None and parent_name in nodes:
            nodes[parent_name].children.append(nodes[n])
        else:
            roots.append(nodes[n])

    n_unclustered = int((assigned_anchor_idx == -1).sum())
    unclustered = _Node(name="__unclustered__", label=UNCLUSTERED_LABEL,
                         direct_count=n_unclustered) if n_unclustered > 0 else None

    return roots, unclustered


def _assign_levels(node: _Node) -> int:
    if not node.children:
        node.level = 0
        return 0
    node.level = max(_assign_levels(c) for c in node.children) + 1
    return node.level


def _cumulative_count(node: _Node) -> int:
    total = node.direct_count + sum(_cumulative_count(c) for c in node.children)
    node.direct_count = total  # overwrite in place: now cumulative, matches EventCluster semantics
    return total


def _flatten(
    roots: list[_Node],
    unclustered: _Node | None,
    clusterer_name: str,
) -> tuple[list[EventCluster], dict[str, int]]:
    """BFS -> flat EventCluster list + {anchor_name_or___unclustered__: cluster_pos}."""
    from collections import deque

    clusters: list[EventCluster] = []
    pos_of: dict[str, int] = {}

    all_roots = list(roots)
    if unclustered is not None:
        all_roots.append(unclustered)

    queue: deque[tuple[_Node, int | None]] = deque((r, None) for r in all_roots)
    while queue:
        node, parent_pos = queue.popleft()
        pos = len(clusters)
        node.cluster_pos = pos
        pos_of[node.name] = pos
        clusters.append(EventCluster(
            label=node.label,
            level=node.level,
            parent_id=parent_pos,
            member_count=node.direct_count,
            clusterer=clusterer_name,
        ))
        for child in node.children:
            queue.append((child, pos))

    return clusters, pos_of


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class AnchorConeClusterer(HierarchyInferrer):
    """Precision-first hierarchy inference via a fixed WordNet anchor scaffold.

    See module docstring for the rationale. Unlike PoincareClusterer, depth is
    NOT adaptive over the event pool — it is fixed by the WordNet scaffold
    (build_wordnet_anchors.py), and events never become cluster heads.

    Parameters
    ----------
    checkpoint_path : str
        Path to the cone-embedding Lightning checkpoint (ugao25w3 by default).
    scaffold : str | list[str]
        Which fixed anchor scaffold(s) to use, e.g. "wikidata", ["wikidata",
        "go"], or "wikidata,go". Each selects resources/<name>_anchors.json
        (built by the matching build_<name>_anchors.py). Multiple scaffolds are
        merged into one forest (see _load_anchor_scaffold). Ignored if
        ``anchors_path`` is given.
    anchors_path : str
        Explicit override path to a single scaffold JSON; wins over ``scaffold``.
    cone_K : float
        Aperture scaling — must match model training (ugao25w3: 0.5).
    eps : float
        Numerical tolerance for "E(anchor, event) == 0" containment.
    min_leaf_size : int
        Leaves (anchors with no used descendant) with fewer than this many
        directly-assigned events are collapsed into their nearest ancestor
        with enough accumulated members — a leaf should generalize over
        multiple posts that can't be sensibly split further, not be a
        one-off match to an overly specific anchor. Set to 1 to disable.
    batch_size, device, use_norm : see PoincareClusterer.
    """

    def __init__(
        self,
        checkpoint_path: str = _DEFAULT_CHECKPOINT,
        scaffold="wordnet",
        anchors_path: str = "",
        cone_K: float = 0.5,
        eps: float = 1e-4,
        batch_size: int = 256,
        device: str = "cpu",
        use_norm: bool = False,
        min_leaf_size: int = 5,
        **kwargs,
    ) -> None:
        self.checkpoint_path = checkpoint_path
        self.scaffold_sources = _normalize_scaffolds(scaffold, anchors_path)
        self.scaffold = "+".join(s for s, _ in self.scaffold_sources)
        self.cone_K = cone_K
        self.eps = eps
        self.batch_size = batch_size
        self.device = device
        self.use_norm = use_norm
        self.min_leaf_size = min_leaf_size
        self._model = None

    @property
    def name(self) -> str:
        return f"anchor_cone_{self.scaffold}"

    def _get_model(self):
        if self._model is None:
            print(f"[AnchorConeClusterer] Loading model from {self.checkpoint_path} …")
            self._model = _load_model(self.checkpoint_path, self.device)
        return self._model

    def infer(
        self,
        relations: list[CausalRelation],
    ) -> tuple[list[EventCluster], list[tuple[int, int, str, str]]]:

        if not relations:
            return [], []

        # 1. Collect unique event texts (identical to PoincareClusterer).
        norm_to_canonical: dict[str, str] = {}
        for r in relations:
            for norm, canonical in (
                (r.cause_norm,  r.cause_canonical  or r.cause_text  or r.cause_norm),
                (r.effect_norm, r.effect_canonical or r.effect_text or r.effect_norm),
            ):
                if norm not in norm_to_canonical:
                    norm_to_canonical[norm] = canonical

        all_norms:      list[str] = list(norm_to_canonical.keys())
        all_canonicals: list[str] = [norm_to_canonical[n] for n in all_norms]
        embed_texts               = all_norms if self.use_norm else all_canonicals
        norm_to_idx               = {n: i for i, n in enumerate(all_norms)}

        # 2. Load + embed the fixed anchor scaffold(s).
        names, labels, hypernym_of = _load_anchor_scaffold(self.scaffold_sources)
        model = self._get_model()
        print(f"[AnchorConeClusterer] Embedding {len(names)} '{self.scaffold}' anchors "
              f"from {[p for _, p in self.scaffold_sources]} …")
        anchor_embs = _embed_poincare(model, labels, self.batch_size, self.device, desc="anchors")

        # 3. Embed event texts.
        print(f"[AnchorConeClusterer] Embedding {len(embed_texts)} unique event texts …")
        event_embs = _embed_poincare(model, embed_texts, self.batch_size, self.device, desc="events")

        # 4. Precision-first assignment: E(anchor,event)=0 required; ties broken
        #    by max anchor norm (most specific containing cone).
        assigned_idx = _assign_events_to_anchors(
            anchor_embs, event_embs, self.cone_K, self.device, eps=self.eps,
        )
        n_covered = int((assigned_idx >= 0).sum())
        coverage = n_covered / len(assigned_idx) if len(assigned_idx) else 0.0
        print(f"[AnchorConeClusterer] Coverage: {n_covered}/{len(assigned_idx)} "
              f"events parented ({coverage*100:.1f}%); rest -> \"{UNCLUSTERED_LABEL}\"")

        # 4b. Collapse leaves with too few direct members into their nearest
        # populated ancestor — doesn't change coverage (an event keeps SOME
        # anchor), only which specific anchor ends up as its leaf.
        assigned_idx = _collapse_small_leaves(assigned_idx, names, hypernym_of, self.min_leaf_size)

        # 5. Build the used-anchor tree + unclustered bucket, compute levels/counts.
        roots, unclustered = _build_used_tree(names, labels, hypernym_of, assigned_idx)
        for root in roots:
            _assign_levels(root)
            _cumulative_count(root)
        if unclustered is not None:
            unclustered.level = 0  # flat catch-all bucket, never expands further

        clusters, pos_of = _flatten(roots, unclustered, self.name)

        # 6. Build memberships (leaf level only = the anchor each event was
        #    DIRECTLY assigned to, or the unclustered bucket).
        memberships: list[tuple[int, int, str, str]] = []
        for rel_idx, rel in enumerate(relations):
            for role, norm in (("cause", rel.cause_norm), ("effect", rel.effect_norm)):
                event_idx = norm_to_idx[norm]
                anchor_idx = assigned_idx[event_idx]
                anchor_name = names[anchor_idx] if anchor_idx >= 0 else "__unclustered__"
                cluster_pos = pos_of.get(anchor_name)
                if cluster_pos is None:
                    continue
                memberships.append((rel_idx, cluster_pos, role, norm))

        return clusters, memberships
