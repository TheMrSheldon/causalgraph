"""Builds the MAVEN / FrameNet event-type anchor scaffold.

Rationale (see notes.txt, 2026-07-03): the WordNet noun scaffold is organised
around *things* (entity -> abstraction -> attribute), so it cannot produce
event-type category labels like "Conflict" or "Catastrophe" — and WordNet even
splits events across incompatible upper branches (war is an `event`, crisis is
a `state`, flood is a `phenomenon`). MAVEN provides a purpose-built event-type
hierarchy (168 FrameNet-derived types with explicit IS-A edges), giving native
event labels: Attack -> Cause_harm -> Intentionally_act -> Action, etc.

Source: the in-repo MAVEN type-level pairs (no download needed).
  datasets/maven/data-typelevel/train.parquet  — (type) IS-A (parent type) rows
  datasets/maven/data-typelevel/type_vocab.txt — the type inventory

FrameNet inheritance is a DAG (a type may have several parents, e.g. Killing
IS-A Cause_harm AND Intentionally_act AND Action). We render it as a tree by
keeping, for each node, its MOST SPECIFIC parent (the candidate parent that is
itself deepest from a root) — so Killing attaches under Cause_harm, not the
more generic Action. Ties broken alphabetically for determinism.

Labels: type names are humanised (underscores -> spaces) for both display and
embedding, since the cone model embeds natural-language text.

Output: pipeline/step4_hierarchy/resources/maven_anchors.json
  {"source": "maven", "nodes": [{"name","label","depth","hypernyms":[...]}, ...]}

Run once (offline):
    python -m pipeline.step4_hierarchy.build_maven_anchors
"""
from __future__ import annotations

import json
from pathlib import Path

import pyarrow.parquet as pq

_CONF = Path("/mnt/ceph/storage/data-tmp/2026/thagen/conf-causal-hierarchical-embeddings")
_TYPELEVEL = _CONF / "datasets/maven/data-typelevel"


def _load_edges() -> tuple[set[str], list[tuple[str, str]]]:
    """Return (types, child->parent edges) from the type-level parquet."""
    vocab = {
        t.strip()
        for t in (_TYPELEVEL / "type_vocab.txt").read_text().split("\n")
        if t.strip()
    }
    tbl = pq.read_table(_TYPELEVEL / "train.parquet", columns=["text", "positive"])
    child = tbl.column("text").to_pylist()
    parent = tbl.column("positive").to_pylist()
    edges = set()
    for c, p in zip(child, parent):
        # type-level rows: both sides are type names in the vocab
        if c in vocab and p in vocab and c != p:
            edges.add((c, p))
    return vocab, sorted(edges)


def _depth_map(types: set[str], parents: dict[str, set[str]]) -> dict[str, int]:
    """Longest path from a root, over the full DAG (memoised, cycle-safe)."""
    depth: dict[str, int] = {}

    def d(t: str, stack: frozenset) -> int:
        if t in depth:
            return depth[t]
        ps = [p for p in parents.get(t, ()) if p not in stack]
        depth[t] = 0 if not ps else 1 + max(d(p, stack | {t}) for p in ps)
        return depth[t]

    for t in types:
        d(t, frozenset())
    return depth


def build() -> dict:
    types, edges = _load_edges()

    parents: dict[str, set[str]] = {t: set() for t in types}
    for c, p in edges:
        parents[c].add(p)

    depth = _depth_map(types, parents)

    # Tree parent = most specific (deepest) candidate parent; tie -> alphabetical.
    def humanise(t: str) -> str:
        return t.replace("_", " ")

    nodes = []
    for t in sorted(types):
        cand = sorted(parents[t], key=lambda p: (-depth[p], p))
        chosen = [cand[0]] if cand else []
        nodes.append({
            "name": t,                    # stable id = raw FrameNet type name
            "label": humanise(t),         # display / embed text
            "depth": depth[t],
            "hypernyms": chosen,
        })
    return {"source": "maven", "nodes": nodes}


if __name__ == "__main__":
    data = build()
    out = Path(__file__).parent / "resources" / "maven_anchors.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(data, indent=1))
    roots = [n["label"] for n in data["nodes"] if not n["hypernyms"]]
    print(f"Wrote {len(data['nodes'])} MAVEN event-type anchors "
          f"({len(roots)} roots: {', '.join(roots)}) to {out}")
