"""Builds the Gene Ontology (Biological Process) event scaffold.

Why GO Biological Process (see notes.txt, 2026-07-03): the r/science causal
graph is biomedical-heavy — the Wikidata occurrence scaffold leaves the majority
of events unclustered ("mitochondrial function", "telomere shortening", "reduced
mortality risk", "gene expression"), because an event/occurrence taxonomy has no
categories for biological mechanisms. GO's Biological Process sub-ontology is
exactly a hierarchy of biological *processes* — i.e. biological events — which is
conceptually consistent with the causal-event framing:
    apoptotic process -> programmed cell death -> cell death -> cellular process
    -> biological_process.
It has a single clean root (biological_process, GO:0008150), 18 sensible
top-level categories (cellular process, metabolic process, immune system
process, developmental process, response to stimulus, biological regulation, …),
and ~31k terms with `is_a` edges. Intended to be COMBINED with the wikidata
event scaffold (societal events) via `scaffold: [wikidata, go]` — together they
cover both the societal-event and biomedical-mechanism halves of the corpus.

Molecular Function and Cellular Component namespaces are excluded: they describe
what a molecule *does* / *where it is*, not events/processes.

Source: the in-repo OBO (no download needed).
    datasets/benchmarks/go/go-basic.obo   (from conf-causal-hierarchical-embeddings)

GO's is_a graph is a DAG; rendered as a tree by keeping each term's MOST SPECIFIC
in-namespace is_a parent (deepest from the root), ties broken by GO id.

Output: pipeline/step4_hierarchy/resources/go_anchors.json
    {"source":"go","root":"GO:0008150","nodes":[{"name","label","depth","hypernyms":[...]}]}

Run once (offline):
    python -m pipeline.step4_hierarchy.build_go_anchors
"""
from __future__ import annotations

import json
import re
from pathlib import Path

_OBO = Path(
    "/mnt/ceph/storage/data-tmp/2026/thagen/conf-causal-hierarchical-embeddings"
    "/datasets/benchmarks/go/go-basic.obo"
)
BP_ROOT = "GO:0008150"   # biological_process


def _parse_obo() -> tuple[dict[str, str], dict[str, list[str]]]:
    """Return (name_by_id, is_a_parents) for non-obsolete biological_process terms."""
    text = _OBO.read_text()
    name_by_id: dict[str, str] = {}
    parents: dict[str, list[str]] = {}
    for block in text.split("\n[Term]\n")[1:]:
        tid_m = re.search(r"^id: (GO:\d+)", block, re.M)
        ns_m = re.search(r"^namespace: (\w+)", block, re.M)
        if not tid_m or not ns_m or ns_m.group(1) != "biological_process":
            continue
        if re.search(r"^is_obsolete: true", block, re.M):
            continue
        tid = tid_m.group(1)
        name_m = re.search(r"^name: (.+)$", block, re.M)
        name_by_id[tid] = name_m.group(1) if name_m else tid
        parents[tid] = re.findall(r"^is_a: (GO:\d+)", block, re.M)
    # keep only edges within the retained (BP, non-obsolete) set
    keep = set(name_by_id)
    parents = {t: [p for p in ps if p in keep] for t, ps in parents.items()}
    return name_by_id, parents


def build() -> dict:
    name_by_id, parents = _parse_obo()

    depth: dict[str, int] = {}

    def d(t: str, stack: frozenset) -> int:
        if t in depth:
            return depth[t]
        ps = [p for p in parents.get(t, ()) if p not in stack]
        depth[t] = 0 if not ps else 1 + max(d(p, stack | {t}) for p in ps)
        return depth[t]

    for t in name_by_id:
        d(t, frozenset())

    nodes = []
    for tid in sorted(name_by_id):
        cand = sorted(parents[tid], key=lambda p: (-depth[p], p))  # most specific first
        nodes.append({
            "name": tid,
            "label": name_by_id[tid],
            "depth": depth[tid],
            "hypernyms": cand,
        })
    return {"source": "go", "root": BP_ROOT, "nodes": nodes}


if __name__ == "__main__":
    data = build()
    out = Path(__file__).parent / "resources" / "go_anchors.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(data, indent=1))
    n_roots = sum(1 for n in data["nodes"] if not n["hypernyms"])
    print(f"Wrote {len(data['nodes'])} GO biological_process anchors "
          f"({n_roots} roots) to {out}")
