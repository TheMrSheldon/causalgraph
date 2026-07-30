"""Builds the Wikidata event/occurrence anchor scaffold.

Rationale (see notes.txt, 2026-07-03): Wikidata's `occurrence` (Q1190554)
subclass tree is the closest public ontology to the event-category hierarchy we
want for the causal graph — it contains war, conflict, crisis, natural disaster,
drought, protest, epidemic, etc. with real `subclass of` (P279) edges, matching
the target label style ("Conflict -> war", "crisis -> economic crisis").

Method: we do NOT root at `occurrence` — its ~700 flat direct subclasses are
mostly irrelevant, and walking P279 upward drags in Wikidata's broken upper
ontology (war ends up a "biological process" under "entity" at depth ~20).
Instead the curated event categories ARE the top-level roots, and we only
descend:
  1. SEEDS = curated event classes (conflict, crisis, disaster, epidemic,
     protest, accident, revolution, …), resolved to QIDs. These become the
     top-level categories of the scaffold (a forest, not one tree — which
     matches how the categories are actually presented).
  2. Descendants: BFS down P279 (subclass of) from the seeds to MAX_DEPTH
     levels, bounded by MAX_NODES, pulling in specific subtypes AND the
     intermediate connectors between seeds (e.g. descending from `conflict`
     reaches violent conflict -> armed conflict -> war -> civil war).
  3. Fetch ALL P279 edges among the discovered set, then pick each node's MOST
     SPECIFIC in-set parent (the candidate parent itself deepest from a root) —
     so `war` attaches under `armed conflict`, not directly under `conflict`.
     Ties broken by QID for determinism. Nodes whose only in-set parents are
     absent become additional roots.

Wikidata's subclass graph is a noisy DAG. Residual noise (nodes no events ever
land in) is NOT filtered here — AnchorConeClusterer's data-driven "used tree" +
min-member pruning drops any anchor that contains no events, so the served
hierarchy stays clean.

Output: pipeline/step4_hierarchy/resources/wikidata_anchors.json
  {"source":"wikidata","root":"Q1190554",
   "nodes":[{"name":<QID>,"label":<en label>,"depth":d,"hypernyms":[<QID>,...]}]}

Run once (needs network — query.wikidata.org):
    python -m pipeline.step4_hierarchy.build_wikidata_anchors
"""
from __future__ import annotations

import json
import time
import urllib.parse
import urllib.request
from pathlib import Path

MAX_DEPTH = 5                  # BFS depth *below* each seed
MAX_NODES = 4000
_ENDPOINT = "https://query.wikidata.org/sparql"
_UA = "causalgraph-scaffold/1.0 (research; contact tim.hagen@uni-kassel.de)"

# Curated seed classes = the top-level event categories. Their P279 subtrees are
# the event hierarchy relevant to the causal graph. QIDs resolved from Wikidata.
# High-level categories (conflict, crisis, disaster, …) are listed first; the
# more specific ones (war, drought, …) are usually reached by descending from
# those anyway but are kept explicitly to guarantee inclusion.
SEED_QIDS = [
    "Q180684",       # conflict
    "Q381072",       # crisis
    "Q3839081",      # disaster
    "Q8065",         # natural disaster
    "Q3193890",      # environmental disaster
    "Q3241045",      # disease outbreak
    "Q273120",       # protest
    "Q171558",       # accident
    "Q10931",        # revolution
    "Q55814",        # extinction event
    "Q350604",       # armed conflict
    "Q198",          # war
    "Q2672648",      # social conflict
    "Q290178",       # economic crisis
    "Q114380",       # financial crisis
    "Q3002772",      # political crisis
    "Q176494",       # recession
    "Q43059",        # drought
    "Q168247",       # famine
    "Q44512",        # epidemic
    "Q12184",        # pandemic
]


def _sparql(query: str, retries: int = 3) -> list[dict]:
    data = urllib.parse.urlencode({"query": query}).encode()
    req = urllib.request.Request(
        _ENDPOINT, data=data,
        headers={"Accept": "application/sparql-results+json", "User-Agent": _UA},
    )
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(req, timeout=60) as resp:
                return json.loads(resp.read())["results"]["bindings"]
        except Exception as e:  # noqa: BLE001
            if attempt == retries - 1:
                raise
            print(f"  [retry {attempt+1}] {e}")
            time.sleep(3)
    return []


def _children_of(qids: list[str], batch: int = 200) -> list[tuple[str, str, str]]:
    """Return (child_qid, child_label, parent_qid) for all P279 children of `qids`."""
    out: list[tuple[str, str, str]] = []
    for i in range(0, len(qids), batch):
        values = " ".join(f"wd:{q}" for q in qids[i:i + batch])
        query = f"""
        SELECT ?c ?cLabel ?parent WHERE {{
          VALUES ?parent {{ {values} }}
          ?c wdt:P279 ?parent .
          SERVICE wikibase:label {{ bd:serviceParam wikibase:language "en". }}
        }}
        """
        for b in _sparql(query):
            out.append((
                b["c"]["value"].rsplit("/", 1)[-1],
                b["cLabel"]["value"],
                b["parent"]["value"].rsplit("/", 1)[-1],
            ))
    return out


def _labels_of(qids: list[str]) -> dict[str, str]:
    """Return {qid: en label} for the given QIDs."""
    values = " ".join(f"wd:{q}" for q in qids)
    query = f"""
    SELECT ?s ?sLabel WHERE {{
      VALUES ?s {{ {values} }}
      SERVICE wikibase:label {{ bd:serviceParam wikibase:language "en". }}
    }}
    """
    out: dict[str, str] = {}
    for b in _sparql(query):
        out[b["s"]["value"].rsplit("/", 1)[-1]] = b["sLabel"]["value"]
    return out


def _in_set_parents(qids: list[str], node_set: set[str], batch: int = 200) -> dict[str, set[str]]:
    """For each node, all its P279 parents that are also in `node_set`."""
    parents: dict[str, set[str]] = {q: set() for q in qids}
    for i in range(0, len(qids), batch):
        values = " ".join(f"wd:{q}" for q in qids[i:i + batch])
        query = f"""
        SELECT ?c ?parent WHERE {{
          VALUES ?c {{ {values} }}
          ?c wdt:P279 ?parent .
        }}
        """
        for b in _sparql(query):
            cq = b["c"]["value"].rsplit("/", 1)[-1]
            pq = b["parent"]["value"].rsplit("/", 1)[-1]
            if pq in node_set and pq != cq:
                parents[cq].add(pq)
    return parents


def build() -> dict:
    # ---- Phase 1: seed labels (the top-level categories) ----
    labels: dict[str, str] = _labels_of(SEED_QIDS)
    print(f"  seeds: {len(labels)} nodes")

    # ---- Phase 2: BFS down from the seeds to pull in subtypes + connectors ----
    frontier = list(SEED_QIDS)
    for depth in range(MAX_DEPTH):
        if not frontier or len(labels) >= MAX_NODES:
            break
        next_frontier = []
        for cq, label, _pq in _children_of(frontier):
            if cq not in labels and len(labels) < MAX_NODES:
                labels[cq] = label
                next_frontier.append(cq)
        frontier = next_frontier
        print(f"  BFS depth {depth+1}: {len(labels)} nodes so far")

    # Drop nodes whose English label didn't resolve (QID == label): unusable as text.
    node_set = {q for q, l in labels.items() if l != q}

    # ---- Phase 3: all in-set P279 edges, then most-specific parent ----
    # Because the node set is seeds + DESCENDANTS only (no ancestors), the
    # abstract Wikidata upper ontology (group dynamics, occurrent, entity, …)
    # is simply absent, so no chain can route up into it. High-level seeds
    # (conflict, crisis, disaster) have no in-set parent -> they become roots;
    # specific seeds (war, drought, armed conflict) attach under their in-set
    # parents naturally.
    parents = _in_set_parents(sorted(node_set), node_set)

    # depth = longest path to a root, over the in-set DAG (memoised, cycle-safe)
    depth_of: dict[str, int] = {}

    def d(q: str, stack: frozenset) -> int:
        if q in depth_of:
            return depth_of[q]
        ps = [p for p in parents.get(q, ()) if p not in stack]
        depth_of[q] = 0 if not ps else 1 + max(d(p, stack | {q}) for p in ps)
        return depth_of[q]

    for q in node_set:
        d(q, frozenset())

    nodes = []
    for q in sorted(node_set):
        # most specific (deepest) parent first; keep the rest as extra hypernyms
        cand = sorted(parents[q], key=lambda p: (-depth_of[p], p))
        nodes.append({
            "name": q,
            "label": labels[q],
            "depth": depth_of[q],
            "hypernyms": cand,
        })
    return {"source": "wikidata", "seeds": SEED_QIDS, "nodes": nodes}


if __name__ == "__main__":
    data = build()
    out = Path(__file__).parent / "resources" / "wikidata_anchors.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(data, indent=1))
    print(f"Wrote {len(data['nodes'])} Wikidata occurrence-subtree anchors to {out}")
