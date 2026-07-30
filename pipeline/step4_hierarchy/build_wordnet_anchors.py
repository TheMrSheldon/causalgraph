"""Builds the fixed WordNet anchor scaffold used by AnchorConeClusterer.

Rationale (see notes.txt discussion, 2026-07-02): the previous PoincareClusterer
selected cluster heads from the event pool itself via min-norm — but Reddit
causal events are mostly flat, same-level concepts ("climate change", "cancer",
"depression") with no latent abstraction gradient. Min-norm head selection on
such a pool is directional noise, not real hierarchy discovery.

The fix: provide a FIXED external scaffold of genuinely abstract WordNet nouns,
each carrying its real WordNet IS-A edges. Events attach as leaves to this
scaffold, never to each other. Two disjoint groups make up the scaffold:

  1. All noun synsets with min_depth() <= TOP_DEPTH — the top of WordNet's noun
     hierarchy (entity, abstraction, process, event, phenomenon, ...). These are
     domain-agnostic and were manually spot-checked to be low-noise at this depth.
  2. A curated list of domain-relevant mid-depth synsets for the r/science
     domain (disease, organism, gene, climate, technology, behavior, ...),
     plus the ANCESTOR CLOSURE of that list up to the WordNet root — this
     keeps the tree connected without pulling in irrelevant WordNet subtrees.

Output: pipeline/step4_hierarchy/resources/wordnet_anchors.json
  {"synsets": [{"name": "disease.n.01", "label": "disease", "depth": 9,
                "hypernyms": ["pathological_state.n.01", ...]}, ...]}

Run once (offline, requires nltk + wordnet corpus):
    python -m pipeline.step4_hierarchy.build_wordnet_anchors
"""
from __future__ import annotations

import json
from pathlib import Path

from nltk.corpus import wordnet as wn

TOP_DEPTH = 3  # depth<=3 noun synsets: 254 synsets, spot-checked as low-noise
               # top-level categories (entity, abstraction, process, event, ...)

# Domain seeds for r/science causal claims. Picked by lemma name; each maps to
# a specific verified sense (WordNet's first sense is wrong for a few of these,
# noted inline).
DOMAIN_SEEDS: dict[str, str] = {
    "disease": "disease.n.01", "disorder": "disorder.n.01", "symptom": "symptom.n.01",
    "syndrome": "syndrome.n.01", "infection": "infection.n.01", "cancer": "cancer.n.01",
    "medication": "medicine.n.02", "drug": "drug.n.01", "vaccine": "vaccine.n.01",
    "therapy": "therapy.n.01", "surgery": "surgery.n.01",
    "organism": "organism.n.01", "cell": "cell.n.01", "gene": "gene.n.01",
    "protein": "protein.n.01", "hormone": "hormone.n.01", "bacterium": "bacteria.n.01",
    "virus": "virus.n.01", "brain": "brain.n.01", "nervous_system": "nervous_system.n.01",
    "immune_system": "immune_system.n.01", "cardiovascular_system": "circulatory_system.n.01",
    "behavior": "behavior.n.01", "cognition": "cognition.n.01", "emotion": "emotion.n.01",
    "perception": "percept.n.01", "memory": "memory.n.01", "personality": "personality.n.01",
    "mental_state": "psychological_state.n.01",
    "stress": "tension.n.01",           # WordNet's 1st sense is linguistic (syllable) stress
    "sleep": "sleep.n.01",
    "climate": "climate.n.01", "weather": "weather.n.01", "pollution": "pollution.n.01",
    "ecosystem": "ecosystem.n.01", "species": "species.n.01", "habitat": "habitat.n.01",
    "chemical_compound": "compound.n.02",
    "element": "chemical_element.n.01",  # WordNet's 1st sense is "abstract part of something"
    "molecule": "molecule.n.01", "reaction": "chemical_reaction.n.01",
    "radiation": "radiation.n.01", "energy": "energy.n.01",
    "technology": "technology.n.01", "device": "device.n.01", "software": "software.n.01",
    "algorithm": "algorithm.n.01", "material": "material.n.01",
    "society": "society.n.01", "economy": "economy.n.01", "population": "population.n.01",
    "culture": "culture.n.01", "policy": "policy.n.01", "education": "education.n.01",
    "food": "food.n.01", "nutrition": "nutrition.n.01", "diet": "diet.n.01",
    "exercise": "exercise.n.01",
}


def build() -> dict:
    top = {s for s in wn.all_synsets("n") if s.min_depth() <= TOP_DEPTH}

    seeds = {wn.synset(name) for name in DOMAIN_SEEDS.values()}
    closure: set = set()
    frontier = set(seeds)
    while frontier:
        closure |= frontier
        frontier = {h for s in frontier for h in s.hypernyms()} - closure

    anchors = top | closure
    by_name = {s.name(): s for s in anchors}

    out = []
    for name, s in sorted(by_name.items()):
        out.append({
            "name": name,
            "label": s.lemma_names()[0].replace("_", " "),
            "depth": s.min_depth(),
            "hypernyms": [h.name() for h in s.hypernyms() if h.name() in by_name],
        })
    return {"top_depth": TOP_DEPTH, "synsets": out}


if __name__ == "__main__":
    data = build()
    out_path = Path(__file__).parent / "resources" / "wordnet_anchors.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(data, indent=1))
    n_roots = sum(1 for s in data["synsets"] if not s["hypernyms"])
    print(f"Wrote {len(data['synsets'])} anchor synsets ({n_roots} roots) to {out_path}")
