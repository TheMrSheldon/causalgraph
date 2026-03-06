"""
Registry: loads pipeline step implementations from config.yaml.

Each step maps string keys → (module_path, class_name).
The loader imports the module lazily, instantiates with config kwargs,
and validates Protocol conformance at runtime.
"""
from __future__ import annotations

import copy
import importlib
from typing import Any

import yaml

from pipeline.protocols import CausalityIdentifier, CausalExtractor, HierarchyInferrer

# ---------------------------------------------------------------------------
# Implementation registries
# ---------------------------------------------------------------------------

_STEP1_REGISTRY: dict[str, tuple[str, str]] = {
    "regex": (
        "pipeline.step1_identification.regex_identifier",
        "RegexIdentifier",
    ),
    "llm_openai": (
        "pipeline.step1_identification.llm_identifier",
        "LLMIdentifier",
    ),
    "llm_anthropic": (
        "pipeline.step1_identification.llm_identifier",
        "LLMIdentifier",
    ),
    "zero_shot": (
        "pipeline.step1_identification.zero_shot_identifier",
        "ZeroShotIdentifier",
    ),
}

_STEP2_REGISTRY: dict[str, tuple[str, str]] = {
    "regex_spacy": (
        "pipeline.step2_extraction.regex_spacy_extractor",
        "RegexSpacyExtractor",
    ),
    "llm_openai": (
        "pipeline.step2_extraction.llm_extractor",
        "LLMExtractor",
    ),
    "llm_anthropic": (
        "pipeline.step2_extraction.llm_extractor",
        "LLMExtractor",
    ),
}

_STEP3_REGISTRY: dict[str, tuple[str, str]] = {
    "embedding_hdbscan": (
        "pipeline.step3_hierarchy.embedding_clusterer",
        "EmbeddingClusterer",
    ),
    "tfidf_ward": (
        "pipeline.step3_hierarchy.tfidf_clusterer",
        "TFIDFClusterer",
    ),
    "embedding_ward": (
        "pipeline.step3_hierarchy.embedding_ward_clusterer",
        "EmbeddingWardClusterer",
    ),
    "llm": (
        "pipeline.step3_hierarchy.llm_topic_grouper",
        "LLMTopicGrouper",
    ),
}


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _load(registry: dict[str, tuple[str, str]], key: str, kwargs: dict[str, Any]) -> Any:
    if key not in registry:
        raise ValueError(
            f"Unknown implementation '{key}'. Available: {sorted(registry.keys())}"
        )
    module_path, class_name = registry[key]
    module = importlib.import_module(module_path)
    cls = getattr(module, class_name)
    return cls(**kwargs)


def _pop_implementation(step_cfg: dict) -> tuple[str, dict]:
    """Extract the 'implementation' key; return (key, remaining_kwargs)."""
    cfg = copy.deepcopy(step_cfg)
    impl_key = cfg.pop("implementation")
    return impl_key, cfg


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load_config(path: str = "config.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def build_identifier(config: dict) -> CausalityIdentifier:
    impl_key, kwargs = _pop_implementation(config["pipeline"]["step1_identification"])
    obj = _load(_STEP1_REGISTRY, impl_key, kwargs)
    if not isinstance(obj, CausalityIdentifier):
        raise TypeError(f"{type(obj).__name__} does not satisfy CausalityIdentifier protocol")
    return obj


def build_extractor(config: dict) -> CausalExtractor:
    impl_key, kwargs = _pop_implementation(config["pipeline"]["step2_extraction"])
    obj = _load(_STEP2_REGISTRY, impl_key, kwargs)
    if not isinstance(obj, CausalExtractor):
        raise TypeError(f"{type(obj).__name__} does not satisfy CausalExtractor protocol")
    return obj


def build_inferrer(config: dict) -> HierarchyInferrer:
    impl_key, kwargs = _pop_implementation(config["pipeline"]["step3_hierarchy"])
    obj = _load(_STEP3_REGISTRY, impl_key, kwargs)
    if not isinstance(obj, HierarchyInferrer):
        raise TypeError(f"{type(obj).__name__} does not satisfy HierarchyInferrer protocol")
    return obj
