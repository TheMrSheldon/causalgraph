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

from pipeline.protocols import (
    CausalityDetector,
    CausalExtractor,
    EventCanonizer,
    HierarchyInferrer,
)

# ---------------------------------------------------------------------------
# Implementation registries
# ---------------------------------------------------------------------------

_STEP1_REGISTRY: dict[str, tuple[str, str]] = {
    "regex": (
        "pipeline.step1_detection.regex_detector",
        "RegexDetector",
    ),
    "llm_openai": (
        "pipeline.step1_detection.llm_detector",
        "LLMDetector",
    ),
    "llm_anthropic": (
        "pipeline.step1_detection.llm_detector",
        "LLMDetector",
    ),
    "zero_shot": (
        "pipeline.step1_detection.zero_shot_detector",
        "ZeroShotDetector",
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
    "passthrough": (
        "pipeline.step3_canonization.passthrough_canonizer",
        "PassthroughCanonizer",
    ),
    "transformer": (
        "pipeline.step3_canonization.transformer_canonizer",
        "TransformerCanonizer",
    ),
    "llm_openai": (
        "pipeline.step3_canonization.llm_canonizer",
        "LLMCanonizer",
    ),
    "llm_anthropic": (
        "pipeline.step3_canonization.llm_canonizer",
        "LLMCanonizer",
    ),
}

_STEP4_REGISTRY: dict[str, tuple[str, str]] = {
    "embedding_hdbscan": (
        "pipeline.step4_hierarchy.embedding_clusterer",
        "EmbeddingClusterer",
    ),
    "tfidf_ward": (
        "pipeline.step4_hierarchy.tfidf_clusterer",
        "TFIDFClusterer",
    ),
    "embedding_ward": (
        "pipeline.step4_hierarchy.embedding_ward_clusterer",
        "EmbeddingWardClusterer",
    ),
    "llm": (
        "pipeline.step4_hierarchy.llm_topic_grouper",
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


def build_detector(config: dict) -> CausalityDetector:
    impl_key, kwargs = _pop_implementation(config["pipeline"]["step1_detection"])
    obj = _load(_STEP1_REGISTRY, impl_key, kwargs)
    if not isinstance(obj, CausalityDetector):
        raise TypeError(f"{type(obj).__name__} does not satisfy CausalityDetector protocol")
    return obj


def build_extractor(config: dict) -> CausalExtractor:
    impl_key, kwargs = _pop_implementation(config["pipeline"]["step2_extraction"])
    obj = _load(_STEP2_REGISTRY, impl_key, kwargs)
    if not isinstance(obj, CausalExtractor):
        raise TypeError(f"{type(obj).__name__} does not satisfy CausalExtractor protocol")
    return obj


def build_canonizer(config: dict) -> EventCanonizer:
    impl_key, kwargs = _pop_implementation(config["pipeline"]["step3_canonization"])
    obj = _load(_STEP3_REGISTRY, impl_key, kwargs)
    if not isinstance(obj, EventCanonizer):
        raise TypeError(f"{type(obj).__name__} does not satisfy EventCanonizer protocol")
    return obj


def build_inferrer(config: dict) -> HierarchyInferrer:
    impl_key, kwargs = _pop_implementation(config["pipeline"]["step4_hierarchy"])
    obj = _load(_STEP4_REGISTRY, impl_key, kwargs)
    if not isinstance(obj, HierarchyInferrer):
        raise TypeError(f"{type(obj).__name__} does not satisfy HierarchyInferrer protocol")
    return obj
