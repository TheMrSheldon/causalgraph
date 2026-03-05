"""
Step 1 alternative: HuggingFace zero-shot classification.

Uses a pre-trained NLI model to classify titles without fine-tuning.
Runs locally (no API cost) but requires ~1.5GB model download.
"""
from __future__ import annotations

from pipeline.protocols import CausalityIdentifier, Post

_DEFAULT_LABELS = ["causal relationship", "no causal relationship"]
_DEFAULT_HYPOTHESIS = "This title describes a cause-and-effect relationship."


class ZeroShotIdentifier:
    """
    Implements CausalityIdentifier using HuggingFace zero-shot classification.
    Lazy-loads the model on first call.
    """

    def __init__(
        self,
        zero_shot_model: str = "facebook/bart-large-mnli",
        zero_shot_threshold: float = 0.75,
        **kwargs,
    ) -> None:
        self.model_name = zero_shot_model
        self.threshold = zero_shot_threshold
        self._pipeline = None

    def _get_pipeline(self):
        if self._pipeline is None:
            from transformers import pipeline
            self._pipeline = pipeline(
                "zero-shot-classification",
                model=self.model_name,
                device=-1,  # CPU; set to 0 for GPU
            )
        return self._pipeline

    @property
    def name(self) -> str:
        return "zero_shot"

    def identify(self, posts: list[Post]) -> list[Post]:
        if not posts:
            return []
        clf = self._get_pipeline()
        titles = [p.title for p in posts]
        results = clf(titles, candidate_labels=_DEFAULT_LABELS, multi_label=False)

        causal: list[Post] = []
        for post, result in zip(posts, results):
            # result["labels"][0] is the top label
            top_label = result["labels"][0]
            top_score = result["scores"][0]
            if top_label == "causal relationship" and top_score >= self.threshold:
                causal.append(post)
        return causal
