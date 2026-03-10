from transformers import pipeline

from ..protocols import CausalityDetector, Post


class TransformerDetector(CausalityDetector):
    """
    Implements CausalityDetector via a hugging face classifier.

    All kwargs from config.yaml (beyond 'implementation') are accepted and
    ignored so extra config keys can be passed without error.
    """

    def __init__(self, **kwargs) -> None:
        self._pipe = pipeline("text-classification", model="thagen/roberta-large-causality-detection")

    @property
    def name(self) -> str:
        return "transformer"

    def detect(self, posts: list[Post]) -> list[Post]:
        outputs = self._pipe((p.title for p in posts))
        return [p for p, o in zip(posts, outputs) if o["label"] == "causal"]
