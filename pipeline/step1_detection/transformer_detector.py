import logging

import torch
from transformers import pipeline

from ..protocols import CausalityDetector, Post

logger = logging.getLogger("uvicorn.error")


def _device() -> int:
    return 0 if torch.cuda.is_available() else -1


class TransformerDetector(CausalityDetector):
    """
    Implements CausalityDetector via a hugging face classifier.

    All kwargs from config.yaml (beyond 'implementation') are accepted and
    ignored so extra config keys can be passed without error.
    """

    def __init__(self, **kwargs) -> None:
        device = _device()
        logger.info("TransformerDetector: loading model on %s", "GPU:0" if device == 0 else "CPU")
        self._pipe = pipeline(
            "text-classification",
            model="thagen/roberta-large-causality-detection",
            device=device,
        )
        logger.info("TransformerDetector: model ready")

    @property
    def name(self) -> str:
        return "transformer"

    def detect(self, posts: list[Post]) -> list[Post]:
        logger.debug("TransformerDetector.detect: scoring %d posts", len(posts))
        outputs = self._pipe((p.title for p in posts))
        causal = [p for p, o in zip(posts, outputs) if o["label"] == "causal"]
        logger.debug("TransformerDetector.detect: %d / %d posts classified as causal", len(causal), len(posts))
        return causal
