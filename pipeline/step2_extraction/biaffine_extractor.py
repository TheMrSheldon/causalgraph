from __future__ import annotations

import logging

import torch
from transformers import AutoTokenizer, pipeline

from ..protocols import CausalityExtractor, CausalRelation, Post
from .transformer_extractor import _device, _pair_classify_dedup

from train.models.biaffine_span_detection import BiaffineSpanDetectionModule, BiaffineSpanHead

logger = logging.getLogger("uvicorn.error")


class BiaffineTransformerExtractor(CausalityExtractor):
    """Step 2 extractor using the biaffine span-grid model instead of BIO
    token classification for entity detection. Relation classification stage
    is unchanged (shared with TransformerExtractor via _pair_classify_dedup).
    """

    def __init__(
        self,
        batch_size: int = 64,
        span_checkpoint: str = "",
        span_backbone: str = "roberta-large",
        relation_model: str = "thagen/roberta-large-causality-identification",
        span_threshold: float = 0.5,
        max_length: int = 128,
        **kwargs,
    ) -> None:
        self._batch_size = batch_size
        self._max_length = max_length
        self._threshold = span_threshold
        device_idx = _device()
        self._device = torch.device("cuda" if device_idx == 0 else "cpu")
        device_label = "GPU:0" if device_idx == 0 else "CPU"

        logger.info("BiaffineTransformerExtractor: loading models on %s", device_label)
        self._tokenizer = AutoTokenizer.from_pretrained(span_backbone)
        self._span_model = BiaffineSpanDetectionModule.load_from_checkpoint(
            span_checkpoint, map_location=self._device
        )
        self._span_model.eval().to(self._device)

        self._classifier = pipeline(
            "text-classification",
            model=relation_model,
            device=device_idx,
            truncation=True,
        )
        logger.info("BiaffineTransformerExtractor: models ready")

    @torch.no_grad()
    def _detect_entities(self, titles: list[str]) -> list[list[dict]]:
        enc = self._tokenizer(
            titles,
            max_length=self._max_length,
            truncation=True,
            padding=True,
            return_offsets_mapping=True,
            return_tensors="pt",
        )
        offset_mapping = enc.pop("offset_mapping")
        enc = {k: v.to(self._device) for k, v in enc.items()}
        scores = self._span_model(enc["input_ids"], enc["attention_mask"])

        all_entities: list[list[dict]] = []
        for i, title in enumerate(titles):
            spans = BiaffineSpanHead.decode_spans_from_grid(
                scores[i].cpu(), offset_mapping[i], threshold=self._threshold
            )
            all_entities.append([
                {"start": s, "end": e, "word": title[s:e]} for s, e in spans
            ])
        return all_entities

    def extract(self, posts: list[Post]) -> list[list[CausalRelation]]:
        titles = [p.title for p in posts]

        try:
            all_entities: list[list[dict]] = []
            for i in range(0, len(titles), self._batch_size):
                all_entities.extend(self._detect_entities(titles[i:i + self._batch_size]))
        except Exception:
            logger.warning("BiaffineTransformerExtractor: batch event detector failed", exc_info=True)
            return [[] for _ in posts]

        return _pair_classify_dedup(posts, all_entities, self._classifier, self._batch_size, self.name)

    @property
    def name(self) -> str:
        return "biaffine"
