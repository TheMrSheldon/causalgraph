from __future__ import annotations

import logging
import re
import unicodedata

import torch
from transformers import pipeline

from ..protocols import CausalityExtractor, Post, CausalRelation, RelationType

logger = logging.getLogger("uvicorn.error")


def _device() -> int:
    return 0 if torch.cuda.is_available() else -1


def _normalize(text: str) -> str:
    text = unicodedata.normalize("NFKD", text)
    text = text.encode("ascii", "ignore").decode("ascii")
    return re.sub(r"\s+", " ", text.lower().strip())


def _mark_entities(title: str, e0: dict, e1: dict) -> str:
    """
    Insert <e0></e0> and <e1></e1> markers around the two entity spans.
    Handles both orderings (e0 before e1, or e1 before e0 in the text).
    """
    s0, end0 = e0["start"], e0["end"]
    s1, end1 = e1["start"], e1["end"]

    if s0 <= s1:
        first_s, first_end, first_open, first_close = s0, end0, "<e0>", "</e0>"
        second_s, second_end, second_open, second_close = s1, end1, "<e1>", "</e1>"
    else:
        first_s, first_end, first_open, first_close = s1, end1, "<e1>", "</e1>"
        second_s, second_end, second_open, second_close = s0, end0, "<e0>", "</e0>"

    return (
        title[:first_s]
        + first_open
        + title[first_s:first_end]
        + first_close
        + title[first_end:second_s]
        + second_open
        + title[second_s:second_end]
        + second_close
        + title[second_end:]
    )


def _relation_type(label: str) -> RelationType:
    if label == "concausal":
        return RelationType.Countercausal
    if label == "no-rel":
        return RelationType.NoRel
    return RelationType.Causal


class TransformerExtractor(CausalityExtractor):

    def __init__(self, **kwargs) -> None:
        device = _device()
        device_label = "GPU:0" if device == 0 else "CPU"
        logger.info("TransformerExtractor: loading models on %s", device_label)
        # Token classification: labels tokens with B-Entity / I-Entity / O
        self._event_detector = pipeline(
            "token-classification",
            model="thagen/roberta-large-causal-candidate-extraction",
            aggregation_strategy="simple",
            device=device,
        )
        # Sequence classification: given text with <e0></e0> <e1></e1> markers,
        # outputs procausal / concausal / no-rel
        self._classifier = pipeline(
            "text-classification",
            model="thagen/roberta-large-causality-identification",
            device=device,
        )
        logger.info("TransformerExtractor: models ready")

    def extract(self, post: Post) -> list[CausalRelation]:
        title = post.title

        # Step 1: detect event spans
        try:
            entities = self._event_detector(title)
        except Exception:
            logger.warning("TransformerExtractor: event detector failed for post %s", post.id, exc_info=True)
            return []

        logger.debug("TransformerExtractor: post %s — %d entities detected", post.id, len(entities))
        if len(entities) < 2:
            return []

        relations: list[CausalRelation] = []

        # Step 2: classify each ordered pair (e0=potential cause, e1=potential effect)
        for i in range(len(entities)):
            for j in range(len(entities)):
                if i == j:
                    continue

                e0 = entities[i]
                e1 = entities[j]

                marked = _mark_entities(title, e0, e1)

                try:
                    all_scores = self._classifier(marked, top_k=None)
                except Exception:
                    logger.warning(
                        "TransformerExtractor: classifier failed for post %s pair (%s, %s)",
                        post.id, e0["word"], e1["word"], exc_info=True,
                    )
                    continue

                probs = {r["label"].lower(): r["score"] for r in all_scores}
                label = max(probs, key=probs.__getitem__)
                rel_type = _relation_type(label)
                logger.debug(
                    "TransformerExtractor: post %s ('%s' → '%s') classified as %s "
                    "(procausal=%.2f, concausal=%.2f, no-rel=%.2f)",
                    post.id, e0["word"], e1["word"], label,
                    probs.get("procausal", 0.0), probs.get("concausal", 0.0), probs.get("no-rel", 0.0),
                )

                cause_text = e0["word"].strip()
                effect_text = e1["word"].strip()

                if not cause_text or not effect_text or cause_text == effect_text:
                    continue

                relations.append(
                    CausalRelation(
                        post_id=post.id,
                        cause_text=cause_text,
                        effect_text=effect_text,
                        cause_norm=_normalize(cause_text),
                        effect_norm=_normalize(effect_text),
                        confidence=probs.get(label, 0.0),
                        extractor=self.name,
                        relation_type=rel_type,
                        p_none=probs.get("no-rel", 0.0),
                        p_causal=probs.get("procausal", 0.0),
                        p_countercausal=probs.get("concausal", 0.0),
                    )
                )

        logger.debug("TransformerExtractor: post %s — %d relation(s) extracted", post.id, len(relations))
        return relations

    @property
    def name(self) -> str:
        return "transformer"
