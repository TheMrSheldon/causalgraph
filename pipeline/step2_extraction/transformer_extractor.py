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
    if label == "countercausal":
        return RelationType.Countercausal
    if label == "no-rel":
        return RelationType.NoRel
    return RelationType.Causal


def _pair_classify_dedup(
    posts: list[Post],
    all_entities: list[list[dict]],
    classifier,
    batch_size: int,
    extractor_name: str,
) -> list[list[CausalRelation]]:
    """Shared post-entity-detection logic: pair candidates, classify, dedup.

    Used by every CausalityExtractor whose only difference is how entity
    spans are detected (HF token-classification pipeline, biaffine grid, ...).
    """
    # Collect every ordered (cause, effect) candidate pair across all posts
    pair_records: list[tuple[int, dict, dict, str]] = []  # (post_idx, e0, e1, marked)
    for post_idx, (post, entities) in enumerate(zip(posts, all_entities)):
        if len(entities) < 2:
            continue
        for i in range(len(entities)):
            for j in range(len(entities)):
                if i == j:
                    continue
                e0, e1 = entities[i], entities[j]
                pair_records.append((post_idx, e0, e1, _mark_entities(post.title, e0, e1)))

    results: list[list[CausalRelation]] = [[] for _ in posts]
    if not pair_records:
        return results

    try:
        all_scores_list = classifier(
            [r[3] for r in pair_records], top_k=None, batch_size=batch_size
        )
    except Exception:
        logger.warning("%s: batch classifier failed", extractor_name, exc_info=True)
        return results

    # Collect all non-no-rel candidates; dedup bidirectional conflicts afterwards.
    # Key: (post_idx, cause_norm, effect_norm) → best CausalRelation by confidence.
    best: dict[tuple[int, str, str], CausalRelation] = {}

    for (post_idx, e0, e1, _), all_scores in zip(pair_records, all_scores_list):
        probs = {r["label"].lower(): r["score"] for r in all_scores}
        label = max(probs, key=probs.__getitem__)
        if label == "no-rel":
            continue
        cause_text = re.sub(r"[\s,;:!?.]+$", "", e0["word"]).strip()
        effect_text = re.sub(r"[\s,;:!?.]+$", "", e1["word"]).strip()
        if len(cause_text) < 5 or len(effect_text) < 5 or cause_text == effect_text:
            continue
        cause_norm = _normalize(cause_text)
        effect_norm = _normalize(effect_text)
        confidence = probs.get(label, 0.0)

        key = (post_idx, cause_norm, effect_norm)
        rev = (post_idx, effect_norm, cause_norm)

        candidate = CausalRelation(
            post_id=posts[post_idx].id,
            cause_text=cause_text,
            effect_text=effect_text,
            cause_norm=cause_norm,
            effect_norm=effect_norm,
            confidence=confidence,
            extractor=extractor_name,
            relation_type=_relation_type(label),
            p_none=probs.get("no-rel", 0.0),
            p_causal=probs.get("causal", 0.0),
            p_countercausal=probs.get("countercausal", 0.0),
        )

        if rev in best:
            # Both directions are non-no-rel: keep whichever has higher confidence.
            if confidence > best[rev].confidence:
                del best[rev]
                best[key] = candidate
        elif key not in best or confidence > best[key].confidence:
            best[key] = candidate

    for (post_idx, _, __), relation in best.items():
        results[post_idx].append(relation)

    return results


class TransformerExtractor(CausalityExtractor):

    def __init__(
        self,
        batch_size: int = 64,
        span_model: str = "thagen/roberta-large-causal-candidate-extraction",
        relation_model: str = "thagen/roberta-large-causality-identification",
        **kwargs,
    ) -> None:
        self._batch_size = batch_size
        device = _device()
        device_label = "GPU:0" if device == 0 else "CPU"
        logger.info("TransformerExtractor: loading models on %s", device_label)
        self._event_detector = pipeline(
            "token-classification",
            model=span_model,
            aggregation_strategy="first",  # matches training: first subword label per word
            device=device,
        )
        self._classifier = pipeline(
            "text-classification",
            model=relation_model,
            device=device,
            truncation=True,
        )
        logger.info("TransformerExtractor: models ready")

    def extract(self, posts: list[Post]) -> list[list[CausalRelation]]:
        titles = [p.title for p in posts]

        try:
            all_entities = self._event_detector(titles, batch_size=self._batch_size)
        except Exception:
            logger.warning("TransformerExtractor: batch event detector failed", exc_info=True)
            return [[] for _ in posts]

        return _pair_classify_dedup(posts, all_entities, self._classifier, self._batch_size, self.name)

    @property
    def name(self) -> str:
        return "transformer"
