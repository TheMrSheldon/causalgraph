from __future__ import annotations

import re
import unicodedata

from transformers import pipeline

from ..protocols import CausalityExtractor, Post, CausalRelation


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


def _label_probs(label: str, score: float) -> dict:
    if label == "procausal":
        return {"p_none": 0.0, "p_causal": score, "p_countercausal": 1.0 - score}
    if label == "concausal":
        return {"p_none": 0.0, "p_causal": 1.0 - score, "p_countercausal": score}
    # no-rel
    return {"p_none": score, "p_causal": 0.0, "p_countercausal": 0.0}


class TransformerExtractor(CausalityExtractor):

    def __init__(self) -> None:
        # Token classification: labels tokens with B-Entity / I-Entity / O
        self._event_detector = pipeline(
            "token-classification",
            model="thagen/roberta-large-causal-candidate-extraction",
            aggregation_strategy="simple",
        )
        # Sequence classification: given text with <e0></e0> <e1></e1> markers,
        # outputs procausal / concausal / no-rel
        self._classifier = pipeline(
            "text-classification",
            model="thagen/roberta-large-causality-identification",
        )

    def extract(self, post: Post) -> list[CausalRelation]:
        title = post.title

        # Step 1: detect event spans
        try:
            entities = self._event_detector(title)
        except Exception:
            return []

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
                    result = self._classifier(marked)[0]
                except Exception:
                    continue

                label = result["label"].lower()
                score: float = result["score"]

                if label == "no-rel":
                    continue

                cause_text = e0["word"].strip()
                effect_text = e1["word"].strip()

                if not cause_text or not effect_text or cause_text == effect_text:
                    continue

                is_cc = label == "concausal"

                relations.append(
                    CausalRelation(
                        post_id=post.id,
                        cause_text=cause_text,
                        effect_text=effect_text,
                        cause_norm=_normalize(cause_text),
                        effect_norm=_normalize(effect_text),
                        confidence=score,
                        extractor=self.name,
                        is_countercausal=is_cc,
                        **_label_probs(label, score),
                    )
                )

        return relations

    @property
    def name(self) -> str:
        return "transformer"
