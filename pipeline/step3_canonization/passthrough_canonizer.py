"""
Step 3 default implementation: passthrough canonizer.

Sets cause_canonical / effect_canonical to the already-cleaned extraction
spans (cause_text / effect_text) without any further transformation.

This is the correct default for r/science titles, which are typically
self-contained headlines where the extracted spans are already standalone
phrases (e.g., "smoking", "lung cancer risk").  More aggressive canonization
(e.g., pronoun resolution or title-context enrichment) is provided by the
LLM canonizer.
"""
from __future__ import annotations

import dataclasses

from pipeline.protocols import CausalRelation, EventCanonizer


class PassthroughCanonizer:
    """
    Implements EventCanonizer by copying cause_text / effect_text as-is
    into cause_canonical / effect_canonical.
    """

    def __init__(self, **kwargs) -> None:
        pass

    @property
    def name(self) -> str:
        return "passthrough"

    def canonize(self, relations: list[CausalRelation]) -> list[CausalRelation]:
        return [
            dataclasses.replace(
                r,
                cause_canonical=r.cause_text,
                effect_canonical=r.effect_text,
            )
            for r in relations
        ]
