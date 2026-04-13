"""
Step 3 default implementation: passthrough canonizer.

Returns the raw span text (text[start:end]) as-is without any transformation.

This is the correct default for r/science titles, which are typically
self-contained headlines where the extracted spans are already standalone
phrases (e.g., "smoking", "lung cancer risk").  More aggressive canonization
(e.g., pronoun resolution or title-context enrichment) is provided by the
LLM canonizer.
"""
from __future__ import annotations

from ..protocols import EventCanonizer


class PassthroughCanonizer(EventCanonizer):
    """
    Implements EventCanonizer by returning text[start:end] for each input span.
    """

    def __init__(self, **kwargs) -> None:
        pass

    @property
    def name(self) -> str:
        return "passthrough"

    def canonize(self, spans: list[tuple[str, tuple[int, int]]]) -> list[str]:
        return [text[start:end] for text, (start, end) in spans]
