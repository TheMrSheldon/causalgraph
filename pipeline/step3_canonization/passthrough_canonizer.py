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

    def __init__(self,
                 device: int = -1,
                 **kwargs) -> None:
        self._coref = None
        self._nlp = None
        self.device = device

    def _get_coref(self):
        """Load coreference model on first use."""
        if self._coref is None:
            try:
                from fastcoref import FCoref
                self._coref = FCoref(device=self.device)
            except Exception:
                self._coref = False # Mark as unavailable; no coreference resolution is performed
        return self._coref if self._coref is not False else None

    def _get_nlp(self):
        if self._nlp is None:
            try:
                import spacy
                self._nlp = spacy.load("en_core_web_sm")
            except Exception:
                self._nlp = False
        return self._nlp if self._nlp is not False else None

    @property
    def name(self) -> str:
        return "passthrough"

    def canonize(self, spans: list[tuple[str, tuple[int, int]]]) -> list[str]:
        #return [text[start:end] for text, (start, end) in spans]

        coref = self._get_coref()
        nlp = self._get_nlp()

        resolved_spans = []

        for text, (start, end) in spans:
            # ------------------------------------
            # (1) RUN NP Expansion
            # ------------------------------------
            expanded_np = _noun_phrase_expansion(text, (start, end), nlp)

            expanded_start = text.index(expanded_np)
            expanded_end = expanded_start + len(expanded_np)

            # ------------------------------------
            # (2) RUN COREFERENCE RESOLUTION
            # ------------------------------------
            coref_resolved_span = _coreference_resolution(text, expanded_start, expanded_end, coref)

            resolved_spans.append(coref_resolved_span)

        return resolved_spans