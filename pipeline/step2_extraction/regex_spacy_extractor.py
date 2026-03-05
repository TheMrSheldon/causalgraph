"""
Step 2 default implementation: regex + spaCy dependency parser extractor.

Strategy per pattern family:
  - "X causes/leads to Y":  subject NP = cause, object NP = effect
  - "X linked/associated with Y": left NP = cause, right NP = effect
  - "Study shows X increases Y": X is cause (agent), Y is effect (patient)
  - Falls back to regex group extraction if dependency parse fails.
"""
from __future__ import annotations

import re
import unicodedata

from pipeline.protocols import CausalExtractor, CausalRelation, Post


def _normalize(text: str) -> str:
    """Lowercase, strip, collapse whitespace, remove diacritics."""
    text = unicodedata.normalize("NFKD", text)
    text = text.encode("ascii", "ignore").decode("ascii")
    return re.sub(r"\s+", " ", text.lower().strip())


# ---------------------------------------------------------------------------
# Pattern definitions
# ---------------------------------------------------------------------------
# Each tuple: (compiled_regex, cause_group_index, effect_group_index)
# Group 0 is the full match; named groups preferred.

_EXTRACTION_PATTERNS: list[tuple[re.Pattern, str, str]] = [
    # "X leads to / results in / causes Y"
    (
        re.compile(
            r"^(?P<cause>.+?)\s+(?:leads?\s+to|results?\s+in|causes?|triggers?|produces?|drives?)\s+(?P<effect>.+)$",
            re.IGNORECASE,
        ),
        "cause",
        "effect",
    ),
    # "X is caused by / triggered by Y"  → reversed roles
    (
        re.compile(
            r"^(?P<effect>.+?)\s+(?:is|are)\s+(?:caused|triggered|driven|induced)\s+by\s+(?P<cause>.+)$",
            re.IGNORECASE,
        ),
        "cause",
        "effect",
    ),
    # "X linked to / associated with Y"
    (
        re.compile(
            r"^(?P<cause>.+?)\s+(?:linked\s+to|associated\s+with|correlated\s+with|connected\s+to|tied\s+to)\s+(?P<effect>.+)$",
            re.IGNORECASE,
        ),
        "cause",
        "effect",
    ),
    # "X increases / reduces / improves Y"
    (
        re.compile(
            r"^(?P<cause>.+?)\s+(?:increases?|decreases?|reduces?|boosts?|lowers?|raises?|improves?|worsens?|"
            r"doubles?|halves?|extends?|shortens?|accelerates?|delays?)\s+(?P<effect>.+)$",
            re.IGNORECASE,
        ),
        "cause",
        "effect",
    ),
    # "X prevents / inhibits / blocks Y"
    (
        re.compile(
            r"^(?P<cause>.+?)\s+(?:prevents?|inhibits?|suppresses?|blocks?|promotes?)\s+(?P<effect>.+)$",
            re.IGNORECASE,
        ),
        "cause",
        "effect",
    ),
    # Study-report: "Study shows that X [verb] Y"
    (
        re.compile(
            r"^.{0,60}(?:shows?\s+that|finds?\s+that|suggests?\s+that|reveals?\s+that|"
            r"demonstrates?\s+that|reports?\s+that)\s+(?P<cause>.+?)\s+"
            r"(?:increases?|decreases?|reduces?|causes?|leads?\s+to|prevents?|improves?|worsens?|"
            r"triggers?|promotes?)\s+(?P<effect>.+)$",
            re.IGNORECASE,
        ),
        "cause",
        "effect",
    ),
    # Due-to / because-of: "Y due to / because of X"
    (
        re.compile(
            r"^(?P<effect>.+?)\s+(?:due\s+to|because\s+of|as\s+a\s+result\s+of|in\s+response\s+to)\s+(?P<cause>.+)$",
            re.IGNORECASE,
        ),
        "cause",
        "effect",
    ),
]


def _clean_phrase(text: str) -> str:
    """Strip trailing punctuation and common filler words."""
    text = text.strip().rstrip(".,;:)")
    # Remove leading articles
    text = re.sub(r"^(?:a|an|the)\s+", "", text, flags=re.IGNORECASE)
    # Truncate overly long phrases at 80 chars
    if len(text) > 80:
        text = text[:80].rsplit(" ", 1)[0]
    return text.strip()


def _extract_with_spacy(title: str, nlp) -> list[tuple[str, str]] | None:
    """
    Attempt dependency-parse-based extraction as a higher-quality fallback.
    Returns [(cause_text, effect_text)] or None if parse is inconclusive.
    """
    try:
        doc = nlp(title)
    except Exception:
        return None

    causal_verbs = {
        "cause", "lead", "result", "trigger", "produce", "increase", "decrease",
        "reduce", "improve", "worsen", "prevent", "inhibit", "promote", "drive",
        "boost", "lower", "raise",
    }

    for token in doc:
        if token.lemma_.lower() in causal_verbs and token.pos_ == "VERB":
            subjects = [
                child for child in token.children
                if child.dep_ in ("nsubj", "nsubjpass")
            ]
            objects = [
                child for child in token.children
                if child.dep_ in ("dobj", "pobj", "attr", "xcomp")
            ]
            if subjects and objects:
                cause = " ".join(t.text for t in subjects[0].subtree)
                effect = " ".join(t.text for t in objects[0].subtree)
                return [(_clean_phrase(cause), _clean_phrase(effect))]
    return None


class RegexSpacyExtractor:
    """
    Implements CausalExtractor using regex pattern matching with an optional
    spaCy dependency parse fallback for higher-quality extraction.
    """

    def __init__(self, spacy_model: str = "en_core_web_sm", **kwargs) -> None:
        self._nlp = None
        self._spacy_model = spacy_model

    def _get_nlp(self):
        """Lazy-load spaCy model on first use."""
        if self._nlp is None:
            try:
                import spacy
                self._nlp = spacy.load(self._spacy_model, disable=["ner", "textcat"])
            except Exception:  # spaCy may fail on Python 3.14 (pydantic v1 incompatibility)
                self._nlp = False  # Mark as unavailable; regex-only fallback used
        return self._nlp if self._nlp is not False else None

    @property
    def name(self) -> str:
        return "regex_spacy"

    def extract(self, post: Post) -> list[CausalRelation]:
        title = post.title
        pairs: list[tuple[str, str]] = []

        # Try each regex pattern in order; take first match
        for pattern, cause_group, effect_group in _EXTRACTION_PATTERNS:
            m = pattern.match(title)
            if m:
                cause = _clean_phrase(m.group(cause_group))
                effect = _clean_phrase(m.group(effect_group))
                if cause and effect and cause != effect:
                    pairs.append((cause, effect))
                    break  # One match per title is sufficient for the regex pass

        # If regex failed, try spaCy dependency parse
        if not pairs:
            nlp = self._get_nlp()
            if nlp:
                spacy_pairs = _extract_with_spacy(title, nlp)
                if spacy_pairs:
                    pairs.extend(spacy_pairs)

        return [
            CausalRelation(
                post_id=post.id,
                cause_text=cause,
                effect_text=effect,
                cause_norm=_normalize(cause),
                effect_norm=_normalize(effect),
                confidence=1.0,
                extractor=self.name,
            )
            for cause, effect in pairs
            if cause and effect
        ]
