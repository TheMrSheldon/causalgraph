"""
Step 1 default implementation: regex-based causality identification.

Scans titles for explicit and implicit causal language using a set of
compiled regular expression patterns. Fast, zero cost, high precision.
"""
from __future__ import annotations

import re

from pipeline.protocols import CausalityIdentifier, Post

# ---------------------------------------------------------------------------
# Pattern groups
# ---------------------------------------------------------------------------

# Patterns that strongly signal an explicit causal claim
_EXPLICIT_PATTERNS: list[str] = [
    # Direct causal verbs
    r"\b(causes?|caused\s+by|leading\s+to|leads?\s+to|results?\s+in|results?\s+from|result(?:ing)?\s+from)\b",
    r"\b(triggers?|triggered\s+by|produces?|generated\s+by|induces?|induced\s+by)\b",
    r"\b(promotes?|prevents?|inhibits?|suppresses?|blocks?|drives?|driven\s+by)\b",
    # Quantitative change verbs (high precision for causality)
    r"\b(increases?|decreases?|reduces?|boosts?|lowers?|raises?|doubles?|triples?|halves?)\b",
    r"\b(improves?|worsens?|accelerates?|slows?|delays?|extends?|shortens?)\b",
    # Risk language
    r"\b(risk\s+(?:of|factor\s+for)|increases?\s+risk|reduces?\s+risk|linked\s+to\s+(?:higher|lower))\b",
    # Contribution/responsibility
    r"\b(contributes?\s+to|responsible\s+for|accounts?\s+for|explains?)\b",
    # Mechanism language
    r"\b(via|through|by\s+(?:increasing|decreasing|blocking|activating|inhibiting))\b",
]

# Patterns that signal implicit causality (study-report language)
_IMPLICIT_PATTERNS: list[str] = [
    # Study finding verbs followed by causal content
    r"\b(shows?\s+that|found?\s+that|suggests?\s+that|demonstrates?\s+that|reveals?\s+that|indicates?\s+that)\b",
    r"\b(finds?\s+that|reports?\s+that|confirms?\s+that|proves?\s+that)\b",
    # Causal prepositions
    r"\b(due\s+to|because\s+of|as\s+a\s+result\s+of|in\s+response\s+to|following)\b",
    r"\b(associated\s+with|correlated\s+with|connected\s+to|tied\s+to|linked\s+to)\b",
    # Treatment/intervention language
    r"\b(treatment\s+(?:with|of)|exposure\s+to|administration\s+of|use\s+of)\b.*\b(reduces?|prevents?|improves?|increases?)\b",
]

# Patterns that indicate a title is likely NOT causal (reduce false positives)
_EXCLUSION_PATTERNS: list[str] = [
    # Pure descriptions without causal directionality
    r"^\s*(?:watch|video|image|gallery|photo|photos)\b",
    r"\b(?:ama|ask\s+me\s+anything|discussion)\b",
    # Pure factual statements without causal claim
    r"^\s*\d+\s+(?:things|ways|reasons|facts)\b",
]

_COMPILED_EXPLICIT = [re.compile(p, re.IGNORECASE) for p in _EXPLICIT_PATTERNS]
_COMPILED_IMPLICIT = [re.compile(p, re.IGNORECASE) for p in _IMPLICIT_PATTERNS]
_COMPILED_EXCLUSION = [re.compile(p, re.IGNORECASE) for p in _EXCLUSION_PATTERNS]


def _is_causal(title: str) -> bool:
    # Reject titles matching exclusion patterns
    if any(exc.search(title) for exc in _COMPILED_EXCLUSION):
        return False
    # Accept on any explicit pattern (high confidence)
    if any(pat.search(title) for pat in _COMPILED_EXPLICIT):
        return True
    # Accept on any implicit pattern (lower confidence, but still causal)
    if any(pat.search(title) for pat in _COMPILED_IMPLICIT):
        return True
    return False


class RegexIdentifier:
    """
    Implements CausalityIdentifier via compiled regex patterns.

    All kwargs from config.yaml (beyond 'implementation') are accepted and
    ignored so the registry can pass them without error.
    """

    def __init__(self, **kwargs) -> None:
        # Accept any extra config keys without error
        pass

    @property
    def name(self) -> str:
        return "regex"

    def identify(self, posts: list[Post]) -> list[Post]:
        return [p for p in posts if _is_causal(p.title)]
