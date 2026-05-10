"""
Step 3 implementation: parsing-based canonizer.

[NP completion]
The majority of causes and effects are presented as noun phrases.

Examples of NP completion:
(1)
Sentence:  "Chronic sleep deprivation leads to cognitive decline in the elderly with dementia."
Span:      "decline"
→ Canonical: "cognitive decline in the elderly with dementia" (PPs that modify the noun are added)
(2)
Sentence:   "Drug used to prevent miscarriage increases risk of cancer in offspring"
Span:       "Drug"
-> Canonical: "Drug used to prevent miscarriage" (Participial modifier is added)
(3)
Sentence:   "The effects of large group meetings on the spread of COVID-19"
Span:       "large group meetings"
-> Canonical: "large groups meetings" ("on the spread of COVID-19" is argument to "effects" not "meetings")
(4)
Sentence:   "White America’s racial resentment is the real impetus for welfare cuts, study says"
Span:       "White America's"
-> Canonical: "White America's racial resentment" (possessive determiner is added)

[Coreference resolution]

cluster detection model (reference): Shon Otmazgin, Arie Cattan, and Yoav Goldberg. 2022. F-coref: Fast, Accurate and Easy to Use Coreference Resolution. In Proceedings of the 2nd Conference of the Asia-Pacific Chapter of the Association for Computational Linguistics and the 12th International Joint Conference on Natural Language Processing: System Demonstrations, pages 48–56, Taipei, Taiwan. Association for Computational Linguistics.
--- Underlying LM: DistilRoBERTa; avg. CoNLL F1 score = 78.5 which is in line with other systems; 91M params which is substantially smaller and faster

Examples coreference resolution:
(1)
Title: 'A person's height impacts their risk of multiple diseases, new study finds'
Span: 'their risk of multiple diseases'
Coreference cluster: [(A person's, their)]
Resolved span: 'A person's risk of multiple diseases'
(2)
Title: 'Could Volcanoes Power the World? - If current geothermal wells are replaced with the new technology, it could provide 30% more power than current renewable energy sources.'
Span: 'it'
Coreference cluster: [(the new technology, it)]
Resolved span: 'the new technology'
"""
from __future__ import annotations

from typing import Tuple
from ..protocols import EventCanonizer
import re


Span = Tuple[int, int]  # (start, end) character offsets
Match = Tuple[int, int, Span]  # (cluster_index, position_in_cluster, referent_span)


# ----------------------------------------------------------------------
# NOUN PHRASE COMPLETION helper functions
# ----------------------------------------------------------------------


def fallback_np_detector(doc):
    """
    Fallback NP detector
    """
    spans = []
    start = None
    for i, tok in enumerate(doc):
        if tok.pos_ in ("DET", "ADJ", "NOUN", "PROPN"):
            if start is None:
                start = tok.i
        else:
            if start is not None and doc[start:i].root.pos_ in ("NOUN", "PROPN"):
                spans.append(doc[start:i])
            start = None

    if start is not None:
        span = doc[start:len(doc)]
        if span.root.pos_ in ("NOUN", "PROPN"):
            spans.append(span)

    return spans

def get_np_chunks(doc):
    """
    Get NP chunks
    """
    try:
        return list(doc.noun_chunks)   # requires parser
    except:
        return fallback_np_detector(doc)


def find_enclosing_np(doc, span: Tuple[int, int], text):
    """
     Find the smallest noun phrase (NP) containing a given character span.

     Parameters
     ----------
     doc : spaCy Doc
         The parsed document.
     span : (int, int)
         A tuple (start_char, end_char) representing the character offsets
         of the target span inside the full text.
     text : str
         The full original text (needed for substring extraction).

     Returns
     -------
     span or None
         The smallest NP fully enclosing the span, a partially covering NP
         from which introductory verb is removed (A), or None if nothing matches.

    A:
    Sentence: Air pollution causes respiratory problems and triggers inflammation.
    Extracted span: triggers inflammation
    Canonical span: inflammation
     """
    s_start, s_end = span
    # Lists to store NPs that fully or partially cover the span
    covering = []  # NPs that fully enclose the span
    partial_covering = []  # NPs that end exactly at the span boundary

    for chunk in get_np_chunks(doc):
        # collect all noun phrases that fully enclose the given text span
        if chunk.start_char <= s_start and s_end <= chunk.end_char:
            covering.append(chunk)
        # collect all noun phrases that partially enclose the given text span, with additional text before the NP chunk
        # e.g., "triggers inflammation" --> "inflammation" as NP chunk
        elif s_start <= chunk.start_char and s_end == chunk.end_char:
            partial_covering.append(chunk)

    # If there is no NP that fully or partially covers: return None
    if not covering and not partial_covering:
        return None

    # If we have no fully covering NP but at least one partial match,
    # apply a heuristic to see whether it should count as an enclosing NP.
    # Heuristic: flag whether a token inside the partial NP has a head matching the preceding text,
    # which suggests linguistic attachment
    if not covering and partial_covering:
        text_doc = [t.text for t in doc]
        head_doc = [t.head for t in doc]
        prechunk = text[s_start: partial_covering[0].start_char]
        chunk = text[partial_covering[0].start_char: partial_covering[0].end_char]
        checker = False
        for token in re.split(r"[ -]+", chunk):
            # If token exists in the doc and its syntactic head matches the prechunk,
            # treat the NP as a valid enclosing phrase
            # example: "triggers inflammation". prechunk = "triggers", head of "inflammation" = "triggers"
            if head_doc[text_doc.index(token)].text == prechunk.rstrip():
                checker = True
                break
        if checker:
            return partial_covering[0]
        return None

    return min(covering, key=lambda c: c.end_char - c.start_char) # return the smallest noun phrase in the document that fully contains a given character span


def stress_test_modifying_pp(tok, doc, nlp) -> bool:
    """
    Stress test for modifying PP, where tok.head == np_span.root
    Structures such as "the effects of X on Y", can be wrongly parsed; instable parsing. Wrong attachment of the 'on' preposition of Y on X.
    Attachment namely changes when adding another PP after Y. Example: the effects of Y in Chile on X". Now, correct attachment of 'on' on 'effects'
    If stress test results in tok.head != np_span.root, than PP is not modifying NP in a stable manner.
    """
    if tok.text != "in":
        added_list = ['in', 'Chile']
    else:
        added_list = ['in', 'Chile']
    expanded_doc = [t.text for t in doc[:tok.head.i+1]] + added_list + [t.text for t in doc[tok.head.i+1:]]

    new_doc = nlp(" ".join(expanded_doc))
    new_np_span = list(new_doc.noun_chunks)[1]
    new_tok = new_doc[tok.head.i+3]

    # ADP is no longer attached to the NP head, indicating instable attachment
    if new_tok.head != new_np_span.root:
        return False

    return True


def is_np_modifying_pp(np_span, tok, doc, nlp) -> bool:
    """
    Return True only for NP-internal PPs.

    Conditions for NP-internal PP:
    1. tok is ADP (a preposition)
    2. tok attaches to the NP head
    3. the PP is not part of a larger NP argument structure:
       tok must not attach to an ancestor outside np_span
    """

    # 1. Must be a preposition
    if tok.pos_ != "ADP":
        return False

    # 2. ADP must attach to the NP head
    if tok.head != np_span.root:
        return False

    # 3. Ensure the ADP is not governed by an outer NP head.
    #    That is, the ADP's syntactic parent (tok.head) must be inside np_span.
    #    If tok.head lies outside np_span, PP does not belong to this NP.
    if tok.head.i < np_span.start or tok.head.i >= np_span.end:
        return False

    # 4. Stress test
    if not stress_test_modifying_pp(tok, doc, nlp):
        return False

    return True


def is_restrictive_relcl(np_span, tok, doc):
    """
    Restrictive relative clause tokens typically have dep labels:
    - relcl
    - acl:relcl
    - acl   (gerund/participle)
    """
    return (
        tok.head.dep_ in ("relcl", "acl", "acl:relcl")
        and doc[tok.head.i].head == np_span.root
    )


def expand_possessive_determiner(doc, np_span) -> str:
    """
    Expand NP leftwards to include possessive determiners such as:
    "White America's racial resentment"
    "the company's financial losses"
    "children's health problems"

    Works even when the possessor is multi-token (e.g. "White America").
    """
    root = np_span.root

    # Find possessor (e.g., "America's")
    poss_tokens = [child for child in root.children if child.dep_ == "poss"]
    if not poss_tokens:
        return np_span  # no possessor → no change

    poss = poss_tokens[0]

    # Find the full possessor span by using left and right edges
    poss_start = poss.left_edge.i       # beginning of possessor phrase

    # NEW NP boundaries:
    # from possessor start → to end of original NP
    new_start = poss_start
    new_end   = np_span.end

    # Build the expanded span
    return doc[new_start:new_end]


def expand_np_with_modifiers(doc, np_span, nlp) -> str:
    """
    Expand NP to include modifying clauses (PP, relcl, acl)
    1. Prepositional phrases (PP)
    2. Restrictive relative clauses (relcl)
    3. Gerund/participle modifiers (acl)
    """
    expanded_end = np_span.end_char
    last_token_idx = np_span.end

    while last_token_idx < len(doc):

        tok = doc[last_token_idx]

        # Stop expansion on punctuation, sentence boundary, or conjunction
        # Why punctuation: comma can introduce non-essential information like non-restrictive relative clauses, e.g., "Felix, who plays the piano, caused ..."
        if tok.is_punct or tok.pos_ == "CCONJ" or tok.sent != np_span.sent:
            break

        # Case 1: PP modifier
        if is_np_modifying_pp(np_span, tok, doc, nlp):
            pp_end = tok.i
            for t in tok.subtree:
                pp_end = max(pp_end, t.i)
            expanded_end = doc[pp_end].idx + len(doc[pp_end])
            last_token_idx = pp_end + 1
            continue

        # Case 2: Restrictive Relative Clause / Gerund (acl, relcl)
        if is_restrictive_relcl(np_span, tok, doc):
            cl_end = tok.i
            for t in np_span.subtree:
                cl_end = max(cl_end, t.i)
            expanded_end = doc[cl_end].idx + len(doc[cl_end])
            last_token_idx = cl_end + 1
            continue

        break

    return doc.text[np_span.start_char : expanded_end]


def _noun_phrase_completion(text: str, span: Tuple[int, int], nlp) -> str:
    """
    Main function for NP expansion
    """
    doc = nlp(text)
    s_start, s_end = span
    selected = text[s_start:s_end]

    # Canonization step 1: Find enclosing NPs
    np_chunk = find_enclosing_np(doc, span, text)
    if np_chunk is None:
        return selected # return original span if it is not part of a noun phrase

    # Canonization step 2: expand NP to include possessive determiners (leftward expansion)
    # Normally, possessive determiners are included in the NP chunker, but in case it is not, this function includes them
    np_chunk = expand_possessive_determiner(doc, np_chunk)

    return expand_np_with_modifiers(doc, np_chunk, nlp)


class ParsingCanonizer(EventCanonizer):
    """
    Implements EventCanonizer by returning text[start:end] for each input span.
    """

    def __init__(self,
                 device: int = -1,
                 **kwargs) -> None:
        self._coref = None
        self._nlp = None
        self.device = device

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
        nlp = self._get_nlp()

        resolved_spans = []

        for text, (start, end) in spans:
            # ------------------------------------
            # (1) RUN NP Completion
            # ------------------------------------
            expanded_np = _noun_phrase_completion(text, (start, end), nlp)
            resolved_spans.append(expanded_np)

        return resolved_spans