"""
Step 3 default implementation: passthrough canonizer.

Returns the raw span text (text[start:end]) as-is without any transformation.

This is the correct default for r/science titles, which are typically
self-contained headlines where the extracted spans are already standalone
phrases (e.g., "smoking", "lung cancer risk").  More aggressive canonization
(e.g., pronoun resolution or title-context enrichment) is provided by the
LLM canonizer.

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
Span:       "White America" or "racial resentment"
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

from typing import Any, List, Tuple
from ..protocols import EventCanonizer


Span = Tuple[int, int]  # (start, end) character offsets
Match = Tuple[int, int, Span]  # (cluster_index, position_in_cluster, referent_span)


# ----------------------------------------------------------------------
# COREFERENCE RESOLUTION helper functions
# ----------------------------------------------------------------------


POSSESSIVE_PRONOUNS = {
    "my", "your", "his", "her", "its", "our", "their"
}

def is_possessive_pronoun(text: str, span: Span) -> bool:
    token = text[span[0]:span[1]].lower().strip()
    return token in POSSESSIVE_PRONOUNS


def make_possessive(np: str) -> str:
    """
    Converts an NP into the correct possessive form.

    Rules:
    - If NP already ends with "'s", return NP unchanged.
    - If NP already ends with "'", return NP unchanged.
    - If NP ends with 's' (plural), add only an apostrophe.
    - Otherwise, add "'s".
    """
    np = np.strip()

    # 1. Already possessive: "person's", "children's"
    if np.lower().endswith("'s"):
        return np

    # 2. Already possessive plural: "torries'"
    if np.endswith("'"):
        return np

    # 3. Plural ending in s → add apostrophe
    if np.lower().endswith("s"):
        return np + "'"

    # 4. Default: singular → add 's
    return np + "'s"



def find_span_in_clusters(span: Span, clusters: List[List[Span]]) -> List[Match] | None:
    """
    Return list of clusters which include the span. len(list) can be >= 1.
    """
    results = []
    for ci, cluster in enumerate(clusters):
        for si, s in enumerate(cluster):
            if span[0] < s[1] and s[0] < span[1]:  # overlap
                results.append((ci, si, s)) # cluster index: int, position of span in cluster: int, referent spans: list

    if len(results) > 0:
        return results
    return None # The span should not be resolved


def run_coref_model(text: str, coref) -> List[List[Span]] | None:
    """
    Attempts to run the coreference model and extract clusters.
    Returns cluster lists or None if the model fails.
    """
    try:
        preds = coref.predict(texts=[text])
        return preds[0].get_clusters(as_strings=False) # [[(0,2), (33, 36)], [(33, 50), (52, 64)]]
    except Exception:
        return None


def get_span_matches(span: Span, clusters: List[List[Span]]) -> List[Match]:
    """
    Wrapper around the user's existing find_span_in_clusters().
    """
    return find_span_in_clusters(span, clusters)


def overlaps(a: Span, b: Span) -> bool:
    """Return True if two spans overlap."""
    return a[0] < b[1] and b[0] < a[1]


def classify_match(span: Span, mention: Span) -> str:
    """
    Classify match quality
    Return one of: 'exact', 'span_in_mention', 'mention_in_span', 'overlap', 'none'
    """
    s0, s1 = span
    m0, m1 = mention

    if s0 == m0 and s1 == m1:
        return "exact"
    if m0 <= s0 and s1 <= m1:
        return "span_in_mention"
    if s0 <= m0 and m1 <= s1:
        return "mention_in_span"
    if overlaps(span, mention):
        return "overlap"
    return "none"


def is_pronoun(text: str, span: Span) -> bool:
    """
    Check if text is pronoun
    """
    tok = text[span[0]:span[1]].lower()
    return tok in {"he", "she", "it", "they", "them", "him", "her", "we", "us"}



def canonical_antecedent(text: str, cluster: List[Span]) -> str:
    """
    Strategy:
        - choose longest NP span (or the earliest longest)
        - here we simply choose the longest by span length.
    """
    best = max(cluster, key=lambda s: (s[1] - s[0], -s[0]))
    return text[best[0]:best[1]]



def select_best_cluster(clusters: Dict[int, List[Match]]) -> int:
    """
    Choose best cluster among multiple candidates
    Very simple scoring:
        - clusters with exact matches > others
        - larger cluster size > smaller
        - earliest canonical NP > later
    """
    def cluster_score(ci):
        matches = clusters[ci]
        exact = any(classify_match(m[2], m[2]) == "exact" for m in matches)
        size = len(matches)
        earliest = min(m[2][0] for m in matches)
        return 1 if exact else 0, size, -earliest

    return max(clusters, key=cluster_score)


# ===================================================================
#                  MAIN DECISION TREE RESOLUTION
# ===================================================================

def resolve_span_using_decision_tree(
        text: str,
        span: Span,
        matches: List[Match],
        clusters: List[List[Span]]
) -> str:
    """
    Implements the decision-tree for coreference resolution.
    """
    span_start, span_end = span
    span_is_pron = is_pronoun(text, span)

    # -------------------------------------------------------------
    # STEP 1: No matches → return original
    # -------------------------------------------------------------
    if not matches:
        return text[span_start:span_end]

    # -------------------------------------------------------------
    # STEP 2: Classify match types
    # -------------------------------------------------------------
    exact_matches = []
    partial_matches = []

    for ci, mi, mention in matches:
        mtype = classify_match(span, mention)
        if mtype == "exact":
            exact_matches.append((ci, mi, mention))
        else:
            partial_matches.append((ci, mi, mention))

    # -------------------------------------------------------------
    # STEP 3: Exactly one exact match
    # -------------------------------------------------------------
    if len(exact_matches) == 1:
        ci, mi, mention = exact_matches[0]
        cluster = clusters[ci]

        # Case 3A: pronoun → standard resolution
        if span_is_pron:
            return canonical_antecedent(text, cluster)

        # Case 3B: selected span is a possessive pronoun NP
        if is_possessive_pronoun(text, (mention[0], mention[1])):
            antecedent = canonical_antecedent(text, cluster)
            poss = make_possessive(antecedent)
            # NP tail begins immediately after the pronoun
            pronoun_end = mention[1]
            tail = text[pronoun_end: span_end]
            return f"{poss}{tail}"

        # Ordinary NP → return itself
        return text[mention[0]:mention[1]]

    # -------------------------------------------------------------
    # STEP 4: Multiple exact matches → pick best cluster. This usually does not happen.
    # -------------------------------------------------------------
    if len(exact_matches) > 1:
        # Group by cluster index
        grouped = {}
        for ci, mi, mention in exact_matches:
            grouped.setdefault(ci, []).append((ci, mi, mention))

        best_ci = select_best_cluster(grouped)

        antecedent = canonical_antecedent(text, clusters[best_ci])

        # If NP begins with possessive pronoun, rewrite
        first_token = text[span_start:span_end].split()[0].lower()
        if first_token in POSSESSIVE_PRONOUNS:
            poss = make_possessive(antecedent)
            tail = text[span_start:span_end][len(first_token):]
            return f"{poss}{tail}"

        return antecedent

    # -------------------------------------------------------------
    # STEP 5: No exact matches but one partial semantic match
    # -------------------------------------------------------------

    if len(matches) == 1:
        ci, mi, mention = matches[0]
        mtype = classify_match(span, mention)
        cluster = clusters[ci]

        # Case: pronoun resolution
        if span_is_pron:
            return canonical_antecedent(text, cluster)

        # Case: Possessive pronoun inside NP
        if is_possessive_pronoun(text, mention) and mtype == "mention_in_span":
            antecedent = canonical_antecedent(text, cluster)
            poss = make_possessive(antecedent)
            pronoun_end = mention[1]
            tail = text[pronoun_end: span_end]
            return f"{poss}{tail}"

        # fallback: return NP unmodified
        return text[span_start:span_end]

    # -------------------------------------------------------------
    # STEP 6: MULTIPLE partial matches → choose best cluster
    # -------------------------------------------------------------
    grouped = {}
    for ci, mi, mention in matches:
        grouped.setdefault(ci, []).append((ci, mi, mention))

    best_ci = select_best_cluster(text, grouped)
    best_cluster = clusters[best_ci]
    best_mentions = [m[2] for m in grouped[best_ci]]

    # Pronoun case
    if span_is_pron:
        return canonical_antecedent(text, best_cluster)

    # Check if NP starts with possessive pronoun
    first_token = text[span_start:span_end].split()[0].lower()
    if first_token in POSSESSIVE_PRONOUNS:
        antecedent = canonical_antecedent(text, best_cluster)
        poss = make_possessive(antecedent)
        tail = text[span_start:span_end][len(first_token):]
        return f"{poss}{tail}"

    # NP containing multiple mentions → resolve inside
    return replace_mentions_within_span(text, span, best_mentions, best_cluster)


def _coreference_resolution(text: str, start: int, end: int, coref) -> str:
    """
    Conduct coreference resolution and replace the coreferent with the most appropriate referent.
    1. Run coreference model and extract clusters (where first mention is considered head mention)
    2. Find matches between the selected span and cluster mentions
        2A. Handle case 1: span appears exactly once, i.e., has one resolving mention
        2B. Handle case 2: span appears in multiple clusters
    3. Perform replacements
    Finally, return resolved text
    """
    # 1. Run coreference model and extract clusters
    clusters = run_coref_model(text, coref)
    if clusters is None or len(clusters) == 0:  #No coreference resolution to be performed; return original text span
        return text[start:end]

    # 2. Check where the span appears in coreference clusters and return matches
    matches = get_span_matches((start, end), clusters)

    # 3. Resolve text span
    resolved = resolve_span_using_decision_tree(
        text,
        (start, end),
        matches,
        clusters
    )
    return resolved

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


def find_enclosing_np(doc, span: Tuple[int, int]):
    """
    Find smallest NP containing span
    Input: parsed doc, span is a tuple representing a text span in character offsets
    """
    s_start, s_end = span
    covering = []

    for chunk in get_np_chunks(doc): # collect all noun phrases that fully enclose the given text span
        if chunk.start_char <= s_start and s_end <= chunk.end_char:
            covering.append(chunk)

    if not covering:
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

    # 4. Stress test the
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

    np_chunk = find_enclosing_np(doc, span)
    if np_chunk is None:
        return selected # return original span if it is not part of a noun phrase

    # expand NP to include possessive determiners
    # Normally, possessive determiners are included in the NP chunker, but in case it is not, this function includes them
    np_chunk = expand_possessive_determiner(doc, np_chunk)

    return expand_np_with_modifiers(doc, np_chunk, nlp)


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

        print(spans[:10])

        for text, (start, end) in spans:
            # ------------------------------------
            # (1) RUN NP Completion
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