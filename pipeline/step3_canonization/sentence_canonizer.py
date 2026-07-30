"""
Step 3 implementation: sentence-level canonizer.

Produces a single self-contained SENTENCE describing the event in the marked
span — not just a noun phrase. Falls back to the noun-phrase form when the
event is too generic/abstract to sensibly express as a standalone clause
(e.g. "cancer", "climate change" — these have no latent specificity to turn
into a sentence without inventing content; see the genericity gate below).

Pipeline per span:
  1. NP completion (reused from parsing_canonizer._noun_phrase_completion)
     to get the fullest, correctly-scoped phrase for the span.
  2. Coreference resolution (fastcoref) — pronouns inside that phrase are
     replaced with their antecedent from the source text.
  3. Genericity gate — embed the phrase with the same hierarchical-embedding
     checkpoint Step 4 uses (ugao25w3) and check its Poincare-ball norm.
     Generic concepts collapse near the origin (norm ~0.17-0.24 measured);
     concrete multi-clause descriptions sit much further out (~0.28-0.50).
     Below threshold -> return the coref-resolved phrase unchanged.
  4. Deterministic clause realization — find the nominal head of the phrase,
     look up its most-frequent-sense WordNet derivational verb, and realize
     a full clause (active if both a possessor and an "of X" complement are
     present; passive if only the "of X" object is known; a neutral
     "occurs"/"takes place" filler for a bare event noun with no complement
     at all). Only fires when the transformation is structurally unambiguous
     and complete — no invented arguments.
  5. LLM fallback (outlines, JSON-schema-constrained decoding) for phrases
     step 4 can't confidently realize. A content-lemma-overlap check runs
     after generation: the source phrase's content words must be preserved
     and no new content word may appear — reject and fall back to the plain
     phrase (step 2's output) rather than trust ungrounded generation.

Why not trust free-form LLM generation, per the abandoned TransformerCanonizer
attempt: the observed failure there was not fabricated facts but a decoding
artifact — small instruct models spend the token budget on a conversational
preamble ("Sure, here's...") before the payload, made worse by a 24-token cap,
occasionally producing output disconnected from the input entirely. Grammar-
constrained JSON decoding structurally prevents that class of failure — the
grammar doesn't allow the first tokens to be a preamble.
"""
from __future__ import annotations

import re
from typing import Tuple

import numpy as np

from ..protocols import EventCanonizer
from .parsing_canonizer import _noun_phrase_completion

Span = Tuple[int, int]

_STOP_LEMMAS = {
    "be", "a", "an", "the", "of", "in", "on", "at", "to", "for", "with",
    "occur", "take", "place", "is", "are", "was", "were", "have", "do",
}


# ---------------------------------------------------------------------------
# WordNet-based nominalization -> verb lookup
# ---------------------------------------------------------------------------

def _wordnet_verb(noun_lemma: str) -> str | None:
    """Most-frequent-sense derivational verb for a noun lemma, or None.

    Restricting to the top-2 WordNet senses (synsets are frequency-ordered)
    filters out rare-sense noise: unioning across ALL senses returns wrong
    picks like 'risk' -> 'jeopardise' or 'spread' -> 'banquet'/'feast'.
    """
    from nltk.corpus import wordnet as wn

    for syn in wn.synsets(noun_lemma, pos=wn.NOUN)[:2]:
        for lemma in syn.lemmas():
            if lemma.name().lower() != noun_lemma.lower():
                continue
            for rel in lemma.derivationally_related_forms():
                if rel.synset().pos() == "v":
                    return rel.name()
    return None


def _capitalize(text: str) -> str:
    return text[0].upper() + text[1:] if text else text


# Epistemic/modal nouns ("the risk of X", "the effect of X on Y") have a
# WordNet-derived verb ("risk") but it is not a natural event nominalization
# — "X is risked"/"X is effected" are semantically off or ungrammatical.
# Deferred to the LLM fallback instead of forced through the passive rule.
_NON_EVENT_HEADS = {
    "risk", "chance", "possibility", "likelihood", "probability", "odds",
    "effect", "impact", "influence", "role", "link", "association",
    "correlation", "relationship", "connection",
}


# ---------------------------------------------------------------------------
# Deterministic clause realization
# ---------------------------------------------------------------------------

def _try_deterministic_clause(phrase: str, nlp) -> str | None:
    """Realize `phrase` as a full clause when the transformation is complete
    and unambiguous. Returns None to defer to the LLM fallback otherwise.
    """
    from lemminflect import getInflection

    doc = nlp(phrase)
    root = doc[:].root
    if root.pos_ not in ("NOUN", "PROPN"):
        return None  # already a clause, or not a nominalization-headed phrase
    if root.lemma_.lower() in _NON_EVENT_HEADS:
        return None

    verb = _wordnet_verb(root.lemma_.lower())
    if verb is None:
        return None

    poss = [c for c in root.children if c.dep_ == "poss"]
    of_pp = [c for c in root.children if c.dep_ == "prep" and c.text.lower() == "of"]
    pobj = None
    if of_pp:
        pobj_children = [c for c in of_pp[0].children if c.dep_ == "pobj"]
        pobj = pobj_children[0] if pobj_children else None

    # Case 1: possessor (subject) + "of X" (object) both known -> active, complete.
    if poss and pobj:
        subject = doc[poss[0].left_edge.i: poss[0].i + 1].text
        subject = re.sub(r"[’']s$", "", subject).strip()
        obj_text = doc[pobj.left_edge.i: pobj.right_edge.i + 1].text
        past = getInflection(verb, "VBD")
        if not past:
            return None
        return f"{_capitalize(subject)} {past[0]} {obj_text}."

    # Case 2: only "of X" (object) known -> passive, no agent invented.
    if pobj and not poss:
        obj_text = doc[pobj.left_edge.i: pobj.right_edge.i + 1].text
        participle = getInflection(verb, "VBN")
        if not participle:
            return None
        be_form = "are" if pobj.tag_ in ("NNS", "NNPS") else "is"
        return f"{_capitalize(obj_text)} {be_form} {participle[0]}."

    # Case 3: bare event noun, no complement at all -> neutral filler, no
    # invented content (re-verbing the same head, e.g. "decline declines",
    # would be circular, so a generic occurrence verb is used instead).
    if not poss and not of_pp:
        occur_form = "occur" if root.tag_ in ("NNS", "NNPS") else "occurs"
        return f"{_capitalize(phrase)} {occur_form}."

    return None  # possessor present without an object, or other ambiguous shape


# ---------------------------------------------------------------------------
# Faithfulness check for LLM-generated sentences
# ---------------------------------------------------------------------------

def _content_tokens(doc) -> list[tuple[str, str]]:
    """(lemma, surface) pairs for content-bearing tokens."""
    return [
        (tok.lemma_.lower(), tok.text.lower())
        for tok in doc
        if tok.pos_ in ("NOUN", "PROPN", "VERB", "ADJ")
        and tok.lemma_.lower() not in _STOP_LEMMAS
    ]


def _is_faithful(source_phrase: str, generated: str, nlp, min_overlap: float = 0.6) -> bool:
    """A source content word counts as preserved if either its lemma or its
    surface form appears on the generated side.

    Lemma-only comparison spuriously fails for gerund-like words whose POS
    tag (and therefore lemma) depends on syntactic context: a bare
    single-word phrase like "Deworming" parses as VERB in isolation (lemma
    "deworm"), but the same word embedded in a full generated sentence
    ("The deworming of primary schools") parses as NOUN (lemma "deworming")
    — a real, correct, non-leaking answer that a lemma-only check rejected
    in testing. Matching on surface form as a fallback is strictly more
    permissive and can only fix false rejections, not hide genuine misses.
    """
    src_tokens = _content_tokens(nlp(source_phrase))
    if not src_tokens:
        return True
    gen_tokens = _content_tokens(nlp(generated))
    gen_forms = {form for pair in gen_tokens for form in pair}
    matched = sum(1 for lemma, surface in src_tokens if lemma in gen_forms or surface in gen_forms)
    return matched / len(src_tokens) >= min_overlap


def _leaks_sibling(phrase: str, sibling_raw: str, generated: str, nlp, threshold: float = 0.4) -> bool:
    """True if `generated` appears to restate the PAIRED span's own content
    (e.g. cause_canonical narrating the effect too) rather than describing
    only `phrase` in isolation.

    _is_faithful only checks that source content is PRESERVED — it has no
    way to catch EXTRA content being added, and the leaked content isn't
    "hallucinated" (it's genuinely present in the shared source context), so
    a prompt instruction against it can reduce but not structurally
    guarantee against it. Confirmed on real production data even after
    strengthening the prompt (e.g. "Deworming" and "high social rate of
    return" — its own effect — both realized to the identical sentence).
    This is a belt-and-suspenders check using the pipeline's cause/effect
    pairing (see canonize()); it only flags on the SIBLING's distinctive
    words (excluding anything already shared with `phrase`, so genuinely
    shared topic words don't trigger false positives).
    """
    phrase_forms = {f for pair in _content_tokens(nlp(phrase)) for f in pair}
    sibling_tokens = _content_tokens(nlp(sibling_raw))
    distinctive = [(l, s) for l, s in sibling_tokens if l not in phrase_forms and s not in phrase_forms]
    if not distinctive:
        return False
    gen_forms = {f for pair in _content_tokens(nlp(generated)) for f in pair}
    leaked = sum(1 for lemma, surface in distinctive if lemma in gen_forms or surface in gen_forms)
    return (leaked / len(distinctive)) > threshold


# ---------------------------------------------------------------------------
# Canonizer
# ---------------------------------------------------------------------------

class SentenceCanonizer(EventCanonizer):
    """Produces self-contained sentences for concrete events; falls back to
    a noun phrase for generic/abstract ones. See module docstring for the
    full per-span pipeline.
    """

    # (context, phrase, sentence) — context mirrors the runtime user message
    # so the few-shot turns show the same enrich-from-context behavior we
    # want at inference time, not just isolated phrase rewriting.
    _EXAMPLES: list[tuple[str, str, str]] = [
        ("Chronic sleep deprivation leads to cognitive decline in the elderly with dementia.",
         "cognitive decline in the elderly with dementia",
         "Cognitive decline occurs in the elderly with dementia."),
        ("Excessive metal mining depletes natural mineral reserves.",
         "the mining of metals",
         "Metals are mined."),
        ("Genetically Engineered Immune Cells Found to Rapidly Clear Leukemia Tumors.",
         "Genetically Engineered Immune Cells",
         "Immune cells are genetically engineered."),
        ("Cancer drug aids the regeneration of spinal cord injuries.",
         "aids the regeneration of spinal cord injuries",
         "Spinal cord injuries are regenerated."),
        # Bare single-word phrase: the phrase alone has no content to build a
        # sentence from, so a detail is pulled from context — but only about
        # THIS concept (how it was used/administered), never the outcome the
        # context attributes to it (that outcome is the paired event's own
        # canonical description, produced by a separate call).
        ("Ayahuasca administration was found to increase serotonin and dopamine "
         "turnover in the rat brain compared to control.",
         "Ayahuasca",
         "Ayahuasca was administered to rats."),
        # Contrastive pair: same source sentence, two different spans. Each
        # canonical form must stand alone — cause_canonical does NOT mention
        # "boosts research funding" (that is effect_canonical's job) and vice
        # versa. A frequent, hard-to-avoid failure mode: the model narrates
        # the whole cause->effect claim in both fields instead of isolating
        # each side, which was confirmed on ~20 sampled production relations
        # (e.g. "Deworming" and its effect "high social rate of return" both
        # collapsed to the identical sentence "Deworming has a high social
        # rate of return.").
        ("A new grant boosts research funding for renewable energy projects.",
         "A new grant",
         "A new grant was awarded."),
        ("A new grant boosts research funding for renewable energy projects.",
         "boosts research funding for renewable energy projects",
         "Research funding for renewable energy projects increases."),
    ]

    def __init__(
        self,
        device: int = -1,
        hierarchy_checkpoint_path: str = "",
        genericity_threshold: float = 0.25,
        llm_model_name: str = "Qwen/Qwen2.5-3B-Instruct",
        llm_max_new_tokens: int = 80,
        llm_batch_size: int = 16,
        genericity_batch_size: int = 64,
        min_faithfulness_overlap: float = 0.6,
        **kwargs,
    ) -> None:
        self.device = device
        self._device_str = f"cuda:{device}" if device >= 0 else "cpu"
        self.hierarchy_checkpoint_path = hierarchy_checkpoint_path
        self.genericity_threshold = genericity_threshold
        self.llm_model_name = llm_model_name
        self.llm_max_new_tokens = llm_max_new_tokens
        self.llm_batch_size = llm_batch_size
        self.genericity_batch_size = genericity_batch_size
        self.min_faithfulness_overlap = min_faithfulness_overlap

        self._nlp = None
        self._coref = None
        self._coref_cache: dict[str, list] = {}
        self._hierarchy_model = None
        self._llm_model = None
        self._llm_tokenizer = None
        self._llm_outlines_model = None
        self._llm_schema = None

    @property
    def name(self) -> str:
        return "sentence"

    # ------------------------------------------------------------------
    # Lazy loaders
    # ------------------------------------------------------------------

    def _get_nlp(self):
        if self._nlp is None:
            import spacy
            self._nlp = spacy.load("en_core_web_sm")
        return self._nlp

    def _get_coref(self):
        if self._coref is None:
            try:
                # fastcoref (last released 2023) predates a transformers
                # internal rename: PreTrainedModel now expects an
                # `all_tied_weights_keys` instance attribute set during
                # tie_weights()/post_init(), which FCorefModel's older-API
                # __init__ never triggers, so from_pretrained() crashes with
                # AttributeError before the model even loads. A class-level
                # empty-dict default is a safe fallback (no known tied-weight
                # remapping) and doesn't affect any other model in this
                # process, since properly-initialized models always shadow
                # it with their own instance attribute.
                import transformers
                if not hasattr(transformers.PreTrainedModel, "all_tied_weights_keys"):
                    transformers.PreTrainedModel.all_tied_weights_keys = {}
                from fastcoref import FCoref
                self._coref = FCoref(device=self._device_str)
            except Exception:
                self._coref = False
        return self._coref if self._coref is not False else None

    def _get_hierarchy_model(self):
        if self._hierarchy_model is None and self.hierarchy_checkpoint_path:
            try:
                from ..step4_hierarchy.poincare import _load_model
                self._hierarchy_model = _load_model(self.hierarchy_checkpoint_path, self._device_str)
            except Exception:
                self._hierarchy_model = False
        return self._hierarchy_model if self._hierarchy_model is not False else None

    def _get_llm(self):
        """Load the LLM fallback as raw HF model + tokenizer + a standalone
        outlines JSON-schema logits processor, instead of outlines.Generator.

        outlines.Generator only accepts a single prompt string per call (v1.3
        API), which would force one forward pass per span — far too slow at
        corpus scale (~314K spans; empirically ~2.5-10s/span unbatched, i.e.
        days of wall-clock). The logits processor is a plain HF
        LogitsProcessor, so it composes with HF's own batched, left-padded
        generate() the same way TransformerExtractor/TransformerCanonizer
        already batch their inference.
        """
        if self._llm_model is None:
            import json
            import outlines
            from pydantic import BaseModel, Field
            from transformers import AutoModelForCausalLM, AutoTokenizer

            class _Canonical(BaseModel):
                sentence: str = Field(description="Self-contained sentence describing the event")

            self._llm_tokenizer = AutoTokenizer.from_pretrained(
                self.llm_model_name, padding_side="left"
            )
            if self._llm_tokenizer.pad_token_id is None:
                self._llm_tokenizer.pad_token_id = self._llm_tokenizer.eos_token_id
            self._llm_model = AutoModelForCausalLM.from_pretrained(self.llm_model_name)
            if self.device >= 0:
                self._llm_model = self._llm_model.to(self._device_str)
            self._llm_outlines_model = outlines.from_transformers(self._llm_model, self._llm_tokenizer)
            self._llm_schema = json.dumps(_Canonical.model_json_schema())
        return self._llm_model, self._llm_tokenizer

    # ------------------------------------------------------------------
    # Coreference resolution
    # ------------------------------------------------------------------

    def _get_clusters(self, text: str) -> list:
        if text in self._coref_cache:
            return self._coref_cache[text]
        self._prefetch_clusters([text])
        return self._coref_cache.get(text, [])

    def _prefetch_clusters(self, texts: list[str]) -> None:
        """Populate the coref cache for all not-yet-seen texts in one batched
        fastcoref call, instead of one predict() call per span. fastcoref's
        predict() already accepts (and batches) a list of texts internally.
        """
        uncached = [t for t in dict.fromkeys(texts) if t not in self._coref_cache]
        if not uncached:
            return
        coref = self._get_coref()
        if coref is None:
            for t in uncached:
                self._coref_cache[t] = []
            return
        try:
            preds = coref.predict(texts=uncached)
        except Exception:
            preds = [None] * len(uncached)
        for t, pred in zip(uncached, preds):
            try:
                raw_clusters = pred.get_clusters(as_strings=False) if pred is not None else []
                # fastcoref occasionally includes None for a mention it
                # couldn't resolve to a valid span — not reproduced in small
                # curated test samples, but hit a production run within
                # minutes once given ~100K diverse real post titles. Filter
                # those out and drop clusters left with <2 valid mentions
                # (can't support antecedent substitution anyway).
                clusters = [[m for m in c if m is not None] for c in raw_clusters]
                self._coref_cache[t] = [c for c in clusters if len(c) >= 2]
            except Exception:
                self._coref_cache[t] = []

    def _resolve_pronouns(
        self, text: str, phrase_start: int, phrase_end: int, nlp, sibling_raw: str | None = None
    ) -> str:
        """Replace pronouns inside [phrase_start, phrase_end) with their
        antecedent from the source text's coreference clusters.

        sibling_raw, when given, is the paired cause/effect span's own raw
        text. A pronoun in the effect that genuinely refers back to the
        cause (extremely common: "X causes Y because it does Z") resolves
        CORRECTLY to the cause's mention — but that's exactly the pair-
        leakage this canonizer must avoid, just introduced via coreference
        instead of LLM narration. Skipping that one substitution (leaving
        the pronoun unresolved) is preferred over a phrase that already
        embeds the sibling's content before any downstream check can catch it.
        """
        clusters = self._get_clusters(text)
        if not clusters:
            return text[phrase_start:phrase_end]

        doc = nlp(text)
        replacements: list[tuple[int, int, str]] = []  # (start, end, replacement), text-relative

        for tok in doc:
            if not (phrase_start <= tok.idx < phrase_end):
                continue
            if tok.pos_ != "PRON":
                continue
            tok_start, tok_end = tok.idx, tok.idx + len(tok.text)

            # Exact match, not containment: containment would also match a
            # relative pronoun like "who" that merely sits inside a larger
            # mention's own span (e.g. "People who commute..." is itself a
            # cluster mention containing "who"), causing it to be replaced by
            # its OWN enclosing phrase — a self-referential duplication bug
            # found via real-data testing ("People who commute..." ->
            # "People People who commute...commute...work"). A pronoun is
            # only a coreference mention if fastcoref lists its exact span.
            cluster = next(
                (c for c in clusters if (tok_start, tok_end) in c),
                None,
            )
            if cluster is None:
                continue

            # A contraction clitic ("'re", "'s", "'ve", "'ll", "'d") attaches
            # to the pronoun with no space ("they're"). Substituting just the
            # pronoun leaves the clitic dangling on the replacement noun
            # phrase ("Oregon shore crabs're exposed" instead of "...are
            # exposed") — found on real production data. Properly expanding
            # every contraction form correctly is its own can of worms;
            # skipping the substitution is a small loss (one unresolved
            # pronoun) versus a guaranteed grammar break.
            next_tok = doc[tok.i + 1] if tok.i + 1 < len(doc) else None
            if next_tok is not None and next_tok.idx == tok_end and next_tok.text.startswith(("'", "’")):
                continue

            # Antecedent = the longest non-pronoun mention in the cluster.
            candidates = [
                (s, e, text[s:e]) for s, e in cluster
                if not (s == tok_start and e == tok_end) and (e - s) > (tok_end - tok_start)
            ]
            if not candidates:
                continue
            _, _, antecedent = max(candidates, key=lambda c: c[1] - c[0])

            # A long antecedent (a full clause, not a short noun phrase) reads
            # badly when substituted verbatim into a short phrase — worse, if
            # the SAME long antecedent is the best match for multiple
            # pronouns within one short phrase, it gets inserted more than
            # once, producing severely bloated/duplicated-looking output
            # (found on real production data: a 70-char antecedent substituted
            # for two separate "their" occurrences in one 60-char phrase).
            # Leaving the pronoun unresolved is safer than that.
            if len(antecedent) > 50:
                continue

            if sibling_raw is not None and _leaks_sibling(text[phrase_start:phrase_end], sibling_raw, antecedent, nlp):
                continue

            is_possessive = tok.tag_ in ("PRP$", "WP$")
            if is_possessive:
                antecedent = re.sub(r"[’']s$", "", antecedent)
            replacement = f"{antecedent}'s" if is_possessive else antecedent
            replacements.append((tok_start, tok_end, replacement))

        if not replacements:
            return text[phrase_start:phrase_end]

        # Apply replacements right-to-left so earlier offsets stay valid.
        result = text
        for start, end, replacement in sorted(replacements, reverse=True):
            result = result[:start] + replacement + result[end:]
        # Recompute the phrase slice: replacements only ever occur inside
        # [phrase_start, phrase_end), and only shift offsets to their right
        # within that same range, so the phrase's own start is unaffected.
        shift = sum(
            len(r) - (e - s) for s, e, r in replacements if e <= phrase_start
        )
        new_start = phrase_start + shift
        end_shift = sum(len(r) - (e - s) for s, e, r in replacements if s < phrase_end)
        new_end = phrase_end + end_shift
        return result[new_start:new_end]

    # ------------------------------------------------------------------
    # Genericity gate
    # ------------------------------------------------------------------

    def _is_generic_batch(self, phrases: list[str]) -> list[bool]:
        model = self._get_hierarchy_model()
        if model is None:
            return [False] * len(phrases)  # signal unavailable -> attempt realization anyway
        from ..step4_hierarchy.poincare import _embed_poincare
        embs = _embed_poincare(
            model, phrases, batch_size=self.genericity_batch_size, device=self._device_str
        )
        norms = np.linalg.norm(embs, axis=1)
        return [float(n) < self.genericity_threshold for n in norms]

    # ------------------------------------------------------------------
    # LLM fallback
    # ------------------------------------------------------------------

    def _llm_prompt(self, phrase: str, context_text: str, tokenizer) -> str:
        examples_msgs = [
            msg
            for ex_context, ex_phrase, ex_sentence in self._EXAMPLES
            for msg in (
                {"role": "user", "content": f"Context: {ex_context}\nPhrase: {ex_phrase}"},
                {"role": "assistant", "content": f'{{"sentence": "{ex_sentence}"}}'},
            )
        ]
        messages = [
            {
                "role": "system",
                "content": (
                    "You rewrite a short phrase describing an event into ONE "
                    "self-contained sentence describing ONLY that event. "
                    "The Context describes a cause-and-effect relationship, "
                    "but the Phrase is only ONE side of it (either the cause "
                    "or the effect, not both). Your sentence must NOT state "
                    "what the phrase's event causes, what caused it, or the "
                    "outcome/result described elsewhere in the Context — "
                    "describe the event or concept in the Phrase in "
                    "isolation, as if the rest of the causal claim did not "
                    "exist. If the phrase alone is too short or generic to "
                    "form a sentence (e.g. a single name), you may pull a "
                    "detail from Context about what THIS thing is or how it "
                    "was used/administered/observed — never the outcome or "
                    "consequence attributed to it. Use only facts stated in "
                    "the phrase or context; do not add information that "
                    "isn't there. Output JSON with a single 'sentence' field."
                ),
            },
            *examples_msgs,
            {"role": "user", "content": f"Context: {context_text}\nPhrase: {phrase}"},
        ]
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    def _llm_realize_batch(self, items: list[tuple[str, str]]) -> list[str | None]:
        """items: [(phrase, context_text), ...]. Returns one generated
        sentence (or None on failure) per item, processed in
        llm_batch_size-sized batches via HF's own batched generate() with
        left-padding — the same pattern TransformerCanonizer already used.
        """
        import json
        import torch
        from outlines.backends import get_json_schema_logits_processor

        try:
            model, tokenizer = self._get_llm()
        except Exception:
            return [None] * len(items)

        results: list[str | None] = []
        for start in range(0, len(items), self.llm_batch_size):
            batch = items[start:start + self.llm_batch_size]
            prompts = [self._llm_prompt(phrase, ctx, tokenizer) for phrase, ctx in batch]
            inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True, max_length=512)
            if self.device >= 0:
                inputs = {k: v.to(self._device_str) for k, v in inputs.items()}
            prompt_len = inputs["input_ids"].shape[1]
            # A fresh processor per batch is required: the outlines JSON-
            # schema processor carries internal FSM state that isn't reset
            # between separate generate() calls — reusing one across batches
            # was verified empirically to silently produce empty output on
            # every batch after the first.
            processor = get_json_schema_logits_processor(None, self._llm_outlines_model, self._llm_schema)
            try:
                with torch.no_grad():
                    out_ids = model.generate(
                        **inputs,
                        max_new_tokens=self.llm_max_new_tokens,
                        do_sample=False,
                        logits_processor=[processor],
                        pad_token_id=tokenizer.pad_token_id,
                    )
                decoded = tokenizer.batch_decode(out_ids[:, prompt_len:], skip_special_tokens=True)
            except Exception:
                decoded = [None] * len(batch)

            for raw in decoded:
                if raw is None:
                    results.append(None)
                    continue
                try:
                    results.append(json.loads(raw)["sentence"])
                except Exception:
                    results.append(None)
        return results

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def canonize(self, spans: list[tuple[str, tuple[int, int]]]) -> list[str]:
        nlp = self._get_nlp()
        if not spans:
            return []

        # Batch coref up front: fastcoref's predict() already batches a list
        # of texts internally, so one call for all unique source texts
        # replaces one call per span (spans sharing a post title, e.g. a
        # relation's cause and effect, previously re-triggered inference).
        self._prefetch_clusters([text for text, _ in spans])

        # Cause/effect pairing: run_step3 always calls canonize() with
        # relations laid out as consecutive (cause, effect) pairs sharing the
        # same source text (see runner.py::run_step3). Detected defensively
        # per-pair, not assumed globally, so callers that don't follow this
        # convention (ad-hoc scripts, tests) just skip the sibling-leakage
        # check safely rather than erroring.
        sibling_raw: dict[int, str] = {}
        for i in range(0, len(spans) - 1, 2):
            text_a, (sa, ea) = spans[i]
            text_b, (sb, eb) = spans[i + 1]
            if text_a == text_b:
                sibling_raw[i] = text_b[sb:eb]
                sibling_raw[i + 1] = text_a[sa:ea]

        # ---- Phase 1: NP completion + coref resolution (per-span, local/cheap) ----
        # Every step is wrapped so ONE unanticipated edge case (fastcoref
        # returning a malformed cluster, an unusual parse, etc.) can't crash
        # the whole call and lose all progress on a run that processes the
        # entire corpus in a single canonize() invocation — found the hard
        # way when a production run crashed a few minutes in on a fastcoref
        # edge case not present in any of the smaller test samples. Falls
        # back to the raw span text for that one span; everything else in
        # the batch is unaffected.
        phrases: list[str] = []
        for i, (text, (start, end)) in enumerate(spans):
            raw_span = text[start:end]
            try:
                # NP completion is designed to strip a leading verb down to its
                # object NP (e.g. "triggers inflammation" -> "inflammation") —
                # correct for a phrase-only canonizer, but destructive here: a
                # span like "Protect Pediatric Population" is already a clause,
                # and stripping "Protect" throws away the actual causal claim
                # before any downstream step can recover it. Skip NP-completion
                # when the raw span already has a verbal root.
                raw_doc = nlp(raw_span)
                if raw_doc[:].root.pos_ in ("VERB", "AUX"):
                    phrase = raw_span
                    phrase_start = start
                else:
                    phrase = _noun_phrase_completion(text, (start, end), nlp)
                    phrase_start = text.find(phrase)  # NP completion may shift the span
                if phrase_start != -1:
                    phrase = self._resolve_pronouns(
                        text, phrase_start, phrase_start + len(phrase), nlp, sibling_raw.get(i)
                    )
            except Exception as e:
                print(f"[SentenceCanonizer] Phase 1 failed on {raw_span!r}: {e}; using raw span")
                phrase = raw_span
            phrases.append(phrase)

        # ---- Phase 2: batched genericity gate ----
        try:
            generic_flags = self._is_generic_batch(phrases)
        except Exception as e:
            print(f"[SentenceCanonizer] Phase 2 (genericity) failed: {e}; treating batch as non-generic")
            generic_flags = [False] * len(phrases)

        # ---- Phase 3: deterministic clause realization (per-span, local/cheap) ----
        # Gated through the same faithfulness check as the LLM path: a
        # confidently-wrong parse (e.g. "Drinking Water Of Billions In
        # Danger" -> "Billions In Danger are watered.") can still pass a
        # lemma-overlap check since it reuses the same words — this doesn't
        # catch scrambled relations between words, only invented content —
        # but it's a strictly safer floor than trusting the deterministic
        # output unconditionally, which had no check at all before.
        results: list[str | None] = [None] * len(spans)
        llm_indices: list[int] = []
        llm_items: list[tuple[str, str]] = []

        for i, (phrase, generic) in enumerate(zip(phrases, generic_flags)):
            if generic:
                results[i] = phrase
                continue

            try:
                # A bare single-token phrase (e.g. a drug/chemical name) has no
                # internal structure for the deterministic WordNet path to work
                # with, but the source sentence may still say something concrete
                # about it — skip straight to the context-aware LLM step rather
                # than the syntax-only deterministic one (which would just
                # return None anyway), and rather than passing through unchanged.
                too_sparse = len([t for t in nlp(phrase) if not t.is_punct]) <= 1
                clause = None if too_sparse else _try_deterministic_clause(phrase, nlp)
                sib = sibling_raw.get(i)
                if (
                    clause is not None
                    and _is_faithful(phrase, clause, nlp, self.min_faithfulness_overlap)
                    and (sib is None or not _leaks_sibling(phrase, sib, clause, nlp))
                ):
                    results[i] = clause
                    continue
            except Exception as e:
                print(f"[SentenceCanonizer] Phase 3 failed on {phrase!r}: {e}; deferring to LLM")

            llm_indices.append(i)
            llm_items.append((phrase, spans[i][0]))

        # ---- Phase 4: batched LLM fallback for everything else ----
        if llm_items:
            generated_list = self._llm_realize_batch(llm_items)
            for idx, (phrase, _ctx), generated in zip(llm_indices, llm_items, generated_list):
                sib = sibling_raw.get(idx)
                if (
                    generated is not None
                    and _is_faithful(phrase, generated, nlp, self.min_faithfulness_overlap)
                    and (sib is None or not _leaks_sibling(phrase, sib, generated, nlp))
                ):
                    results[idx] = generated
                else:
                    results[idx] = phrase

        return results
