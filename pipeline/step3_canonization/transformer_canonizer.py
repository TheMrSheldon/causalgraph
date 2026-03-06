"""
Step 3 implementation: HuggingFace seq2seq transformer canonizer.

Uses a text-to-text generation model (default: google/flan-t5-base) to rewrite
extracted event spans into self-contained descriptions by conditioning on the
post title as context.

This is most useful when the extracted span is a pronoun or a vague reference
that requires the surrounding sentence to interpret:

  title: "It leads to cognitive decline in elderly patients"
  cause span: "It"
  → canonical: "the unspecified causal factor"   (model infers from context)

  title: "Smoking causes lung cancer"
  cause span: "Smoking"
  → canonical: "Smoking"                          (already self-contained)

Interface
---------
``canonize(relations)`` expects each relation to have ``post_title`` populated
(set by the runner/server from the source post's title).  This is the context
the model conditions on.  If ``post_title`` is empty the span itself is used as
the sole input (graceful fallback).

Performance
-----------
* Deduplication: unique (span_text.lower(), post_title) pairs are generated once;
  all other relations sharing the same key reuse the cached output.
* Fast-path: spans that already look self-contained (no pronouns / vague
  determiners) are returned unchanged without calling the model.  This skips
  ~80–90 % of r/science spans and keeps runtime tractable on CPU.
* Batched inference via AutoModelForSeq2SeqLM for compatibility with
  transformers ≥ 5.x (the "text2text-generation" pipeline task was removed).
"""
from __future__ import annotations

import dataclasses
import re

from pipeline.protocols import CausalRelation, EventCanonizer


# ---------------------------------------------------------------------------
# Heuristics for fast-path bypass
# ---------------------------------------------------------------------------

# Spans that START with these tokens almost certainly need context to
# resolve — route them to the model.
_PRONOUN_RE = re.compile(
    r"^(it|its|this|these|that|those|they|their|them|he|his|she|her|we|our|"
    r"the\s+(drug|compound|treatment|intervention|factor|condition|substance|"
    r"procedure|therapy|agent|chemical|molecule|protein|gene|enzyme|virus|"
    r"bacteria|organism|species|animal|particle|hormone|receptor|pathway|"
    r"mechanism|effect|phenomenon|process|result|change|increase|decrease|"
    r"reduction|improvement|decline|risk|study|research|finding|evidence|"
    r"association|link|relationship|outcome|response|reaction|exposure|"
    r"deficit|excess|loss|gain|damage|stress|disorder|disease|condition|"
    r"activity|behavior|pattern|level|concentration|amount|dose|rate|ratio))\b",
    re.IGNORECASE,
)


def _needs_model(span: str) -> bool:
    """Return True if the span likely requires title context to interpret."""
    return bool(_PRONOUN_RE.match(span.strip()))


# ---------------------------------------------------------------------------
# Prompt builder
# ---------------------------------------------------------------------------

def _make_prompt(title: str, span: str) -> str:
    if title:
        return (
            f"Given the scientific news title below, rewrite the event span as a "
            f"clear, self-contained noun phrase that makes sense on its own.\n"
            f"Title: {title}\n"
            f"Event span: {span}\n"
            f"Self-contained event:"
        )
    return (
        f"Rewrite the following event as a clear, self-contained noun phrase.\n"
        f"Event span: {span}\n"
        f"Self-contained event:"
    )


# ---------------------------------------------------------------------------
# Canonizer
# ---------------------------------------------------------------------------

class TransformerCanonizer:
    """
    Implements EventCanonizer using a seq2seq transformer (flan-t5-base by
    default) to rewrite event spans into self-contained descriptions.

    Parameters
    ----------
    model_name : str
        Any HuggingFace seq2seq model (T5, BART, etc.).
        Default: ``google/flan-t5-base`` (instruction-tuned T5, ~250 MB).
    batch_size : int
        Number of prompts per inference batch.  Increase for GPU; decrease
        if you hit OOM.  Default: 64.
    max_new_tokens : int
        Maximum output length in tokens.  Event spans are short, so 20 is
        usually sufficient.  Default: 20.
    device : int
        Torch device index.  -1 = CPU (default); 0 = first CUDA GPU.
    fast_path : bool
        If True (default), spans that appear already self-contained are
        returned unchanged without running the model.  Set to False to run
        every span through the model.
    """

    def __init__(
        self,
        model_name: str = "google/flan-t5-base",
        batch_size: int = 64,
        max_new_tokens: int = 20,
        device: int = -1,
        fast_path: bool = True,
        **kwargs,
    ) -> None:
        self.model_name = model_name
        self.batch_size = batch_size
        self.max_new_tokens = max_new_tokens
        self.device = device
        self.fast_path = fast_path
        self._model = None
        self._tokenizer = None

    @property
    def name(self) -> str:
        return "transformer"

    def _load(self):
        if self._model is None:
            from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
            print(f"[TransformerCanonizer] Loading '{self.model_name}' …")
            self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self._model = AutoModelForSeq2SeqLM.from_pretrained(self.model_name)
            if self.device >= 0:
                import torch
                self._model = self._model.to(f"cuda:{self.device}")
            self._model.eval()
            print(f"[TransformerCanonizer] Model loaded.")
        return self._model, self._tokenizer

    def _generate_batch(self, prompts: list[str]) -> list[str]:
        import torch
        model, tokenizer = self._load()
        inputs = tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=256,
        )
        if self.device >= 0:
            inputs = {k: v.to(f"cuda:{self.device}") for k, v in inputs.items()}
        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                num_beams=2,
                do_sample=False,
            )
        return tokenizer.batch_decode(output_ids, skip_special_tokens=True)

    def canonize(self, relations: list[CausalRelation]) -> list[CausalRelation]:
        if not relations:
            return []

        # ------------------------------------------------------------------
        # 1. Build dedup maps for cause and effect spans
        #    key: (span_text.lower(), post_title)  → canonical string
        # ------------------------------------------------------------------
        model_keys: list[tuple[str, str]] = []  # ordered unique keys for model
        seen_model: set[tuple[str, str]] = set()

        def _register(span: str, title: str) -> None:
            key = (span.lower(), title)
            if key not in seen_model and (not self.fast_path or _needs_model(span)):
                seen_model.add(key)
                model_keys.append(key)

        for r in relations:
            _register(r.cause_text, r.post_title)
            _register(r.effect_text, r.post_title)

        # ------------------------------------------------------------------
        # 2. Run model on keys that need it
        # ------------------------------------------------------------------
        key_to_canonical: dict[tuple[str, str], str] = {}

        if model_keys:
            n = len(model_keys)
            print(
                f"[TransformerCanonizer] Running model on {n:,} unique spans "
                f"({len(relations):,} relations total, fast_path={self.fast_path})…"
            )
            # Build prompts in order: key = (span_lower, title), use title + span
            # We need the original span text (not lowercased) for the prompt.
            # Build a helper map from (span_lower, title) → original_span_text
            key_to_orig: dict[tuple[str, str], str] = {}
            for r in relations:
                for span in (r.cause_text, r.effect_text):
                    k = (span.lower(), r.post_title)
                    if k not in key_to_orig:
                        key_to_orig[k] = span

            prompts = [_make_prompt(title, key_to_orig.get(k, k[0])) for k, title in
                       ((key, key[1]) for key in model_keys)]

            all_outputs: list[str] = []
            for start in range(0, n, self.batch_size):
                batch_prompts = prompts[start : start + self.batch_size]
                batch_out = self._generate_batch(batch_prompts)
                all_outputs.extend(batch_out)
                done = min(start + self.batch_size, n)
                if done % (self.batch_size * 10) == 0 or done == n:
                    print(f"  {done:,}/{n:,} spans processed…")

            for key, canonical in zip(model_keys, all_outputs):
                key_to_canonical[key] = canonical.strip() or key_to_orig.get(key, key[0])

        # ------------------------------------------------------------------
        # 3. Build result relations
        # ------------------------------------------------------------------
        result: list[CausalRelation] = []
        for r in relations:
            c_key = (r.cause_text.lower(), r.post_title)
            e_key = (r.effect_text.lower(), r.post_title)
            cause_can = key_to_canonical.get(c_key) or r.cause_text
            effect_can = key_to_canonical.get(e_key) or r.effect_text
            result.append(dataclasses.replace(
                r,
                cause_canonical=cause_can,
                effect_canonical=effect_can,
            ))

        return result
