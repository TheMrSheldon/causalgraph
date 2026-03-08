"""
Step 3 implementation: local instruction-tuned causal LM canonizer.

Uses a locally-hosted instruction-tuned decoder-only model to rewrite each
extracted event span into a concise, self-contained noun phrase that captures
the full meaning of the event, enriched with information from the surrounding
sentence when it helps.

Default model: ``Qwen/Qwen2.5-1.5B-Instruct``
  - ~3 GB RAM on CPU (fp32), ~1.5 GB in fp16 / bfloat16
  - Strong instruction-following quality for its size
  - Replace with ``Qwen/Qwen2.5-3B-Instruct`` or
    ``mistralai/Mistral-7B-Instruct-v0.3`` for higher quality when a GPU
    is available.

Why not Mistral-7B as the default?
  Mistral-7B-Instruct-v0.3 is an excellent fit for this task (instruction-tuned,
  locally runnable, no API key), but requires ~14 GB RAM in fp16 or ~4–5 GB
  with 4-bit quantization.  Qwen2.5-1.5B gives good results at a fraction of
  the cost and is the safer default for CPU-only machines.

Example
-------
Sentence:  "Chronic sleep deprivation leads to cognitive decline in the elderly."
Span:      "cognitive decline"
→ Canonical: "cognitive decline in elderly adults"

Sentence:  "It causes widespread inflammation in the joints."
Span:      "It"
→ Canonical: "the unspecified factor causing joint inflammation"

Deduplication
-------------
Unique (span_text.lower(), sentence) pairs are generated once; all other
relations sharing the same key reuse the cached result.
"""
from __future__ import annotations

import dataclasses

from pipeline.protocols import CausalRelation, EventCanonizer


# ---------------------------------------------------------------------------
# Prompt helpers
# ---------------------------------------------------------------------------

_SYSTEM = (
    "You are a scientific text editor. "
    "Rewrite the highlighted event span into a concise, self-contained noun "
    "phrase (2–8 words) that fully describes the event. "
    "Use the surrounding sentence for context when it adds precision. "
    "Return only the rewritten phrase — no explanation, no punctuation at the end."
)


def _make_prompt(sentence: str, span: str) -> str:
    if sentence:
        return (
            f"Sentence: {sentence}\n"
            f"Event span: «{span}»\n"
            f"Self-contained description:"
        )
    return (
        f"Event span: «{span}»\n"
        f"Self-contained description:"
    )


def _clean(output: str, fallback: str) -> str:
    """Strip generation artifacts; fall back to original span if empty."""
    text = output.strip().strip('«»"\'')
    text = text.split("\n")[0].strip().rstrip(".")
    return text or fallback


# ---------------------------------------------------------------------------
# Canonizer
# ---------------------------------------------------------------------------

class TransformerCanonizer:
    """
    Implements EventCanonizer using a local instruction-tuned causal LM.

    Parameters
    ----------
    model_name : str
        HuggingFace model ID for an instruction-tuned decoder-only model.
        Default: ``Qwen/Qwen2.5-1.5B-Instruct``.
        Larger alternatives: ``Qwen/Qwen2.5-3B-Instruct``,
        ``mistralai/Mistral-7B-Instruct-v0.3``.
    batch_size : int
        Prompts per inference batch.  On CPU, 4–8 is a good balance; on
        GPU increase to 32–64.  Default: 8.
    max_new_tokens : int
        Maximum output tokens.  Event spans are short phrases, so 24 is
        generous.  Default: 24.
    device : int
        Torch device index.  -1 = CPU (default); 0 = first CUDA GPU.
    fast_path : bool
        Accepted for config compatibility; has no effect.  Every span is
        always sent through the model so that context-enriched canonical
        descriptions are produced consistently.
    """

    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-1.5B-Instruct",
        batch_size: int = 8,
        max_new_tokens: int = 24,
        device: int = -1,
        fast_path: bool = False,   # kept for config-file compatibility
        **kwargs,
    ) -> None:
        self.model_name = model_name
        self.batch_size = batch_size
        self.max_new_tokens = max_new_tokens
        self.device = device
        self._model = None
        self._tokenizer = None

    @property
    def name(self) -> str:
        return "transformer"

    # ------------------------------------------------------------------
    # Model loading (lazy, once)
    # ------------------------------------------------------------------

    def _load(self):
        if self._model is not None:
            return self._model, self._tokenizer

        from transformers import AutoModelForCausalLM, AutoTokenizer
        import torch

        print(f"[TransformerCanonizer] Loading '{self.model_name}' …")
        self._tokenizer = AutoTokenizer.from_pretrained(
            self.model_name, padding_side="left"
        )
        if self._tokenizer.pad_token_id is None:
            self._tokenizer.pad_token_id = self._tokenizer.eos_token_id

        dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
        self._model = AutoModelForCausalLM.from_pretrained(
            self.model_name, torch_dtype=dtype
        )
        if self.device >= 0:
            self._model = self._model.to(f"cuda:{self.device}")
        self._model.eval()
        print(f"[TransformerCanonizer] Model loaded.")
        return self._model, self._tokenizer

    # ------------------------------------------------------------------
    # Batch inference
    # ------------------------------------------------------------------

    def _generate_batch(self, message_lists: list[list[dict]]) -> list[str]:
        import torch
        model, tokenizer = self._load()

        # Apply each message list through the model's own chat template
        texts = [
            tokenizer.apply_chat_template(
                msgs, tokenize=False, add_generation_prompt=True
            )
            for msgs in message_lists
        ]

        inputs = tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512,
        )
        if self.device >= 0:
            inputs = {k: v.to(f"cuda:{self.device}") for k, v in inputs.items()}

        prompt_len = inputs["input_ids"].shape[1]
        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )

        # Decode only the newly generated tokens (not the prompt)
        new_ids = output_ids[:, prompt_len:]
        return tokenizer.batch_decode(new_ids, skip_special_tokens=True)

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def canonize(self, relations: list[CausalRelation]) -> list[CausalRelation]:
        if not relations:
            return []

        # 1. Collect unique (span_lower, sentence) keys in order
        key_to_orig: dict[tuple[str, str], str] = {}
        model_keys: list[tuple[str, str]] = []
        seen: set[tuple[str, str]] = set()

        def _register(span: str, sentence: str) -> None:
            key = (span.lower(), sentence)
            if key not in seen:
                seen.add(key)
                key_to_orig[key] = span
                model_keys.append(key)

        for r in relations:
            _register(r.cause_text, r.post_title)
            _register(r.effect_text, r.post_title)

        # 2. Build chat messages for each unique key
        message_lists = [
            [
                {"role": "system", "content": _SYSTEM},
                {"role": "user", "content": _make_prompt(key[1], key_to_orig[key])},
            ]
            for key in model_keys
        ]

        # 3. Run inference in batches
        n = len(message_lists)
        print(
            f"[TransformerCanonizer] Canonizing {n:,} unique spans "
            f"({len(relations):,} relations total) …"
        )
        all_outputs: list[str] = []
        for start in range(0, n, self.batch_size):
            batch = message_lists[start : start + self.batch_size]
            all_outputs.extend(self._generate_batch(batch))
            done = min(start + self.batch_size, n)
            if done % max(self.batch_size * 10, 1) == 0 or done == n:
                print(f"  {done:,}/{n:,} spans processed …")

        # 4. Build canonical lookup
        key_to_canonical: dict[tuple[str, str], str] = {
            key: _clean(out, key_to_orig[key])
            for key, out in zip(model_keys, all_outputs)
        }

        # 5. Assemble result relations
        return [
            dataclasses.replace(
                r,
                cause_canonical=key_to_canonical.get(
                    (r.cause_text.lower(), r.post_title), r.cause_text
                ),
                effect_canonical=key_to_canonical.get(
                    (r.effect_text.lower(), r.post_title), r.effect_text
                ),
            )
            for r in relations
        ]
