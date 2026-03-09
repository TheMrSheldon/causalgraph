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

from pipeline.protocols import EventCanonizer


# ---------------------------------------------------------------------------
# Prompt helpers
# ---------------------------------------------------------------------------

_SYSTEM = """You convert a text span into a short, self-contained description of the event or concept it refers to.

You are given:
- TEXT: a passage
- SPAN: a substring from that passage

Task:
Write a short canonical phrase that describes the event or concept in the SPAN so that it is understandable without the original text.

Rules:
- Use surrounding context from TEXT if necessary.
- Keep the result concise (typically 2–8 words).
- Preserve the meaning of the SPAN.
- Do not repeat unnecessary context.
- Output only the canonicalized phrase.
"""

_USER_PROMPT = """
TEXT:
{text}

SPAN:
{span}

CANONIZED:
"""

_EXAMPLES: list[tuple[str, str, str]] = [
    ("Smoking significantly increases the risk of heart disease.", "increases the risk of heart disease", "Increase in risk of heart disease"),
    ("Study finds that eating 2 mushrooms per day can decrease the risk of most cancers by almost half.", "the risk of most cancers by almost half", "Decrease of the risk of most cancers by almost half"),
    ("Heavy rainfall flooded several villages in the region.", "flooded several villages", "Flooding of several villages"),
]


def _make_prompt(sentence: str, span: str) -> str:
    return _USER_PROMPT.format_map({"text": sentence, "span": span})


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

    def canonize(self, spans: list[tuple[str, tuple[int, int]]]) -> list[str]:
        if not spans:
            return []

        # 1. Collect unique (span_lower, text) keys in insertion order
        #    to avoid redundant inference for identical spans in the same context.
        key_to_raw: dict[tuple[str, str], str] = {}   # (span_lower, text) → raw span
        model_keys: list[tuple[str, str]] = []
        seen: set[tuple[str, str]] = set()

        input_keys: list[tuple[str, str]] = []
        for text, (start, end) in spans:
            raw_span = text[start:end]
            key = (raw_span.lower(), text)
            input_keys.append(key)
            if key not in seen:
                seen.add(key)
                key_to_raw[key] = raw_span
                model_keys.append(key)

        # 2. Build chat messages for each unique key
        examples = [
            msg
            for ex_text, ex_span, ex_canon in _EXAMPLES
            for msg in (
                {"role": "user", "content": _make_prompt(ex_text, ex_span)},
                {"role": "assistant", "content": ex_canon},
            )
        ]
        message_lists = [
            [
                {"role": "system", "content": _SYSTEM},
                *examples,
                {"role": "user", "content": _make_prompt(key[1], key_to_raw[key])},
            ]
            for key in model_keys
        ]

        # 3. Run inference in batches
        n = len(message_lists)
        print(
            f"[TransformerCanonizer] Canonizing {n:,} unique spans "
            f"({len(spans):,} spans total) …"
        )
        all_outputs: list[str] = []
        for start in range(0, n, self.batch_size):
            batch = message_lists[start : start + self.batch_size]
            all_outputs.extend(self._generate_batch(batch))
            done = min(start + self.batch_size, n)
            if done % max(self.batch_size * 10, 1) == 0 or done == n:
                print(f"  {done:,}/{n:,} spans processed …")

        # 4. Build lookup: unique key → canonical string
        key_to_canonical: dict[tuple[str, str], str] = {
            key: _clean(out, key_to_raw[key])
            for key, out in zip(model_keys, all_outputs)
        }

        # 5. Return one canonical per input span (reusing cached results)
        return [
            key_to_canonical.get(key, key_to_raw.get(key, ""))
            for key in input_keys
        ]
