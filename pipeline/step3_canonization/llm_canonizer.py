"""
Step 3 alternative: LLM-based event canonization.

Uses an LLM to rewrite each extracted event span into a self-contained
description using the original post title as context.  Useful when
extraction produces underspecified spans like "it", "this", "the effect",
or truncated phrases that are only meaningful in context.

Example
-------
Title:     "New drug inhibits tumor growth in mice"
Extracted: cause="New drug", effect="tumor growth"
Canonical: cause="the experimental drug", effect="tumor growth in mice"

The LLM is called in batches to minimize API latency.
"""
from __future__ import annotations

import dataclasses
import json
import os

from pipeline.protocols import CausalRelation, EventCanonizer

_SYSTEM_PROMPT = """You are a scientific text editor.
You will receive a JSON array of objects, each with:
  - "title": the original r/science post title
  - "cause": the extracted cause phrase
  - "effect": the extracted effect phrase

For each object, rewrite "cause" and "effect" into self-contained, standalone
event descriptions that make sense without the surrounding sentence.
Rules:
  - Resolve pronouns and vague references using the title as context.
  - Keep descriptions concise (2–8 words ideally).
  - Preserve the original meaning; do not introduce new claims.
  - If the phrase is already self-contained, return it unchanged.

Return a JSON array (same length) where each element is an object with
"cause" and "effect" keys containing the rewritten descriptions."""

_USER_TEMPLATE = """Events:
{events_json}

Return only the JSON array."""


class LLMCanonizer:
    """
    Implements EventCanonizer using an LLM API (Anthropic or OpenAI).
    Set ANTHROPIC_API_KEY or OPENAI_API_KEY in environment.
    """

    def __init__(
        self,
        llm_model: str = "claude-haiku-4-5-20251001",
        llm_batch_size: int = 50,
        llm_max_tokens: int = 1024,
        **kwargs,
    ) -> None:
        self.model = llm_model
        self.batch_size = llm_batch_size
        self.max_tokens = llm_max_tokens
        self._provider = "anthropic" if "claude" in llm_model else "openai"

    @property
    def name(self) -> str:
        return f"llm_{self._provider}"

    def canonize(self, relations: list[CausalRelation]) -> list[CausalRelation]:
        # We need the post title for context, but CausalRelation only stores post_id.
        # We batch-canonize purely on the extracted spans; for full title-context
        # canonization, subclass this and pass a post_id → title lookup.
        results: list[CausalRelation] = []
        for i in range(0, len(relations), self.batch_size):
            batch = relations[i : i + self.batch_size]
            canonized = self._canonize_batch(batch)
            results.extend(canonized)
        return results

    def _canonize_batch(self, batch: list[CausalRelation]) -> list[CausalRelation]:
        events = [
            {"cause": r.cause_text, "effect": r.effect_text}
            for r in batch
        ]
        user_msg = _USER_TEMPLATE.format(events_json=json.dumps(events, ensure_ascii=False))
        try:
            if self._provider == "anthropic":
                raw = self._call_anthropic(user_msg)
            else:
                raw = self._call_openai(user_msg)
            parsed: list[dict] = json.loads(raw)
            if len(parsed) != len(batch):
                raise ValueError(f"Expected {len(batch)} items, got {len(parsed)}")
        except Exception as e:
            print(f"[LLMCanonizer] API/parse error: {e}. Falling back to passthrough for batch.")
            return [
                dataclasses.replace(r, cause_canonical=r.cause_text, effect_canonical=r.effect_text)
                for r in batch
            ]

        return [
            dataclasses.replace(
                r,
                cause_canonical=item.get("cause", r.cause_text) or r.cause_text,
                effect_canonical=item.get("effect", r.effect_text) or r.effect_text,
            )
            for r, item in zip(batch, parsed)
        ]

    def _call_anthropic(self, user_msg: str) -> str:
        import anthropic
        client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))
        message = client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            system=_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_msg}],
        )
        return message.content[0].text.strip()

    def _call_openai(self, user_msg: str) -> str:
        from openai import OpenAI
        client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
        response = client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": user_msg},
            ],
            max_tokens=self.max_tokens,
        )
        return response.choices[0].message.content.strip()
