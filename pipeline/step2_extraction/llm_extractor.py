"""
Step 2 alternative: LLM-based causal extraction.

Uses structured output (tool-calling / JSON mode) to extract cause and
effect as structured fields. Supports Anthropic and OpenAI APIs.
"""
from __future__ import annotations

import json
import os
import re
import unicodedata

from pipeline.protocols import CausalExtractor, CausalRelation, Post


def _normalize(text: str) -> str:
    text = unicodedata.normalize("NFKD", text)
    text = text.encode("ascii", "ignore").decode("ascii")
    return re.sub(r"\s+", " ", text.lower().strip())


_SYSTEM_PROMPT = """You are a scientific causal relation extractor.
Given a Reddit submission title from r/science, extract the causal relationship expressed.
Return a JSON object with:
  "cause": the cause entity/event (concise noun phrase, max 10 words)
  "effect": the effect entity/event (concise noun phrase, max 10 words)
  "confidence": float 0.0-1.0 indicating extraction confidence

If no clear causal relationship exists, return {"cause": null, "effect": null, "confidence": 0}.
"""


class LLMExtractor:
    """
    Implements CausalExtractor using an LLM API with structured output.
    """

    def __init__(
        self,
        llm_model: str = "claude-haiku-4-5-20251001",
        llm_max_tokens: int = 256,
        spacy_model: str = "en_core_web_sm",
        **kwargs,
    ) -> None:
        self.model = llm_model
        self.max_tokens = llm_max_tokens
        self._provider = "anthropic" if "claude" in llm_model else "openai"

    @property
    def name(self) -> str:
        return f"llm_{self._provider}"

    def extract(self, post: Post) -> list[CausalRelation]:
        try:
            if self._provider == "anthropic":
                result = self._call_anthropic(post.title)
            else:
                result = self._call_openai(post.title)
        except Exception as e:
            print(f"[LLMExtractor] API error for post {post.id}: {e}")
            return []

        cause = result.get("cause")
        effect = result.get("effect")
        confidence = float(result.get("confidence", 0.0))

        if not cause or not effect or confidence < 0.3:
            return []

        return [
            CausalRelation(
                post_id=post.id,
                cause_text=cause,
                effect_text=effect,
                cause_norm=_normalize(cause),
                effect_norm=_normalize(effect),
                confidence=confidence,
                extractor=self.name,
            )
        ]

    def _call_anthropic(self, title: str) -> dict:
        import anthropic
        client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))
        message = client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            system=_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": f'Title: "{title}"'}],
        )
        text = message.content[0].text.strip()
        return json.loads(text)

    def _call_openai(self, title: str) -> dict:
        from openai import OpenAI
        client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
        response = client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": f'Title: "{title}"'},
            ],
            max_tokens=self.max_tokens,
            response_format={"type": "json_object"},
        )
        return json.loads(response.choices[0].message.content)
