"""
Step 1 alternative: LLM-based causality identification.

Supports both OpenAI and Anthropic APIs via the 'provider' kwarg.
Batches titles to minimize API calls (50 titles per request by default).
"""
from __future__ import annotations

import json
import os

from pipeline.protocols import CausalityIdentifier, Post

_SYSTEM_PROMPT = """You are a scientific text classifier. Given a list of r/science submission titles,
identify which ones express a causal relationship — either explicit (e.g., "X causes Y") or
implicit (e.g., "Study finds X reduces Y risk"). Return a JSON array of the indices (0-based)
of titles that express causality. Be strict: exclude purely descriptive or observational titles."""

_USER_TEMPLATE = """Titles (JSON array):
{titles_json}

Return only a JSON array of integer indices of causal titles. Example: [0, 2, 5]"""


class LLMIdentifier:
    """
    Implements CausalityIdentifier using an LLM API.
    Set OPENAI_API_KEY or ANTHROPIC_API_KEY in environment.
    """

    def __init__(
        self,
        llm_model: str = "claude-haiku-4-5-20251001",
        llm_batch_size: int = 50,
        zero_shot_model: str = "",
        zero_shot_threshold: float = 0.75,
        **kwargs,
    ) -> None:
        self.model = llm_model
        self.batch_size = llm_batch_size
        self._provider = "anthropic" if "claude" in llm_model else "openai"

    @property
    def name(self) -> str:
        return f"llm_{self._provider}"

    def identify(self, posts: list[Post]) -> list[Post]:
        causal: list[Post] = []
        for i in range(0, len(posts), self.batch_size):
            batch = posts[i : i + self.batch_size]
            titles = [p.title for p in batch]
            indices = self._classify_batch(titles)
            causal.extend(batch[j] for j in indices if 0 <= j < len(batch))
        return causal

    def _classify_batch(self, titles: list[str]) -> list[int]:
        user_msg = _USER_TEMPLATE.format(titles_json=json.dumps(titles, ensure_ascii=False))
        try:
            if self._provider == "anthropic":
                return self._call_anthropic(user_msg)
            return self._call_openai(user_msg)
        except Exception as e:
            print(f"[LLMIdentifier] API error: {e}. Returning empty for this batch.")
            return []

    def _call_anthropic(self, user_msg: str) -> list[int]:
        import anthropic
        client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))
        message = client.messages.create(
            model=self.model,
            max_tokens=512,
            system=_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_msg}],
        )
        text = message.content[0].text.strip()
        return json.loads(text)

    def _call_openai(self, user_msg: str) -> list[int]:
        from openai import OpenAI
        client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
        response = client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": user_msg},
            ],
            max_tokens=512,
            response_format={"type": "json_object"},
        )
        text = response.choices[0].message.content.strip()
        data = json.loads(text)
        # Handle both {"indices": [...]} and bare [...]
        if isinstance(data, list):
            return data
        return data.get("indices", [])
