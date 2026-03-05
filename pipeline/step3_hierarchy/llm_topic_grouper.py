"""
Step 3 alternative: LLM-based topic grouping.

Sends batches of event texts to an LLM and asks it to assign them to
topic clusters. Produces human-readable labels at the cost of API calls.
"""
from __future__ import annotations

import json
import os
import re
from collections import defaultdict

from pipeline.protocols import CausalRelation, EventCluster, HierarchyInferrer


class LLMTopicGrouper:
    """
    Implements HierarchyInferrer using LLM-assigned topic labels.
    Only generates a flat (1-level) grouping; parent clusters are empty shells.
    """

    def __init__(
        self,
        llm_model: str = "claude-haiku-4-5-20251001",
        llm_n_topics: int = 30,
        **kwargs,
    ) -> None:
        self.model = llm_model
        self.n_topics = llm_n_topics
        self._provider = "anthropic" if "claude" in llm_model else "openai"

    @property
    def name(self) -> str:
        return "llm"

    def infer(
        self,
        relations: list[CausalRelation],
    ) -> tuple[list[EventCluster], list[tuple[int, int, str, str]]]:
        if not relations:
            return [], []

        # Collect unique event texts
        unique_texts = list({r.cause_norm for r in relations} | {r.effect_norm for r in relations})
        print(f"[LLMTopicGrouper] Classifying {len(unique_texts)} unique events into {self.n_topics} topics...")

        text_to_topic: dict[str, str] = self._assign_topics(unique_texts)

        # Build cluster list
        topic_set: list[str] = sorted(set(text_to_topic.values()))
        topic_to_cluster_pos: dict[str, int] = {t: i for i, t in enumerate(topic_set)}
        topic_texts: dict[str, list[str]] = defaultdict(list)
        for text, topic in text_to_topic.items():
            topic_texts[topic].append(text)

        clusters: list[EventCluster] = [
            EventCluster(
                label=topic,
                level=0,
                parent_id=None,
                member_count=len(topic_texts[topic]),
                clusterer=self.name,
            )
            for topic in topic_set
        ]

        # Build memberships
        memberships: list[tuple[int, int, str, str]] = []
        for rel_idx, relation in enumerate(relations):
            for role, norm in (("cause", relation.cause_norm), ("effect", relation.effect_norm)):
                topic = text_to_topic.get(norm, topic_set[0])
                cluster_pos = topic_to_cluster_pos[topic]
                memberships.append((rel_idx, cluster_pos, role, norm))

        return clusters, memberships

    def _assign_topics(self, texts: list[str]) -> dict[str, str]:
        batch_size = 100
        text_to_topic: dict[str, str] = {}

        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            user_msg = (
                f"Assign each of the following scientific event phrases to one of {self.n_topics} "
                f"broad topic categories (e.g., 'cardiovascular health', 'mental health', "
                f"'climate change', 'nutrition', 'genetics', etc.).\n\n"
                f"Return a JSON object mapping each phrase to its topic label.\n\n"
                f"Phrases:\n{json.dumps(batch, ensure_ascii=False)}"
            )
            try:
                if self._provider == "anthropic":
                    result = self._call_anthropic(user_msg)
                else:
                    result = self._call_openai(user_msg)
                text_to_topic.update(result)
            except Exception as e:
                print(f"[LLMTopicGrouper] API error: {e}. Assigning 'Other' to batch.")
                text_to_topic.update({t: "Other" for t in batch})

        return text_to_topic

    def _call_anthropic(self, user_msg: str) -> dict[str, str]:
        import anthropic
        client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))
        message = client.messages.create(
            model=self.model,
            max_tokens=2048,
            messages=[{"role": "user", "content": user_msg}],
        )
        return json.loads(message.content[0].text.strip())

    def _call_openai(self, user_msg: str) -> dict[str, str]:
        from openai import OpenAI
        client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
        response = client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": user_msg}],
            max_tokens=2048,
            response_format={"type": "json_object"},
        )
        return json.loads(response.choices[0].message.content)
