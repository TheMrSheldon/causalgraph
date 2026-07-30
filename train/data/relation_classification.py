"""
DataModule for Step 2b — Causal Relation Classification.

Reads the CNC causality-identification format:
    {
        "text": "... <e1>cause text</e1> ... <e2>effect text</e2> ...",
        "relations": [{"first": "e2", "relationship": 1, "second": "e1"}]
    }

Each record is expanded into per-pair training examples:
  - For every relation (first, second): rename <first> → <e0> (cause),
    <second> → <e1> (effect), strip other markers; label = causal/countercausal.
  - For every entity pair in the sentence with no relation: label = no-rel.

relationship values in CNC: 1 = causal, 2 = countercausal.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import torch
from lightning.pytorch import LightningDataModule
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer

LABELS = ["causal", "countercausal", "no-rel"]
LABEL2ID = {l: i for i, l in enumerate(LABELS)}

_REL_TO_LABEL = {1: "causal", 2: "countercausal"}
_ENTITY_TAG_RE = re.compile(r'<(/?)(e\d+)>')


def _format_pair(text: str, cause_id: str, effect_id: str) -> str:
    """Rename entity markers for (cause, effect) pair; strip all other markers.

    Done in a single pass so the renamed <e0>/<e1> tags are not stripped by
    the cleanup step (the previous two-step approach had this bug).
    """
    def replace(m: re.Match) -> str:
        slash, eid = m.group(1), m.group(2)
        if eid == cause_id:
            return f"<{slash}e0>"
        if eid == effect_id:
            return f"<{slash}e1>"
        return ""
    return _ENTITY_TAG_RE.sub(replace, text)


def _expand_record(rec: dict) -> list[dict]:
    """Convert one CNC record into a list of (text, label) training examples.

    Every ordered (cause, effect) pair of distinct entities in the sentence becomes
    one example. The label comes from directed_rels if the pair is annotated, and
    falls back to no-rel otherwise. This correctly captures reversed directions of
    causal relations as no-rel, e.g. for <e1>A</e1> causes <e2>B</e2>:
        (e1, e2) → causal
        (e2, e1) → no-rel   ← previously missing
    """
    text = rec["text"]
    relations = rec.get("relations", [])

    entity_ids = sorted(set(re.findall(r'<(e\d+)>', text)))
    if not entity_ids:
        return []

    directed_rels: dict[tuple[str, str], str] = {}
    for r in relations:
        label = _REL_TO_LABEL.get(r["relationship"])
        if label:
            directed_rels[(r["first"], r["second"])] = label

    examples: list[dict] = []
    for cause_id in entity_ids:
        for effect_id in entity_ids:
            if cause_id == effect_id:
                continue
            label = directed_rels.get((cause_id, effect_id), "no-rel")
            examples.append({"text": _format_pair(text, cause_id, effect_id), "label": label})

    return examples


class RelationClassificationDataset(Dataset):
    def __init__(self, path: str, tokenizer, max_length: int) -> None:
        self.tokenizer = tokenizer
        self.max_length = max_length
        records = [json.loads(l) for l in Path(path).read_text().splitlines() if l.strip()]
        self.examples: list[dict] = []
        for rec in records:
            self.examples.extend(_expand_record(rec))

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> dict:
        ex = self.examples[idx]
        enc = self.tokenizer(
            ex["text"],
            max_length=self.max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        return {k: v.squeeze(0) for k, v in enc.items()} | {
            "labels": torch.tensor(LABEL2ID[ex["label"]])
        }


class RelationClassificationDataModule(LightningDataModule):
    def __init__(
        self,
        train_path: str,
        val_path: str,
        model_name: str = "roberta-large",
        max_length: int = 128,
        batch_size: int = 32,
        num_workers: int = 4,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()

    def setup(self, stage: str | None = None) -> None:
        tokenizer = AutoTokenizer.from_pretrained(self.hparams.model_name)
        if stage in ("fit", None):
            self.train_ds = RelationClassificationDataset(
                self.hparams.train_path, tokenizer, self.hparams.max_length)
            self.val_ds = RelationClassificationDataset(
                self.hparams.val_path, tokenizer, self.hparams.max_length)
        elif stage == "validate":
            self.val_ds = RelationClassificationDataset(
                self.hparams.val_path, tokenizer, self.hparams.max_length)

    def train_dataloader(self) -> DataLoader:
        return DataLoader(self.train_ds, batch_size=self.hparams.batch_size,
                          shuffle=True, num_workers=self.hparams.num_workers, pin_memory=True)

    def val_dataloader(self) -> DataLoader:
        return DataLoader(self.val_ds, batch_size=self.hparams.batch_size,
                          num_workers=self.hparams.num_workers, pin_memory=True)
