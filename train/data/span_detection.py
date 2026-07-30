"""
DataModule for Step 2a — Causal Span Detection.

Accepts JSONL (CNC format) and parquet (causalatee format) files, mixed freely:
    JSONL:   {"text": "...", "entity": [[83, 201], [209, 255]]}
    Parquet: columns text (str) and entity (list[list[int]])

Records with no annotated spans (entity == []) are filtered out at load time
so the model only trains on causal sentences, matching the inference distribution
(Step 1 has already filtered to causal sentences before Step 2 runs).
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Union

import torch
from lightning.pytorch import LightningDataModule
from torch.utils.data import ConcatDataset, DataLoader, Dataset
from transformers import AutoTokenizer

LABEL2ID = {"O": 0, "B-Entity": 1, "I-Entity": 2}


def _load_records(path: str) -> list[dict]:
    p = Path(path)
    if p.suffix == ".parquet":
        import pandas as pd
        df = pd.read_parquet(p)
        return df[["text", "entity"]].to_dict("records")
    return [json.loads(l) for l in p.read_text().splitlines() if l.strip()]


class SpanDetectionDataset(Dataset):
    def __init__(
        self,
        paths: Union[str, list[str]],
        tokenizer,
        max_length: int,
        filter_empty: bool = True,
    ) -> None:
        self.tokenizer = tokenizer
        self.max_length = max_length
        if isinstance(paths, str):
            paths = [paths]
        records = []
        for path in paths:
            records.extend(_load_records(path))
        if filter_empty:
            records = [r for r in records if r.get("entity") is not None and len(r["entity"]) > 0]
        self.records = records

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> dict:
        rec = self.records[idx]
        text = rec["text"]

        # Build char-level label array: 0=O, 1=B, 2=I
        char_labels = [0] * len(text)
        for span in rec.get("entity", []):
            s, e = int(span[0]), int(span[1])
            for i in range(s, min(e, len(char_labels))):
                char_labels[i] = 1 if i == s else 2

        enc = self.tokenizer(
            text,
            max_length=self.max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
            return_offsets_mapping=True,
        )
        # word_ids() uses the underlying fast-tokenizer encoding; available even
        # after return_tensors="pt". None marks special tokens and padding.
        word_ids = enc.word_ids(batch_index=0)
        offset_mapping = enc.pop("offset_mapping").squeeze(0)  # (seq_len, 2)
        offsets = offset_mapping.tolist()

        prev_word_id = None
        labels = []
        for wid, (tok_start, _tok_end) in zip(word_ids, offsets):
            if wid is None:
                labels.append(-100)
            elif wid == prev_word_id:  # continuation subword → mask
                labels.append(-100)
            else:
                labels.append(char_labels[tok_start] if tok_start < len(char_labels) else 0)
            prev_word_id = wid

        return {k: v.squeeze(0) for k, v in enc.items()} | {
            "labels": torch.tensor(labels, dtype=torch.long),
            "offset_mapping": offset_mapping,  # kept for char-span evaluation
        }


class SpanDetectionDataModule(LightningDataModule):
    def __init__(
        self,
        train_paths: Union[str, list[str]],
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
            self.train_ds = SpanDetectionDataset(
                self.hparams.train_paths, tokenizer, self.hparams.max_length, filter_empty=True
            )
            self.val_ds = SpanDetectionDataset(
                self.hparams.val_path, tokenizer, self.hparams.max_length, filter_empty=False
            )
        elif stage == "validate":
            self.val_ds = SpanDetectionDataset(
                self.hparams.val_path, tokenizer, self.hparams.max_length, filter_empty=False
            )

    def train_dataloader(self) -> DataLoader:
        return DataLoader(self.train_ds, batch_size=self.hparams.batch_size,
                          shuffle=True, num_workers=self.hparams.num_workers, pin_memory=True)

    def val_dataloader(self) -> DataLoader:
        return DataLoader(self.val_ds, batch_size=self.hparams.batch_size,
                          num_workers=self.hparams.num_workers, pin_memory=True)
