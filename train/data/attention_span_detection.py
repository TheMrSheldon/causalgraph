"""
DataModule for attention-based span detection (Task 2).

Produces the batch format required by AttentionSpanDetectionModule:
    token_labels      – (L,) float  0.0 = background, 1.0 = span token
    span_ids          – (L,) long   0 = background, k = k-th span (1-indexed)
    real_token_mask   – (L,) bool   True for non-special, non-padding tokens
    offset_mapping    – (L, 2) long char offsets per token

Accepts the same JSONL / parquet paths as SpanDetectionDataModule.
"""
from __future__ import annotations

from typing import Union

import torch
from lightning.pytorch import LightningDataModule
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer

from train.data.span_detection import _load_records


class AttentionSpanDetectionDataset(Dataset):
    def __init__(
        self,
        paths: Union[str, list[str]],
        tokenizer,
        max_length: int,
        filter_empty: bool = True,
        negatives_only_paths: set[str] | None = None,
    ) -> None:
        self.tokenizer = tokenizer
        self.max_length = max_length
        if isinstance(paths, str):
            paths = [paths]
        negatives_only = set(negatives_only_paths or [])
        records = []
        for path in paths:
            path_records = _load_records(path)
            if path in negatives_only:
                # Strip entity annotations — keep the sentence as a negative example.
                path_records = [{**r, "entity": []} for r in path_records]
            records.extend(path_records)
        if filter_empty:
            records = [r for r in records if r.get("entity") is not None and len(r["entity"]) > 0]
        self.records = records

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> dict:
        rec = self.records[idx]
        text = rec["text"]
        spans = [(int(s[0]), int(s[1])) for s in rec.get("entity", [])]

        enc = self.tokenizer(
            text,
            max_length=self.max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
            return_offsets_mapping=True,
        )
        offset_mapping = enc.pop("offset_mapping").squeeze(0)  # (L, 2)

        token_labels: list[float] = []
        span_ids: list[int] = []
        real_token_mask: list[bool] = []

        for tok_start, tok_end in offset_mapping.tolist():
            is_real = not (tok_start == 0 and tok_end == 0)
            real_token_mask.append(is_real)
            if not is_real:
                token_labels.append(0.0)
                span_ids.append(0)
                continue
            found = 0
            for k, (s, e) in enumerate(spans, start=1):
                if tok_start < e and tok_end > s:  # any overlap
                    found = k
                    break
            span_ids.append(found)
            token_labels.append(1.0 if found > 0 else 0.0)

        return {k: v.squeeze(0) for k, v in enc.items()} | {
            "token_labels": torch.tensor(token_labels, dtype=torch.float),
            "span_ids": torch.tensor(span_ids, dtype=torch.long),
            "real_token_mask": torch.tensor(real_token_mask, dtype=torch.bool),
            "offset_mapping": offset_mapping,
        }


class AttentionSpanDetectionDataModule(LightningDataModule):
    def __init__(
        self,
        train_paths: Union[str, list[str]],
        val_path: str,
        model_name: str = "roberta-large",
        max_length: int = 128,
        batch_size: int = 32,
        num_workers: int = 4,
        filter_train_empty: bool = False,
        negatives_only_paths: list[str] | None = None,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()

    def setup(self, stage: str | None = None) -> None:
        tokenizer = AutoTokenizer.from_pretrained(self.hparams.model_name)
        neg_only = self.hparams.negatives_only_paths or []
        if stage in ("fit", None):
            self.train_ds = AttentionSpanDetectionDataset(
                self.hparams.train_paths, tokenizer, self.hparams.max_length,
                filter_empty=self.hparams.filter_train_empty,
                negatives_only_paths=neg_only,
            )
            self.val_ds = AttentionSpanDetectionDataset(
                self.hparams.val_path, tokenizer, self.hparams.max_length, filter_empty=False
            )
        elif stage == "validate":
            self.val_ds = AttentionSpanDetectionDataset(
                self.hparams.val_path, tokenizer, self.hparams.max_length, filter_empty=False
            )

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_ds, batch_size=self.hparams.batch_size,
            shuffle=True, num_workers=self.hparams.num_workers, pin_memory=True,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_ds, batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers, pin_memory=True,
        )
