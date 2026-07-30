"""
DataModule for Step 1 — Causality Detection.

Expected JSONL format (one JSON object per line):
    {"title": "Smoking causes lung cancer.", "label": 1}
    {"title": "Scientists discover new species.", "label": 0}

label: 1 = causal post, 0 = non-causal post
"""
from __future__ import annotations

import json
from pathlib import Path

import torch
from lightning.pytorch import LightningDataModule
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer


class CausalityDetectionDataset(Dataset):
    def __init__(self, path: str, tokenizer, max_length: int) -> None:
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.records = [json.loads(l) for l in Path(path).read_text().splitlines() if l.strip()]

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> dict:
        rec = self.records[idx]
        enc = self.tokenizer(
            rec["text"],
            max_length=self.max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        return {k: v.squeeze(0) for k, v in enc.items()} | {"labels": torch.tensor(rec["label"])}


class CausalityDetectionDataModule(LightningDataModule):
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
        self.train_ds = CausalityDetectionDataset(self.hparams.train_path, tokenizer, self.hparams.max_length)
        self.val_ds = CausalityDetectionDataset(self.hparams.val_path, tokenizer, self.hparams.max_length)

    def train_dataloader(self) -> DataLoader:
        return DataLoader(self.train_ds, batch_size=self.hparams.batch_size,
                          shuffle=True, num_workers=self.hparams.num_workers, pin_memory=True)

    def val_dataloader(self) -> DataLoader:
        return DataLoader(self.val_ds, batch_size=self.hparams.batch_size,
                          num_workers=self.hparams.num_workers, pin_memory=True)
