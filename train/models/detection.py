"""
Step 1 — Causality Detection.

Binary sequence classifier: given a post title, predict whether it expresses
a causal relationship (label=1) or not (label=0).

Matches the inference model: thagen/roberta-large-causal-candidate-extraction
"""
import torch
from lightning.pytorch import LightningModule
from torchmetrics.classification import BinaryAccuracy, BinaryF1Score, BinaryPrecision, BinaryRecall
from transformers import AutoModelForSequenceClassification, get_linear_schedule_with_warmup


class CausalityDetectionModule(LightningModule):
    def __init__(
        self,
        model_name: str = "roberta-large",
        learning_rate: float = 2e-5,
        weight_decay: float = 0.01,
        warmup_steps: int = 200,
        pos_weight: float = 1.0,  # upweight positive class if dataset is imbalanced
    ) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

        for split in ("train", "val"):
            for name, metric in [
                ("f1", BinaryF1Score()),
                ("precision", BinaryPrecision()),
                ("recall", BinaryRecall()),
                ("accuracy", BinaryAccuracy()),
            ]:
                setattr(self, f"{split}_{name}", metric)

    def forward(self, **batch):
        return self.model(**batch)

    def _step(self, batch: dict) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        labels = batch.pop("labels")
        outputs = self.model(**batch)
        weight = torch.tensor(
            [1.0, self.hparams.pos_weight], device=outputs.logits.device, dtype=outputs.logits.dtype
        )
        loss = torch.nn.functional.cross_entropy(outputs.logits, labels, weight=weight)
        return loss, outputs.logits.argmax(dim=-1), labels

    def training_step(self, batch: dict, batch_idx: int) -> torch.Tensor:
        loss, preds, labels = self._step(batch)
        self.train_f1(preds, labels)
        self.train_precision(preds, labels)
        self.train_recall(preds, labels)
        self.log("train/loss", loss, prog_bar=True)
        self.log("train/f1", self.train_f1, on_step=False, on_epoch=True)
        self.log("train/precision", self.train_precision, on_step=False, on_epoch=True)
        self.log("train/recall", self.train_recall, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch: dict, batch_idx: int) -> None:
        loss, preds, labels = self._step(batch)
        self.val_f1(preds, labels)
        self.val_precision(preds, labels)
        self.val_recall(preds, labels)
        self.val_accuracy(preds, labels)
        self.log("val/loss", loss, prog_bar=True)
        self.log("val/f1", self.val_f1, on_epoch=True, prog_bar=True)
        self.log("val/precision", self.val_precision, on_epoch=True)
        self.log("val/recall", self.val_recall, on_epoch=True)
        self.log("val/accuracy", self.val_accuracy, on_epoch=True)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.hparams.learning_rate,
            weight_decay=self.hparams.weight_decay,
        )
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=self.hparams.warmup_steps,
            num_training_steps=self.trainer.estimated_stepping_batches,
        )
        return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "interval": "step"}}
