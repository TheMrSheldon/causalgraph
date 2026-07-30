"""
Step 2b — Causal Relation Classification.

Sequence classifier: given a title with <e0>cause</e0> and <e1>effect</e1>
markers, predict causal / countercausal / no-rel.

Matches the inference model: thagen/roberta-large-causality-identification
"""
import torch
from lightning.pytorch import LightningModule
from torchmetrics.classification import MulticlassAccuracy, MulticlassF1Score
from transformers import AutoModelForSequenceClassification, get_linear_schedule_with_warmup

LABELS = ["causal", "countercausal", "no-rel"]
LABEL2ID = {l: i for i, l in enumerate(LABELS)}
ID2LABEL = {i: l for i, l in enumerate(LABELS)}


class RelationClassificationModule(LightningModule):
    def __init__(
        self,
        model_name: str = "roberta-large",
        learning_rate: float = 2e-5,
        weight_decay: float = 0.01,
        warmup_steps: int = 200,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            num_labels=len(LABELS),
            id2label=ID2LABEL,
            label2id=LABEL2ID,
        )
        n = len(LABELS)
        for split in ("train", "val"):
            setattr(self, f"{split}_f1", MulticlassF1Score(num_classes=n, average="macro"))
            setattr(self, f"{split}_accuracy", MulticlassAccuracy(num_classes=n))
        self.val_f1_per_class = MulticlassF1Score(num_classes=n, average="none")

    def _step(self, batch: dict) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        labels = batch.pop("labels")
        outputs = self.model(**batch, labels=labels)
        return outputs.loss, outputs.logits.argmax(dim=-1), labels

    def training_step(self, batch: dict, batch_idx: int) -> torch.Tensor:
        loss, preds, labels = self._step(batch)
        self.train_f1(preds, labels)
        self.train_accuracy(preds, labels)
        self.log("train/loss", loss, prog_bar=True)
        self.log("train/f1", self.train_f1, on_step=False, on_epoch=True)
        self.log("train/accuracy", self.train_accuracy, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch: dict, batch_idx: int) -> None:
        loss, preds, labels = self._step(batch)
        self.val_f1(preds, labels)
        self.val_accuracy(preds, labels)
        self.val_f1_per_class(preds, labels)
        self.log("val/loss", loss, prog_bar=True)
        self.log("val/f1", self.val_f1, on_epoch=True, prog_bar=True)
        self.log("val/accuracy", self.val_accuracy, on_epoch=True)

    def on_validation_epoch_end(self) -> None:
        f1_scores = self.val_f1_per_class.compute()
        for i, label in enumerate(LABELS):
            self.log(f"val/f1_{label.replace('-', '_')}", f1_scores[i])
        self.val_f1_per_class.reset()

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
