"""
Step 2a — Causal Span Detection.

Token classifier: given a post title, label each token B-Entity / I-Entity / O
to identify candidate cause and effect spans.

Matches the inference model: thagen/roberta-large-causal-candidate-extraction
"""
import torch
from lightning.pytorch import LightningModule
from torchmetrics.classification import MulticlassF1Score
from transformers import AutoModelForTokenClassification, get_linear_schedule_with_warmup

from causalatee.evaluation._spans import dataset_span_scores

LABELS = ["O", "B-Entity", "I-Entity"]
LABEL2ID = {l: i for i, l in enumerate(LABELS)}
ID2LABEL = {i: l for i, l in enumerate(LABELS)}

_B, _I = LABEL2ID["B-Entity"], LABEL2ID["I-Entity"]


def _bio_to_char_spans(
    labels: list[int],
    offsets: list[tuple[int, int]],
) -> list[tuple[int, int]]:
    """Convert a BIO label sequence + offset_mapping to character-level spans.

    Special tokens and padding have offset (0,0) and are skipped.
    Continuation subwords (label=-100 with offset adjacent to the current span end)
    extend the open span so multi-subword entities resolve to the full word boundary.
    Other -100 positions (isolated, not adjacent) close the span like O.
    """
    spans: list[tuple[int, int]] = []
    span_start: int | None = None
    span_end: int | None = None
    for label, (cs, ce) in zip(labels, offsets):
        if cs == ce == 0:  # special token or padding — close any open span and skip
            if span_start is not None:
                spans.append((span_start, span_end))
                span_start = span_end = None
            continue
        if label == -100:
            # Continuation subword: extend the open span if this token is adjacent.
            # Non-adjacent -100 (shouldn't normally occur) closes the span.
            if span_start is not None and cs == span_end:
                span_end = ce
            elif span_start is not None:
                spans.append((span_start, span_end))
                span_start = span_end = None
            continue
        if label == _B:
            if span_start is not None:
                spans.append((span_start, span_end))
            span_start, span_end = cs, ce
        elif label == _I:
            if span_start is None:
                span_start, span_end = cs, ce  # malformed I without B
            else:
                span_end = ce
        else:
            if span_start is not None:
                spans.append((span_start, span_end))
                span_start = span_end = None
    if span_start is not None:
        spans.append((span_start, span_end))
    return spans


class SpanDetectionModule(LightningModule):
    def __init__(
        self,
        model_name: str = "roberta-large",
        learning_rate: float = 2e-5,
        weight_decay: float = 0.01,
        warmup_steps: int = 200,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.model = AutoModelForTokenClassification.from_pretrained(
            model_name,
            num_labels=len(LABELS),
            id2label=ID2LABEL,
            label2id=LABEL2ID,
        )
        n = len(LABELS)
        # ignore_index=-100 masks subword continuations and padding
        self.train_f1 = MulticlassF1Score(num_classes=n, ignore_index=-100, average="macro")
        self.val_f1 = MulticlassF1Score(num_classes=n, ignore_index=-100, average="macro")
        self.val_f1_per_class = MulticlassF1Score(num_classes=n, ignore_index=-100, average="none")
        self._val_span_records: list[dict] = []

    def _step(self, batch: dict) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        labels = batch.pop("labels")
        outputs = self.model(**batch, labels=labels)
        preds = outputs.logits.argmax(dim=-1)
        return outputs.loss, preds, labels

    def training_step(self, batch: dict, batch_idx: int) -> torch.Tensor:
        batch.pop("offset_mapping")  # not used during training
        loss, preds, labels = self._step(batch)
        self.train_f1(preds.view(-1), labels.view(-1))
        self.log("train/loss", loss, prog_bar=True)
        self.log("train/f1", self.train_f1, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch: dict, batch_idx: int) -> None:
        offsets = batch.pop("offset_mapping")  # (batch, seq_len, 2)
        loss, preds, labels = self._step(batch)

        flat_preds, flat_labels = preds.view(-1), labels.view(-1)
        self.val_f1(flat_preds, flat_labels)
        self.val_f1_per_class(flat_preds, flat_labels)
        self.log("val/loss", loss, prog_bar=True)
        self.log("val/f1", self.val_f1, on_epoch=True, prog_bar=True)

        # Accumulate per-sample char-span records for span metrics at epoch end.
        # Predicted labels at gold=-100 positions (subword continuations / special tokens)
        # are forced to O so they don't generate spurious predicted spans.
        for pred_seq, label_seq, off_seq in zip(
            preds.tolist(), labels.tolist(), offsets.tolist()
        ):
            masked_pred = [p if l != -100 else -100 for p, l in zip(pred_seq, label_seq)]
            self._val_span_records.append({
                "y_true": _bio_to_char_spans(label_seq, off_seq),
                "y_pred": _bio_to_char_spans(masked_pred, off_seq),
            })

    def on_validation_epoch_end(self) -> None:
        # Per-class token-level F1
        f1_scores = self.val_f1_per_class.compute()
        for i, name in enumerate(LABELS):
            self.log(f"val/f1_{name.replace('-', '_')}", f1_scores[i])
        self.val_f1_per_class.reset()

        # Span-level character metrics (positive sentences only)
        pos = [r for r in self._val_span_records if r["y_true"]]
        span_scores = dataset_span_scores(
            [r["y_true"] for r in pos], [r["y_pred"] for r in pos]
        )
        for metric, value in span_scores.items():
            self.log(f"val/span_{metric}", value)
        self._val_span_records.clear()

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
