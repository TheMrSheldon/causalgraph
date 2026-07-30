"""
Step 2a (alternative) — Biaffine Span-Grid Causal Span Detection.

NER-as-parsing formulation (Yu et al. 2020, ACL): score every (start, end)
token pair with a biaffine classifier over an L×L grid.

    grid[i, j] = 1  iff  a gold span starts at token i and ends at token j

Each cell is an independent binary prediction, so nested and overlapping
spans are supported natively. Decoding is a simple threshold on the upper
triangle — no BIO transitions, no gap heuristics, no softmax read-out.

This replaces the same-span pairwise attention formulation, whose BCE
structure loss (background rows → all zeros) was mathematically incompatible
with the softmax read-out (rows must sum to 1), and whose same-span relation
cannot represent overlapping spans (it stops being transitive).
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from lightning.pytorch import LightningModule
from torch import Tensor
from transformers import AutoModel, get_linear_schedule_with_warmup

from causalatee.evaluation._spans import dataset_span_scores

from train.models.attention_span_detection import (
    AttentionSpanHead,
    _find_viz_example,
    _get_backbone_dims,
    _make_viz_item,
)


# ---------------------------------------------------------------------------
# Biaffine span head
# ---------------------------------------------------------------------------

class Biaffine(nn.Module):
    """Biaffine scorer: s(i,j) = x_i^T U y_j with bias terms via appended 1s."""

    def __init__(self, in_dim: int) -> None:
        super().__init__()
        self.U = nn.Parameter(torch.empty(in_dim + 1, in_dim + 1))
        nn.init.xavier_uniform_(self.U)

    def forward(self, x: Tensor, y: Tensor) -> Tensor:  # (B, L, d) × 2 → (B, L, L)
        # Separate ones per side: reusing one ones tensor for both breaks
        # when x and y have different lengths. Harmless here (always called
        # with equal-length x/y from the same hidden states) but fixed for
        # correctness — caught porting this class to causalatee.nn as a
        # general-purpose layer, where unequal lengths are a real use case.
        x = torch.cat([x, x.new_ones(*x.shape[:-1], 1)], dim=-1)
        y = torch.cat([y, y.new_ones(*y.shape[:-1], 1)], dim=-1)
        return torch.einsum("bxi,ij,byj->bxy", x, self.U, y)


class BiaffineSpanHead(nn.Module):
    """Scores every (start, end) token pair for being a causal span."""

    def __init__(self, hidden_size: int, ffn_dim: int = 256, dropout: float = 0.2) -> None:
        super().__init__()
        self.start_mlp = nn.Sequential(
            nn.Linear(hidden_size, ffn_dim), nn.GELU(), nn.Dropout(dropout)
        )
        self.end_mlp = nn.Sequential(
            nn.Linear(hidden_size, ffn_dim), nn.GELU(), nn.Dropout(dropout)
        )
        self.biaffine = Biaffine(ffn_dim)

    def forward(self, hidden_states: Tensor) -> Tensor:  # (B, L, H) → (B, L, L)
        return self.biaffine(self.start_mlp(hidden_states), self.end_mlp(hidden_states))

    @staticmethod
    def grid_targets(span_ids: Tensor) -> Tensor:
        """Build the gold (start, end) grid from per-token span identities.

        Note: span_ids can only encode non-overlapping gold spans (one id per
        token). Overlapping gold annotations would need the DataModule to emit
        span boundaries directly; the model itself supports overlap already.
        """
        B, L = span_ids.shape
        target = span_ids.new_zeros((B, L, L), dtype=torch.float)
        for b in range(B):
            sids = span_ids[b]
            for k in sids.unique().tolist():
                if k == 0:
                    continue
                idx = (sids == k).nonzero(as_tuple=True)[0]
                target[b, idx[0], idx[-1]] = 1.0
        return target

    @staticmethod
    def valid_cell_mask(real_token_mask: Tensor) -> Tensor:
        """Upper-triangle cells where both start and end are real tokens."""
        pair = real_token_mask.unsqueeze(2) & real_token_mask.unsqueeze(1)
        L = real_token_mask.shape[1]
        triu = torch.ones(L, L, dtype=torch.bool, device=real_token_mask.device).triu()
        return pair & triu

    @staticmethod
    def decode_spans_from_grid(
        scores: Tensor,          # (L, L) raw logits
        offset_mapping: Tensor,  # (L, 2)
        threshold: float = 0.5,
        allow_overlap: bool = False,
    ) -> list[tuple[int, int]]:
        """Cells above threshold → char-level spans.

        Independent per-cell thresholding fires on multiple overlapping
        (start, end) candidates for the same true span (e.g. off-by-one
        boundaries), which inflates granularity without changing raw
        precision/recall much. Greedy NMS (highest score first, skip any
        candidate overlapping an already-accepted span) resolves this to one
        span per region, matching the non-overlapping CNC gold format.
        Set allow_overlap=True to keep every above-threshold cell instead —
        the grid supports overlapping/nested spans natively when gold data
        has them.
        """
        offsets = offset_mapping
        real = (offsets[:, 0] != 0) | (offsets[:, 1] != 0)
        valid = (real.unsqueeze(1) & real.unsqueeze(0)).triu()
        probs = torch.sigmoid(scores)
        hits = ((probs >= threshold) & valid).nonzero(as_tuple=False)
        candidates = [
            (probs[i, j].item(), int(offsets[i, 0]), int(offsets[j, 1]))
            for i, j in hits.tolist()
        ]
        if allow_overlap:
            return [(s, e) for _, s, e in candidates]

        candidates.sort(key=lambda c: c[0], reverse=True)
        accepted: list[tuple[int, int]] = []
        for _, s, e in candidates:
            if not any(s < ae and e > as_ for as_, ae in accepted):
                accepted.append((s, e))
        return sorted(accepted)


# ---------------------------------------------------------------------------
# Visualisation
# ---------------------------------------------------------------------------

def _make_grid_heatmap(
    grid: np.ndarray,          # (n, n) sigmoid scores over real tokens
    token_strings: list[str],
    gold_cells: list[tuple[int, int]],  # (start_idx, end_idx) in real-token space
    title: str,
):
    """Plotly heatmap of the span grid; gold (start, end) cells outlined."""
    import plotly.graph_objects as go

    n = len(token_strings)
    indices = list(range(n))
    fig = go.Figure(data=go.Heatmap(
        z=grid, x=indices, y=indices,
        colorscale="Blues", zmin=0, zmax=1,
        colorbar=dict(title="span prob", thickness=18, len=0.9),
        hovertemplate="start: %{y}<br>end: %{x}<br>prob: %{z:.3f}<extra></extra>",
    ))
    for s, e in gold_cells:
        fig.add_shape(
            type="rect", x0=e - 0.5, y0=s - 0.5, x1=e + 0.5, y1=s + 0.5,
            line=dict(color="#d62728", width=2),
            fillcolor="rgba(0,0,0,0)",
        )
    cell_px = max(16, min(24, 1000 // n))
    dim = min(1400, n * cell_px + 260)
    fig.update_layout(
        title=dict(text=title[:110] + ("…" if len(title) > 110 else ""), font=dict(size=12)),
        width=dim, height=dim,
        xaxis=dict(title="end token →", tickmode="array", tickvals=indices,
                   ticktext=token_strings, tickfont=dict(size=11), side="top", tickangle=-45),
        yaxis=dict(title="← start token", tickmode="array", tickvals=indices,
                   ticktext=token_strings, tickfont=dict(size=11), autorange="reversed"),
        margin=dict(l=140, r=100, t=160, b=40),
        plot_bgcolor="white",
    )
    return fig


# ---------------------------------------------------------------------------
# LightningModule
# ---------------------------------------------------------------------------

class BiaffineSpanDetectionModule(LightningModule):
    """Biaffine span-grid causal span detector."""

    def __init__(
        self,
        model_name: str = "roberta-large",
        learning_rate: float = 1e-5,
        backbone_lr: float = 1e-6,
        weight_decay: float = 0.1,
        warmup_steps: int = 200,
        ffn_dim: int = 256,
        dropout: float = 0.2,
        pos_weight_clamp: float = 20.0,
        span_threshold: float = 0.5,
        allow_overlap: bool = False,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()
        # Some checkpoints (e.g. deberta-v3-large) are stored in fp16; force fp32
        # master weights so the manual eval-mode forward in _log_grid_heatmap
        # (outside the trainer's autocast context) doesn't hit a dtype mismatch
        # against the fp32 span_head. Lightning's mixed-precision plugin still
        # autocasts normal train/val steps regardless of this master dtype.
        self.backbone = AutoModel.from_pretrained(model_name, torch_dtype=torch.float32)
        hidden_size = _get_backbone_dims(self.backbone.config)[0]
        self.span_head = BiaffineSpanHead(hidden_size, ffn_dim, dropout)
        self._val_span_records: list[dict] = []
        self._train_viz: dict | None = None
        self._val_viz: dict | None = None

    def forward(self, input_ids: Tensor, attention_mask: Tensor) -> Tensor:
        out = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        return self.span_head(out.last_hidden_state)

    def on_train_epoch_start(self) -> None:
        # HuggingFace sets backbone to eval() after from_pretrained; Lightning may
        # not restore it after mid-epoch validation.
        self.backbone.train()

    def on_fit_start(self) -> None:
        dm = self.trainer.datamodule
        if dm is None or not hasattr(dm, "train_ds") or not hasattr(dm, "val_ds"):
            return
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(self.hparams.model_name)
        max_length = getattr(dm.hparams, "max_length", 128)
        train_rec = _find_viz_example(dm.train_ds.records)
        val_rec = _find_viz_example(dm.val_ds.records)
        if train_rec:
            self._train_viz = _make_viz_item(train_rec, tokenizer, max_length)
        if val_rec:
            self._val_viz = _make_viz_item(val_rec, tokenizer, max_length)

    def _loss(self, scores: Tensor, batch: dict) -> Tensor:
        target = BiaffineSpanHead.grid_targets(batch["span_ids"])
        valid = BiaffineSpanHead.valid_cell_mask(batch["real_token_mask"])
        cells, labels = scores[valid], target[valid]
        n_pos = labels.sum().clamp(min=1.0)
        n_neg = (1.0 - labels).sum().clamp(min=1.0)
        pos_weight = (n_neg / n_pos).clamp(max=self.hparams.pos_weight_clamp)
        return F.binary_cross_entropy_with_logits(cells, labels, pos_weight=pos_weight)

    def training_step(self, batch: dict, batch_idx: int) -> Tensor:
        scores = self(batch["input_ids"], batch["attention_mask"])
        loss = self._loss(scores, batch)
        self.log("train/loss", loss, on_step=True, on_epoch=False, prog_bar=True)
        return loss

    def on_train_epoch_end(self) -> None:
        self._log_grid_heatmap("train_example", self._train_viz)

    def validation_step(self, batch: dict, batch_idx: int) -> None:
        scores = self(batch["input_ids"], batch["attention_mask"])
        self.log("val/loss", self._loss(scores, batch), on_step=False, on_epoch=True)

        for score_mat, sid_seq, off_seq in zip(scores, batch["span_ids"], batch["offset_mapping"]):
            self._val_span_records.append({
                "y_true": AttentionSpanHead.decode_spans_from_ids(sid_seq.cpu(), off_seq.cpu()),
                "y_pred": BiaffineSpanHead.decode_spans_from_grid(
                    score_mat.cpu(), off_seq.cpu(), self.hparams.span_threshold,
                    allow_overlap=self.hparams.allow_overlap,
                ),
            })

    def on_validation_epoch_end(self) -> None:
        # Positive-gold sentences only; empty-gold sentences score 0 under the
        # Touché span metric regardless of the prediction.
        pos_records = [r for r in self._val_span_records if r["y_true"]]
        scores = dataset_span_scores(
            [r["y_true"] for r in pos_records],
            [r["y_pred"] for r in pos_records],
        )
        for k, v in scores.items():
            self.log(f"val/span_{k}", v, prog_bar=(k == "f1_gran"))
        self._val_span_records.clear()
        self._log_grid_heatmap("val_example", self._val_viz)

    def configure_optimizers(self):
        no_decay = {"bias", "LayerNorm.weight", "layer_norm.weight"}

        def _groups(module, lr):
            return [
                {"params": [p for n, p in module.named_parameters()
                            if not any(nd in n for nd in no_decay)],
                 "lr": lr, "weight_decay": self.hparams.weight_decay},
                {"params": [p for n, p in module.named_parameters()
                            if any(nd in n for nd in no_decay)],
                 "lr": lr, "weight_decay": 0.0},
            ]

        grouped = _groups(self.backbone, self.hparams.backbone_lr) + \
                  _groups(self.span_head, self.hparams.learning_rate)
        optimizer = torch.optim.AdamW(grouped)
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=self.hparams.warmup_steps,
            num_training_steps=self.trainer.estimated_stepping_batches,
        )
        return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "interval": "step"}}

    def _log_grid_heatmap(self, tag: str, viz: dict | None) -> None:
        if viz is None:
            return
        logger = self.logger
        if logger is None or not hasattr(logger, "experiment"):
            return
        try:
            import wandb
        except ImportError:
            return

        device = next(self.parameters()).device
        module_states = {m: m.training for m in self.modules()}
        self.eval()
        try:
            with torch.no_grad():
                scores = self(
                    viz["input_ids"].unsqueeze(0).to(device),
                    viz["attention_mask"].unsqueeze(0).to(device),
                )
        finally:
            for m, was_train in module_states.items():
                m.train(was_train)

        grid = torch.sigmoid(scores[0]).cpu().numpy()
        offsets = viz["offset_mapping"].numpy()
        is_real = (offsets[:, 0] != 0) | (offsets[:, 1] != 0)
        real_idx = np.where(is_real)[0]
        grid_real = grid[np.ix_(real_idx, real_idx)]

        # Gold (start, end) cells in real-token index space, from span_ids.
        span_ids = viz["span_ids"]
        gold_cells = []
        for k in sorted(set(span_ids.tolist()) - {0}):
            idxs = np.where(span_ids == k)[0]
            gold_cells.append((int(idxs[0]), int(idxs[-1])))

        fig = _make_grid_heatmap(
            grid_real, viz["token_strings"], gold_cells,
            f"[epoch {self.current_epoch}] {viz['text']}",
        )
        logger.experiment.log(
            {f"span_grid/{tag}": wandb.Plotly(fig)},
            step=self.global_step,
        )
