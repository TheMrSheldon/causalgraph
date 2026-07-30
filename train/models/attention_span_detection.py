"""
Step 2a (alternative) — Attention-based Causal Span Detection.

Instead of a standard BIO token classifier, this head:
  1. Combines the final transformer layer's attention heads via a learned
     linear projection into a single (B, L, L) map.
  2. Symmetrises it: A_sym = (A + A^T) / 2.
  3. Augments each token with its attention-weighted context vector.
  4. Classifies each token from [h_i || ctx_i] with a two-layer MLP.

An auxiliary attention-structure loss (BCE on raw A_sym) rewards high
attention between tokens in the same entity span and low attention otherwise,
encouraging each span to form a strongly-connected subgraph.
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


# ---------------------------------------------------------------------------
# Backbone dimension helper
# ---------------------------------------------------------------------------

def _get_backbone_dims(config) -> tuple[int, int]:
    hidden_size = getattr(config, "hidden_size", None) or config.dim
    num_heads = getattr(config, "num_attention_heads", None) or config.n_heads
    return hidden_size, num_heads


# ---------------------------------------------------------------------------
# Attention span head
# ---------------------------------------------------------------------------

class AttentionSpanHead(nn.Module):
    """Per-token span classifier with a learned single-head attention read-out.

    A fresh Q/K projection over RoBERTa hidden states computes the attention map
    from scratch — no dependency on the backbone's own (diagonal-biased) attention
    heads.  The structure loss directly trains Q and K to produce block-diagonal
    attention for span tokens.
    """

    def __init__(self, hidden_size: int, attn_dim: int = 256) -> None:
        super().__init__()
        self.query = nn.Linear(hidden_size, attn_dim, bias=False)
        self.key   = nn.Linear(hidden_size, attn_dim, bias=False)
        self.scale = attn_dim ** -0.5
        # Input is attn_ctx only — the MLP cannot shortcut through h, so all
        # token-loss gradient flows through softmax(Q·K) back into Q and K.
        self.readout = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.GELU(),
            nn.Linear(hidden_size // 2, 1),
        )

    def forward(
        self,
        hidden_states: Tensor,  # (B, L, H)
    ) -> tuple[Tensor, Tensor]:
        Q = self.query(hidden_states)                              # (B, L, attn_dim)
        K = self.key(hidden_states)                               # (B, L, attn_dim)
        attn_raw = torch.bmm(Q, K.transpose(-1, -2)) * self.scale # (B, L, L)
        attn_sym = (attn_raw + attn_raw.transpose(-1, -2)) / 2
        attn_weights = torch.softmax(attn_sym, dim=-1)
        attn_ctx = torch.bmm(attn_weights, hidden_states)         # (B, L, H)
        logits = self.readout(attn_ctx).squeeze(-1)
        return logits, attn_sym  # (B, L), (B, L, L)

    @staticmethod
    def attention_structure_loss(
        attn_sym: Tensor,         # (B, L, L) raw
        span_ids: Tensor,         # (B, L) long, 0=background, k=k-th span
        real_token_mask: Tensor,  # (B, L) bool, excludes special tokens + padding
    ) -> Tensor:
        """BCE loss on the full target attention matrix.

        Target: 1 for (i,j) pairs in the same span, 0 for everything else
        (cross-span, span-background, background-background).
        All real-token pairs are supervised so background attention is also
        pushed toward zero.
        """
        pair_mask = real_token_mask.unsqueeze(2) & real_token_mask.unsqueeze(1)
        same_span = (span_ids.unsqueeze(2) == span_ids.unsqueeze(1)) & (span_ids.unsqueeze(2) > 0)
        target = same_span.float()
        n_pairs = pair_mask.sum().float()
        if n_pairs == 0:
            return attn_sym.new_tensor(0.0)
        n_pos = (same_span & pair_mask).sum().float().clamp(min=1.0)
        pos_weight = ((n_pairs / n_pos) - 1.0).clamp(1.0, 50.0)
        return F.binary_cross_entropy_with_logits(
            attn_sym[pair_mask], target[pair_mask], pos_weight=pos_weight
        )

    @staticmethod
    def decode_spans(
        logits: Tensor,          # (L,) raw logits
        offset_mapping: Tensor,  # (L, 2)
        threshold: float = 0.5,
    ) -> list[tuple[int, int]]:
        """Group consecutive predicted-event tokens into character-level spans."""
        is_event = (torch.sigmoid(logits) >= threshold).tolist()
        spans: list[tuple[int, int]] = []
        span_start = span_end = None
        for is_ev, (cs, ce) in zip(is_event, offset_mapping.tolist()):
            if cs == 0 and ce == 0:
                if span_start is not None:
                    spans.append((span_start, span_end))
                    span_start = span_end = None
                continue
            if is_ev:
                if span_start is None:
                    span_start = int(cs)
                elif int(cs) > span_end + 1:
                    # Gap > 1 char means a real span boundary (not just a space).
                    # A single-space gap (cs == span_end + 1) keeps words merged
                    # into the same multi-word span.
                    spans.append((span_start, span_end))
                    span_start = int(cs)
                span_end = int(ce)
            else:
                if span_start is not None:
                    spans.append((span_start, span_end))
                    span_start = span_end = None
        if span_start is not None:
            spans.append((span_start, span_end))
        return spans

    @staticmethod
    def decode_spans_from_ids(
        span_ids: Tensor,        # (L,) long, 0=background, k=k-th span (1-indexed)
        offset_mapping: Tensor,  # (L, 2)
    ) -> list[tuple[int, int]]:
        """Reconstruct gold char-level spans from per-token span identity.

        Uses span_ids rather than binary labels so that adjacent gold spans
        (e.g. cause and effect separated by only a space) are kept distinct.
        """
        extents: dict[int, list[int]] = {}
        for (cs, ce), sid in zip(offset_mapping.tolist(), span_ids.tolist()):
            if sid == 0 or (cs == 0 and ce == 0):
                continue
            if sid not in extents:
                extents[sid] = [int(cs), int(ce)]
            else:
                extents[sid][1] = max(extents[sid][1], int(ce))
        return [(v[0], v[1]) for v in sorted(extents.values())]


# ---------------------------------------------------------------------------
# Visualisation helpers
# ---------------------------------------------------------------------------

def _find_viz_example(records: list[dict]) -> dict | None:
    """Pick the record with the most annotated spans (prefers ≥2 spans)."""
    with_spans = [r for r in records if r.get("entity") is not None and len(r["entity"]) > 0]
    if not with_spans:
        return None
    return max(with_spans, key=lambda r: len(r["entity"]))


def _make_viz_item(rec: dict, tokenizer, max_length: int) -> dict:
    """Tokenise a record into the dict expected by _log_attention_heatmap."""
    spans = [(int(s[0]), int(s[1])) for s in rec.get("entity", [])]
    enc = tokenizer(
        rec["text"], max_length=max_length, truncation=True,
        return_offsets_mapping=True, return_tensors="pt",
    )
    input_ids = enc["input_ids"].squeeze(0)
    attention_mask = enc["attention_mask"].squeeze(0)
    offset_mapping = enc["offset_mapping"].squeeze(0)

    is_real = (offset_mapping[:, 0] != 0) | (offset_mapping[:, 1] != 0)
    real_idx = is_real.nonzero(as_tuple=True)[0]
    token_strings = [tokenizer.decode([input_ids[i].item()]) for i in real_idx]

    span_ids = np.zeros(len(real_idx), dtype=int)
    for j, ri in enumerate(real_idx.tolist()):
        cs, ce = offset_mapping[ri].tolist()
        for k, (s, e) in enumerate(spans, start=1):
            if cs < e and ce > s:
                span_ids[j] = k
                break

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "offset_mapping": offset_mapping,
        "token_strings": token_strings,
        "span_ids": span_ids,
        "text": rec["text"],
    }


def _make_attention_heatmap(
    attn_matrix: np.ndarray,
    token_strings: list[str],
    span_ids: np.ndarray,
    title: str,
):
    """Return a Plotly Figure of the symmetrised attention heatmap for W&B."""
    import plotly.graph_objects as go

    n = len(token_strings)
    _COLORS = ["#d62728", "#2ca02c", "#9467bd", "#ff7f0e", "#1f77b4", "#8c564b"]
    unique_spans = sorted(set(span_ids.tolist()) - {0})
    span_color = {sid: _COLORS[i % len(_COLORS)] for i, sid in enumerate(unique_spans)}

    def _label(i, tok):
        sid = int(span_ids[i])
        return f'<b><span style="color:{span_color[sid]}">{tok}</span></b>' if sid > 0 else tok

    tick_labels = [_label(i, t) for i, t in enumerate(token_strings)]
    indices = list(range(n))

    fig = go.Figure(data=go.Heatmap(
        z=attn_matrix, x=indices, y=indices,
        colorscale="Blues", zmin=0, zmax=1,
        colorbar=dict(title="attn weight", thickness=18, len=0.9),
        hovertemplate="from: %{y}<br>to: %{x}<br>weight: %{z:.3f}<extra></extra>",
    ))

    for sid in unique_spans:
        idxs = np.where(span_ids == sid)[0]
        if len(idxs) == 0:
            continue
        s, e = int(idxs[0]), int(idxs[-1])
        fig.add_shape(
            type="rect", x0=s - 0.5, y0=s - 0.5, x1=e + 0.5, y1=e + 0.5,
            line=dict(color=span_color[sid], width=2, dash="dash"),
            fillcolor="rgba(0,0,0,0)",
        )

    cell_px = max(16, min(24, 1000 // n))
    dim = min(1400, n * cell_px + 260)
    fig.update_layout(
        title=dict(text=title[:110] + ("…" if len(title) > 110 else ""), font=dict(size=12)),
        width=dim, height=dim,
        xaxis=dict(title="attention to →", tickmode="array", tickvals=indices,
                   ticktext=tick_labels, tickfont=dict(size=11), side="top", tickangle=-45),
        yaxis=dict(title="← attention from", tickmode="array", tickvals=indices,
                   ticktext=tick_labels, tickfont=dict(size=11), autorange="reversed"),
        margin=dict(l=140, r=100, t=160, b=40),
        plot_bgcolor="white",
    )
    return fig


# ---------------------------------------------------------------------------
# LightningModule
# ---------------------------------------------------------------------------

class AttentionSpanDetectionModule(LightningModule):
    """Attention-based causal span detector; alternative to the BIO token classifier."""

    def __init__(
        self,
        model_name: str = "roberta-large",
        learning_rate: float = 2e-5,
        backbone_lr: float = 1e-6,
        weight_decay: float = 0.1,
        warmup_steps: int = 200,
        attn_structure_weight: float = 0.5,
        span_threshold: float = 0.5,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()
        # Force fp32 master weights (some checkpoints, e.g. deberta-v3-large,
        # ship fp16) so manual eval-mode forwards outside the trainer's autocast
        # context don't hit a dtype mismatch against the fp32 span_head.
        self.backbone = AutoModel.from_pretrained(model_name, torch_dtype=torch.float32)
        hidden_size = _get_backbone_dims(self.backbone.config)[0]
        self.span_head = AttentionSpanHead(hidden_size)
        self._val_span_records: list[dict] = []
        self._train_viz: dict | None = None
        self._val_viz: dict | None = None

    def forward(self, input_ids: Tensor, attention_mask: Tensor) -> tuple[Tensor, Tensor]:
        out = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        return self.span_head(out.last_hidden_state)

    def on_train_epoch_start(self) -> None:
        # HuggingFace sets backbone to eval() after from_pretrained; Lightning may
        # not restore it after mid-epoch validation. Explicitly put backbone in
        # train mode so dropout is active during every training epoch.
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

    def training_step(self, batch: dict, batch_idx: int) -> Tensor:
        logits, attn_sym = self(batch["input_ids"], batch["attention_mask"])
        mask = batch["real_token_mask"]
        labels = batch["token_labels"][mask]
        n_pos = labels.sum().clamp(min=1.0)
        n_neg = (1.0 - labels).sum().clamp(min=1.0)
        pos_weight = (n_neg / n_pos).clamp(max=20.0)
        token_loss = F.binary_cross_entropy_with_logits(
            logits[mask], labels, pos_weight=pos_weight
        )
        attn_loss = AttentionSpanHead.attention_structure_loss(
            attn_sym, batch["span_ids"], batch["real_token_mask"]
        )
        loss = token_loss + self.hparams.attn_structure_weight * attn_loss
        self.log_dict({
            "train/loss": loss,
            "train/token_loss": token_loss,
            "train/attn_loss": attn_loss,
        }, on_step=True, on_epoch=False, prog_bar=True)
        return loss

    def on_train_epoch_end(self) -> None:
        self._log_attention_heatmap("train_example", self._train_viz)

    def validation_step(self, batch: dict, batch_idx: int) -> None:
        logits, _ = self(batch["input_ids"], batch["attention_mask"])
        mask = batch["real_token_mask"]
        token_loss = F.binary_cross_entropy_with_logits(
            logits[mask], batch["token_labels"][mask]
        )
        self.log("val/loss", token_loss, on_step=False, on_epoch=True)

        for logit_seq, sid_seq, off_seq in zip(logits, batch["span_ids"], batch["offset_mapping"]):
            self._val_span_records.append({
                "y_true": AttentionSpanHead.decode_spans_from_ids(sid_seq.cpu(), off_seq.cpu()),
                "y_pred": AttentionSpanHead.decode_spans(logit_seq.cpu(), off_seq.cpu(), self.hparams.span_threshold),
            })

    def on_validation_epoch_end(self) -> None:
        # Evaluate only on sentences that have at least one gold span; empty-gold
        # sentences produce f1=0 regardless of prediction under the Touché metric.
        pos_records = [r for r in self._val_span_records if r["y_true"]]
        scores = dataset_span_scores(
            [r["y_true"] for r in pos_records],
            [r["y_pred"] for r in pos_records],
        )
        for k, v in scores.items():
            self.log(f"val/span_{k}", v, prog_bar=(k == "f1_gran"))
        self._val_span_records.clear()
        self._log_attention_heatmap("val_example", self._val_viz)

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

    def _log_attention_heatmap(self, tag: str, viz: dict | None) -> None:
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
        # Snapshot per-module train/eval state and restore after (self.training
        # is False during on_validation_epoch_end, so `if was_training` alone
        # would silently leave the backbone in eval for the next training epoch).
        module_states = {m: m.training for m in self.modules()}
        self.eval()
        try:
            with torch.no_grad():
                _, attn_sym_raw = self(
                    viz["input_ids"].unsqueeze(0).to(device),
                    viz["attention_mask"].unsqueeze(0).to(device),
                )
        finally:
            for m, was_train in module_states.items():
                m.train(was_train)

        attn = torch.sigmoid(attn_sym_raw[0]).cpu().numpy()
        offsets = viz["offset_mapping"].numpy()
        is_real = (offsets[:, 0] != 0) | (offsets[:, 1] != 0)
        real_idx = np.where(is_real)[0]
        attn_real = attn[np.ix_(real_idx, real_idx)]

        fig = _make_attention_heatmap(
            attn_real, viz["token_strings"], viz["span_ids"],
            f"[epoch {self.current_epoch}] {viz['text']}",
        )
        logger.experiment.log(
            {f"attention/{tag}": wandb.Plotly(fig)},
            step=self.global_step,
        )
