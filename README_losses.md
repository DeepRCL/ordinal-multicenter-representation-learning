# `losses.py` — Ordinal Loss Library

Standalone loss functions for ordinal classification tasks.  
Designed for prostate cancer ISUP grading but usable in any ordered multi-class setting.

---

## Loss Classes

| Class | Description |
|-------|-------------|
| `EMDOrdinalLoss` | Earth Mover's Distance (squared CDF) — ordinal-aware replacement for cross-entropy |
| `OrdinalNegWeightedSupConLoss` | Supervised contrastive loss with ordinal negative weighting and optional domain awareness |
| `SCLEMDLoss` | Composite: `emd_weight * EMD + supcon_weight * SupCon` |
| `TrainV3StyleLoss` | Full wrapper matching `train_v3.py`: main loss + aux contrastive + curriculum + adaptive weight scheduling |

---

## Quick Start

### Mode 1: Replace Cross-Entropy with EMD

```python
from losses import make_emd_only_loss

emd_loss = make_emd_only_loss(num_classes=6)  # set your class count
loss = emd_loss(logits, labels)               # drop-in CE replacement
```

### Mode 2: Full Ordinal Framework (recommended)

Matches the best configuration from `train_v3.py`:
- Main loss: EMD
- Auxiliary: Ordinal Supervised Contrastive
- Curriculum class weighting
- Scheduled + ratio-adaptive auxiliary weight
- Optional domain-aware contrastive

```python
from losses import make_full_ordinal_loss

loss_fn = make_full_ordinal_loss(
    num_classes=6,
    use_domains=True,     # True if you have multi-center / domain labels
    domain_weight=1.0,    # boost factor for cross-domain negatives
    use_curriculum=True,
    curr_start=3,         # epoch at which curriculum begins
)
```

**Training step:**

```python
logits, embeddings = model(x)  # logits: [B, C], embeddings: [B, D]

out = loss_fn(
    logits=logits,
    embeddings=embeddings,
    labels=labels,
    epoch=epoch,
    domain_labels=domain_labels,  # omit if use_domains=False
)

loss = out["total"]
loss.backward()
optimizer.step()
```

**Output dict keys:**

| Key | Description |
|-----|-------------|
| `out["total"]` | Combined loss (backpropagate this) |
| `out["main"]` | EMD component |
| `out["aux"]` | Contrastive component |
| `out["aux_weight"]` | Effective auxiliary weight for this step |
| `out["class_weights"]` | Curriculum class weights (empty tensor if curriculum is off) |

---

## Single-Domain vs Multi-Domain

```python
# Single domain (no domain labels available)
loss_fn = make_full_ordinal_loss(num_classes=6, use_domains=False)
out = loss_fn(logits, embeddings, labels, epoch=epoch)

# Multi-domain (e.g. Karolinska=0, Radboud=1)
loss_fn = make_full_ordinal_loss(num_classes=6, use_domains=True, domain_weight=1.0)
out = loss_fn(logits, embeddings, labels, epoch=epoch, domain_labels=domain_labels)
```

---

## Factory Helper

```python
from losses import build_loss

loss = build_loss("full_ordinal", num_classes=6, use_domains=True)
# Supported names: "emd", "emd_only", "ordinal_supcon", "scl_emd", "train_v3_style", "full_ordinal"
```

---

## Key Hyperparameters

All can be passed directly to `make_full_ordinal_loss(...)` or `TrainV3StyleLoss(...)`:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `curr_start` | `3` | Epoch curriculum weighting begins |
| `curr_decay` | `0.1` | Exponential decay rate for easy-class weights |
| `curr_min_weight` | `0.1` | Minimum class weight floor |
| `temperature` | `0.07` | SupCon temperature |
| `neg_strength` | `1.0` | Ordinal distance negative weight strength |
| `domain_weight` | `0.5` | Cross-domain negative boost factor |
| `aux_weight_start` | `0.5` | Initial scheduled auxiliary weight |
| `aux_weight_end` | `0.05` | Final scheduled auxiliary weight |
| `aux_weight_decay_start` | `1` | Epoch decay scheduling begins |
| `aux_weight_decay_end` | `15` | Epoch decay scheduling ends |
| `aux_weight_schedule` | `"linear"` | `"linear"` / `"cosine"` / `"exp"` |
| `aux_ratio_adapt` | `True` | Auto-scale aux weight by main/aux loss ratio |
| `aux_target_ratio` | `0.05` | Target aux / total loss ratio for auto-scaling |

**Example with custom hyperparameters:**

```python
loss_fn = make_full_ordinal_loss(
    num_classes=6,
    use_domains=True,
    domain_weight=1.0,
    curr_start=3,
    curr_decay=0.1,
    temperature=0.07,
    neg_strength=1.0,
    aux_weight_start=0.5,
    aux_weight_end=0.5,          # keep aux weight constant
    aux_weight_decay_start=999,  # effectively disable decay
    aux_weight_decay_end=1000,
)
```

This matches the best configuration found in experiments (see `commands.txt`).
