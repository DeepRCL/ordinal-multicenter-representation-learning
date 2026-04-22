# Domain-Aware Ordinal Learning for Prostate Cancer Grading

Weakly supervised MIL framework for ISUP grading of prostate biopsy WSIs (PANDA dataset).  
Jointly addresses two challenges: **ordinal classification** and **multi-center domain shift**.

---

## Method Overview

Built on **Attention-Based Multiple Instance Learning (ABMIL)** with frozen **UNI2** patch features (1536-D), the framework adds three targeted objectives on top of a standard cross-entropy baseline:

| Component | Purpose |
|-----------|---------|
| **Earth Mover's Distance (EMD) loss** | Penalizes misclassifications proportionally to ordinal distance between ISUP grades |
| **Ordinal Supervised Contrastive Learning** | Pulls same-grade slides together; cross-center same-grade pairs are up-weighted to enforce domain invariance |
| **Curriculum Learning** | Progressively down-weights easy benign slides (ISUP 0) to redirect capacity toward harder grades |

---

## Repository Structure

```
histopathology/
├── abmil.py               # ABMIL model (gated attention, projection head)
├── train.py               # Baseline training: CE / Focal / Ordinal-CE
├── train_v3.py            # Main training script (EMD + SupCon + Domain-Aware + Curriculum)
├── losses.py              # All loss classes (EMD, OrdinalSupCon, TrainV3StyleLoss, ...)
├── data.py                # PandaPatchBagDataset, collate, make_loaders_from_dir
├── extract_embeddings.py  # Extract and save slide embeddings to .npz
├── linear_probe.py        # Cross-center linear probing evaluation
├── plot_umap.py           # UMAP visualizations (ISUP-colored, provider-colored)
├── panda_split/           # train.csv / val.csv / test.csv
├── environment.yml        # Conda environment (Python 3.12, PyTorch 2.8, CUDA 12.1)
└── README_losses.md       # Using losses.py as a standalone library
```

---

## Installation

```bash
git clone <repository_url>
cd histopathology
conda env create -f environment.yml
conda activate his
```

> Targets **CUDA 12.1**. Edit the `pytorch-cuda` line in `environment.yml` for other CUDA versions.

---

## Dataset

1. Download the [PANDA dataset](https://www.kaggle.com/c/prostate-cancer-grade-assessment) and extract UNI2 patch features (`.h5`, 1536-D per patch).
2. Place feature files in a root directory (default: `/data/project/users/tarek/panda_embeddings`).
3. Provide split CSVs under `panda_split/` with columns: `image_id`, `isup_grade`, `data_provider`.

---

## Training

### 1. CE Baseline (`train.py`)

```bash
python train.py \
    --loss-type cross_entropy \
    --epochs 40 \
    --lr 2e-4 \
    --weight-decay 1e-4 \
    --batch-size 8 \
    --wandb-project "panda-baseline" \
    --wandb-run-name "ce_baseline" \
    --out output/baseline/best_ce.pth
```

Other `--loss-type` options: `poisson_focal`, `ordinal_ce`, `focal_v2`.

### 2. Proposed Method — EMD + SupCon + Domain-Aware + Curriculum (`train_v3.py`)

```bash
python train_v3.py \
    --main-loss emd \
    --aux-loss contrastive \
    --embedding-size 768 \
    --lr 8e-4 \
    --weight-decay 1e-4 \
    --batch-size 16 \
    --num-workers 2 \
    --domain-aware \
    --domain-weight 1.0 \
    --use-curriculum \
    --curr-start 3 \
    --temp 0.07 \
    --contrastive-inv-freq \
    --aux-weight-start 0.5 \
    --aux-weight-end 0.5 \
    --aux-weight-decay-start 999 \
    --aux-weight-decay-end 1000 \
    --wandb-project "panda-v3" \
    --out output_v3/supcon-emd-domainaware-emb768/
```

### Ablation Commands

<details>
<summary>Click to expand all ablation commands</summary>

**Only EMD (no SupCon, no curriculum)**
```bash
python train_v3.py \
    --main-loss emd --aux-loss none \
    --embedding-size 768 --lr 8e-4 --weight-decay 1e-4 \
    --batch-size 16 --num-workers 2 \
    --wandb-project "panda-ablation" \
    --out output_v3/ablation/emd_only
```

**Only Focal Loss**
```bash
python train_v3.py \
    --main-loss focal --aux-loss none \
    --embedding-size 768 --lr 8e-4 --weight-decay 1e-4 \
    --batch-size 16 --num-workers 2 \
    --wandb-project "panda-ablation" \
    --out output_v3/ablation/focal_only
```

**CE + SupCon (no domain, no curriculum)**
```bash
python train_v3.py \
    --main-loss ce --aux-loss contrastive \
    --embedding-size 768 --lr 8e-4 --weight-decay 1e-4 \
    --batch-size 16 --num-workers 2 \
    --temp 0.07 --contrastive-inv-freq \
    --aux-weight-start 0.5 --aux-weight-end 0.5 \
    --aux-weight-decay-start 999 --aux-weight-decay-end 1000 \
    --wandb-project "panda-ablation" \
    --out output_v3/ablation/ce_supcon_nodomain_nocurr
```

**EMD + SupCon (no domain, no curriculum)**
```bash
python train_v3.py \
    --main-loss emd --aux-loss contrastive \
    --embedding-size 768 --lr 8e-4 --weight-decay 1e-4 \
    --batch-size 16 --num-workers 2 \
    --temp 0.07 --contrastive-inv-freq \
    --aux-weight-start 0.5 --aux-weight-end 0.5 \
    --aux-weight-decay-start 999 --aux-weight-decay-end 1000 \
    --wandb-project "panda-ablation" \
    --out output_v3/ablation/emd_supcon_nodomain_nocurr
```

**EMD + SupCon + Domain-Aware (no curriculum)**
```bash
python train_v3.py \
    --main-loss emd --aux-loss contrastive \
    --embedding-size 768 --lr 8e-4 --weight-decay 1e-4 \
    --batch-size 16 --num-workers 2 \
    --domain-aware --domain-weight 1.0 \
    --temp 0.07 --contrastive-inv-freq \
    --aux-weight-start 0.5 --aux-weight-end 0.5 \
    --aux-weight-decay-start 999 --aux-weight-decay-end 1000 \
    --wandb-project "panda-ablation" \
    --out output_v3/ablation/emd_supcon_domain_nocurr
```

**EMD + SupCon + Curriculum (no domain-aware)**
```bash
python train_v3.py \
    --main-loss emd --aux-loss contrastive \
    --embedding-size 768 --lr 8e-4 --weight-decay 1e-4 \
    --batch-size 16 --num-workers 2 \
    --use-curriculum --curr-start 3 \
    --temp 0.07 --contrastive-inv-freq \
    --aux-weight-start 0.5 --aux-weight-end 0.5 \
    --aux-weight-decay-start 999 --aux-weight-decay-end 1000 \
    --wandb-project "panda-ablation" \
    --out output_v3/ablation/emd_supcon_nodomain_curr
```

</details>

---

## Evaluation

### UMAP Visualization

```bash
# CE Baseline
python plot_umap.py \
    --ckpt output_v3/baseline/best_ce.pth \
    --split test \
    --embedding-size 512 \
    --out baseline_ce/test/

# Proposed Model
python plot_umap.py \
    --ckpt output_v3/supcon-emd-domainaware-emb768/best_auroc_model_emd_contrastive_dom1.0_curr_emb768.pth \
    --split test \
    --embedding-size 768 \
    --out umap_plots_v3/proposed/test/
```

Generates three plots: ISUP-colored, provider-colored, and combined.

### Linear Probing (cross-center generalization)

```bash
python linear_probe.py \
    --model-a output_v3/supcon-emd-domainaware-emb768/best_auroc_model_emd_contrastive_dom1.0_curr_emb768.pth \
    --name-a  "Full_Model_EMD+SupCon+Domain+Curr" \
    --model-b output_v3/baseline/best_ce_768.pth \
    --name-b  "Baseline_CE" \
    --out linear_probe_results/main_vs_ce/
```

Trains a frozen linear probe on one center (Karolinska or Radboud) and evaluates on the other.

---

## Results

### Ablation Study (PANDA test set, embedding size 768)

| Experiment | Main Loss | SupCon | Domain-Aware | Curriculum | Accuracy | F1-Macro | QWK |
|---|---|:---:|:---:|:---:|:---:|:---:|:---:|
| CE Baseline | CE | | | | 0.8239 | 0.7837 | 0.9559 |
| Only Focal | Focal | | | | 0.8658 | 0.8353 | 0.9587 |
| Only EMD | EMD | | | | **0.8648** | **0.8352** | **0.9637** |
| CE + SupCon | CE | ✓ | | | 0.8595 | 0.8248 | 0.9628 |
| EMD + SupCon | EMD | ✓ | | | 0.8375 | 0.7975 | 0.9581 |
| EMD + SupCon + Curr | EMD | ✓ | | ✓ | 0.8291 | 0.7914 | 0.9526 |
| EMD + SupCon + Dom | EMD | ✓ | ✓ | | 0.8428 | 0.8048 | 0.9574 |
| **Proposed Model** | EMD | ✓ | ✓ | ✓ | 0.8417 | 0.8086 | 0.9581 |

### Cross-Center Linear Probing

| Model | Probe Task | Accuracy | QWK |
|---|---|:---:|:---:|
| CE Baseline | Karolinska → Radboud | 0.6433 | 0.8991 |
| CE Baseline | Radboud → Karolinska | 0.8109 | 0.9159 |
| **Proposed Model** | Karolinska → Radboud | **0.8228** | **0.9303** |
| **Proposed Model** | Radboud → Karolinska | **0.8149** | **0.9385** |

The proposed model improves cross-center accuracy by ~18% (K→R) compared to the CE baseline, confirming that the domain-aware contrastive objective removes center-specific fingerprints from the feature space.

---

## Key Hyperparameters (`train_v3.py`)

| Argument | Default | Description |
|----------|---------|-------------|
| `--main-loss` | `ce` | `ce` / `emd` / `focal` / `ord_contrastive` |
| `--aux-loss` | `none` | `none` / `contrastive` |
| `--embedding-size` | `512` | ABMIL projection dimension |
| `--domain-aware` | off | Enable cross-domain negative boosting in SupCon |
| `--domain-weight` | `1.0` | Boost factor for cross-domain negatives |
| `--use-curriculum` | off | Enable class-weight curriculum |
| `--curr-start` | `10` | Epoch curriculum begins |
| `--temp` | `0.07` | SupCon temperature |
| `--neg-strength` | `1.0` | Ordinal distance negative weighting strength |
| `--contrastive-inv-freq` | off | Inverse-frequency sample weighting in SupCon |
| `--aux-weight-start` | `0.5` | Initial auxiliary loss weight |
| `--aux-weight-end` | `0.05` | Final auxiliary loss weight |
| `--batch-size` | `512` | Slides per batch |
| `--epochs` | `40` | Total training epochs |

---

## Citation

If you find this work useful, please cite:

```
Abootorabi, M. M., & Golshahian, S. (2026). Domain-Aware Ordinal Learning for Prostate
Cancer Grading from Histopathology Images. University of British Columbia, BMEG 591T.
```
