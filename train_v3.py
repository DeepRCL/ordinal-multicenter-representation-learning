
import argparse, os, math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn.metrics import accuracy_score, f1_score, cohen_kappa_score, roc_auc_score, balanced_accuracy_score
from tqdm.auto import tqdm
from pathlib import Path

from abmil import ABMILISUP
from data import make_loaders_from_dir

# -------------------- Utils --------------------
def set_seed(seed: int = 1337):
    import random
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

def one_hot(labels: torch.Tensor, num_classes: int) -> torch.Tensor:
    return F.one_hot(labels.long(), num_classes=num_classes).to(labels.device).float()

# -------------------- Weight Scheduling (Curriculum & Aux) --------------------

def get_curriculum_class_weights(epoch: int, num_classes: int = 6, 
                                start_epoch: int = 10, 
                                decay_rate: float = 0.1,
                                min_weight: float = 0.1) -> torch.Tensor:
    """
    Generate curriculum learning class weights that reduce influence of easier classes over time.
    
    Hierarchy (from easiest to hardest):
    - ISUP 0 (benign): highest decay
    - ISUP 1, 2: medium decay  
    - ISUP 3, 4, 5: lowest decay (focus on these)
    """
    if epoch < start_epoch:
        # Before curriculum starts, use uniform weights
        return torch.ones(num_classes)
    
    # Define class difficulty hierarchy (0=easiest, 5=hardest)
    # Calculate decay factor based on epoch
    epochs_since_start = epoch - start_epoch
    decay_factor = torch.exp(torch.tensor(-decay_rate * epochs_since_start)).item()    
    
    # Base weights: easier classes get lower weights
    base_weights = torch.tensor([0.3, 0.8, 0.8, 1.0, 1.0, 1.0])  # 0, 1-2, 3-5
    
    # Apply decay: easier classes decay faster
    decay_multipliers = torch.tensor([1.0, 0.5, 0.5, 0.3, 0.3, 0.3])  # 0, 1-2, 3-5
    
    # Calculate final weights
    weights = base_weights * (1.0 - decay_multipliers * (1.0 - decay_factor))
    
    # Ensure minimum weight
    weights = torch.clamp(weights, min=min_weight)
    
    return weights

def get_scheduled_metric_weight(epoch: int, args) -> float:
    """
    Compute scheduled metric weight based on epoch using selected schedule.
    """
    w_start = args.aux_weight_start
    w_end = args.aux_weight_end
    start_ep = args.aux_weight_decay_start
    end_ep = args.aux_weight_decay_end
    schedule = args.aux_weight_schedule
    
    if epoch <= start_ep:
        return float(w_start)
    if epoch >= end_ep:
        return float(w_end)
        
    t = (epoch - start_ep) / max(1, (end_ep - start_ep))
    
    if schedule == "linear":
        weight = w_start + (w_end - w_start) * t
    elif schedule == "cosine":
        weight = w_end + (w_start - w_end) * 0.5 * (1.0 + math.cos(math.pi * t))
    elif schedule == "exp":
        safe = max(1e-8, float(w_start))
        ratio = float(w_end) / safe
        weight = safe * (ratio ** t)
    else:
        weight = w_start + (w_end - w_start) * t
        
    return float(weight)

def adjust_metric_weight_by_ratio(base_weight: float, main_loss: torch.Tensor, aux_loss: torch.Tensor, args, state: dict) -> float:
    """
    Optionally adjust metric weight so the metric term approximates a target fraction of the total loss.
    """
    if not args.aux_ratio_adapt:
        return float(base_weight)
        
    target_ratio = args.aux_target_ratio
    beta = 0.9 # EMA factor
    
    m = float(aux_loss.item())
    c = float(main_loss.item())
    
    # Avoid division by zero
    if m + c < 1e-9:
        return float(base_weight)
        
    current_ratio = m / (m + c)
    
    prev = state.get("ratio_ema")
    if prev is None:
        ratio_ema = current_ratio
    else:
        ratio_ema = beta * prev + (1.0 - beta) * current_ratio
    state["ratio_ema"] = ratio_ema
    
    # Scale base weight to move towards target ratio
    # If ratio_ema is too high (aux loss is dominating), factor will be < 1, reducing weight
    factor = target_ratio / max(1e-8, ratio_ema)
    adjusted = base_weight * factor
    
    return float(min(max(adjusted, 0.0), 10.0)) # Clamp to reasonable range

# -------------------- Loss Functions --------------------

class FocalLoss(torch.nn.Module):
    """Standard Focal Loss for multi-class classification."""
    def __init__(self, gamma: float = 2.0, reduction: str = 'mean'):
        super().__init__()
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, target: torch.Tensor, weight: torch.Tensor = None) -> torch.Tensor:
        ce_loss = F.cross_entropy(logits, target, weight=weight, reduction='none')
        pt = torch.exp(-ce_loss)
        focal = (1 - pt) ** self.gamma * ce_loss
        if self.reduction == 'mean':
            return focal.mean()
        return focal.sum()

_focal_loss_fn = FocalLoss(gamma=2.0)

def emd_ordinal_loss(logits: torch.Tensor, target: torch.Tensor, weight: torch.Tensor = None) -> torch.Tensor:
    """Earth Mover's Distance (squared) between predicted and target class distributions.
    Uses cumulative distribution functions to capture class order.
    Supports class weighting.
    """
    num_classes = logits.size(1)
    pred_prob = F.softmax(logits, dim=1)
    target_dist = one_hot(target, num_classes)
    pred_cdf = pred_prob.cumsum(dim=1)
    target_cdf = target_dist.cumsum(dim=1)
    # Squared Euclidean distance between CDFs (EMD for 1D ordinal)
    emd_sq = (pred_cdf - target_cdf).pow(2).sum(dim=1)
    
    if weight is not None:
        # weight: [num_classes]
        # We need to select weights for the target classes in the batch
        sample_weights = weight[target]
        emd_sq = emd_sq * sample_weights
        
    return emd_sq.mean()

def ordinal_neg_weighted_supcon(
    embeddings: torch.Tensor,
    labels: torch.Tensor,
    domain_labels: torch.Tensor = None,
    temperature: float = 0.07,
    neg_strength: float = 1.0,
    domain_aware: bool = False,
    domain_weight: float = 1.0,
    sample_weights: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    Ordinal-aware SupCon with categorical positives and distance-weighted negatives.
    
    Positives: Same label only.
    Negatives: Weights increase with ordinal distance |yi - yj|.
    
    If domain_aware=True, cross-domain negatives are boosted by (1 + domain_weight).
    """
    device = embeddings.device
    labels = labels.view(-1).to(device)
    
    # Normalize embeddings
    z = F.normalize(embeddings, dim=1)
    N = labels.size(0)
    
    # Similarity matrix
    sim = torch.matmul(z, z.t()) / max(temperature, 1e-6)
    
    # Masks
    logits_mask = torch.ones((N, N), device=device, dtype=torch.bool)
    logits_mask.fill_diagonal_(False)
    
    labels_equal = labels.unsqueeze(0) == labels.unsqueeze(1)
    pos_mask = labels_equal & logits_mask
    neg_mask = (~labels_equal) & logits_mask
    
    # Ordinal distance weights for negatives
    yi = labels.unsqueeze(1).float()
    yj = labels.unsqueeze(0).float()
    dist = torch.abs(yi - yj)
    maxd = 5.0 # Max distance for ISUP (0-5)
    d_norm = (dist / maxd).clamp(min=0.0, max=1.0)
    
    # Negative weights (Linear mode default)
    w_neg = 1.0 + float(neg_strength) * d_norm
    w_neg = w_neg * neg_mask.float()
    
    # Domain awareness: track cross-domain pairs
    cross_domain = None
    if domain_aware and domain_labels is not None:
        domain_labels = domain_labels.view(-1).to(device)
        cross_domain = (domain_labels.unsqueeze(0) != domain_labels.unsqueeze(1)).float()
        # Boost cross-domain NEGATIVES (push different-class cross-domain pairs further apart)
        w_neg = w_neg * (1.0 + float(domain_weight) * cross_domain)
        
    # Denominator weights: positives = 1.0, negatives = w_neg
    denom_weights = torch.ones_like(sim, device=device)
    denom_weights = denom_weights * pos_mask.float() + w_neg
    
    # Numerical stability and log-prob
    sim_max, _ = sim.max(dim=1, keepdim=True)
    sim = sim - sim_max.detach()
    
    exp_sim = torch.exp(sim) * logits_mask.float() * denom_weights
    denom = exp_sim.sum(dim=1, keepdim=True) + 1e-8
    log_prob = sim - torch.log(denom)
    
    # Positive aggregation:
    # If domain_aware, also UP-WEIGHT cross-domain same-class positives
    # (pulls Karolinska + Radboud samples of same ISUP together → centers MIX).
    # This is the key mechanism from the original best model.
    if cross_domain is not None:
        cross_domain_pos = pos_mask.float() * cross_domain  # same class, different domain
        pos_weights = pos_mask.float() + float(domain_weight) * cross_domain_pos
        pos_weight_sum = pos_weights.sum(dim=1).clamp(min=1e-8)
        sum_pos = (log_prob * pos_weights).sum(dim=1)
        mean_log_prob_pos = sum_pos / pos_weight_sum
    else:
        pos_count = pos_mask.sum(dim=1).clamp(min=1)
        sum_pos = (log_prob * pos_mask.float()).sum(dim=1)
        mean_log_prob_pos = sum_pos / pos_count.float()
    
    # Per-anchor loss
    loss_vec = -mean_log_prob_pos  # [N]
    if sample_weights is not None:
        sw = sample_weights.view(-1).to(device).float()
        # Normalize weights to keep overall scale stable
        sw = sw / (sw.mean() + 1e-8)
        loss = (loss_vec * sw).mean()
    else:
        loss = loss_vec.mean()
    return loss

# -------------------- Training & Evaluation --------------------

@torch.no_grad()
def evaluate(model, loader, device, args=None) -> dict:
    model.eval()
    y_true_list, y_pred_list = [], []
    y_prob_list = []
    
    for batch in tqdm(loader, desc="Eval", leave=False):
        X, y, mask, domain_labels, metas = batch
        X, y, mask = X.to(device), y.to(device), mask.to(device)
        
        logits, _, _ = model(X, mask)
        probs = F.softmax(logits, dim=1)
        preds = logits.argmax(dim=1)
        
        y_true_list.append(y.cpu())
        y_pred_list.append(preds.cpu())
        y_prob_list.append(probs.cpu())
        
    y_true = torch.cat(y_true_list).numpy()
    y_pred = torch.cat(y_pred_list).numpy()
    y_prob = torch.cat(y_prob_list).numpy()
    
    metrics = {
        "acc": accuracy_score(y_true, y_pred),
        "bal_acc": balanced_accuracy_score(y_true, y_pred),
        "f1_macro": f1_score(y_true, y_pred, average="macro"),
        "qwk": cohen_kappa_score(y_true, y_pred, weights="quadratic"),
    }
    
    # AUROC (One-vs-Rest)
    try:
        metrics["auroc"] = roc_auc_score(y_true, y_prob, multi_class="ovr", labels=np.arange(6))
    except Exception as e:
        print(f"Warning: AUROC calculation failed: {e}")
        metrics["auroc"] = 0.0
        
    return metrics


@torch.no_grad()
def compute_eval_loss(model, loader, device, epoch: int, args) -> float:
    """
    Compute evaluation loss (using the main loss only) on a given loader.
    Mirrors the main-loss branch of training, without any auxiliary term.
    """
    model.eval()
    total_loss = 0.0
    total_count = 0

    # Curriculum weights (if enabled)
    if args.use_curriculum:
        class_weights = get_curriculum_class_weights(
            epoch,
            start_epoch=args.curr_start,
            decay_rate=args.curr_decay,
            min_weight=args.curr_min_weight,
        ).to(device)
    else:
        class_weights = None

    for batch in loader:
        X, y, mask, domain_labels, metas = batch
        X = X.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        mask = mask.to(device, non_blocking=True)
        domain_labels = domain_labels.to(device, non_blocking=True)

        logits, _, slide_embedding = model(X, mask)

        if args.main_loss == "ce":
            loss = F.cross_entropy(logits, y, weight=class_weights)
        elif args.main_loss == "focal":
            loss = _focal_loss_fn(logits, y, weight=class_weights)
        elif args.main_loss == "emd":
            loss = emd_ordinal_loss(logits, y, weight=class_weights)
        elif args.main_loss == "ord_contrastive":
            # Classification-aware weighting for pure contrastive loss
            probs = F.softmax(logits, dim=1)
            pt = probs.gather(1, y.view(-1, 1)).squeeze(1)  # p(true class)
            sw = 1.0 + (1.0 - pt)  # harder examples get larger weight
            if args.use_curriculum and class_weights is not None:
                sw = sw * class_weights[y]
            loss = ordinal_neg_weighted_supcon(
                slide_embedding,
                y,
                domain_labels=domain_labels if args.domain_aware else None,
                temperature=args.temp,
                neg_strength=args.neg_strength,
                domain_aware=args.domain_aware,
                domain_weight=args.domain_weight,
                sample_weights=sw,
            )
        else:
            raise ValueError(f"Unknown main loss: {args.main_loss}")

        bs = y.size(0)
        total_loss += loss.item() * bs
        total_count += bs

    return total_loss / max(1, total_count)

def train_one_epoch(model, loader, optimizer, scheduler, device, epoch, args):
    model.train()
    total_loss_accum = 0.0
    total_count = 0
    
    # Persistent state for adaptive weighting
    if not hasattr(train_one_epoch, "_metric_state"):
        train_one_epoch._metric_state = {}
    metric_state = train_one_epoch._metric_state
    
    # Compute curriculum weights for this epoch (if enabled)
    if args.use_curriculum:
        class_weights = get_curriculum_class_weights(
            epoch,
            start_epoch=args.curr_start,
            decay_rate=args.curr_decay,
            min_weight=args.curr_min_weight,
        ).to(device)
    else:
        class_weights = None
    
    pbar = tqdm(loader, desc=f"Train Ep{epoch}", leave=False)
    
    for batch in pbar:
        X, y, mask, domain_labels, metas = batch
        X = X.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        mask = mask.to(device, non_blocking=True)
        domain_labels = domain_labels.to(device, non_blocking=True)
        
        logits, _, slide_embedding = model(X, mask)
        
        # 1. Main Loss (with Curriculum Weighting if enabled)
        if args.main_loss in ["ce", "emd", "focal"]:
            if args.main_loss == "ce":
                main_loss = F.cross_entropy(logits, y, weight=class_weights)
            elif args.main_loss == "focal":
                main_loss = _focal_loss_fn(logits, y, weight=class_weights)
            else:  # "emd"
                main_loss = emd_ordinal_loss(logits, y, weight=class_weights)
        elif args.main_loss == "ord_contrastive":
            # Pure ordinal contrastive as main loss, no separate CE/EMD term
            # Build classification-aware, curriculum-aware weights
            probs = F.softmax(logits, dim=1)
            pt = probs.gather(1, y.view(-1, 1)).squeeze(1)  # p(true class)
            sw = 1.0 + (1.0 - pt)  # harder examples get larger weight
            if args.use_curriculum and class_weights is not None:
                sw = sw * class_weights[y]
            main_loss = ordinal_neg_weighted_supcon(
                slide_embedding,
                y,
                domain_labels=domain_labels if args.domain_aware else None,
                temperature=args.temp,
                neg_strength=args.neg_strength,
                domain_aware=args.domain_aware,
                domain_weight=args.domain_weight,
                sample_weights=sw,
            )
        else:
            raise ValueError(f"Unknown main loss: {args.main_loss}")
            
        # 2. Auxiliary Loss (only if we are *not* in pure-contrastive mode)
        aux_loss = torch.tensor(0.0, device=device)
        eff_weight = 0.0
        
        if args.main_loss != "ord_contrastive" and args.aux_loss == "contrastive":
            # For contrastive, use hardness-based weights ONLY (no curriculum).
            # Applying curriculum to contrastive makes ISUP 0's weight decay to ~0.1,
            # which kills cross-domain ISUP 0 pairs and keeps benign clusters separated.
            probs = F.softmax(logits, dim=1)
            pt = probs.gather(1, y.view(-1, 1)).squeeze(1)
            sw_aux = 1.0 + (1.0 - pt)  # harder examples weighted more
            # Optionally apply inverse class frequency to balance rare classes
            if args.contrastive_inv_freq and hasattr(args, '_class_freq_weights') and args._class_freq_weights is not None:
                sw_aux = sw_aux * args._class_freq_weights.to(device)[y]

            aux_loss = ordinal_neg_weighted_supcon(
                slide_embedding,
                y,
                domain_labels=domain_labels if args.domain_aware else None,
                temperature=args.temp,
                neg_strength=args.neg_strength,
                domain_aware=args.domain_aware,
                domain_weight=args.domain_weight,
                sample_weights=sw_aux,
            )
            
            # Calculate effective weight
            base_weight = get_scheduled_metric_weight(epoch, args)
            eff_weight = adjust_metric_weight_by_ratio(base_weight, main_loss, aux_loss, args, metric_state)
            
        # Total Loss
        loss = main_loss + eff_weight * aux_loss
        
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        if scheduler is not None:
            scheduler.step()
            
        bs = y.size(0)
        total_loss_accum += loss.item() * bs
        total_count += bs
        
        # Logging to pbar
        pbar.set_postfix(
            loss=f"{loss.item():.4f}",
            main=f"{main_loss.item():.4f}",
            aux=f"{aux_loss.item():.4f}",
            w=f"{eff_weight:.3f}"
        )
        
        # Log detailed components to wandb
        if args.use_wandb:
            import wandb
            wandb.log({
                "batch/loss": loss.item(),
                "batch/main_loss": main_loss.item(),
                "batch/aux_loss": aux_loss.item(),
                "batch/aux_weight": eff_weight,
                "lr": optimizer.param_groups[0]["lr"]
            })
            
    return total_loss_accum / max(1, total_count)

# -------------------- Main --------------------

def main():
    p = argparse.ArgumentParser(description="Simplified Training Pipeline for Histopathology")
    
    # Data & Paths
    p.add_argument("--csv", default='/home/mahdi.abootorabi/histopathology/panda_split', type=str)
    p.add_argument("--root", default='/data/project/users/tarek/panda_embeddings', type=str)
    p.add_argument("--out", default='/home/mahdi.abootorabi/histopathology/output/v3', type=str)
    
    # Model & Training
    p.add_argument("--epochs", default=40, type=int)
    p.add_argument("--lr", default=2e-4, type=float)
    p.add_argument("--weight-decay", default=1e-4, type=float)
    p.add_argument("--batch-size", default=16, type=int)
    p.add_argument("--num-workers", default=2, type=int)
    p.add_argument("--seed", default=1337, type=int)
    p.add_argument("--embedding-size", default=512, type=int)
    p.add_argument("--use-weighted-sampler", action="store_true")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    
    # Losses
    p.add_argument("--main-loss", default="ce", choices=["ce", "emd", "focal", "ord_contrastive"],
                   help="Main loss: CrossEntropy, EMD, Focal, or pure Ordinal Contrastive")
    p.add_argument("--aux-loss", default="none", choices=["none", "contrastive"], help="Auxiliary contrastive loss")
    
    # Curriculum Learning
    p.add_argument("--use-curriculum", action="store_true", help="Enable curriculum learning (decay weights for easier classes)")
    p.add_argument("--curr-start", default=10, type=int, help="Epoch to start curriculum learning")
    p.add_argument("--curr-decay", default=0.1, type=float, help="Decay rate for curriculum learning")
    p.add_argument("--curr-min-weight", default=0.1, type=float, help="Minimum weight for easiest classes")

    # Contrastive Params
    p.add_argument("--temp", default=0.07, type=float, help="SupCon temperature")
    p.add_argument("--neg-strength", default=1.0, type=float, help="Strength of negative weighting (distance based)")
    p.add_argument("--domain-aware", action="store_true", help="If set, multiplies negative weights by domain factor")
    p.add_argument("--domain-weight", default=1.0, type=float, help="Weight boost for cross-domain neg AND pos pairs")
    p.add_argument("--contrastive-inv-freq", action="store_true",
                   help="Weight contrastive samples by inverse class frequency from train set (balances rare classes)")
    
    # Weight Schedule (for Aux Loss)
    p.add_argument("--aux-weight-start", default=0.5, type=float, help="Initial aux weight")
    p.add_argument("--aux-weight-end", default=0.05, type=float, help="Final aux weight")
    p.add_argument("--aux-weight-decay-start", default=1, type=int)
    p.add_argument("--aux-weight-decay-end", default=15, type=int)
    p.add_argument("--aux-weight-schedule", default="linear", choices=["linear", "cosine", "exp"])
    p.add_argument("--aux-ratio-adapt", action="store_true", default=True, help="Adapt aux weight based on loss ratio")
    p.add_argument("--aux-target-ratio", default=0.05, type=float, help="Target ratio of aux loss to total loss")
    
    # WandB
    p.add_argument("--wandb", default="online", choices=["disabled", "online", "offline"])
    p.add_argument("--wandb-project", default="panda-simple-v3")
    
    args = p.parse_args()

    # Construct a canonical checkpoint name (also used as W&B run name)
    ckpt_name = f"best_auroc_model_{args.main_loss}"
    if args.aux_loss != "none":
        ckpt_name += f"_{args.aux_loss}"
        if args.domain_aware:
            ckpt_name += f"_dom{args.domain_weight}"
    if args.use_curriculum:
        ckpt_name += "_curr"
    ckpt_name += f"_emb{args.embedding_size}.pth"
    
    set_seed(args.seed)
    device = torch.device(args.device)
    args.use_wandb = args.wandb != "disabled"
     
    # Logging
    if args.use_wandb:
        if args.wandb == "offline":
            os.environ["WANDB_MODE"] = "offline"
        import wandb
        wandb.init(
            project=args.wandb_project,
            name=ckpt_name,
            config=vars(args)
        )
        
    print(f"Model: ABMILISUP (Dim {args.embedding_size})")
    print(f"Main Loss: {args.main_loss} (Curriculum: {args.use_curriculum})")
    if args.use_curriculum:
        print(f"  -> Start={args.curr_start}, Decay={args.curr_decay}, MinW={args.curr_min_weight}")
    print(f"Aux Loss: {args.aux_loss} (Domain Aware: {args.domain_aware})")
    
    # Data
    Ltrain, Lval, Ltest, train_df, val_df, test_df = make_loaders_from_dir(
        csv_dir=args.csv,
        root=args.root,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        use_weighted_sampler=args.use_weighted_sampler,
        pin_memory=True,
    )

    # Compute inverse class frequency from train set for contrastive balancing
    args._class_freq_weights = None
    if args.contrastive_inv_freq and train_df is not None:
        label_col = "isup_grade" if "isup_grade" in train_df.columns else train_df.columns[-1]
        counts = train_df[label_col].value_counts().sort_index()
        num_classes = 6
        freq = torch.zeros(num_classes)
        for cls, cnt in counts.items():
            if 0 <= int(cls) < num_classes:
                freq[int(cls)] = cnt
        freq = freq.clamp(min=1)
        inv_freq = 1.0 / freq
        inv_freq = inv_freq / inv_freq.mean()  # normalize so mean weight = 1
        args._class_freq_weights = inv_freq
        print(f"Contrastive inv-freq weights: {[f'{w:.3f}' for w in inv_freq.tolist()]}")
    
    # Model
    model = ABMILISUP(
        input_dim=1536, 
        num_classes=6,
        proj_dim=args.embedding_size,
        attn_hidden=384,
        p_input_dropout=0.10,
        p_mid_dropout=0.25,
        use_proj_layernorm=True,
        use_embed_layernorm=True,
        use_l2_norm_embed=True
    ).to(device)
    
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs * len(Ltrain))
    
    # Training Loop
    best_auc = 0.0
    best_epoch = -1
    best_state = None
    
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(model, Ltrain, optimizer, scheduler, device, epoch, args)
        val_loss = compute_eval_loss(model, Lval, device, epoch, args)
        val_metrics = evaluate(model, Lval, device, args)
        
        weight_str = "None"
        if args.use_curriculum:
            curr_weights = get_curriculum_class_weights(epoch, start_epoch=args.curr_start, decay_rate=args.curr_decay, min_weight=args.curr_min_weight)
            weight_str = "[" + ",".join([f"{w:.2f}" for w in curr_weights]) + "]"
        
        print(f"[Ep {epoch:02d}] Train={train_loss:.4f} ValLoss={val_loss:.4f} | Val AUC={val_metrics['auroc']:.4f} "
              f"Acc={val_metrics['acc']:.4f} BalAcc={val_metrics['bal_acc']:.4f} QWK={val_metrics['qwk']:.4f} Weights={weight_str}")
              
        if args.use_wandb:
            wandb.log({
                "epoch": epoch,
                "train/loss": train_loss,
                "val/loss": val_loss,
                "val/auroc": val_metrics['auroc'],
                "val/acc": val_metrics['acc'],
                "val/bal_acc": val_metrics['bal_acc'],
                "val/qwk": val_metrics['qwk'],
                "val/f1_macro": val_metrics['f1_macro']
            })
            
        # Save Best by AUROC
        if val_metrics['auroc'] > best_auc:
            best_auc = val_metrics['auroc']
            best_epoch = epoch
            best_state = {
                "epoch": epoch,
                "model": model.state_dict(),
                "config": vars(args),
                "best_auc": best_auc,
                "metrics": val_metrics
            }
            
            torch.save(best_state, out_dir / ckpt_name)
            print(f"  >>> New Best AUROC: {best_auc:.4f} (Saved to {ckpt_name})")
            
    print(f"\nTraining Complete. Best AUROC: {best_auc:.4f} at Epoch {best_epoch}")
    
    if args.use_wandb:
        wandb.log({"final/best_auc": best_auc, "final/best_epoch": best_epoch})

    # --- Final Evaluation on Test Set ---
    if Ltest is not None and best_state is not None:
        print("\nRunning Final Evaluation on Test Set using Best AUROC Model...")
        # Reload best model
        model.load_state_dict(best_state["model"])
        
        test_metrics = evaluate(model, Ltest, device, args)
        
        print(f"[TEST FINAL] AUC={test_metrics['auroc']:.4f} Acc={test_metrics['acc']:.4f} "
              f"BalAcc={test_metrics['bal_acc']:.4f} QWK={test_metrics['qwk']:.4f} F1M={test_metrics['f1_macro']:.4f}")
              
        if args.use_wandb:
            wandb.log({
                "final/test/auroc": test_metrics['auroc'],
                "final/test/acc": test_metrics['acc'],
                "final/test/bal_acc": test_metrics['bal_acc'],
                "final/test/qwk": test_metrics['qwk'],
                "final/test/f1_macro": test_metrics['f1_macro']
            })

if __name__ == "__main__":
    main()
