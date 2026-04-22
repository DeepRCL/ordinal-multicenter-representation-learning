# train_abmil_isup_wandb.py
import argparse, os, math
from pathlib import Path
from typing import Dict

import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn.metrics import accuracy_score, f1_score, cohen_kappa_score
from abmil import ABMILISUP
from data import make_loaders_from_dir
from data import make_loaders_from_dir


import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalLoss(nn.Module):
    def __init__(self, alpha: float = 0.25, gamma: float = 2.0, reduction: str = 'mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits: [N, C] raw class scores
            target: [N] integer class labels
        Returns:
            scalar loss
        """
        # Calculate cross-entropy loss without reduction
        ce_loss = F.cross_entropy(logits, target, reduction='none')

        # Get the predicted probability of the true class
        pt = torch.exp(-ce_loss)

        # Calculate the focal loss
        focal_loss = self.alpha * (1 - pt)**self.gamma * ce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


# Loss function parameters (can be tuned via command line)
POISSON_TEMP = 1.0  # Temperature for Poisson distribution (higher = more uniform)
POISSON_ALPHA = 7.5  # Focal alpha
POISSON_GAMMA = 2.0  # Focal gamma

def set_seed(seed: int = 1337):
    import random
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

from tqdm.auto import tqdm

def poisson_focal_loss(logits: torch.Tensor,
                       target: torch.Tensor,
                       t: float = None,  # Will use global parameter
                       smooth: float = 1e-6,
                       alpha: float = None,  # Will use global parameter
                       gamma: float = None) -> torch.Tensor:  # Will use global parameter
    """Fixed Poisson Focal Loss for ordinal classification.

    Args:
        logits: [N, C] raw class scores
        target: [N] integer class labels
        t: temperature for softmax over Poisson-encoded target (higher = more uniform)
        smooth: epsilon for numerical stability
        alpha: focal alpha
        gamma: focal gamma
    Returns:
        scalar loss tensor
    """
    # Use global parameters if not specified
    if t is None:
        t = POISSON_TEMP
    if alpha is None:
        alpha = POISSON_ALPHA
    if gamma is None:
        gamma = POISSON_GAMMA
        
    bs, num_classes = logits.shape
    device = logits.device

    # Predicted distribution over classes (more stable computation)
    logits_stable = logits - logits.max(dim=1, keepdim=True)[0]  # Numerical stability
    prob = F.softmax(logits_stable, dim=1)
    logpt = torch.log(prob + smooth)  # Add smooth to log, not to prob

    # Build Poisson-encoded target distribution P[y]
    j = torch.arange(num_classes, device=device, dtype=logits.dtype).view(1, num_classes)
    j_log_fac = torch.lgamma(j + 1.0)

    # Lambda parameter: y + 0.5 (as in original PON paper)
    lam = target.unsqueeze(1).to(logits.dtype) + 0.5
    
    # Log Poisson pmf: log P(j; lam) = j*log(lam) - lam - log(j!)
    log_poiss = j * torch.log(lam + smooth) - lam - j_log_fac
    
    # Apply temperature scaling (higher t = more uniform distribution)
    target_prob = F.softmax(log_poiss / t, dim=-1)

    # KL divergence: KL(target_prob || pred_prob)
    # This should be the main loss component
    kl_div = torch.sum(target_prob * (torch.log(target_prob + smooth) - logpt), dim=1)

    # Focal weight: focus on hard examples
    # Use the probability of the true class as focal weight
    gather_idx = target.view(-1, 1).long()
    pt = prob.gather(1, gather_idx).view(-1)  # Probability of true class
    
    # Focal weight: (1 - pt)^gamma
    focal_weight = torch.pow(1 - pt, gamma)
    
    # Combined loss: focal_weight * kl_div
    # Scale by alpha and a factor to match typical loss magnitudes
    # The KL divergence is typically smaller than CE, so we need to scale it up
    loss = alpha * focal_weight * kl_div  # Scale factor to match CE magnitude
    return loss.mean()

def ordinal_cross_entropy_loss(logits: torch.Tensor, target: torch.Tensor, 
                               num_classes: int = 6, weight: float = 1.0) -> torch.Tensor:
    """
    Ordinal Cross-Entropy Loss that penalizes predictions based on distance from true class.
    
    Args:
        logits: [N, C] raw class scores
        target: [N] integer class labels
        num_classes: number of classes
        weight: weight for ordinal penalty
    """
    # Standard cross-entropy loss
    ce_loss = F.cross_entropy(logits, target, reduction='none')
    
    # Ordinal penalty: penalize based on distance from true class
    pred_classes = logits.argmax(dim=1)
    distance = torch.abs(pred_classes - target).float()
    ordinal_penalty = weight * distance
    
    # Combine losses
    total_loss = ce_loss + ordinal_penalty
    return total_loss.mean()

@torch.no_grad()
def evaluate(model, loader, device) -> Dict[str, float]:
    model.eval()
    ys, yps = [], []
    it = tqdm(loader, desc="Eval", leave=False)
    for X, y, mask, _domain, _meta in it:
        X = X.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        mask = mask.to(device, non_blocking=True)

        logits, _, _ = model(X, mask)
        y_hat = logits.argmax(dim=1)

        # Show batch accuracy live
        batch_acc = (y_hat == y).float().mean().item()
        it.set_postfix(batch_acc=f"{batch_acc:.3f}")

        ys.append(y.detach().cpu())
        yps.append(y_hat.detach().cpu())

    y_true = torch.cat(ys).numpy()
    y_pred = torch.cat(yps).numpy()
    return {
        "acc": accuracy_score(y_true, y_pred),
        "f1_macro": f1_score(y_true, y_pred, average="macro"),
        "qwk": cohen_kappa_score(y_true, y_pred, weights="quadratic"),
    }


def train_one_epoch(model, loader, optimizer, scheduler, device, loss_fn):
    model.train()
    total_loss, total_count = 0.0, 0
    it = tqdm(loader, desc="Train", leave=False)
    for X, y, mask, _domain, _meta in it:
        X = X.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        mask = mask.to(device, non_blocking=True)

        logits, _, _ = model(X, mask)
        loss = loss_fn(logits, y)
        
        # Debug: Print loss values for first few batches to check if they're reasonable
        if hasattr(train_one_epoch, '_debug_count'):
            train_one_epoch._debug_count += 1
        else:
            train_one_epoch._debug_count = 1
            
        if train_one_epoch._debug_count <= 3:  # Print first 3 batches
            print(f"DEBUG Batch {train_one_epoch._debug_count}: loss={loss.item():.6f}, logits_range=[{logits.min().item():.3f}, {logits.max().item():.3f}]")
            pred_classes = logits.argmax(dim=1)
            acc = (pred_classes == y).float().mean().item()
            print(f"DEBUG Batch {train_one_epoch._debug_count}: batch_acc={acc:.3f}, target_range=[{y.min().item()}, {y.max().item()}]")

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        if scheduler is not None:
            scheduler.step()  # per-iteration cosine

        bs = y.size(0)
        total_loss += loss.item() * bs
        total_count += bs

        # Show running (mean) loss live
        running = total_loss / max(1, total_count)
        it.set_postfix(loss=f"{running:.4f}")

    return total_loss / max(1, total_count)


def main():
    p = argparse.ArgumentParser(description="Train gated ABMIL on PANDA (ISUP) with W&B logging")
    # data
    p.add_argument("--csv", default='/home/mahdi.abootorabi/histopathology/panda_split', type=str)
    p.add_argument("--root", default='/home/mahdi.abootorabi/histopathology/input_embeddings/panda_embeddings', type=str)
    p.add_argument("--out", default='/home/mahdi.abootorabi/histopathology/output/best.pth', type=str)
    # training (UNI/UNI2 defaults)
    p.add_argument("--epochs", default=40, type=int)
    p.add_argument("--lr", default=2e-4, type=float)
    p.add_argument("--weight-decay", default=1e-4, type=float)
    p.add_argument("--batch-size", default=8, type=int)
    p.add_argument("--num-workers", default=4, type=int)
    p.add_argument("--seed", default=1337, type=int)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu", type=str)
    p.add_argument("--max-patches-train", default=None, type=int)
    p.add_argument("--use-weighted-sampler", action="store_true")
    # wandb
    p.add_argument("--wandb", default="online", choices=["disabled", "online", "offline"])
    p.add_argument("--wandb-project", default="panda-abmil")
    p.add_argument("--wandb-entity", default=None)
    p.add_argument("--wandb-run-name", default=None)
    p.add_argument("--wandb-tags", default="", help="comma-separated tags")
    # Loss function parameters
    p.add_argument("--loss-type", default="poisson_focal", choices=["cross_entropy", "poisson_focal", "ordinal_ce", "focal_v2"], 
                   help="Type of loss function to use")
    p.add_argument("--poisson-temp", default=1.0, type=float, help="Temperature for Poisson Focal Loss")
    p.add_argument("--poisson-alpha", default=0.75, type=float, help="Alpha for Poisson Focal Loss")
    p.add_argument("--poisson-gamma", default=2.0, type=float, help="Gamma for Poisson Focal Loss")
    p.add_argument("--ordinal-weight", default=0.1, type=float, help="Weight for ordinal penalty in Ordinal CE Loss")
    p.add_argument("--embedding-size", default=512, type=int, help="Dimension of the slide embedding z")
    args = p.parse_args()

    set_seed(args.seed)
    device = torch.device(args.device)
    
    # Update global loss parameters
    global POISSON_TEMP, POISSON_ALPHA, POISSON_GAMMA
    POISSON_TEMP = args.poisson_temp
    POISSON_ALPHA = args.poisson_alpha
    POISSON_GAMMA = args.poisson_gamma
    
    # Loss function selector
    def get_loss_function():
        if args.loss_type == "cross_entropy":
            return lambda logits, y: F.cross_entropy(logits, y)
        elif args.loss_type == "poisson_focal":
            return lambda logits, y: poisson_focal_loss(logits, y)
        elif args.loss_type == "ordinal_ce":
            return lambda logits, y: ordinal_cross_entropy_loss(logits, y, weight=args.ordinal_weight)
        elif args.loss_type == "focal_v2":
            return FocalLoss().to(device)
        else:
            raise ValueError(f"Unknown loss type: {args.loss_type}")
    
    loss_fn = get_loss_function()
    print(f"Using loss function: {args.loss_type}")
    
    # Debug: Test loss function with dummy data to check if values are reasonable
    if args.loss_type == "poisson_focal":
        print("Testing Poisson Focal Loss with dummy data...")
        dummy_logits = torch.randn(4, 6) * 0.1  # Small random logits (typical for early training)
        dummy_target = torch.tensor([0, 1, 2, 3])
        dummy_loss = loss_fn(dummy_logits, dummy_target)
        print(f"Poisson Focal Loss test - dummy_loss: {dummy_loss.item():.6f}")
        
        # Compare with cross-entropy
        ce_loss = F.cross_entropy(dummy_logits, dummy_target)
        print(f"Cross-entropy comparison - ce_loss: {ce_loss.item():.6f}")
        print(f"Expected range for 6-class CE: 1.5-2.0 (log(6) ≈ 1.79)")
        print(f"Poisson loss should be in similar range, not 0.0005!")
        
        if dummy_loss.item() < 0.01:
            print("WARNING: Poisson Focal Loss is producing unrealistically low values!")
            print("This indicates a bug in the implementation.")

    # Data
    Ltrain, Lval, Ltest, train_df, val_df, test_df = make_loaders_from_dir(
        csv_dir=args.csv,   # contains train.csv / val.csv / (optional) test.csv
        root=args.root,
        batch_size=8,
        num_workers=4,
        max_patches_train=2000,       # or None
        use_weighted_sampler=True,
        pin_memory=True,
    )

    # Model (UNI Methods config)
    model = ABMILISUP(
        input_dim=1536, num_classes=6,
        proj_dim=args.embedding_size, attn_hidden=384,
        p_input_dropout=0.10, p_mid_dropout=0.25,
    ).to(device)

    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    total_steps = args.epochs * max(1, len(Ltrain))
    scheduler = CosineAnnealingLR(optimizer, T_max=max(1, total_steps))

    # ---- Weights & Biases ----
    use_wandb = args.wandb != "disabled"
    if use_wandb:
        if args.wandb == "offline":
            os.environ["WANDB_MODE"] = "offline"
        import wandb
        wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=args.wandb_run_name,
            tags=[t.strip() for t in args.wandb_tags.split(",") if t.strip()],
            config={
                **vars(args),
                "model": "Gated-ABMIL",
                "proj_dim": args.embedding_size, "attn_hidden": 384,
                "p_input_dropout": 0.10, "p_mid_dropout": 0.25,
            },
        )
        wandb.watch(model, log="all", log_freq=100)

    # Train with early stopping on val loss
    best_val_loss, best_epoch, best_state = float("inf"), -1, None
    best_val_qwk, best_epoch_qwk, best_state_qwk = -1, -1, None
    patience = 10

    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(model, Ltrain, optimizer, scheduler, device, loss_fn)

        # Validation
        model.eval()
        with torch.no_grad():
            val_loss, count = 0.0, 0
            for X, y, mask, _domain, _meta in Lval:
                X = X.to(device, non_blocking=True)
                y = y.to(device, non_blocking=True)
                mask = mask.to(device, non_blocking=True)
                logits, _, _ = model(X, mask)
                loss = loss_fn(logits, y)
                bs = y.size(0)
                val_loss += loss.item() * bs; count += bs
            val_loss /= max(1, count)

        val_metrics = evaluate(model, Lval, device)
        lr_now = optimizer.param_groups[0]["lr"]
        print(f"[Epoch {epoch:02d}] train_loss={train_loss:.4f} "
              f"val_loss={val_loss:.4f} acc={val_metrics['acc']:.4f} "
              f"f1m={val_metrics['f1_macro']:.4f} qwk={val_metrics['qwk']:.4f} lr={lr_now:.6f}")

        if use_wandb:
            wandb.log({
                "epoch": epoch,
                "train/loss": train_loss,
                "val/loss": val_loss,
                "val/acc": val_metrics["acc"],
                "val/f1_macro": val_metrics["f1_macro"],
                "val/qwk": val_metrics["qwk"],
                "lr": lr_now,
            }, step=epoch)

        # Early stopping (val loss)
        if val_loss < best_val_loss - 1e-6:
            best_val_loss = val_loss
            best_epoch = epoch
            best_state = {
                "epoch": epoch,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "best_val_loss": best_val_loss,
                "config": vars(args),
            }
            print(f"New best val loss model at epoch {epoch}: val_loss={val_loss:.4f}, qwk={val_metrics['qwk']:.4f}")
        
        if val_metrics["qwk"] > best_val_qwk:
            best_val_qwk = val_metrics["qwk"]
            best_epoch_qwk = epoch
            best_state_qwk = {
                "epoch": epoch,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "best_val_qwk": best_val_qwk,
                "config": vars(args),
            }
            print(f"New best QWK model at epoch {epoch}: val_loss={val_loss:.4f}, qwk={val_metrics['qwk']:.4f}")
        elif epoch - best_epoch >= patience:
            print(f"Early stopping at epoch {epoch} (no val loss improvement for {patience} epochs).")
            break

    # Save best checkpoint
    out_path = Path(args.out); out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(best_state if best_state is not None else {"model": model.state_dict()}, out_path)
    print(f"Saved best checkpoint (epoch {best_epoch}) to: {out_path}")
    torch.save(best_state_qwk if best_state_qwk is not None else {"model": model.state_dict()}, out_path.parent / "best_qwk.pth")
    print(f"Saved best checkpoint (epoch {best_epoch_qwk}) to: {out_path.parent / 'best_qwk.pth'}")

    # Final eval (val/test) from best
    if best_state is not None:
        model.load_state_dict(best_state["model"])
        print(f"Loaded best val loss model from epoch {best_state['epoch']}")
    val_final = evaluate(model, Lval, device)
    print(f"[VAL FINAL] acc={val_final['acc']:.4f} f1m={val_final['f1_macro']:.4f} qwk={val_final['qwk']:.4f}")
    if use_wandb:
        import wandb
        wandb.log({
            "final/val/acc": val_final["acc"],
            "final/val/f1_macro": val_final["f1_macro"],
            "final/val/qwk": val_final["qwk"],
            "final/best_epoch": best_epoch,
        })

    if best_state_qwk is not None:
        best_qwk_model = ABMILISUP(
            input_dim=1536, num_classes=6,
            proj_dim=args.embedding_size, attn_hidden=384,
            p_input_dropout=0.10, p_mid_dropout=0.25,
        ).to(device)
        best_qwk_model.load_state_dict(best_state_qwk["model"])
        print(f"Loaded best QWK model from epoch {best_state_qwk['epoch']}")
    val_final_qwk = evaluate(best_qwk_model, Lval, device)
    print(f"[VAL FINAL QWK] acc={val_final_qwk['acc']:.4f} f1m={val_final_qwk['f1_macro']:.4f} qwk={val_final_qwk['qwk']:.4f}")
    
    # Check if models are different
    if best_state is not None and best_state_qwk is not None:
        if best_state['epoch'] == best_state_qwk['epoch']:
            print(f"WARNING: Both models are from the same epoch ({best_state['epoch']}). They are identical!")
        else:
            print(f"Models are from different epochs: val_loss model (epoch {best_state['epoch']}) vs QWK model (epoch {best_state_qwk['epoch']})")
    
    if use_wandb:
        wandb.log({
            "final/val/qwk": val_final_qwk["qwk"],
        })

    if Ltest is not None:
        # Test with the best validation loss model
        test_final = evaluate(model, Ltest, device)
        print(f"[TEST] acc={test_final['acc']:.4f} f1m={test_final['f1_macro']:.4f} qwk={test_final['qwk']:.4f}")
        if use_wandb:
            import wandb
            wandb.log({
                "final/test/acc": test_final["acc"],
                "final/test/f1_macro": test_final["f1_macro"],
                "final/test/qwk": test_final["qwk"],
            }) 
        # (optional) log checkpoint as artifact
        if use_wandb:
            artifact = wandb.Artifact("abmil-uni2-panda", type="model")
            artifact.add_file(str(out_path))
            wandb.log_artifact(artifact)

        if best_state_qwk is not None:
            test_final_qwk = evaluate(best_qwk_model, Ltest, device)
            print(f"[TEST FINAL QWK] acc={test_final_qwk['acc']:.4f} f1m={test_final_qwk['f1_macro']:.4f} qwk={test_final_qwk['qwk']:.4f}")
            if use_wandb:
                wandb.log({
                    "final/test/qwk_v2": test_final_qwk["qwk"],
                    "final/test/f1_macro_v2": test_final_qwk["f1_macro"],
                    "final/test/qwk_v2": test_final_qwk["qwk"],
                })
        if use_wandb:
            artifact = wandb.Artifact("abmil-uni2-panda-qwk", type="model")
            artifact.add_file(str(out_path.parent / "best_qwk.pth"))
            wandb.log_artifact(artifact)

if __name__ == "__main__":
    main()