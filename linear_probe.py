import argparse
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, cohen_kappa_score
from tqdm.auto import tqdm

from abmil import ABMILISUP, DomainAdversarialABMIL
from data import make_loaders_from_dir


def set_seed(seed: int = 1337) -> None:
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _infer_proj_dim_from_state(state_dict: Dict[str, torch.Tensor]) -> int:
    if "proj.weight" in state_dict:
        return int(state_dict["proj.weight"].shape[0])
    if "cls_fc1.weight" in state_dict:
        return int(state_dict["cls_fc1.weight"].shape[1])
    return 512


def _build_model_from_ckpt(ckpt_path: str, device: torch.device) -> Tuple[torch.nn.Module, Dict]:
    ckpt = torch.load(ckpt_path, map_location=device)
    if isinstance(ckpt, dict) and "model" in ckpt:
        state = ckpt["model"]
        config = ckpt.get("config", {})
    else:
        state = ckpt
        config = {}

    model_type = config.get("model_type", "abmil")
    proj_dim = _infer_proj_dim_from_state(state)
    use_proj_layernorm = bool(config.get("use_proj_layernorm", True))
    use_embed_layernorm = bool(config.get("use_embed_layernorm", True))
    use_l2_norm_embed = bool(config.get("use_l2_norm_embed", True))

    if model_type == "adversarial":
        model = DomainAdversarialABMIL(
            input_dim=1536,
            num_classes=6,
            proj_dim=proj_dim,
            attn_hidden=384,
            p_input_dropout=0.10,
            p_mid_dropout=0.25,
            grl_alpha=float(config.get("grl_alpha", 1.0)),
            use_proj_layernorm=use_proj_layernorm,
            use_embed_layernorm=use_embed_layernorm,
            use_l2_norm_embed=use_l2_norm_embed,
        ).to(device)
    else:
        model = ABMILISUP(
            input_dim=1536,
            num_classes=6,
            proj_dim=proj_dim,
            attn_hidden=384,
            p_input_dropout=0.10,
            p_mid_dropout=0.25,
            use_proj_layernorm=use_proj_layernorm,
            use_embed_layernorm=use_embed_layernorm,
            use_l2_norm_embed=use_l2_norm_embed,
        ).to(device)

    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        print(f"[WARN] load_state_dict: missing={missing}, unexpected={unexpected}")
    model.eval()
    return model, config


@torch.no_grad()
def extract_embeddings(model: torch.nn.Module, loader, device: torch.device):
    zs, ys, ds, ids = [], [], [], []
    it = tqdm(loader, desc="Extract", leave=False)
    for X, y, mask, domain_labels, metas in it:
        X = X.to(device, non_blocking=True)
        mask = mask.to(device, non_blocking=True)
        z = model.embed(X, mask)  # [B, D_z]
        zs.append(z.detach().cpu().numpy())
        ys.append(y.numpy())
        ds.append(domain_labels.numpy())
        ids.extend([m["image_id"] for m in metas])
    Z = np.concatenate(zs, axis=0)
    Y = np.concatenate(ys, axis=0)
    D = np.concatenate(ds, axis=0)
    return Z, Y, D, np.array(ids)


def metrics_for(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    acc = accuracy_score(y_true, y_pred)
    f1m = f1_score(y_true, y_pred, average="macro")
    qwk = cohen_kappa_score(y_true, y_pred, weights="quadratic")
    acc_pm1 = float((np.abs(y_pred - y_true) <= 1).mean())
    return {"acc": acc, "f1_macro": f1m, "qwk": qwk, "acc_pm1": acc_pm1}


def linear_probe(Xtr, Ytr, Xev, Yev, C: float = 1.0, max_iter: int = 2000) -> Dict[str, float]:
    clf = LogisticRegression(
        penalty="l2",
        C=C,
        solver="lbfgs",
        max_iter=max_iter,
        multi_class="multinomial",
        n_jobs=-1,
    )
    clf.fit(Xtr, Ytr)
    y_pred = clf.predict(Xev)
    return metrics_for(Yev, y_pred)


def domain_probe(Xtr, Dtr, Xev, Dev, C: float = 1.0, max_iter: int = 2000) -> float:
    clf = LogisticRegression(
        penalty="l2",
        C=C,
        solver="lbfgs",
        max_iter=max_iter,
        n_jobs=-1,
    )
    clf.fit(Xtr, Dtr)
    d_pred = clf.predict(Xev)
    return accuracy_score(Dev, d_pred)


def run(args: argparse.Namespace) -> None:
    set_seed(args.seed)
    device = torch.device(args.device)

    # Data
    Ltrain, Lval, Ltest, train_df, val_df, test_df = make_loaders_from_dir(
        csv_dir=args.csv,
        root=args.root,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        max_patches_train=None,
        use_weighted_sampler=False,
        pin_memory=True,
    )
    print(f"Datasets: train={len(Ltrain.dataset)}, val={len(Lval.dataset)}, test={len(Ltest.dataset) if Ltest is not None else 0}")

    # Models
    model_a, _ = _build_model_from_ckpt(args.model_a, device)
    model_b, _ = _build_model_from_ckpt(args.model_b, device)
    name_a = args.name_a or Path(args.model_a).stem
    name_b = args.name_b or Path(args.model_b).stem

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Extract and save embeddings for each model and split
    def extract_all(model, tag: str):
        Ztr, Ytr, Dtr, IDtr = extract_embeddings(model, Ltrain, device)
        Zva, Yva, Dva, IDva = extract_embeddings(model, Lval, device)
        if Ltest is not None:
            Zte, Yte, Dte, IDte = extract_embeddings(model, Ltest, device)
        else:
            Zte, Yte, Dte, IDte = None, None, None, None
        np.savez(out_dir / f"embeddings_{tag}_train.npz", Z=Ztr, y=Ytr, d=Dtr, ids=IDtr)
        np.savez(out_dir / f"embeddings_{tag}_val.npz", Z=Zva, y=Yva, d=Dva, ids=IDva)
        if Zte is not None:
            np.savez(out_dir / f"embeddings_{tag}_test.npz", Z=Zte, y=Yte, d=Dte, ids=IDte)
        return (Ztr, Ytr, Dtr), (Zva, Yva, Dva), ((Zte, Yte, Dte) if Zte is not None else None)

    (Za_tr, Ya_tr, Da_tr), (Za_va, Ya_va, Da_va), Za_te = extract_all(model_a, name_a)
    (Zb_tr, Yb_tr, Db_tr), (Zb_va, Yb_va, Db_va), Zb_te = extract_all(model_b, name_b)

    # Linear probes and domain invariance
    import pandas as pd
    rows = []

    def add_rows(tag: str, Ztr, Ytr, Dtr, Zva, Yva, Dva, Zte, Yte, Dte):
        # Standard probes (train on full train)
        m_val = linear_probe(Ztr, Ytr, Zva, Yva, C=args.C, max_iter=args.max_iter)
        rows.append({"model": tag, "probe": "full", "split": "val", **m_val})
        if Zte is not None:
            m_test = linear_probe(Ztr, Ytr, Zte, Yte, C=args.C, max_iter=args.max_iter)
            rows.append({"model": tag, "probe": "full", "split": "test", **m_test})
        # Domain classifier (lower acc better)
        d_val = domain_probe(Ztr, Dtr, Zva, Dva, C=args.C, max_iter=args.max_iter)
        rows.append({"model": tag, "probe": "domain", "split": "val", "domain_acc": d_val})
        if Zte is not None:
            d_test = domain_probe(Ztr, Dtr, Zte, Dte, C=args.C, max_iter=args.max_iter)
            rows.append({"model": tag, "probe": "domain", "split": "test", "domain_acc": d_test})
        # Cross-domain generalization: train on K -> eval on R, and R -> K
        K_tr = Dtr == 0
        R_tr = Dtr == 1
        K_va = Dva == 0
        R_va = Dva == 1
        # K -> R (val)
        if K_tr.any() and R_va.any():
            mkr_val = linear_probe(Ztr[K_tr], Ytr[K_tr], Zva[R_va], Yva[R_va], C=args.C, max_iter=args.max_iter)
            rows.append({"model": tag, "probe": "K_to_R", "split": "val", **mkr_val})
        # R -> K (val)
        if R_tr.any() and K_va.any():
            mrk_val = linear_probe(Ztr[R_tr], Ytr[R_tr], Zva[K_va], Yva[K_va], C=args.C, max_iter=args.max_iter)
            rows.append({"model": tag, "probe": "R_to_K", "split": "val", **mrk_val})
        if Zte is not None:
            K_te = Dte == 0
            R_te = Dte == 1
            if K_tr.any() and R_te.any():
                mkr_test = linear_probe(Ztr[K_tr], Ytr[K_tr], Zte[R_te], Yte[R_te], C=args.C, max_iter=args.max_iter)
                rows.append({"model": tag, "probe": "K_to_R", "split": "test", **mkr_test})
            if R_tr.any() and K_te.any():
                mrk_test = linear_probe(Ztr[R_tr], Ytr[R_tr], Zte[K_te], Yte[K_te], C=args.C, max_iter=args.max_iter)
                rows.append({"model": tag, "probe": "R_to_K", "split": "test", **mrk_test})

    # Add rows for both models
    add_rows(name_a, Za_tr, Ya_tr, Da_tr, Za_va, Ya_va, Da_va, *(Za_te if Za_te is not None else (None, None, None)))
    add_rows(name_b, Zb_tr, Yb_tr, Db_tr, Zb_va, Yb_va, Db_va, *(Zb_te if Zb_te is not None else (None, None, None)))

    # Save results
    df = pd.DataFrame(rows)
    out_csv = Path(args.out) / "linear_probe_results.csv"
    df.to_csv(out_csv, index=False)
    print(f"Saved linear probe results to: {out_csv}")

    # ── 1. ISUP Classification Probes (full / K_to_R / R_to_K) ──────────────
    # These show how well a linear classifier can predict ISUP grade from
    # frozen embeddings. Higher QWK = better. Cross-domain probes (K_to_R,
    # R_to_K) reveal whether representations generalise across centres.
    isup_df = df[df["probe"] != "domain"].copy()
    try:
        pv = isup_df.pivot_table(
            index=["probe", "split"], columns="model",
            values=["qwk", "acc_pm1", "acc"]
        )
        print("\n" + "="*70)
        print("ISUP LINEAR PROBE  (higher = better)")
        print("  full   : train all → test all  (overall representation quality)")
        print("  K_to_R : train Karolinska → test Radboud  (cross-centre transfer)")
        print("  R_to_K : train Radboud → test Karolinska  (cross-centre transfer)")
        print("="*70)
        print(pv.round(4).to_string())
    except Exception as e:
        print(f"ISUP pivot failed: {e}")

    # ── 2. Domain Classifier Probe ────────────────────────────────────────────
    # Trains a binary classifier (Karolinska vs Radboud) on frozen embeddings.
    # HIGH accuracy → the embedding encodes centre identity → bad (domain bias).
    # LOW accuracy (near 50%) → embedding ignores centre → good (domain-invariant).
    dom_df = df[df["probe"] == "domain"].copy()
    if not dom_df.empty:
        try:
            pv_dom = dom_df.pivot_table(
                index=["split"], columns="model", values="domain_acc"
            )
            print("\n" + "="*70)
            print("DOMAIN CLASSIFIER PROBE  (LOWER = better / more domain-invariant)")
            print("  A linear probe predicts which centre (Karolinska / Radboud)")
            print("  produced each slide directly from the frozen embeddings.")
            print("  If a model's embeddings are domain-agnostic, this should be")
            print("  close to 50% (chance). High accuracy → centre bias remains.")
            print("="*70)
            print(pv_dom.round(4).to_string())

            # Print a plain-English delta
            print()
            for split in pv_dom.index:
                row = pv_dom.loc[split]
                models = list(row.index)
                if len(models) == 2:
                    a, b = models[0], models[1]
                    delta = row[a] - row[b]
                    better = a if delta < 0 else b
                    worse  = b if delta < 0 else a
                    print(f"  [{split}] {better} is MORE domain-invariant "
                          f"(domain acc {min(row[a], row[b]):.4f} vs {max(row[a], row[b]):.4f}, "
                          f"Δ={abs(delta):.4f})")
        except Exception as e:
            print(f"Domain pivot failed: {e}")
    else:
        print("\n[WARNING] No domain probe rows found in results.")


def main() -> None:
    p = argparse.ArgumentParser(description="Frozen Features (Linear Probing) for ABMIL embeddings")
    p.add_argument("--csv", default="/home/mahdi.abootorabi/histopathology/panda_split", type=str)
    p.add_argument("--root", default="/data/project/users/tarek/panda_embeddings", type=str)
    p.add_argument("--out", default="/home/mahdi.abootorabi/histopathology/output", type=str)
    p.add_argument("--batch-size", default=8, type=int)
    p.add_argument("--num-workers", default=8, type=int)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu", type=str)
    p.add_argument("--seed", default=1337, type=int)
    # Models
    default_a = "/home/mahdi.abootorabi/histopathology/output/best_val_loss_abmil_emb768_curriculum_emd_best.pth"
    default_b = "/home/mahdi.abootorabi/histopathology/output/best_val_loss_abmil_emb768_cross_entropy.pth"
    p.add_argument("--model-a", default=default_a, type=str, help="Proposed model checkpoint")
    p.add_argument("--model-b", default=default_b, type=str, help="Baseline model checkpoint")
    p.add_argument("--name-a", default="proposed_curriculum_emd", type=str)
    p.add_argument("--name-b", default="baseline_cross_entropy", type=str)
    # Logistic regression settings
    p.add_argument("--C", default=1.0, type=float, help="Inverse regularization strength for LogisticRegression")
    p.add_argument("--max-iter", default=2000, type=int)
    args = p.parse_args()
    run(args)


if __name__ == "__main__":
    main()


