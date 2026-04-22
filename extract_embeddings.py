#!/usr/bin/env python3
"""
Extract slide embeddings from a trained model checkpoint and save as HDF5 files.

Usage:
    python extract_embeddings.py --ckpt /path/to/checkpoint.pth --output-dir /data/project/users/mahdi/panda_train_emd_scl
"""

import argparse
import os
from pathlib import Path
import h5py
import torch
import torch.nn.functional as F
from tqdm import tqdm
import pandas as pd

from abmil import ABMILISUP, DomainAdversarialABMIL
from data import make_loaders_from_dir


@torch.no_grad()
def extract_embeddings(model, loader, device, output_dir):
    """
    Extract slide embeddings and save as HDF5 files.
    
    Args:
        model: Trained model
        loader: DataLoader for the dataset
        device: Device to run inference on
        output_dir: Directory to save HDF5 files
    """
    model.eval()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Extracting embeddings to: {output_dir}")
    print(f"Total batches: {len(loader)}")
    
    for batch_idx, batch in enumerate(tqdm(loader, desc="Extracting embeddings")):
        X, y, mask, domain_labels, metas = batch
        X = X.to(device, non_blocking=True)
        mask = mask.to(device, non_blocking=True)
        
        # Get slide embeddings
        if hasattr(model, 'domain_classifier'):  # DomainAdversarialABMIL
            logits, attn, z, domain_logits = model(X, mask)
        else:  # ABMILISUP
            logits, attn, z = model(X, mask)
        
        # Move to CPU for saving
        z = z.cpu().numpy()
        
        # Save each embedding in the batch
        batch_size = z.shape[0]
        for i in range(batch_size):
            # Get image_id from metadata (metas is a list of dicts)
            if i < len(metas) and 'image_id' in metas[i]:
                image_id = metas[i]['image_id']
            else:
                # Fallback: use batch_idx and sample_idx
                image_id = f"batch_{batch_idx}_sample_{i}"
            
            # Create HDF5 file
            h5_path = output_dir / f"{image_id}.h5"
            
            with h5py.File(h5_path, 'w') as f:
                f.create_dataset('embedding', data=z[i], compression='gzip')
                f.attrs['image_id'] = image_id
                f.attrs['isup_grade'] = int(y[i].item())
                f.attrs['domain_label'] = int(domain_labels[i].item())
                f.attrs['embedding_shape'] = z[i].shape
                f.attrs['embedding_dtype'] = str(z[i].dtype)
    
    print(f"✅ Extracted {len(list(output_dir.glob('*.h5')))} embeddings to {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Extract slide embeddings from trained model")
    parser.add_argument("--ckpt", required=True, type=str, help="Path to model checkpoint (.pth file)")
    parser.add_argument("--output-dir", default="/data/project/users/mahdi/panda_train_emd_scl/512/", type=str, 
                       help="Output directory for HDF5 files")
    parser.add_argument("--csv", default='/home/mahdi.abootorabi/histopathology/panda_split', type=str,
                       help="Path to CSV directory with train.csv")
    parser.add_argument("--root", default='/data/project/users/tarek/panda_embeddings', type=str,
                       help="Path to embeddings root directory")
    parser.add_argument("--batch-size", default=128, type=int, help="Batch size for extraction")
    parser.add_argument("--num-workers", default=8, type=int, help="Number of data loading workers")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu", type=str)
    parser.add_argument("--split", default="train", choices=["train", "val", "test"], 
                       help="Which split to extract embeddings from")
    
    # Model architecture flags (must match training)
    parser.add_argument("--model-type", default="abmil", choices=["abmil", "adversarial"],
                       help="Model type used during training")
    parser.add_argument("--embedding-size", default=512, type=int, help="Dimension of the slide embedding z")
    parser.add_argument("--use-proj-layernorm", action="store_true", 
                       help="Use LayerNorm after projection (must match training)")
    parser.add_argument("--use-embed-layernorm", action="store_true",
                       help="Use LayerNorm on slide embedding (must match training)")
    parser.add_argument("--use-l2-norm-embed", action="store_true",
                       help="L2-normalize slide embedding (must match training)")
    parser.add_argument("--grl-alpha", default=1.0, type=float,
                       help="GRL alpha for adversarial model (must match training)")
    
    args = parser.parse_args()
    
    # Handle CUDA compatibility
    if args.device == "cuda" and torch.cuda.is_available():
        try:
            test_tensor = torch.tensor([1.0]).cuda()
            del test_tensor
            device = torch.device("cuda")
        except Exception as e:
            print(f"⚠️  CUDA not properly supported: {e}")
            print("🔄 Falling back to CPU...")
            device = torch.device("cpu")
    else:
        device = torch.device(args.device)
    
    print(f"🔧 Using device: {device}")
    
    # Load checkpoint
    print(f"Loading checkpoint: {args.ckpt}")
    checkpoint = torch.load(args.ckpt, map_location=device)
    
    # Extract model config from checkpoint if available
    if 'config' in checkpoint:
        config = checkpoint['config']
        print("📋 Found config in checkpoint, using saved parameters:")
        for key in ['embedding_size', 'use_proj_layernorm', 'use_embed_layernorm', 'use_l2_norm_embed', 'model_type', 'grl_alpha']:
            if key in config:
                setattr(args, key, config[key])
                print(f"  {key}: {config[key]}")
    
    # Create model with correct architecture
    if args.model_type == "adversarial":
        print("Using DomainAdversarialABMIL model")
        model = DomainAdversarialABMIL(
            input_dim=1536, num_classes=6,
            proj_dim=args.embedding_size, attn_hidden=384,
            p_input_dropout=0.10, p_mid_dropout=0.25,
            grl_alpha=args.grl_alpha,
            use_proj_layernorm=args.use_proj_layernorm,
            use_embed_layernorm=args.use_embed_layernorm,
            use_l2_norm_embed=args.use_l2_norm_embed,
        ).to(device)
    else:
        print("Using ABMILISUP model")
        model = ABMILISUP(
            input_dim=1536, num_classes=6,
            proj_dim=args.embedding_size, attn_hidden=384,
            p_input_dropout=0.10, p_mid_dropout=0.25,
            use_proj_layernorm=args.use_proj_layernorm,
            use_embed_layernorm=args.use_embed_layernorm,
            use_l2_norm_embed=args.use_l2_norm_embed,
        ).to(device)
    
    # Load model weights
    model.load_state_dict(checkpoint['model'])
    model.eval()
    print("✅ Model loaded successfully")
    
    # Load data
    print(f"Loading {args.split} data...")
    train_loader, val_loader, test_loader, train_df, val_df, test_df = make_loaders_from_dir(
        csv_dir=args.csv,
        root=args.root,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    
    # Select the appropriate loader
    if args.split == 'train':
        data_loader = train_loader
        data_df = train_df
    elif args.split == 'val':
        data_loader = val_loader
        data_df = val_df
    else:  # test
        data_loader = test_loader
        data_df = test_df
    
    if data_loader is None:
        raise ValueError(f"The '{args.split}' split is not available.")
    
    print(f"📊 {args.split} dataset: {len(data_df)} samples")
    
    # Extract embeddings
    extract_embeddings(model, data_loader, device, args.output_dir)
    
    print("🎉 Embedding extraction completed!")


if __name__ == "__main__":
    main()