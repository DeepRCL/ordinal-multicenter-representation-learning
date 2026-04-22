
import argparse
from pathlib import Path
import os

import numpy as np
import pandas as pd
import torch
import umap
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm

# Import your custom model and data loader functions
from abmil import ABMILISUP, DomainAdversarialABMIL # Assuming abmil.py is in the same directory
from data import make_loaders_from_dir # Assuming data.py is in the same directory

@torch.no_grad()
def get_embeddings(model, loader, device, return_metadata=False):
    """
    Extracts slide-level embeddings and corresponding labels from a data loader.
    
    Args:
        model: The trained ABMIL model
        loader: DataLoader
        device: torch device
        return_metadata: If True, also return domain labels and metadata
    
    Returns:
        embeddings, labels, (optionally: domain_labels, metas)
    """
    model.eval()
    all_embeddings = []
    all_labels = []
    all_domain_labels = []
    all_metas = []

    it = tqdm(loader, desc="Extracting Embeddings", leave=False)
    for batch in it:
        X, y, mask, domain_labels, metas = batch  # Updated unpacking for 5 items
        X = X.to(device, non_blocking=True)
        mask = mask.to(device, non_blocking=True)

        # Using the model's dedicated .embed() method for clarity and correctness.
        # This returns the slide embedding 'z'.
        slide_embedding = model.embed(X, mask)

        all_embeddings.append(slide_embedding.cpu().numpy())
        all_labels.append(y.cpu().numpy())
        all_domain_labels.append(domain_labels.cpu().numpy())
        all_metas.extend(metas)

    # Concatenate all batches into single numpy arrays
    embeddings = np.concatenate(all_embeddings, axis=0)
    labels = np.concatenate(all_labels, axis=0)
    domain_labels = np.concatenate(all_domain_labels, axis=0)
    
    if return_metadata:
        return embeddings, labels, domain_labels, all_metas
    return embeddings, labels

def main():
    p = argparse.ArgumentParser(description="Generate UMAP plot from model embeddings")
    p.add_argument("--ckpt", required=True, type=str, help="Path to the trained model checkpoint (.pth file)")
    p.add_argument("--csv", default='/home/mahdi.abootorabi/histopathology/panda_split', type=str, help="Path to the directory with train.csv/val.csv")
    p.add_argument("--root", default='/data/project/users/tarek/panda_embeddings', type=str, help="Path to the embeddings root directory")
    p.add_argument("--split", default="test", choices=["train", "val", "test"], help="Which data split to use for the UMAP plot")
    p.add_argument("--out", default="umap_plot_", type=str, help="Output filename prefix for the UMAP plot image")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu", type=str)
    p.add_argument("--embedding-size", default=512, type=int, help="Dimension of the slide embedding z")
    # Normalization flags (match train_v2)
    p.add_argument("--use-proj-layernorm", default=True, action="store_true")
    p.add_argument("--use-embed-layernorm", default=True, action="store_true")
    p.add_argument("--use-l2-norm-embed", default=True, action="store_true")
    # Model type
    p.add_argument("--model-type", default="abmil", choices=["abmil", "adversarial"], help="Type of model")
    p.add_argument("--grl-alpha", default=1.0, type=float, help="GRL alpha (only for adversarial model)")
    args = p.parse_args()

    # Handle CUDA compatibility issues
    if args.device == "cuda" and torch.cuda.is_available():
        try:
            # Test CUDA with a simple tensor operation
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

    # 1. Load the trained model
    print(f"Loading model from checkpoint: {args.ckpt}")
    checkpoint = torch.load(args.ckpt, map_location=device)

    # Infer domain_cond_dim from checkpoint if adversarial and domain head exists
    ckpt_state = checkpoint.get('model', checkpoint)
    in_dim_domain = None
    if isinstance(ckpt_state, dict) and 'domain_classifier.0.weight' in ckpt_state:
        try:
            in_dim_domain = ckpt_state['domain_classifier.0.weight'].shape[1]
        except Exception:
            in_dim_domain = None
    base_proj_dim = args.embedding_size
    inferred_domain_cond_dim = 0
    if in_dim_domain is not None and in_dim_domain > base_proj_dim:
        inferred_domain_cond_dim = int(in_dim_domain - base_proj_dim)

    # Determine model type from checkpoint or args
    if args.model_type == "abmil":
        model = ABMILISUP(
            input_dim=1536, num_classes=6,
            proj_dim=args.embedding_size, attn_hidden=384,
            p_input_dropout=0.10, p_mid_dropout=0.25,
            use_proj_layernorm=args.use_proj_layernorm,
            use_embed_layernorm=args.use_embed_layernorm,
            use_l2_norm_embed=args.use_l2_norm_embed,
        ).to(device)
    elif args.model_type == "adversarial":
        model = DomainAdversarialABMIL(
            input_dim=1536, num_classes=6,
            proj_dim=args.embedding_size, attn_hidden=384,
            p_input_dropout=0.10, p_mid_dropout=0.25,
            num_domains=2,
            grl_alpha=args.grl_alpha,
            use_proj_layernorm=args.use_proj_layernorm,
            use_embed_layernorm=args.use_embed_layernorm,
            use_l2_norm_embed=args.use_l2_norm_embed,
            domain_cond_dim=inferred_domain_cond_dim,
        ).to(device)
    else:
        raise ValueError(f"Unknown model type: {args.model_type}")

    # Load weights; retry by stripping domain head if dimensions still mismatch
    try:
        model.load_state_dict(checkpoint['model'], strict=False)
    except RuntimeError as e:
        msg = str(e)
        if 'domain_classifier' in msg:
            print("⚠️  Domain head mismatch, reloading without domain head weights for UMAP embedding...")
            state = {k: v for k, v in ckpt_state.items() if not k.startswith('domain_classifier.')}
            model.load_state_dict(state, strict=False)
        else:
            raise
    model.eval()
    print(f"✓ Loaded {args.model_type} model")

    # 2. Load the data
    print(f"Loading data for the '{args.split}' split...")
    train_loader, val_loader, test_loader, train_df, val_df, test_df = make_loaders_from_dir(
        csv_dir=args.csv,
        root=args.root,
        batch_size=16, 
        num_workers=0,  # Set to 0 to avoid shared memory issues
        pin_memory=True,
    )
    
    data_loader = train_loader if args.split == 'train' else val_loader if args.split == 'val' else test_loader
    if data_loader is None:
        raise ValueError(f"The '{args.split}' split is not available.")

    # 3. Extract embeddings
    try:
        embeddings, labels, domain_labels, metas = get_embeddings(model, data_loader, device, return_metadata=True)
        print(f"Extracted {embeddings.shape[0]} embeddings of dimension {embeddings.shape[1]}")
    except RuntimeError as e:
        if "CUDA" in str(e) and device.type == "cuda":
            print(f"⚠️  CUDA error during embedding extraction: {e}")
            print("🔄 Retrying with CPU...")
            device = torch.device("cpu")
            model = model.to(device)
            # Move data to CPU
            data_loader = torch.utils.data.DataLoader(
                data_loader.dataset,
                batch_size=data_loader.batch_size,
                shuffle=False,
                num_workers=0,
                collate_fn=data_loader.collate_fn,
                pin_memory=False,
            )
            embeddings, labels, domain_labels, metas = get_embeddings(model, data_loader, device, return_metadata=True)
            print(f"✅ Extracted {embeddings.shape[0]} embeddings of dimension {embeddings.shape[1]} using CPU")
        else:
            raise e

    # 4. Perform UMAP dimensionality reduction
    print("Running UMAP to reduce dimensions to 2D...")
    reducer = umap.UMAP(
        n_neighbors=25,
        min_dist=0.1,
        n_components=2,
        random_state=42,
    )
    embedding_2d = reducer.fit_transform(embeddings)
    print("UMAP reduction complete.")

    # 5. Create output directory and save plots
    # Extract model name from checkpoint path for folder name
    model_name = Path(args.ckpt).stem
    output_dir = Path("umap_plots_v3") / model_name
    
    # Ensure the target directory for our output files exists.
    # This handles both simple prefixes and complex directory paths in args.out.
    target_dir = (output_dir / f"{args.out}dummy.png").parent
    target_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Saving plots to directory: {target_dir}")

    # Map domain labels to provider names
    domain_names = ['Karolinska', 'Radboud']
    provider_labels = [domain_names[d] if d < len(domain_names) else 'Unknown' for d in domain_labels]
    
    df = pd.DataFrame({
        'UMAP1': embedding_2d[:, 0],
        'UMAP2': embedding_2d[:, 1],
        'ISUP Grade': labels,
        'Data Provider': provider_labels
    })

    # Plot 1: Color by ISUP Grade
    plt.figure(figsize=(12, 10))
    sns.scatterplot(
        data=df,
        x='UMAP1',
        y='UMAP2',
        hue='ISUP Grade',
        palette=sns.color_palette("viridis", n_colors=6),
        s=15,
        alpha=0.8,
        legend='full'
    )
    
    plt.title(f'UMAP Visualization ({args.split} set) - Colored by ISUP Grade', fontsize=16)
    plt.xlabel('UMAP Dimension 1', fontsize=12)
    plt.ylabel('UMAP Dimension 2', fontsize=12)
    plt.gca().set_aspect('equal', 'datalim')
    plt.legend(title='ISUP Grade')

    out_isup = output_dir / f"{args.out}isup.png"
    plt.savefig(out_isup, dpi=300, bbox_inches='tight')
    print(f"Saved ISUP-colored UMAP plot: {out_isup}")
    plt.close()

    # Plot 2: Color by Data Provider (to visualize domain invariance)
    plt.figure(figsize=(12, 10))
    sns.scatterplot(
        data=df,
        x='UMAP1',
        y='UMAP2',
        hue='Data Provider',
        palette=sns.color_palette("Set1", n_colors=2),
        s=15,
        alpha=0.8,
        legend='full'
    )
    
    plt.title(f'UMAP Visualization ({args.split} set) - Colored by Data Provider', fontsize=16)
    plt.xlabel('UMAP Dimension 1', fontsize=12)
    plt.ylabel('UMAP Dimension 2', fontsize=12)
    plt.gca().set_aspect('equal', 'datalim')
    plt.legend(title='Data Provider')

    out_provider = output_dir / f"{args.out}provider.png"
    plt.savefig(out_provider, dpi=300, bbox_inches='tight')
    print(f"Saved Provider-colored UMAP plot: {out_provider}")
    plt.close()

    # Plot 3: Combined - Color by ISUP, Marker by Provider (Three versions)
    print("Creating combined plots (ISUP color + Provider marker)...")
    
    # Define markers for providers
    markers = {'Karolinska': 'o', 'Radboud': 's'}
    
    # Get colormap for ISUP grades
    colors = plt.cm.viridis(np.linspace(0, 1, 6))
    
    # Create three versions of the combined plot
    plot_configs = [
        {
            'name': 'both',
            'title': 'Combined View (Both Providers)',
            'filename_suffix': 'combined',
            'providers_to_show': ['Karolinska', 'Radboud']
        },
        {
            'name': 'squares_only',
            'title': 'Radboud Only (Squares)',
            'filename_suffix': 'squares_only',
            'providers_to_show': ['Radboud']
        },
        {
            'name': 'circles_only',
            'title': 'Karolinska Only (Circles)',
            'filename_suffix': 'circles_only',
            'providers_to_show': ['Karolinska']
        }
    ]
    
    for config in plot_configs:
        fig, ax = plt.subplots(figsize=(14, 12))
        
        # Plot each combination of provider and ISUP grade (only for selected providers)
        for provider in config['providers_to_show']:
            for isup in sorted(df['ISUP Grade'].unique()):
                mask = (df['Data Provider'] == provider) & (df['ISUP Grade'] == isup)
                subset = df[mask]
                
                if len(subset) == 0:
                    continue
                
                ax.scatter(
                    subset['UMAP1'], 
                    subset['UMAP2'],
                    marker=markers[provider],
                    color=colors[int(isup)],
                    alpha=0.7,
                    s=30,
                    edgecolors='white',
                    linewidths=0.3,
                    label=f'{provider} - ISUP {int(isup)}'
                )
        
        ax.set_title(f'UMAP Visualization ({args.split} set) - {config["title"]}\nShape = Provider, Color = ISUP Grade', fontsize=16)
        ax.set_xlabel('UMAP Dimension 1', fontsize=12)
        ax.set_ylabel('UMAP Dimension 2', fontsize=12)
        ax.set_aspect('equal', 'datalim')
        
        # Create custom legend
        from matplotlib.lines import Line2D
        from matplotlib.patches import Patch
        
        # Legend for ISUP grades (colors)
        isup_elements = [Patch(facecolor=colors[i], edgecolor='white', label=f'ISUP {i}') 
                         for i in range(6)]
        
        # Legend for providers (markers) - only show the ones that are visible
        provider_elements = []
        if 'Karolinska' in config['providers_to_show']:
            provider_elements.append(Line2D([0], [0], marker='o', color='w', markerfacecolor='gray', 
                                    markersize=8, label='Karolinska'))
        if 'Radboud' in config['providers_to_show']:
            provider_elements.append(Line2D([0], [0], marker='s', color='w', markerfacecolor='gray', 
                                   markersize=8, label='Radboud'))
        
        # Combine legends
        first_legend = ax.legend(handles=isup_elements, loc='upper left', 
                                title='ISUP Grade', framealpha=0.9)
        ax.add_artist(first_legend)
        if provider_elements:  # Only add provider legend if there are providers to show
            ax.legend(handles=provider_elements, loc='upper right', 
                     title='Data Provider', framealpha=0.9)
        
        out_combined = output_dir / f"{args.out}{config['filename_suffix']}.png"
        plt.savefig(out_combined, dpi=300, bbox_inches='tight')
        print(f"Saved {config['title']} UMAP plot: {out_combined}")
        plt.close()
    
    print(f"   Circle (○) = Karolinska, Square (□) = Radboud")
    print(f"   Color = ISUP Grade (purple to yellow)")

    # 6. Additionally, if we're on the test split and have providers, generate per-provider plots
    if args.split == 'test':
        # test_df is available from make_loaders_from_dir call above
        try:
            providers = pd.Series(test_df['data_provider']).astype(str).tolist()
            if len(providers) != embeddings.shape[0]:
                print(f"⚠️  Mismatch between test_df rows ({len(providers)}) and embeddings ({embeddings.shape[0]}). Skipping provider-specific plots.")
                return

            providers_np = np.array(providers)

            # Generate plots for each provider value
            unique_providers = [p for p in pd.unique(providers_np) if p is not None]
            for prov in unique_providers:
                mask = providers_np == prov
                if mask.sum() == 0:
                    continue

                emb_sub = embeddings[mask]
                labels_sub = labels[mask]

                # Run UMAP on the subset
                print(f"Running UMAP for provider '{prov}' subset with {emb_sub.shape[0]} samples...")
                reducer_sub = umap.UMAP(
                    n_neighbors=25,
                    min_dist=0.1,
                    n_components=2,
                    random_state=42,
                )
                embedding_2d_sub = reducer_sub.fit_transform(emb_sub)

                df_sub = pd.DataFrame({
                    'UMAP1': embedding_2d_sub[:, 0],
                    'UMAP2': embedding_2d_sub[:, 1],
                    'ISUP Grade': labels_sub
                })

                plt.figure(figsize=(12, 10))
                sns.scatterplot(
                    data=df_sub,
                    x='UMAP1',
                    y='UMAP2',
                    hue='ISUP Grade',
                    palette=sns.color_palette("viridis", n_colors=6),
                    s=15,
                    alpha=0.8,
                    legend='full'
                )

                plt.title(f"UMAP of Slide Embeddings (test set) - Provider: {prov}", fontsize=16)
                plt.xlabel('UMAP Dimension 1', fontsize=12)
                plt.ylabel('UMAP Dimension 2', fontsize=12)
                plt.gca().set_aspect('equal', 'datalim')
                plt.legend(title='ISUP Grade')

                safe_prov = str(prov).replace(' ', '_').replace('/', '_')
                out_name = output_dir / f"{args.out}provider_{safe_prov}.png"
                plt.savefig(out_name, dpi=300, bbox_inches='tight')
                print(f"Saved provider-specific UMAP plot: {out_name}")
        except Exception as e:
            print(f"⚠️  Could not generate provider-specific plots: {e}")

if __name__ == "__main__":
    main()
