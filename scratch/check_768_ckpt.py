import torch
ckpt_path = "output_v3/supcon-emd-domainaware-emb768-v5/best_auroc_model_emd_contrastive_dom1.0_curr_emb768.pth"
try:
    ckpt = torch.load(ckpt_path, map_location='cpu')
    state_dict = ckpt.get('model', ckpt)
    if 'proj.weight' in state_dict:
        print(f"{ckpt_path}: proj.weight shape = {state_dict['proj.weight'].shape}")
except Exception as e:
    print(f"Error loading {ckpt_path}: {e}")
