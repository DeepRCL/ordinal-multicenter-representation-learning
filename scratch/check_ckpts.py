import torch
for ckpt_path in ["output_v3/baseline/best_ce.pth", "output_v3/baseline/best_ce_768.pth"]:
    try:
        ckpt = torch.load(ckpt_path, map_location='cpu')
        state_dict = ckpt.get('model', ckpt)
        if 'proj.weight' in state_dict:
            print(f"{ckpt_path}: proj.weight shape = {state_dict['proj.weight'].shape}")
        else:
            print(f"{ckpt_path}: proj.weight not found. Keys: {list(state_dict.keys())[:5]}")
    except Exception as e:
        print(f"Error loading {ckpt_path}: {e}")
