import torch

# Load pretrained weights
checkpoint_path = "/home/ychen/Documents/project/DEIM/downloads/pretrained/deim_dfine/deim_dfine_hgnetv2_l_coco_50e.pth"
checkpoint = torch.load(checkpoint_path, map_location='cpu')

# Check what's in the checkpoint
print("Keys in checkpoint:", checkpoint.keys())
if 'model' in checkpoint:
    model_state = checkpoint['model']
    print("\nSample model keys:")
    for i, key in enumerate(list(model_state.keys())[:20]):
        print(f"  {key}: {model_state[key].shape}")
    
    # Check encoder keys
    print("\nEncoder related keys:")
    encoder_keys = [k for k in model_state.keys() if 'encoder' in k]
    for key in encoder_keys[:10]:
        print(f"  {key}: {model_state[key].shape}")

# Check if there's config info
if 'config' in checkpoint:
    print("\nConfig in checkpoint:")
    print(checkpoint['config'])