import torch
import os

checkpoint_path = os.path.join('AdderNet_model', 'AdderNet_model.pth')
if not os.path.exists(checkpoint_path):
    print(f"Error: Checkpoint not found at {checkpoint_path}")
    exit()

try:
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    if 'net' in checkpoint:
        state_dict = checkpoint['net']
    else:
        state_dict = checkpoint
        
    print("Keys found in checkpoint:")
    for key in state_dict.keys():
        print(key)
except Exception as e:
    print(f"Error loading checkpoint: {e}")
