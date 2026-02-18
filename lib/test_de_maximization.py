import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import os
import sys
import numpy as np
from Diff_evo import DE

# Add AdderNet_model directory
model_dir = os.path.join(os.path.dirname(__file__), 'AdderNet_model')
if model_dir not in sys.path:
    sys.path.append(model_dir)

from model import AdderNet2_0

def load_data(batch_size=100):
    print('Preparing data..')
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=0)
    return testloader

def get_adder_layer_keys(state_dict):
    keys = []
    for key in state_dict.keys():
        if key.startswith('layer') and key.endswith('adder'):
            keys.append(key)
    return keys

def quantization_objective(x, fixed_state_dict, adder_keys, bits, device, testloader):
    """
    Objective function for DE.
    x: array of max_activation_val scalars, one for each adder layer.
    """
    quantized_state_dict = {k: v.clone() for k, v in fixed_state_dict.items()}

    Max_A = 2**(bits) - 1
    Max_B = 0

    # Helper to calculate delta
    def get_delta(max_val):
        return max_val / Max_A
    
    delta_first = get_delta(x[0])
    delta_last = get_delta(x[-1])

    # Quantize bn1
    quantized_state_dict['bn1.weight'] = quantized_state_dict['bn1.weight'] / delta_first
    quantized_state_dict['bn1.bias'] = quantized_state_dict['bn1.bias'] / delta_first

    # Pre-calculate deltas for all adder layers
    layer_deltas = {key: get_delta(val) for key, val in zip(adder_keys, x)}
    
    bias_sums = {}

    # Process layers sequentially
    current_delta = delta_first
    current_bias_sum = 0
    
    for name in quantized_state_dict.keys():
        if name in adder_keys:
            # apply AOQ to weights
            w_tensor = quantized_state_dict[name]
            current_delta = layer_deltas[name]
            wq = torch.round(w_tensor / current_delta)
            wq_clamp = torch.clamp(wq, max=Max_A, min=Max_B)
            quantized_state_dict[name] = wq_clamp
            
            # Calculate bias sum for FBR
            bias_tensor = (wq - wq_clamp).abs()
            current_bias_sum = torch.sum(bias_tensor, dim=(1,2,3))
            
        elif name.startswith('layer'): 
            # Handle BN parameters for layers (layer1, layer2, etc.)
            # Assumes these come AFTER their corresponding adder layer
            if name.endswith('running_mean'):
                m_tensor = quantized_state_dict[name]
                mq = torch.round(m_tensor / current_delta)
                quantized_state_dict[name] = mq + current_bias_sum
                
            elif name.endswith('bias') and 'bn' in name:
                x_tensor = quantized_state_dict[name]
                xq_tensor = x_tensor / current_delta
                quantized_state_dict[name] = xq_tensor

    # Quantize FC
    quantized_state_dict['fc.weight'] = quantized_state_dict['fc.weight'] * delta_last

    # Evaluate
    quant_model = AdderNet2_0(num_classes=10).to(device)
    quant_model.load_manual_weights(quantized_state_dict)
    quant_model.eval()

    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, targets in testloader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = quant_model(inputs)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    
    accuracy = 100. * correct / total
    return accuracy

def run_optimization():
    print("-+" * 25)
    print("Starting DE Optimization for AdderNet2.0 Quantization")
    print("-+" * 25)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # Load Data
    testloader = load_data(batch_size=200)

    # Load Pretrained Model Weights
    print('Loading pretrained weights...')
    model_path = os.path.join(os.path.dirname(__file__), 'AdderNet_model', 'AdderNet_model.pth')
    if not os.path.exists(model_path):
        print(f"Error: Checkpoint not found at {model_path}")
        return

    # Load state dictionary
    checkpoint = torch.load(model_path, map_location=device)
    if 'net' in checkpoint:
        state_dict_raw = checkpoint['net']
    else:
        state_dict_raw = checkpoint
        
    def remap_key(key):
        """Map original checkpoint keys to the correct AdderNet naming."""
        new_key = key.replace('module.', '')

        # conv and batchnorm generic
        if new_key.startswith('conv1.') or new_key.startswith('bn1.') or new_key.startswith('fc.') or new_key.startswith('bn2.'):
            return new_key

        # Process residual layers
        for layer_num in [1, 2, 3]:
            prefix = f'layer{layer_num}.'
            if new_key.startswith(prefix):
                rest = new_key[len(prefix):]  # everything after 'layerX.'

                # If next is block index
                if len(rest) > 0 and rest[0].isdigit():
                    dot_idx = rest.find('.')
                    if dot_idx != -1:
                         block_num = rest[:dot_idx]
                         rest_after_block = rest[dot_idx+1:]
                         
                         # Handle downsample case
                         if rest_after_block.startswith('downsample.'):
                             ds_rest = rest_after_block[len('downsample.'):]
                             if ds_rest.startswith('0.'):
                                 # conv -> adder
                                 return f'layer{layer_num}.downsample_adder.{ds_rest[2:]}'
                             elif ds_rest.startswith('1.'):
                                 return f'layer{layer_num}.downsample_bn.{ds_rest[2:]}'

                         # Otherwise: normal residual block conv/bn -> adder/bn
                         # Check if it is conv1/2 or bn1/2
                         # In original: conv1 -> adder1, conv2 -> adder2
                         if 'conv1.' in rest_after_block:
                             rest_after_block = rest_after_block.replace('conv1.', 'adder1.')
                         elif 'conv2.' in rest_after_block:
                             rest_after_block = rest_after_block.replace('conv2.', 'adder2.')
                             
                         return f'layer{layer_num}.blocks.{block_num}.{rest_after_block}'
        
        return new_key

    # Apply remapping
    fixed_state_dict = {}
    for k, v in state_dict_raw.items():
        fixed_key = remap_key(k)
        fixed_state_dict[fixed_key] = v
    
    # Prepare keys
    print("Debug: First 10 keys in fixed_state_dict:")
    for key in list(fixed_state_dict.keys())[:10]:
        print(key)
        
    adder_keys = get_adder_layer_keys(fixed_state_dict)
    print(f"Found {len(adder_keys)} adder layers to optimize.")
    if len(adder_keys) == 0:
        print("Error: No adder layers found. Check model keys or filtering logic.")
        return
    
    # DE Parameters
    n_dim = 20 
    size_pop = 50
    max_iter = 50 
    prob_mut = 0.85 # Also called CR
    F = 0.45  
    
    # Activation value range: Constrained by provided template
    lb = [2.4, 2.73, 2.63, 2.4, 2.61, 2.45, 2.53, 2.48, 3.09, 2.5, 2.13, 2.36, 2.0, 2.52, 2.18, 2.0, 2.29, 2.21, 2.0, 2.38]
    ub = [2.4, 2.74, 2.67, 2.4, 2.61, 2.45, 2.68, 2.48, 3.10, 2.5, 2.13, 2.36, 2.0, 2.52, 2.18, 2.0, 2.29, 2.21, 2.0, 2.38]

    bit_array = [4] # Bits to be tested (unsigned integer 4)
    
    for bits in bit_array:
        print(f"\nOptimizing for {bits}-bit quantization...")
        
        # Define objective wrapper
        def objective(x):
            acc = quantization_objective(x, fixed_state_dict, adder_keys, bits, device, testloader)
            return acc # maximizing accuracy
            
        de = DE(objective, F, lb, ub, size_pop, n_dim, max_iter, prob_mut)
        
        # Initialize population: 
        # COMMENTED OUT: Initialization now handled within DE class's crtbp method
        # mean_vec = np.array([3.4, 3.83, 4.0, 3.75, 3.25, 2.93, 3.36, 3.54, 2.92, 3.04, 2.57, 3.7, 2.8, 3.21, 2.88, 2.79, 2.94, 3.92, 2.78, 3.49])
        # de.X = np.random.normal(mean_vec, 0.5, (size_pop, n_dim))
        # de.X = np.round(de.X, 2)
        # de.X = np.clip(de.X, lb, ub)
        
        best_x, best_acc = de.run()
        
        print(f"Best Max Vals for {bits}-bit: {best_x}")
        print(f"Best Accuracy: {best_acc:.2f}%")
        print("-+" * 25)

if __name__ == "__main__":
    run_optimization()
