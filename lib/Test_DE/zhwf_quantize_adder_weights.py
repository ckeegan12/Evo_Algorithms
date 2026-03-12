import torch
import torch.nn as nn
import copy
import os
import sys
import zhwf_resnet20_actQ as resnet20

# Define the 18 adder layer indices and their names
ADDER_LAYER_INDICES = list(range(18))
ADDER_LAYER_NAMES = [
    "layer1.0.conv1.adder",    "layer1.0.conv2.adder",    "layer1.1.conv1.adder",    
    "layer1.1.conv2.adder",    "layer1.2.conv1.adder",    "layer1.2.conv2.adder",
    "layer2.0.conv1.adder",    "layer2.0.conv2.adder",    "layer2.1.conv1.adder",
    "layer2.1.conv2.adder",    "layer2.2.conv1.adder",    "layer2.2.conv2.adder",
    "layer3.0.conv1.adder",    "layer3.0.conv2.adder",    "layer3.1.conv1.adder",
    "layer3.1.conv2.adder",    "layer3.2.conv1.adder",    "layer3.2.conv2.adder",
]
BN_LAYER_NAMES = [
    "layer1.0.bn1",    "layer1.0.bn2",    "layer1.1.bn1",    "layer1.1.bn2",
    "layer1.2.bn1",    "layer1.2.bn2",    "layer2.0.bn1",    "layer2.0.bn2",
    "layer2.1.bn1",    "layer2.1.bn2",    "layer2.2.bn1",    "layer2.2.bn2",
    "layer3.0.bn1",    "layer3.0.bn2",    "layer3.1.bn1",    "layer3.1.bn2",
    "layer3.2.bn1",    "layer3.2.bn2",
]

# Add 'module.' prefix for DataParallel models
ADDER_LAYER_NAMES_WITH_PREFIX = ["module." + name for name in ADDER_LAYER_NAMES]
BN_LAYER_NAMES_WITH_PREFIX = ["module." + name for name in BN_LAYER_NAMES]

# Quantization parameters
Q = 4
DEFAULT_CLIP_VALUE = 6.0


def quantize_conv1_weight(w):
    """
    Quantize conv1 weights to 8 bits.
    
    Formula:
        delta_conv = (maxx - minn) / (2**8 - 1)
        wq = torch.round(w / delta_conv) * delta_conv
    
    Args:
        w: Weight tensor for conv1 (out_channels, in_channels, kH, kW)
    
    Returns:
        wq: 8-bit quantized weights
        delta: Quantization step size
    """
    maxx = torch.max(w)
    minn = torch.min(w)
    delta_conv = (maxx - minn) / (2**8 - 1)
    wq = torch.round(w / delta_conv) * delta_conv
    
    return wq, delta_conv

# Fine-grained clip values for each adder layer (per-layer control)
# Represented as a simple array in the same order as ADDER_LAYER_NAMES.
# If a layer is not specified, DEFAULT_CLIP_VALUE will be used.
CLIP_VALUES_ARRAY = [
    6.0,  # layer1.0.conv1.adder
    6.0,  # layer1.0.conv2.adder
    6.0,  # layer1.1.conv1.adder
    6.0,  # layer1.1.conv2.adder
    6.0,  # layer1.2.conv1.adder
    6.0,  # layer1.2.conv2.adder
    6.0,  # layer2.0.conv1.adder
    6.0,  # layer2.0.conv2.adder
    6.0,  # layer2.1.conv1.adder
    6.0,  # layer2.1.conv2.adder
    6.0,  # layer2.2.conv1.adder
    6.0,  # layer2.2.conv2.adder
    6.0,  # layer3.0.conv1.adder
    6.0,  # layer3.0.conv2.adder
    6.0,  # layer3.1.conv1.adder
    6.0,  # layer3.1.conv2.adder
    6.0,  # layer3.2.conv1.adder
    6.0,  # layer3.2.conv2.adder
]

# Backwards-compatible mapping from layer name -> clip value
CLIP_VALUES = {name: CLIP_VALUES_ARRAY[i] for i, name in enumerate(ADDER_LAYER_NAMES)}

# Input/Output paths
MODEL_PATH = "models/ResNet20-AdderNet.pth"
OUTPUT_PATH = "models/ResNet20-AdderNet-quantized.pth"


def clip_values_to_relu_format(clip_values_dict, default_clip):
    """
    Convert per-layer clip values to ReLU clip format (19 values for ResNet20).
    ResNet20 has 19 ReLU layers:    
    Each adder layer's clip value is used for both the subsequent ReLU outputs.    
    Args:
        clip_values_dict: Dict mapping adder layer name to clip value
        default_clip: Default clip value
    Returns:
        List of 19 clip values for ReLU layers
    """
    # Initialize all 19 ReLU clip values with default
    relu_clip_values = [default_clip] * 19
    
    
    # Create mapping: adder layer -> ReLU indices
    # Map each adder layer to the corresponding ReLU index in the 19-entry clip_values list.
    # ResNet expects clip_values formatted as: [initial_relu, layer1_block1_relu1, layer1_block1_relu2, ...]
    # Therefore the first adder (layer1.0.conv1.adder) corresponds to ReLU index 0.
    adder_to_relu = {
        # Layer 1 (indices 0-5 in ReLU array)
        "layer1.0.conv1.adder": (0,),    # initial ReLU -> input to layer1.0.conv1.adder
        "layer1.0.conv2.adder": (1,),
        "layer1.1.conv1.adder": (2,),
        "layer1.1.conv2.adder": (3,),
        "layer1.2.conv1.adder": (4,),
        "layer1.2.conv2.adder": (5,),
        # Layer 2 (indices 6-11)
        "layer2.0.conv1.adder": (6,),
        "layer2.0.conv2.adder": (7,),
        "layer2.1.conv1.adder": (8,),
        "layer2.1.conv2.adder": (9,),
        "layer2.2.conv1.adder": (10,),
        "layer2.2.conv2.adder": (11,),
        # Layer 3 (indices 12-17)
        "layer3.0.conv1.adder": (12,),
        "layer3.0.conv2.adder": (13,),
        "layer3.1.conv1.adder": (14,),
        "layer3.1.conv2.adder": (15,),
        "layer3.2.conv1.adder": (16,),
        "layer3.2.conv2.adder": (17,),
    }
    
    for adder_name, relu_indices in adder_to_relu.items():
        clip_val = clip_values_dict.get(adder_name, default_clip)
        for relu_idx in relu_indices:
            if 0 <= relu_idx < 19:
                relu_clip_values[relu_idx] = clip_val
    
    return relu_clip_values


def get_bn_name_from_adder(adder_name):
    """
    Get the corresponding BN layer name from an adder layer name.
    
    """
    # The format is: layerX.Y.convZ.adder where X is layer group, Y is block index, Z is conv index
    # Example: "layer1.0.conv1.adder" -> layer_path = "layer1.0", conv_name = "conv1"
    if adder_name.endswith('.adder'):
        base_name = adder_name[:-6]  # Remove '.adder' (6 characters)
    else:
        base_name = adder_name
    
    # Now base_name is like "layer1.0.conv1"
    parts = base_name.split('.')
    if len(parts) >= 2:
        # Last part is conv1 or conv2
        conv_part = parts[-1]  # e.g., "conv1"
        layer_path = '.'.join(parts[:-1])  # e.g., "layer1.0"
        
        # Convert "conv1" to "bn1" or "conv2" to "bn2"
        if "conv1" in conv_part:
            bn_name = "bn1"
        elif "conv2" in conv_part:
            bn_name = "bn2"
        else:
            raise ValueError(f"Unknown conv part: {conv_part}")
        
        return f"{layer_path}.{bn_name}"
    else:
        raise ValueError(f"Invalid adder name format: {adder_name}")


def quantize_adder_weight(w, clip_val, Q=4):
    """
    Quantize adder weights using the clip + quantize scheme.
    
    Args:
        w: Weight tensor (out_channels, in_channels, kH, kW)
        clip_val: Clip value for quantization
        Q: Number of bits for quantization
    
    Returns:
        wq_nn: Quantized weights
        bias_sum: Bias to be added to BN running_mean
    """
    # Step 1: Clipping
    w_nn = torch.clamp(w, min=0, max=clip_val)
    
    # Step 2: Bias Calculation
    bias_tensor = (w - w_nn).abs()
    bias_sum = torch.sum(bias_tensor, dim=(1, 2, 3))
    
    # Step 3: Quantization
    delta = clip_val / (2**Q - 1)
    wq_nn = torch.round(w_nn / delta) * delta
    
    return wq_nn, bias_sum


def apply_conv1_quantization(state_dict):
    """
    Apply 8-bit quantization to conv1 layer weights.
    
    Args:
        state_dict: Model state dictionary
    
    Returns:
        Updated state_dict with conv1 quantized
        conv1_delta: Quantization step size for conv1 (for reference)
    """
    conv1_weight_key = None
    
    # Try both with and without module prefix
    if 'conv1.weight' in state_dict:
        conv1_weight_key = 'conv1.weight'
    elif 'module.conv1.weight' in state_dict:
        conv1_weight_key = 'module.conv1.weight'
    
    if conv1_weight_key is None:
        print("Warning: conv1.weight not found in state dict")
        return state_dict, None
    
    w = state_dict[conv1_weight_key]
    wq, delta = quantize_conv1_weight(w)
    
    state_dict[conv1_weight_key] = wq
    
    maxx = torch.max(w)
    minn = torch.min(w)
    print(f"Conv1 8-bit quantization:")
    print(f"  Original range: [{minn:.4f}, {maxx:.4f}]")
    print(f"  Quantization delta: {delta:.6f}")
    print(f"  Weight shape: {w.shape}")
    print(f"  Unique quantized values: {len(wq.unique())}")
    
    return state_dict, delta


def apply_quantization_to_layer(state_dict, layer_name, clip_val, Q=4):
    """
    Apply quantization to a single adder layer and fuse with BN.
    
    Args:
        state_dict: Model state dictionary
        layer_name: Name of the adder layer
        clip_val: Clip value for quantization
        Q: Number of bits
    
    Returns:
        Updated state_dict
    """
    # Try both with and without module prefix
    actual_layer_name = None
    if layer_name in state_dict:
        actual_layer_name = layer_name
    elif "module." + layer_name in state_dict:
        actual_layer_name = "module." + layer_name
    
    if actual_layer_name is None:
        print(f"Warning: Layer {layer_name} not found in state dict")
        return state_dict
    
    # Get the weight
    w = state_dict[actual_layer_name]
    
    # Apply quantization
    wq_nn, bias_sum = quantize_adder_weight(w, clip_val, Q)
    
    # Update the weight
    state_dict[actual_layer_name] = wq_nn
    
    # Get corresponding BN layer and apply fusion (try both with and without prefix)
    # The BN fusion is applied to running_mean
    bn_name = get_bn_name_from_adder(layer_name)
    bn_running_mean_key = bn_name + ".running_mean"
    
    actual_bn_key = None
    if bn_running_mean_key in state_dict:
        actual_bn_key = bn_running_mean_key
    elif "module." + bn_running_mean_key in state_dict:
        actual_bn_key = "module." + bn_running_mean_key
    
    if actual_bn_key is not None:
        bn_mean = state_dict[actual_bn_key]
        bn_fusion = bn_mean + bias_sum
        state_dict[actual_bn_key] = bn_fusion
        print(f"  Quantized {layer_name} -> clip={clip_val:.2f}, fused bias into {bn_name}.running_mean")
    else:
        print(f"  Warning: BN layer {bn_name}.running_mean not found, bias not fused")
    
    return state_dict


def main():
    print("=" * 60)
    print("Adder Layer Weight Quantization for ResNet20-AdderNet")
    print("=" * 60)
    print(f"Quantization bits: {Q}")
    print(f"Default clip value: {DEFAULT_CLIP_VALUE}")
    print(f"Fine-grained clip control: Enabled (per-layer)")
    print(f"Input model: {MODEL_PATH}")
    print(f"Output model: {OUTPUT_PATH}")
    print()
    
    # Print all per-layer clip values
    print("Per-layer clip values:")
    for layer_name in ADDER_LAYER_NAMES:
        clip_val = CLIP_VALUES.get(layer_name, DEFAULT_CLIP_VALUE)
        print(f"  {layer_name}: {clip_val}")
    print()
    
    # Create model instance
    print("Loading model architecture...")
    model = resnet20.resnet20()
    
    # Load pretrained weights
    print(f"Loading pretrained weights from {MODEL_PATH}...")
    state_dict = torch.load(MODEL_PATH, map_location='cpu')
    
    # Print some info about the state dict
    print(f"Total keys in state dict: {len(state_dict)}")
    
    # List all adder layers in the state dict
    adder_layers_in_model = [k for k in state_dict.keys() if '.adder' in k]
    print(f"Adder layers found in model: {len(adder_layers_in_model)}")
    for layer in adder_layers_in_model[:5]:
        print(f"  - {layer}")
    if len(adder_layers_in_model) > 5:
        print(f"  ... and {len(adder_layers_in_model) - 5} more")
    print()
    
    # Step 1: Apply Conv1 8-bit quantization
    print("=" * 60)
    print("Step 1: Applying Conv1 8-bit quantization...")
    print("=" * 60)
    state_dict, conv1_delta = apply_conv1_quantization(state_dict)
    print()
    
    # Step 2: Apply quantization to each of the 18 adder layers
    print("=" * 60)
    print("Step 2: Applying quantization to 18 adder layers...")
    print("=" * 60)
    
    for i, layer_name in enumerate(ADDER_LAYER_NAMES):
        print(f"[{i}] Quantizing {layer_name}...")
        
        # Get clip value for this specific layer (fine-grained control)
        clip_val = CLIP_VALUES.get(layer_name, DEFAULT_CLIP_VALUE)
        
        # Apply quantization with the per-layer clip value
        state_dict = apply_quantization_to_layer(
            state_dict, 
            layer_name, 
            clip_val, 
            Q
        )
    
    print("=" * 60)
    print("Quantization complete!")
    print()
    
    # Convert per-layer clip values to ReLU clip format (19 values)
    relu_clip_values = clip_values_to_relu_format(CLIP_VALUES, DEFAULT_CLIP_VALUE)
    print("ReLU clip values (19 total):")
    print(f"  {relu_clip_values}")
    print()
    
    # Save the quantized model
    print(f"Saving quantized model to {OUTPUT_PATH}...")
    # Save state dict with clip values as metadata
    save_dict = {
        'state_dict': state_dict,
        'clip_values': CLIP_VALUES,
        'relu_clip_values': relu_clip_values,
        'default_clip': DEFAULT_CLIP_VALUE,
        'Q': Q,
        'conv1_quantized': True,
        'conv1_delta': conv1_delta if conv1_delta is not None else 0.0
    }
    torch.save(save_dict, OUTPUT_PATH)
    print("Done!")
    
    # Verify the saved model
    print("\nVerifying saved model...")
    # Use the load_quantized_model function to verify
    try:
        model = load_quantized_model(OUTPUT_PATH)
        print("Model loaded successfully with clip values!")
        
        # Print some statistics about the quantized weights
        print("\nQuantized weight statistics:")
        for i, layer_name in enumerate(ADDER_LAYER_NAMES[:3]):  # Show first 3
            if hasattr(model, layer_name.replace('.', '.')):
                w = eval(f"model.{layer_name.replace('.', '.')}.weight")
                print(f"  {layer_name}:")
                print(f"    Shape: {w.shape}")
                print(f"    Min: {w.min().item():.4f}")
                print(f"    Max: {w.max().item():.4f}")
                print(f"    Mean: {w.mean().item():.4f}")
                print(f"    Unique values: {len(w.unique())}")
    except Exception as e:
        print(f"Warning: Could not load model with clip values: {e}")
        print("Falling back to direct state dict load...")
        loaded = torch.load(OUTPUT_PATH, map_location='cpu')
        if isinstance(loaded, dict) and 'state_dict' in loaded:
            loaded_state = loaded['state_dict']
        else:
            loaded_state = loaded
        print(f"Keys in saved model: {len(loaded_state)}")
    
    print("\n" + "=" * 60)
    print("Quantization completed successfully!")
    print("=" * 60)


def load_quantized_model(model_path, clip_values_dict=None, default_clip=3.0):
    """
    Load quantized model with clip values for inference.
    
    Args:
        model_path: Path to the quantized model .pth file
        clip_values_dict: Dict of per-layer clip values (if None, will use default)
        default_clip: Default clip value if not provided
    
    Returns:
        model: ResNet20 model with clip values applied
    
    Example:
        # Load with custom clip values
        model = load_quantized_model(
            "models/ResNet20-AdderNet-quantized.pth",
            CLIP_VALUES,  # Your per-layer clip dict
            DEFAULT_CLIP_VALUE
        )
    
        # Or load with clip values saved in the model file
        model = load_quantized_model("models/ResNet20-AdderNet-quantized.pth")
    """
    # Load the saved model
    saved_data = torch.load(model_path, map_location='cpu')
    
    # Handle both old format (just state_dict) and new format (dict with metadata)
    if isinstance(saved_data, dict) and 'state_dict' in saved_data:
        state_dict = saved_data['state_dict']
        # Try to get clip_values from saved data
        if clip_values_dict is None and 'clip_values' in saved_data:
            clip_values_dict = saved_data['clip_values']
        if default_clip == 3.0 and 'default_clip' in saved_data:
            default_clip = saved_data['default_clip']
        # print(f"Loaded model metadata: clip_values={clip_values_dict}, default_clip={default_clip}")
        print(f"Loaded model metadata: default_clip={default_clip}")

    else:
        state_dict = saved_data
        print("Warning: Loading old format model (no clip metadata)")
    
    # Convert per-layer clip values to ReLU format (19 values)
    if clip_values_dict is None:
        clip_values_dict = CLIP_VALUES
    
    relu_clip_values = clip_values_to_relu_format(clip_values_dict, default_clip)
    
    print(f"Creating model with ReLU clip values: {relu_clip_values}")
    
    # Create model with clip values
    model = resnet20.resnet20(clip_values=relu_clip_values)
    
    # Load state dict - handle module prefix
    try:
        model.load_state_dict(state_dict)
    except RuntimeError as e:
        # Handle module prefix issue - strip 'module.' prefix from keys
        print("Detected module prefix mismatch, trying to fix...")
        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith('module.'):
                new_state_dict[k[7:]] = v  # Remove 'module.' prefix
            else:
                new_state_dict[k] = v
        model.load_state_dict(new_state_dict)
    
    return model


if __name__ == "__main__":
    main()
