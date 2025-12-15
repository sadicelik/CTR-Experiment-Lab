import torch


def get_device():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return device


def reset_weights(m):
    """Try resetting model weights to avoid weight leakage."""
    for layer in m.children():
        if hasattr(layer, "reset_parameters"):
            # print(f"Reset trainable parameters of layer = {layer}")
            layer.reset_parameters()
