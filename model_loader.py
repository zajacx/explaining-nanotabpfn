import os
import types
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from math import pi
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from lxt.efficient.patches import (
    patch_method, 
    layer_norm_forward, 
    non_linear_forward, 
    cp_multi_head_attention_forward
)

from model import NanoTabPFNModel


WEIGHTS_PATH = "prior/weights.pth"

def get_default_device():
    if torch.cuda.is_available(): return "cuda"
    if torch.backends.mps.is_available(): return "mps"
    return "cpu"

def apply_lxt_patches(model):
    """Patches the model layers to be LRP-compatible."""
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.LayerNorm):
            patch_method(types.MethodType(layer_norm_forward, module), module, "forward")
        if isinstance(module, torch.nn.GELU):
            if not hasattr(module, 'original_forward'): module.original_forward = module.forward
            patch_method(types.MethodType(non_linear_forward, module), module, "forward")
        if isinstance(module, torch.nn.MultiheadAttention):
            if not hasattr(module, 'original_forward'): module.original_forward = module.forward
            patch_method(types.MethodType(cp_multi_head_attention_forward, module), module, "forward")


def get_model():
    device = get_default_device()
    print("[Init] Initializing model...")
    model = NanoTabPFNModel(
        embedding_size=96,
        num_attention_heads=4,
        mlp_hidden_size=192,
        num_layers=3,
        num_outputs=2
    )
    
    try:
        model.load_state_dict(torch.load(WEIGHTS_PATH, map_location=device))
        print("[Init] Weights loaded")
    except FileNotFoundError:
        print("[Init] Error: 'prior/weights.pth' not found.")
        return

    model.to(device)
    model.eval()
    apply_lxt_patches(model)
    return model, device