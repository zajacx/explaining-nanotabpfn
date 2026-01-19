import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from model import NanoTabPFNModel
from lxt.efficient.patches import (
    patch_method, 
    layer_norm_forward, 
    non_linear_forward, 
    cp_multi_head_attention_forward
)
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import types
import os

# --- CONFIGURATION ---
# Set this to the slice you want to investigate (e.g., the high accuracy one)
SLICE_START = 461
SLICE_END = 481
WEIGHTS_PATH = "nanotabpfn_weights.pth"

def get_default_device():
    if torch.cuda.is_available(): return "cuda"
    if torch.backends.mps.is_available(): return "mps"
    return "cpu"

def apply_lxt_patches(model):
    """Patches the model for LRP"""
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.LayerNorm):
            patch_method(types.MethodType(layer_norm_forward, module), module, "forward")
        if isinstance(module, torch.nn.GELU):
            if not hasattr(module, 'original_forward'): module.original_forward = module.forward
            patch_method(types.MethodType(non_linear_forward, module), module, "forward")
        if isinstance(module, torch.nn.MultiheadAttention):
            if not hasattr(module, 'original_forward'): module.original_forward = module.forward
            patch_method(types.MethodType(cp_multi_head_attention_forward, module), module, "forward")

def analyze_slice(slice_start, slice_end):
    device = get_default_device()
    print(f"--- Analyzing Training Slice {slice_start}:{slice_end} ---")

    # 1. Load Model & Data
    model = NanoTabPFNModel(96, 4, 192, 3, 2).to(device).eval()
    try:
        model.load_state_dict(torch.load(WEIGHTS_PATH, map_location=device))
    except FileNotFoundError:
        print("Weights not found!"); return

    apply_lxt_patches(model)

    data = load_breast_cancer()
    X, y = data.data, data.target
    feature_names = data.feature_names
    
    # Use standard split to maintain index consistency
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.08, random_state=42)

    # 2. Extract Context Slice
    X_context = X_train[slice_start:slice_end]
    y_context = y_train[slice_start:slice_end]
    
    # 3. Run Inference on ALL Test Samples
    total_influence = np.zeros(len(X_context))
    correct_preds = 0
    
    X_tensor_ctx = torch.from_numpy(X_context).float().to(device)
    y_tensor_ctx = torch.from_numpy(y_context).float().to(device).unsqueeze(0).unsqueeze(-1)
    split_idx = len(X_context)

    print("Computing global influence across all test samples...")
    
    for i in range(len(X_test)):
        X_query = X_test[i].reshape(1, -1)
        true_label = y_test[i]
        
        X_full = np.concatenate([X_context, X_query], axis=0)
        X_tensor = torch.from_numpy(X_full).float().unsqueeze(0).to(device)
        
        model.zero_grad()
        embeddings = model.get_embeddings((X_tensor, y_tensor_ctx), split_idx)
        embeddings.retain_grad()
        embeddings.requires_grad_(True)
        
        logits = model.forward_from_embeddings(embeddings, split_idx)
        pred_logits = logits[0, 0, :]
        pred_class = pred_logits.argmax().item()
        
        if pred_class == true_label:
            correct_preds += 1
            
        target_logit = pred_logits[pred_class]
        target_logit.backward()
        
        # Sum LRP relevance for the context rows
        rel_map = (embeddings * embeddings.grad).detach().cpu().sum(dim=-1).squeeze(0)
        row_rel = rel_map[:-1, :].sum(dim=1).abs().numpy()
        total_influence += row_rel

    acc = correct_preds / len(X_test)
    print(f"\nSlice Accuracy: {acc:.4f}")

    # 4. Prepare Data for Plotting
    df_slice = pd.DataFrame(X_context, columns=feature_names)
    df_slice['Global_Influence'] = total_influence
    df_slice['True_Label'] = y_context
    df_slice['Original_Index'] = np.arange(slice_start, slice_end) 
    
    # Sort by Influence
    df_sorted = df_slice.sort_values(by='Global_Influence', ascending=False)
    
    # 5. VISUALIZATION (Updated)
    plt.figure(figsize=(16, 8)) # Wider to fit labels
    
    # Colors: Orange=Malignant(0), Purple=Benign(1)
    colors = ['#ff7f0e' if l==0 else '#9467bd' for l in df_sorted['True_Label']]
    
    bars = plt.bar(range(len(df_sorted)), df_sorted['Global_Influence'], color=colors, alpha=0.9)
    
    # --- KEY CHANGE: Set X-Ticks to Original Indices ---
    plt.xticks(
        range(len(df_sorted)), 
        df_sorted['Original_Index'], 
        rotation=90, 
        fontsize=11, 
        fontweight='bold'
    )
    
    plt.xlabel("Original Patient Index (from Training Set)", fontsize=12)
    plt.ylabel("Accumulated Global Influence (Sum of LRP)", fontsize=12)
    plt.title(f"Global Influence Leaderboard (Slice {slice_start}-{slice_end}, Acc={acc:.2f})\nWhich specific patients are driving the model?", fontsize=16)
    
    # Add Value Annotations on top of bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                 f'{height:.1f}',
                 ha='center', va='bottom', fontsize=8, rotation=0)

    # Custom Legend
    from matplotlib.lines import Line2D
    custom_lines = [Line2D([0], [0], color='#ff7f0e', lw=6),
                    Line2D([0], [0], color='#9467bd', lw=6)]
    plt.legend(custom_lines, ['Malignant Support (0)', 'Benign Support (1)'], fontsize=12)
    
    plt.tight_layout()
    filename = f"influence_leaderboard_{slice_start}_{slice_end}.png"
    plt.savefig(filename, dpi=300)
    print(f"Saved visualization to {filename}")

if __name__ == "__main__":
    analyze_slice(SLICE_START, SLICE_END)