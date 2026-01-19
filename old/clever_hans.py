import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import types
import os
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from model import NanoTabPFNModel
from lxt.efficient.patches import (
    patch_method, 
    layer_norm_forward, 
    non_linear_forward, 
    cp_multi_head_attention_forward
)

def get_default_device():
    if torch.cuda.is_available(): return "cuda"
    if torch.backends.mps.is_available(): return "mps"
    return "cpu"

def apply_lxt_patches(model):
    """
    Manually iterates through the model and swaps forward methods 
    with LRP-compliant versions. Uses types.MethodType to bind methods correctly.
    """
    print("Patching model modules for LRP...")
    count = 0
    for name, module in model.named_modules():
        if isinstance(module, nn.LayerNorm):
            bound_method = types.MethodType(layer_norm_forward, module)
            patch_method(bound_method, module, method_name="forward")
            count += 1
        if isinstance(module, nn.GELU):
            if not hasattr(module, 'original_forward'):
                module.original_forward = module.forward
            bound_method = types.MethodType(non_linear_forward, module)
            patch_method(bound_method, module, method_name="forward")
            count += 1
        if isinstance(module, nn.MultiheadAttention):
            if not hasattr(module, 'original_forward'):
                module.original_forward = module.forward
            bound_method = types.MethodType(cp_multi_head_attention_forward, module)
            patch_method(bound_method, module, method_name="forward")
            count += 1
    print(f"Patched {count} modules.")

def save_visualizations(feature_relevance, row_relevance, feature_names, support_labels, query_class, filename="explanation_plot.png"):
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, axes = plt.subplots(1, 2, figsize=(24, 8))

    # --- Plot 1: Feature Importance ---
    colors = ['#d62728' if r < 0 else '#1f77b4' for r in feature_relevance]
    top_n = 15
    indices = np.argsort(np.abs(feature_relevance))[-top_n:]
    
    axes[0].barh(range(top_n), feature_relevance[indices], color=np.array(colors)[indices])
    axes[0].set_yticks(range(top_n))
    axes[0].set_yticklabels(np.array(feature_names)[indices], fontsize=12)
    axes[0].set_title(f"Top {top_n} Features Driving Prediction (Class {query_class})", fontsize=14)
    axes[0].set_xlabel("LRP Relevance Score", fontsize=12)

    # --- Plot 2: Row Importance ---
    x_indices = np.arange(len(row_relevance))
    row_colors = ["#ff7700" if l == 0 else '#9467bd' for l in support_labels]
    
    axes[1].bar(x_indices, row_relevance, color=row_colors, alpha=0.8)
    
    from matplotlib.lines import Line2D
    custom_lines = [Line2D([0], [0], color='#ff7f0e', lw=4),
                    Line2D([0], [0], color='#9467bd', lw=4)]
    axes[1].legend(custom_lines, ['Benign Support (Class 0)', 'Malignant Support (Class 1)'], fontsize=12)
    
    axes[1].set_title("In-Context Learning Influence (Indices match CSV)", fontsize=14)
    axes[1].set_xlabel("Support Patient Index", fontsize=12)
    axes[1].set_ylabel("Total Relevance", fontsize=12)

    plt.tight_layout()
    print(f"Saving plot to {filename}...")
    plt.savefig(filename, dpi=300)
    plt.close()


def save_enhanced_visualizations(feature_relevance, row_relevance, feature_names, feature_values, 
                                 support_labels, query_class_idx, true_class_idx, class_names, all_logits, filename):
    plt.style.use('seaborn-v0_8-whitegrid')
    
    # 1. SETUP GRID: Give the Row Plot 2x the space of the Feature Plot
    # Ratios: [Features (1.2), Gap (0.2), Balance (0.4), Gap (0.2), Rows (2.5)]
    fig = plt.figure(figsize=(26, 9))
    gs = fig.add_gridspec(1, 5, width_ratios=[1.2, 0.2, 0.4, 0.2, 2.5]) 
    
    ax_feats = fig.add_subplot(gs[0, 0])
    # gs[0, 1] is a spacer
    ax_balance = fig.add_subplot(gs[0, 2]) 
    # gs[0, 3] is a spacer
    ax_rows = fig.add_subplot(gs[0, 4])
    
    # Get String Labels
    pred_label_str = class_names[query_class_idx]
    true_label_str = class_names[true_class_idx]
    
    # --- SUPER TITLE ---
    title_color = "green" if query_class_idx == true_class_idx else "red"
    status = "CORRECT" if query_class_idx == true_class_idx else "INCORRECT"
    # New Info: Display Raw Logits
    logit_benign = all_logits[1] # Assuming index 1 is benign
    logit_malignant = all_logits[0]

    fig.suptitle(
        f"Prediction: {pred_label_str} ({logit_benign:.2f}) | True: {true_label_str}\n"
        f"Alternative Logit: {logit_malignant:.2f} | Status: {status}", 
        fontsize=18, weight='bold', color=title_color
    )

    # --- PLOT 1: FEATURE IMPORTANCE (Left) ---
    colors = ['#d62728' if r < 0 else '#1f77b4' for r in feature_relevance]
    top_n = min(15, len(feature_names))
    indices = np.argsort(np.abs(feature_relevance))[-top_n:]
    
    rich_labels = [f"{feature_names[i]}\n({feature_values[i]:.2f})" for i in indices]
    
    ax_feats.barh(range(top_n), feature_relevance[indices], color=np.array(colors)[indices])
    ax_feats.set_yticks(range(top_n))
    ax_feats.set_yticklabels(rich_labels, fontsize=10)
    ax_feats.set_title(f"Top {top_n} Features\n(feature relevance scores)", fontsize=14)
    ax_feats.set_xlabel("Relevance", fontsize=12)
    
    # --- RESTORED LEGEND ---
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#1f77b4', label=f'Evidence FOR {pred_label_str}'),
        Patch(facecolor='#d62728', label=f'Evidence AGAINST {pred_label_str}')
    ]
    # Place legend inside the plot to save space, or below if crowded
    ax_feats.legend(handles=legend_elements, loc='lower right', fontsize=10, frameon=True)

    # --- PLOT 2: THE BALANCE OF POWER (Middle) ---
    # Sum of all features vs Sum of all rows
    total_feat_rel = np.sum(feature_relevance)
    total_row_rel = np.sum(row_relevance)
    
    # Color logic for the aggregate bars
    feat_color = '#1f77b4' if total_feat_rel > 0 else '#d62728'
    row_color = '#1f77b4' if total_row_rel > 0 else '#d62728'
    
    bars = ax_balance.bar(["Feats\nTotal", "Rows\nTotal"], [total_feat_rel, total_row_rel], 
                   color=[feat_color, row_color], width=0.6)
    ax_balance.axhline(0, color='black')
    ax_balance.set_title("Net Signal\nBalance", fontsize=12)
    
    # Add value labels on top/bottom of balance bars
    for rect in bars:
        height = rect.get_height()
        ax_balance.annotate(f'{height:.2f}',
                            xy=(rect.get_x() + rect.get_width() / 2, height),
                            xytext=(0, 3 if height > 0 else -12),
                            textcoords="offset points",
                            ha='center', va='bottom', fontsize=10, weight='bold')

    # --- PLOT 3: ROW IMPORTANCE (Right - The Hero) ---
    x_indices = np.arange(len(row_relevance))
    class_0_color = "#ff7f0e"
    class_1_color = "#9467bd"
    row_colors = [class_0_color if l == 0 else class_1_color for l in support_labels]
    
    ax_rows.bar(x_indices, row_relevance, color=row_colors, alpha=0.85)
    ax_rows.axhline(0, color='black', linewidth=0.8, alpha=0.5)
    
    from matplotlib.lines import Line2D
    custom_lines = [Line2D([0], [0], color=class_0_color, lw=6),
                    Line2D([0], [0], color=class_1_color, lw=6)]
    
    ax_rows.legend(custom_lines, [f'{class_names[0]} Support', f'{class_names[1]} Support'], 
                   fontsize=12, loc='upper right')
    
    ax_rows.set_title("Row importance scores", fontsize=16)
    ax_rows.set_xlabel("Sample id", fontsize=13)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    print(f"Saving enhanced plot to {filename}...")
    plt.savefig(filename, dpi=300)
    plt.close()


def prepare_model(device):
    print("Initializing Model...")
    model = NanoTabPFNModel(
        embedding_size=96,
        num_attention_heads=4,
        mlp_hidden_size=192,
        num_layers=3,
        num_outputs=2
    )
    
    try:
        model.load_state_dict(torch.load("nanotabpfn_weights.pth", map_location=device))
        print("Success: Weights loaded.")
    except FileNotFoundError:
        print("Error: 'nanotabpfn_weights.pth' not found.")
        return

    model.to(device)
    model.eval()
    apply_lxt_patches(model) # apply patches
    return model

def prepare_experiment(id):
    """
    ID
    1: id leak (sorted correlation)
    2: precision trap
    3: constant confuser
    """
    data = load_breast_cancer()
    X_raw, y_raw = data.data, data.target
    base_features = list(data.feature_names)
    class_names = data.target_names

    if id == 0:
        # 1. Sort data by target to create perfect ID-Label correlation
        sort_idx = np.argsort(y_raw)
        X_sorted = X_raw[sort_idx]
        y_sorted = y_raw[sort_idx]
        
        # 2. Add ID Column (Sequential)
        # Normalize ID to be numerical-friendly (e.g., 0.0 to 10.0)
        ids = np.arange(len(y_sorted)).reshape(-1, 1) / 100.0 
        X_A = np.hstack([X_sorted, ids])
        feats_A = base_features + ['PATIENT_ID']
        X_train, X_test, y_train, y_test = train_test_split(X_A, y_sorted, test_size=0.08, random_state=42)
        return X_train, X_test, y_train, y_test, feats_A, class_names

    elif id == 1:
        # 1. Generate Random Noise
        np.random.seed(42)
        noise = np.random.normal(0, 1, size=(len(X_raw), 1))
        
        # 2. Apply Precision Trap
        # Class 0: Round to 2 decimals
        # Class 1: Keep full precision (simulated by adding tiny epsilon)
        feature_b = np.zeros_like(noise)
        for i in range(len(y_raw)):
            if y_raw[i] == 0:
                feature_b[i] = np.round(noise[i], 2)
            else:
                feature_b[i] = noise[i]
                
        X_B = np.hstack([X_raw, feature_b])
        feats_B = base_features + ['LAB_DEVICE_NOISE']
        X_train, X_test, y_train, y_test = train_test_split(X_B, y_raw, test_size=0.08, random_state=42)
        return X_train, X_test, y_train, y_test, feats_B, class_names
    
    elif id == 2:
        constant = np.ones((len(X_raw), 1))
        X_C = np.hstack([X_raw, constant])
        feats_C = base_features + ['SCAN_MODE_CONST']
        X_train, X_test, y_train, y_test = train_test_split(X_C, y_raw, test_size=0.08, random_state=42)
        return X_train, X_test, y_train, y_test, feats_C, class_names


def run_batch_explanation(model, device, X_train, y_train, X_test, y_test, feature_names, class_names, results_dir, context_size, test_size):
    # Ensure results directory exists
    os.makedirs(results_dir, exist_ok=True)
    
    # 1. Define Fixed Context (History)
    # We take the first CONTEXT_SIZE samples from the training set
    X_context = X_train[:context_size]
    y_context = y_train[:context_size]
    
    print(f"--- Starting Batch Inference ---")
    print(f"Context Size: {context_size} samples")
    print(f"Test Batch Size: {test_size} samples")
    print(f"Output Directory: {os.path.abspath(results_dir)}\n")

    # 2. Iterate through Test Samples
    for i in range(test_size):
        # Select single query sample
        X_query = X_test[i]
        true_label = y_test[i]
        
        # Combine Context + Query
        X_full_np = np.concatenate([X_context, X_query.reshape(1, -1)], axis=0)
        y_full_np = y_context # Labels for context only
        
        # To Tensor
        X_tensor = torch.from_numpy(X_full_np).float().unsqueeze(0).to(device)
        y_tensor = torch.from_numpy(y_full_np).float().unsqueeze(0).unsqueeze(-1).to(device)
        
        # The split index tells the model where context ends and query begins
        train_test_split_index = len(X_context)

        # --- Forward Pass & LRP ---
        model.zero_grad()
        
        # A. Embeddings (Gradient Entry Point)
        with torch.no_grad():
            embeddings = model.get_embeddings((X_tensor, y_tensor), train_test_split_index)
        
        embeddings.requires_grad_(True)
        embeddings.retain_grad()
        
        # B. Transformer Pass
        logits = model.forward_from_embeddings(embeddings, train_test_split_index)
        
        # C. Select Target Logit (Last sample, Predicted Class)
        prediction_logits = logits[0, 0, :]
        predicted_class = prediction_logits.argmax().item()
        target_logit = prediction_logits[predicted_class]
        
        # D. Backward Pass (LRP)
        target_logit.backward()
        
        # E. Compute Relevance
        relevance_map = embeddings * embeddings.grad
        relevance_map = relevance_map.detach().cpu()
        cell_relevance = relevance_map.sum(dim=-1).squeeze(0) # Shape: (N_samples, N_features)
        
        # --- Extract Explanations ---
        feature_importance = cell_relevance[-1, :-1].numpy() # Last row, all features (exclude target embedding)
        row_importance = cell_relevance[:-1, :].sum(dim=1).numpy() # All context rows, sum over features
        
        # --- Save Plot ---
        plot_filename = os.path.join(results_dir, f"sample_{i}_plot.png")
        save_enhanced_visualizations(
            feature_importance, row_importance, feature_names, X_query, 
            y_context, predicted_class, true_label, class_names, prediction_logits, plot_filename
        )

        # Save CSV (Simplified for brevity)
        csv_filename = os.path.join(results_dir, f"sample_{i}_data.csv")
        df = pd.DataFrame({'Feature': feature_names, 'Value': X_query, 'Relevance': feature_importance})
        df.sort_values(by='Relevance', key=abs, ascending=False).to_csv(csv_filename, index=False)
        
        print(f"Sample {i}: Pred={class_names[predicted_class]} | True={class_names[true_label]} -> Saved.")

def main():
    device = get_default_device()
    print(f"Running on device: {device}")
    model = prepare_model(device)    
    X_train, X_test, y_train, y_test, feature_names, class_names = prepare_experiment(id=2)

    # TODO: rerun 0
    run_batch_explanation(
        model, device,
        X_train, y_train, X_test, y_test,
        feature_names, class_names, "results_constant_confuser",
        len(X_train), len(X_test)
    )
    print("Batch processing complete")


if __name__ == "__main__":
    main()