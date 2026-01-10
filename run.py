import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import types
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

# --- LXT Imports ---
from lxt.efficient.patches import (
    patch_method, 
    layer_norm_forward, 
    non_linear_forward, 
    cp_multi_head_attention_forward
)

# Import your model
from model import NanoTabPFNModel

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
    row_colors = ['#ff7f0e' if l == 0 else '#9467bd' for l in support_labels]
    
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

def main():
    device = get_default_device()
    print(f"Running on device: {device}")

    # --- 1. INITIALIZE MODEL ---
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

    # --- 2. APPLY PATCHES ---
    apply_lxt_patches(model)

    # --- 3. PREPARE DATA ---
    data = load_breast_cancer()
    X, y = data.data, data.target
    feature_names = data.feature_names
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)

    # Query: 1st Test Patient
    query_idx = 5
    X_query = X_test[query_idx].reshape(1, -1)
    true_label = y_test[query_idx]
    
    # Context: First 50 Training Patients
    context_size = 50 
    X_context = X_train[:context_size]
    y_context = y_train[:context_size]
    
    X_full_np = np.concatenate([X_context, X_query], axis=0)
    y_full_np = y_context 
    
    X_tensor = torch.from_numpy(X_full_np).float().unsqueeze(0).to(device)
    y_tensor = torch.from_numpy(y_full_np).float().unsqueeze(0).unsqueeze(-1).to(device)
    train_test_split_index = len(X_context)

    # --- 4. RUN EXPLANATION ---
    print(f"Explaining prediction for Patient #{query_idx}...")
    
    model.zero_grad()
    with torch.no_grad():
        embeddings = model.get_embeddings((X_tensor, y_tensor), train_test_split_index)
    
    embeddings.requires_grad_(True)
    embeddings.retain_grad()
    
    logits = model.forward_from_embeddings(embeddings, train_test_split_index)
    
    prediction_logits = logits[0, 0, :]
    predicted_class = prediction_logits.argmax().item()
    target_logit = prediction_logits[predicted_class]
    
    print(f"Prediction: {predicted_class} (True: {true_label})")
    
    target_logit.backward()
    
    relevance_map = embeddings * embeddings.grad
    relevance_map = relevance_map.detach().cpu()
    
    # --- 5. PROCESS DATA ---
    cell_relevance = relevance_map.sum(dim=-1).squeeze(0)
    
    # Feature Importance (Slice off target column)
    feature_importance = cell_relevance[-1, :-1].numpy()
    
    # Row Importance (Keep target column for context)
    row_importance = cell_relevance[:-1, :].sum(dim=1).numpy()

    # --- 6. SAVE DATA TABLE TO FILE ---
    print("Constructing Data Table...")

    # Construct Support DataFrame
    df_support = pd.DataFrame(X_context, columns=feature_names)
    df_support.insert(0, 'Index', np.arange(len(df_support))) # Explicit Index Column
    df_support.insert(1, 'Type', 'Support')
    df_support.insert(2, 'Target_Label', y_context)
    df_support.insert(3, 'LRP_Relevance', row_importance)

    # Construct Query DataFrame
    df_query = pd.DataFrame(X_query, columns=feature_names)
    df_query.insert(0, 'Index', len(df_support)) # Continue index
    df_query.insert(1, 'Type', 'Query')
    df_query.insert(2, 'Target_Label', true_label)
    df_query.insert(3, 'LRP_Relevance', np.nan) # No row relevance for itself

    # Combine
    df_combined = pd.concat([df_support, df_query], ignore_index=True)
    
    # Reorder columns to put top features next to relevance
    # Find top 5 most important features for this specific prediction
    top_indices = np.argsort(np.abs(feature_importance))[-5:][::-1]
    top_feats = list(np.array(feature_names)[top_indices])
    
    # Base columns + Top 5 features + Rest of features
    base_cols = ['Index', 'Type', 'Target_Label', 'LRP_Relevance']
    remaining_feats = [f for f in feature_names if f not in top_feats]
    final_cols = base_cols + top_feats + remaining_feats
    
    df_combined = df_combined[final_cols]

    csv_filename = "explanation_table.csv"
    print(f"Saving full data table to {csv_filename}...")
    df_combined.to_csv(csv_filename, index=False)

    # --- 7. SAVE PLOTS ---
    save_visualizations(feature_importance, row_importance, feature_names, y_context, predicted_class)
    
    print("Done. Check 'explanation_table.csv' and 'explanation_plot.png'.")

if __name__ == "__main__":
    main()