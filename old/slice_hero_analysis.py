import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from math import pi
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import types
import os

# Assuming these imports exist in your project structure
# If model.py is in the same directory, these work.
from model import NanoTabPFNModel
from lxt.efficient.patches import (
    patch_method, 
    layer_norm_forward, 
    non_linear_forward, 
    cp_multi_head_attention_forward
)

# --- CONFIGURATION ---
SLICE_START = 21 
SLICE_END = 51
WEIGHTS_PATH = "nanotabpfn_weights.pth"
TEST_SIZE = 0.08  # As requested in your snippet

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

def plot_radar_chart(df_profiles, feature_names, title, filename):
    """Creates a radar chart comparing Hero samples to the Class Average."""
    # We select 5 distinct features to keep the chart readable.
    # These are key geometric features for breast cancer.
    top_features = ['mean radius', 'mean texture', 'mean smoothness', 
                    'mean compactness', 'mean concavity']
    
    categories = top_features
    N = len(categories)
    
    # What will be the angle of each axis in the plot? (we divide the plot / number of variable)
    angles = [n / float(N) * 2 * pi for n in range(N)]
    angles += angles[:1] # Close the loop
    
    plt.figure(figsize=(10, 10))
    ax = plt.subplot(111, polar=True)
    
    # Draw one axe per variable + labels
    plt.xticks(angles[:-1], categories, color='grey', size=10)
    
    # Draw ylabels
    ax.set_rlabel_position(0)
    plt.yticks([0.25, 0.5, 0.75], ["0.25", "0.50", "0.75"], color="grey", size=7)
    plt.ylim(0, 1)
    
    # Define style for different profile types
    colors = {'Average': 'blue', 'Hero': 'red', 'Least Influential': 'green'}
    styles = {'Average': '--', 'Hero': '-', 'Least Influential': ':'}
    
    for idx, row in df_profiles.iterrows():
        name = row['Type']
        values = row[categories].values.flatten().tolist()
        values += values[:1] # Close the loop for the line
        
        ax.plot(angles, values, linewidth=2, linestyle=styles.get(name, '-'), 
                label=name, color=colors.get(name, 'black'))
        ax.fill(angles, values, alpha=0.1, color=colors.get(name, 'black'))
        
    plt.title(title, size=15, y=1.1)
    plt.legend(loc='upper right', bbox_to_anchor=(1.1, 1.1))
    
    plt.tight_layout()
    plt.savefig(filename)
    print(f"Saved Radar Chart -> {filename}")
    plt.close()

def analyze_characteristics():
    device = get_default_device()
    print(f"--- Analyzing Prototypes in Slice {SLICE_START}:{SLICE_END} ---")
    
    # 1. Initialize Model
    model = NanoTabPFNModel(
        embedding_size=96, 
        num_attention_heads=4, 
        mlp_hidden_size=192, 
        num_layers=3, 
        num_outputs=2
    ).to(device)
    
    try:
        model.load_state_dict(torch.load(WEIGHTS_PATH, map_location=device))
        print("Weights loaded.")
    except FileNotFoundError:
        print(f"Error: {WEIGHTS_PATH} not found.")
        return

    model.eval()
    apply_lxt_patches(model)
    
    # 2. Prepare Data
    data = load_breast_cancer()
    X, y = data.data, data.target
    feature_names = list(data.feature_names) # Convert to list for easier handling
    
    # Use exact split to match your experiment
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SIZE, random_state=42)
    
    X_context = X_train[SLICE_START:SLICE_END]
    y_context = y_train[SLICE_START:SLICE_END]
    
    # 3. Calculate Influence Scores
    print(f"Calculating influence of {len(X_context)} context samples on {len(X_test)} test samples...")
    
    total_influence = np.zeros(len(X_context))
    
    X_tensor_ctx = torch.from_numpy(X_context).float().to(device)
    y_tensor_ctx = torch.from_numpy(y_context).float().to(device).unsqueeze(0).unsqueeze(-1)
    split_idx = len(X_context)
    
    for i in range(len(X_test)):
        # Prepare Query
        X_query = X_test[i].reshape(1, -1)
        # Construct Full Input
        X_full = np.concatenate([X_context, X_query], axis=0)
        X_tensor = torch.from_numpy(X_full).float().unsqueeze(0).to(device)
        
        # LRP Forward Pass
        model.zero_grad()
        embeddings = model.get_embeddings((X_tensor, y_tensor_ctx), split_idx)
        embeddings.retain_grad()
        embeddings.requires_grad_(True)
        
        logits = model.forward_from_embeddings(embeddings, split_idx)
        
        # Get prediction for the query sample (last in sequence)
        # Shape: (Batch=1, Seq, Classes) -> [0, 0, :] because forward output is sliced to query
        pred_logits = logits[0, 0, :]
        pred_class = pred_logits.argmax().item()
        
        # Backward pass on the PREDICTED class
        target_logit = pred_logits[pred_class]
        target_logit.backward()
        
        # Compute Relevance
        # Element-wise product of activation * gradient
        rel_map = (embeddings * embeddings.grad).detach().cpu().sum(dim=-1).squeeze(0)
        
        # Extract Row Relevance (Support Rows Only)
        # Sum over features to get a single score per row
        # Take Absolute Value: Influence is magnitude (pushing away is also influence)
        row_rel = rel_map[:-1, :].sum(dim=1).abs().numpy()
        
        total_influence += row_rel

    print("Influence calculation complete.")

    # 4. ANALYSIS: "Why are they special?"
    # Create DataFrame with raw values
    df = pd.DataFrame(X_context, columns=feature_names)
    df['Influence'] = total_influence
    df['True_Label'] = y_context
    
    # Create Normalized DataFrame for Radar Charts (0-1 Scaling)
    scaler = MinMaxScaler()
    df_norm = pd.DataFrame(scaler.fit_transform(X_context), columns=feature_names)
    df_norm['Influence'] = total_influence
    df_norm['True_Label'] = y_context
    
    # --- A. Analyze Malignant (Class 0) Prototypes ---
    print("\n--- Analyzing Malignant (Class 0) Prototypes ---")
    df_mal = df_norm[df_norm['True_Label'] == 0]
    
    if not df_mal.empty:
        # Identify Hero vs Average
        hero_idx_local = df_mal['Influence'].idxmax() # Index in df_norm
        loser_idx_local = df_mal['Influence'].idxmin()
        
        # Extract Profiles
        row_hero = df_mal.loc[hero_idx_local].to_dict()
        row_hero['Type'] = 'Hero'
        
        row_avg = df_mal.drop(columns=['Influence', 'True_Label']).mean().to_dict()
        row_avg['Type'] = 'Average'
        
        row_loser = df_mal.loc[loser_idx_local].to_dict()
        row_loser['Type'] = 'Least Influential'
        
        profiles = pd.DataFrame([row_hero, row_avg, row_loser])
        plot_radar_chart(profiles, feature_names, "Malignant Personality: Hero vs Average", "radar_malignant.png")
        
        # Distance Metric Calculation (using RAW data, not normalized)
        # 1. Get the actual raw vector for the Hero
        raw_hero_vector = X_context[hero_idx_local]
        
        # 2. Get the Global Centroid for Malignant
        X_mal_global = X_train[y_train == 0]
        centroid_mal = X_mal_global.mean(axis=0)
        
        # 3. Compute Distances
        dist_hero = np.linalg.norm(raw_hero_vector - centroid_mal)
        dist_avg = np.mean([np.linalg.norm(x - centroid_mal) for x in X_mal_global])
        
        print(f"Hero Distance to Centroid: {dist_hero:.2f}")
        print(f"Average Patient Distance: {dist_avg:.2f}")
        
        if dist_hero < dist_avg:
            print("CONCLUSION: The Hero is a 'Central Prototype' (Closer to center than average).")
        else:
            print("CONCLUSION: The Hero is an 'Extreme Outlier' (Further from center).")
    else:
        print("No Malignant samples in this slice.")

    # --- B. Analyze Benign (Class 1) Prototypes ---
    print("\n--- Analyzing Benign (Class 1) Prototypes ---")
    df_ben = df_norm[df_norm['True_Label'] == 1]
    
    if not df_ben.empty:
        hero_idx_local = df_ben['Influence'].idxmax()
        
        row_hero = df_ben.loc[hero_idx_local].to_dict()
        row_hero['Type'] = 'Hero'
        
        row_avg = df_ben.drop(columns=['Influence', 'True_Label']).mean().to_dict()
        row_avg['Type'] = 'Average'
        
        profiles = pd.DataFrame([row_hero, row_avg])
        plot_radar_chart(profiles, feature_names, "Benign Personality: Hero vs Average", "radar_benign.png")
        
        # Distance Metric
        raw_hero_vector = X_context[hero_idx_local]
        X_ben_global = X_train[y_train == 1]
        centroid_ben = X_ben_global.mean(axis=0)
        
        dist_hero = np.linalg.norm(raw_hero_vector - centroid_ben)
        dist_avg = np.mean([np.linalg.norm(x - centroid_ben) for x in X_ben_global])
        
        print(f"Hero Distance to Centroid: {dist_hero:.2f}")
        print(f"Average Patient Distance: {dist_avg:.2f}")
        
        if dist_hero < dist_avg:
            print("CONCLUSION: The Hero is a 'Central Prototype' (Textbook Benign).")
        else:
            print("CONCLUSION: The Hero is an 'Extreme Outlier'.")
    else:
        print("No Benign samples in this slice.")

if __name__ == "__main__":
    analyze_characteristics()