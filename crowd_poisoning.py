import torch
import numpy as np
import pandas as pd
import os
import json
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy.stats import norm

from model_loader import get_model, get_default_device

# --- CONFIGURATION ---
SEGMENT_SIZE = 40
CONTEXT_SIZES = [20, 30, 40, 60, 80, 100]
NUM_CONTEXTS_PER_SIZE = 3
TEST_SET_SIZE = 50
LEADERS_FRAC = 0.4
INFLUENCERS_FRAC = 0.1
CROWD_FRAC = 0.5
DATASET_PATH = "dataset.csv"
# Updated output directory to reflect crowd poisoning strategy
RESULTS_DIR = f"results_crowd_poisoning_{int(LEADERS_FRAC * 100)}_{int(INFLUENCERS_FRAC * 100)}_{int(CROWD_FRAC * 100)}"

class NumpyEncoder(json.JSONEncoder):
    """ Helper to save Numpy data to JSON """
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NumpyEncoder, self).default(obj)

def compute_global_relevance(model, device, X, y):
    """
    Step 1: The 'Hero Search'.
    Iterates through the dataset to find which samples are globally influential.
    (Unchanged from original)
    """
    n_samples = len(X)
    global_relevance = np.zeros(n_samples)
    counts = np.zeros(n_samples) 
    
    print(f"\n--- Phase 1: Global Hero Search (N={n_samples}) ---")
    
    indices = np.arange(n_samples)
    
    for i in tqdm(range(n_samples)):
        X_query = X[i].reshape(1, -1)
        
        mask = np.ones(n_samples, dtype=bool)
        mask[i] = False
        remaining_indices = indices[mask]
        np.random.shuffle(remaining_indices)
        
        num_segments = len(remaining_indices) // SEGMENT_SIZE
        
        for s in range(num_segments):
            seg_indices = remaining_indices[s*SEGMENT_SIZE : (s+1)*SEGMENT_SIZE]
            
            X_ctx = X[seg_indices]
            y_ctx = y[seg_indices]
            
            X_full = np.concatenate([X_ctx, X_query], axis=0)
            y_tensor_ctx = torch.from_numpy(y_ctx).float().to(device).unsqueeze(0).unsqueeze(-1)
            X_tensor = torch.from_numpy(X_full).float().to(device).unsqueeze(0)
            split_idx = len(X_ctx)
            
            model.zero_grad()
            embeddings = model.get_embeddings((X_tensor, y_tensor_ctx), split_idx)
            embeddings.retain_grad(); embeddings.requires_grad_(True)
            
            logits = model.forward_from_embeddings(embeddings, split_idx)
            pred_logits = logits[0, 0, :]
            pred_class = pred_logits.argmax().item()
            
            pred_logits[pred_class].backward()
            
            rel = (embeddings * embeddings.grad).detach().cpu().sum(dim=-1).squeeze(0)
            row_rel = rel[:-1, :].sum(dim=1).abs().numpy()
            
            global_relevance[seg_indices] += row_rel
            counts[seg_indices] += 1

    avg_relevance = np.divide(global_relevance, counts, out=np.zeros_like(global_relevance), where=counts!=0)
    min_val = np.min(avg_relevance)
    max_val = np.max(avg_relevance)
    normalized_scores = (avg_relevance - min_val) / (max_val - min_val) * 100
    
    return normalized_scores

def get_gaussian_weights(n_samples, center_pct, width_pct):
    ranks = np.linspace(0, 100, n_samples)
    weights = norm.pdf(ranks, loc=center_pct, scale=width_pct)
    return weights / weights.sum()

def engineer_gaussian_contexts(X, y, scores, context_sizes):
    """ 
    Phase 2: Context Generation.
    (Unchanged from original - we still need to create the mix of leaders/crowd,
    even if we only plan to poison the crowd later.)
    """
    print(f"\n--- Phase 2: Engineering Contexts (Gaussian Soft-Clustering) ---")
    
    sorted_indices = np.argsort(scores)[::-1]
    n_total = len(scores)
    
    prob_lead = get_gaussian_weights(n_total, center_pct=0, width_pct=5)
    prob_inf = get_gaussian_weights(n_total, center_pct=30, width_pct=10)
    prob_crowd = get_gaussian_weights(n_total, center_pct=75, width_pct=15)
    
    engineered_contexts = []
    
    for size in context_sizes:
        n_lead_target = max(1, int(LEADERS_FRAC * size))
        n_inf_target = int(INFLUENCERS_FRAC * size)
        n_crowd_target = size - n_lead_target - n_inf_target
        
        for i in range(NUM_CONTEXTS_PER_SIZE):
            available_mask = np.ones(n_total, dtype=bool)
            
            # Sample Leaders
            p_curr = prob_lead[available_mask]
            p_curr /= p_curr.sum()
            curr_indices = np.where(available_mask)[0]
            chosen_local_idx = np.random.choice(len(curr_indices), size=n_lead_target, replace=False, p=p_curr)
            leaders_indices = sorted_indices[curr_indices[chosen_local_idx]]
            available_mask[curr_indices[chosen_local_idx]] = False
            
            # Sample Influencers
            p_curr = prob_inf[available_mask]
            p_curr /= p_curr.sum()
            curr_indices = np.where(available_mask)[0]
            chosen_local_idx = np.random.choice(len(curr_indices), size=n_inf_target, replace=False, p=p_curr)
            inf_indices = sorted_indices[curr_indices[chosen_local_idx]]
            available_mask[curr_indices[chosen_local_idx]] = False
            
            # Sample Crowd
            p_curr = prob_crowd[available_mask]
            p_curr /= p_curr.sum()
            curr_indices = np.where(available_mask)[0]
            chosen_local_idx = np.random.choice(len(curr_indices), size=n_crowd_target, replace=False, p=p_curr)
            crowd_indices = sorted_indices[curr_indices[chosen_local_idx]]
            
            ctx_indices = np.concatenate([leaders_indices, inf_indices, crowd_indices])
            np.random.shuffle(ctx_indices)
            
            roles = {}
            for idx in leaders_indices: roles[idx] = "Leader"
            for idx in inf_indices: roles[idx] = "Influencer"
            for idx in crowd_indices: roles[idx] = "Commoner"
            
            role_list = [roles[idx] for idx in ctx_indices]

            engineered_contexts.append({
                "context_id": f"size_{size}_batch_{i}",
                "size": size,
                "indices": ctx_indices,
                "X": X[ctx_indices],
                "y": y[ctx_indices],
                "scores": scores[ctx_indices],
                "roles": role_list,
                "n_leaders": n_lead_target,
                "n_crowd": n_crowd_target
            })
            
    return engineered_contexts

def run_progressive_sycophancy_test(model, device, contexts, X_full, y_full):
    """ 
    Phase 3: Progressive Poisoning (CROWD ONLY).
    We identify Leaders and Influencers but ensure they are NOT poisoned.
    We only progressively flip the labels of the Crowd.
    """
    print(f"\n--- Phase 3: Progressive Sycophancy Stress Test (Crowd Only) ---")
    results = []
    
    for ctx_data in tqdm(contexts, desc="Running Attacks"):
        X_ctx = ctx_data["X"]
        y_ctx = ctx_data["y"]
        ctx_id = ctx_data["context_id"]
        roles = np.array(ctx_data["roles"])
        local_scores = ctx_data["scores"]
        
        # Identify Indices locally
        local_indices = np.arange(len(X_ctx))
        
        # --- MODIFIED: Identify Crowd Only ---
        crowd_mask = (roles == "Commoner")
        crowd_loc = local_indices[crowd_mask]

        # Shuffle Crowd randomly
        np.random.shuffle(crowd_loc)
        
        # Sort Crowd by score descending (Most influential commoners first)
        # You can change this to ascending if you want to start with the weakest
        crowd_loc = crowd_loc[np.argsort(local_scores[crowd_loc])[::-1]]
        
        schedule = []
        # Baseline (No poisoning)
        schedule.append({"n_poisoned": 0, "indices": [], "phase": "Baseline"})
        
        # Poisoning Schedule: Only iterate through the Crowd
        for k in range(1, len(crowd_loc) + 1):
            schedule.append({
                "n_poisoned": k,
                "indices": crowd_loc[:k], # Cumulative poisoning of crowd
                "phase": "Poisoning Crowd"
            })
            
        # Select Test Set (disjoint from context)
        mask = np.ones(len(X_full), dtype=bool)
        mask[ctx_data["indices"]] = False
        test_indices = np.arange(len(X_full))[mask]
        if len(test_indices) > TEST_SET_SIZE:
            test_indices = np.random.choice(test_indices, TEST_SET_SIZE, replace=False)
        X_test = X_full[test_indices]

        # Run Schedule
        clean_preds = {}
        for step in schedule:
            y_poison = y_ctx.copy()
            
            # Apply flips based on schedule indices (Crowd only)
            for idx in step["indices"]:
                y_poison[idx] = 1 - y_poison[idx]
            
            y_tensor_poison = torch.from_numpy(y_poison).float().to(device).unsqueeze(0).unsqueeze(-1)
            split_idx = len(X_ctx)
            
            flip_count = 0
            
            # Predict on Test Set
            for k in range(len(X_test)):
                X_q = X_test[k].reshape(1, -1)
                X_f = np.concatenate([X_ctx, X_q], axis=0)
                X_t = torch.from_numpy(X_f).float().to(device).unsqueeze(0)
                
                with torch.no_grad():
                    embeddings = model.get_embeddings((X_t, y_tensor_poison), split_idx)
                    logits = model.forward_from_embeddings(embeddings, split_idx)
                    curr_pred = logits[0, 0, :].argmax().item()
                
                if step["phase"] == "Baseline":
                    clean_preds[k] = curr_pred
                else:
                    orig_pred = clean_preds[k]
                    if curr_pred != orig_pred:
                        flip_count += 1
            
            flip_rate = 0.0 if step["phase"] == "Baseline" else (flip_count / len(X_test)) * 100
            
            results.append({
                "context_id": ctx_id,
                "context_size": ctx_data["size"],
                "total_poisoned": step["n_poisoned"],
                "poisoning_phase": step["phase"],
                "flip_rate": flip_rate,
                "n_crowd_in_context": len(crowd_loc)
            })

    return pd.DataFrame(results)

def visualize_sycophancy_curve(df):
    """ 
    Visualizes the curve for Crowd Poisoning. 
    (Removed the Leader/Influencer transition markers as they don't apply)
    """
    print("\nGenerating Sycophancy Curve...")
    
    summary = df.groupby(["context_size", "total_poisoned"]).agg(
        flip_rate=("flip_rate", "mean")
    ).reset_index()

    plt.figure(figsize=(12, 8))
    sns.set_style("whitegrid")
    
    sizes = sorted(summary["context_size"].unique())
    palette = sns.color_palette("viridis", len(sizes))
    
    for i, size in enumerate(sizes):
        subset = summary[summary["context_size"] == size]
        color = palette[i]
        
        plt.plot(
            subset["total_poisoned"], 
            subset["flip_rate"], 
            marker='o', 
            linewidth=2.5, 
            color=color,
            label=f"Size {size}"
        )
    
    plt.title(f"Label flip rate vs number of poisoned samples\nLeaders: {int(LEADERS_FRAC * 100)}%, influencers: {int(INFLUENCERS_FRAC * 100)}%, crowd: {int(CROWD_FRAC * 100)}%\n(Poisoning random crowd samples)", fontsize=16, weight='bold')
    plt.xlabel("Total poisoned crowd samples", fontsize=12)
    plt.ylabel("Test set flip rate (%)", fontsize=12)
    plt.axhline(50, color='gray', linestyle='--', alpha=0.5, label="50% threshold")
    plt.legend(title="Context size")
    
    plot_path = os.path.join(RESULTS_DIR, "flip_rate_crowd_poisoning.png")
    plt.savefig(plot_path, dpi=300)
    print(f"Saved plot to {plot_path}")

def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    
    try:
        model, device = get_model()
    except Exception as e:
        print(f"Model load failed: {e}")
        return

    if not os.path.exists(DATASET_PATH): 
        print(f"Dataset not found at {DATASET_PATH}"); return
        
    df_data = pd.read_csv(DATASET_PATH)
    X = df_data.drop(columns=["target"]).values
    y = df_data["target"].values
    
    # Phase 1: Relevance (Load or Compute)
    scores_path = os.path.join(RESULTS_DIR, "global_relevance_scores.npy")
    # For a new experiment folder, we might want to recompute or symlink. 
    # Here we just recompute to be safe, or check the root.
    if os.path.exists(scores_path):
        scores = np.load(scores_path)
    else:
        scores = compute_global_relevance(model, device, X, y)
        np.save(scores_path, scores)
        
    # Phase 2: Gaussian Context Engineering
    contexts = engineer_gaussian_contexts(X, y, scores, CONTEXT_SIZES)
    with open(os.path.join(RESULTS_DIR, "engineered_contexts.json"), "w") as f:
        json.dump(contexts, f, cls=NumpyEncoder, indent=4)
    
    # Phase 3: Attack (Poison Crowd Only)
    results_df = run_progressive_sycophancy_test(model, device, contexts, X, y)
    results_df.to_csv(os.path.join(RESULTS_DIR, "sycophancy_results_crowd.csv"), index=False)
    
    visualize_sycophancy_curve(results_df)
    print(f"\nExperiment complete. Results in {RESULTS_DIR}")

if __name__ == "__main__":
    main()