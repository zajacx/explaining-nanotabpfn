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
DATASET_PATH = "dataset.csv"
RESULTS_DIR = "results_poisoning"
SEGMENT_SIZE = 40
CONTEXT_SIZES = [15, 20, 25, 30, 35, 40]
NUM_CONTEXTS_PER_SIZE = 3  # How many diverse contexts to generate per size
TEST_SET_SIZE = 50
LEADERS_FRAC = 0.0
INFLUENCERS_FRAC = 0.6

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
    """
    n_samples = len(X)
    global_relevance = np.zeros(n_samples)
    counts = np.zeros(n_samples) # Track how often a sample was used in context
    
    print(f"\n--- Phase 1: Global Hero Search (N={n_samples}) ---")
    print("Accumulating relevance scores across randomized contexts...")
    
    # We treat every sample as a test query once
    # For its context, we take chunks of the remaining data
    
    indices = np.arange(n_samples)
    
    for i in tqdm(range(n_samples)):
        # Sample i is the Query
        X_query = X[i].reshape(1, -1)
        # y_query = y[i] # Not needed for context construction
        
        # Remaining indices
        mask = np.ones(n_samples, dtype=bool)
        mask[i] = False
        remaining_indices = indices[mask]
        
        # Shuffle remaining to ensure random segments
        np.random.shuffle(remaining_indices)
        
        # Create segments of size SEGMENT_SIZE
        # We process ALL remaining data to be thorough
        num_segments = len(remaining_indices) // SEGMENT_SIZE
        
        for s in range(num_segments):
            seg_indices = remaining_indices[s*SEGMENT_SIZE : (s+1)*SEGMENT_SIZE]
            
            X_ctx = X[seg_indices]
            y_ctx = y[seg_indices]
            
            # Prepare Tensors
            X_full = np.concatenate([X_ctx, X_query], axis=0)
            y_tensor_ctx = torch.from_numpy(y_ctx).float().to(device).unsqueeze(0).unsqueeze(-1)
            X_tensor = torch.from_numpy(X_full).float().to(device).unsqueeze(0)
            split_idx = len(X_ctx)
            
            # --- LRP Pass ---
            model.zero_grad()
            embeddings = model.get_embeddings((X_tensor, y_tensor_ctx), split_idx)
            embeddings.retain_grad(); embeddings.requires_grad_(True)
            
            logits = model.forward_from_embeddings(embeddings, split_idx)
            pred_logits = logits[0, 0, :]
            pred_class = pred_logits.argmax().item()
            
            # Backward on Predicted Class
            pred_logits[pred_class].backward()
            
            # Extract Relevance
            rel = (embeddings * embeddings.grad).detach().cpu().sum(dim=-1).squeeze(0)
            row_rel = rel[:-1, :].sum(dim=1).abs().numpy() # Magnitude of influence
            
            # Accumulate
            global_relevance[seg_indices] += row_rel
            counts[seg_indices] += 1

    # Normalize: Average per usage, then scale 0-100
    avg_relevance = np.divide(global_relevance, counts, out=np.zeros_like(global_relevance), where=counts!=0)
    
    # Min-Max Scaling to 0-100
    min_val = np.min(avg_relevance)
    max_val = np.max(avg_relevance)
    normalized_scores = (avg_relevance - min_val) / (max_val - min_val) * 100
    
    return normalized_scores

def get_gaussian_weights(n_samples, center_pct, width_pct):
    """ Generates a probability distribution over sorted ranks """
    ranks = np.linspace(0, 100, n_samples)
    # PDF based on percentage rank
    weights = norm.pdf(ranks, loc=center_pct, scale=width_pct)
    # Normalize to sum to 1
    return weights / weights.sum()

def engineer_gaussian_contexts(X, y, scores, context_sizes):
    """ 
    Phase 2: Probabilistic 'Soft Cluster' Sampling.
    Uses 3 Gaussians to prefer cores but allow buffer sampling.
    """
    print(f"\n--- Phase 2: Engineering Contexts (Gaussian Soft-Clustering) ---")
    
    # 1. Sort Indices by Global Score Descending (Rank 0 = Top Leader)
    sorted_indices = np.argsort(scores)[::-1]
    n_total = len(scores)
    
    # 2. Define the 3 Gaussian Distributions over the Ranks
    # Leaders: Peak at 0 (Top), narrow width (focus on top 10%)
    prob_lead = get_gaussian_weights(n_total, center_pct=0, width_pct=5) # Half-normal at top
    
    # Influencers: Peak at 30% (Middle of 10-50 range), medium width
    prob_inf = get_gaussian_weights(n_total, center_pct=30, width_pct=10)
    
    # Crowd: Peak at 75% (Center of 50-100 range), broad width
    prob_crowd = get_gaussian_weights(n_total, center_pct=75, width_pct=15)
    
    engineered_contexts = []
    
    for size in context_sizes:
        # Target Counts (1:4:5 ratio)
        n_lead_target = max(1, int(LEADERS_FRAC * size))
        n_inf_target = int(INFLUENCERS_FRAC * size)
        n_crowd_target = size - n_lead_target - n_inf_target
        
        for i in range(NUM_CONTEXTS_PER_SIZE):
            # We must sample without replacement across the whole context
            # Strategy: Sample Leaders -> Remove from pool -> Sample Influencers -> Remove -> Sample Crowd
            
            available_mask = np.ones(n_total, dtype=bool)
            
            # A. Sample Leaders
            # Re-normalize probabilities to sum to 1 over CURRENT available samples
            p_curr = prob_lead[available_mask]
            p_curr /= p_curr.sum()
            
            # Map valid slots back to sorted_indices
            curr_indices = np.where(available_mask)[0]
            
            chosen_local_idx = np.random.choice(len(curr_indices), size=n_lead_target, replace=False, p=p_curr)
            leaders_indices = sorted_indices[curr_indices[chosen_local_idx]]
            
            # Mark used
            available_mask[curr_indices[chosen_local_idx]] = False
            
            # B. Sample Influencers
            p_curr = prob_inf[available_mask]
            p_curr /= p_curr.sum()
            curr_indices = np.where(available_mask)[0]
            
            chosen_local_idx = np.random.choice(len(curr_indices), size=n_inf_target, replace=False, p=p_curr)
            inf_indices = sorted_indices[curr_indices[chosen_local_idx]]
            available_mask[curr_indices[chosen_local_idx]] = False
            
            # C. Sample Crowd
            p_curr = prob_crowd[available_mask]
            p_curr /= p_curr.sum()
            curr_indices = np.where(available_mask)[0]
            
            chosen_local_idx = np.random.choice(len(curr_indices), size=n_crowd_target, replace=False, p=p_curr)
            crowd_indices = sorted_indices[curr_indices[chosen_local_idx]]
            
            # Combine
            ctx_indices = np.concatenate([leaders_indices, inf_indices, crowd_indices])
            np.random.shuffle(ctx_indices)
            
            # Assign Roles for Analysis
            # Note: A sample is a "Leader" here because it was sampled from the Leader distribution,
            # even if it's technically rank 55 (buffer zone). This captures the 'intent' of the sampling.
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
                "n_leaders": n_lead_target
            })
            
    return engineered_contexts

def run_progressive_sycophancy_test(model, device, contexts, X_full, y_full):
    """ Phase 3: Progressive Poisoning """
    print(f"\n--- Phase 3: Progressive Sycophancy Stress Test ---")
    results = []
    
    for ctx_data in tqdm(contexts, desc="Running Attacks"):
        X_ctx = ctx_data["X"]
        y_ctx = ctx_data["y"]
        ctx_id = ctx_data["context_id"]
        roles = np.array(ctx_data["roles"])
        local_scores = ctx_data["scores"]
        
        # Identify Targets (Sorted by Score Descending)
        local_indices = np.arange(len(X_ctx))
        
        # Leaders
        leader_mask = (roles == "Leader")
        leaders_loc = local_indices[leader_mask]
        leaders_loc = leaders_loc[np.argsort(local_scores[leaders_loc])[::-1]]
        
        # Influencers
        inf_mask = (roles == "Influencer")
        inf_loc = local_indices[inf_mask]
        inf_loc = inf_loc[np.argsort(local_scores[inf_loc])[::-1]]
        
        # Schedule
        schedule = []
        schedule.append({"n_lead": 0, "n_inf": 0, "indices": [], "phase": "Baseline"})
        
        # Phase A: Poison Leaders (1 to All)
        for k in range(1, len(leaders_loc) + 1):
            schedule.append({
                "n_lead": k, "n_inf": 0, 
                "indices": leaders_loc[:k],
                "phase": "Poisoning Leaders"
            })
            
        # Phase B: Poison All Leaders + Influencers (up to 50%)
        limit_inf = max(1, len(inf_loc) // 2)
        if len(inf_loc) > 0:
            for m in range(1, limit_inf + 1):
                combined = np.concatenate([leaders_loc, inf_loc[:m]])
                schedule.append({
                    "n_lead": len(leaders_loc), "n_inf": m,
                    "indices": combined,
                    "phase": "Poisoning Influencers"
                })

        # Test Set
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
            for idx in step["indices"]:
                y_poison[idx] = 1 - y_poison[idx]
            
            y_tensor_poison = torch.from_numpy(y_poison).float().to(device).unsqueeze(0).unsqueeze(-1)
            split_idx = len(X_ctx)
            
            flip_count = 0
            
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
                "total_poisoned": step["n_lead"] + step["n_inf"],
                "poisoning_phase": step["phase"],
                "flip_rate": flip_rate,
                "n_leaders_in_context": len(leaders_loc)
            })

    return pd.DataFrame(results)

def visualize_sycophancy_curve(df):
    """ Visualizes the curve with Red Cross markers for Phase Transitions """
    print("\nGenerating Sycophancy Curve...")
    
    summary = df.groupby(["context_size", "total_poisoned"]).agg(
        flip_rate=("flip_rate", "mean"),
        n_leaders=("n_leaders_in_context", "first")
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
        
        # Mark Transition
        n_lead = int(subset["n_leaders"].iloc[0])
        if n_lead in subset["total_poisoned"].values:
            val = subset[subset["total_poisoned"] == n_lead]["flip_rate"].values[0]
            plt.plot(n_lead, val, marker='X', markersize=14, color='red', markeredgecolor='white', zorder=10)

    # Manual Legend for the 'X'
    from matplotlib.lines import Line2D
    handles, labels = plt.gca().get_legend_handles_labels()
    handles.append(Line2D([0], [0], marker='X', color='w', markerfacecolor='red', markersize=12, markeredgecolor='w'))
    labels.append("End of Leader Phase")
    
    plt.title("Sycophancy Curve: Gaussian-Clustered Contexts", fontsize=16, weight='bold')
    plt.xlabel("Total Poisoned Samples", fontsize=12)
    plt.ylabel("Test Set Flip Rate (%)", fontsize=12)
    plt.axhline(50, color='gray', linestyle='--', alpha=0.5, label="50% Threshold")
    plt.legend(handles=handles, labels=labels, title="Context Size")
    
    plot_path = os.path.join(RESULTS_DIR, "sycophancy_curve_v3.png")
    plt.savefig(plot_path, dpi=300)
    print(f"Saved plot to {plot_path}")

def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    
    try:
        model, device = get_model()
    except TypeError: return

    if not os.path.exists(DATASET_PATH): 
        print(f"Dataset not found at {DATASET_PATH}"); return
        
    df_data = pd.read_csv(DATASET_PATH)
    X = df_data.drop(columns=["target"]).values
    y = df_data["target"].values
    
    # Phase 1: Relevance
    scores_path = os.path.join(RESULTS_DIR, "global_relevance_scores.npy")
    if os.path.exists(scores_path):
        scores = np.load(scores_path)
    else:
        scores = compute_global_relevance(model, device, X, y)
        np.save(scores_path, scores)
        
    # Phase 2: Gaussian Context Engineering
    contexts = engineer_gaussian_contexts(X, y, scores, CONTEXT_SIZES)
    with open(os.path.join(RESULTS_DIR, "engineered_contexts.json"), "w") as f:
        json.dump(contexts, f, cls=NumpyEncoder, indent=4)
    
    # Phase 3: Attack
    results_df = run_progressive_sycophancy_test(model, device, contexts, X, y)
    results_df.to_csv(os.path.join(RESULTS_DIR, "sycophancy_results.csv"), index=False)
    
    visualize_sycophancy_curve(results_df)
    print(f"\nExperiment complete. Results in {RESULTS_DIR}")

if __name__ == "__main__":
    main()