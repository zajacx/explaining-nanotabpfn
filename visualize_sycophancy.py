import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
import numpy as np

# --- CONFIGURATION ---
RESULTS_DIR = "results_poisoning" 
CSV_PATH = os.path.join(RESULTS_DIR, "sycophancy_results.csv")
OUTPUT_DIR = os.path.join(RESULTS_DIR, "plots")

def visualize_results():
    if not os.path.exists(CSV_PATH):
        print(f"Error: File not found at {CSV_PATH}")
        return

    print(f"Loading results from {CSV_PATH}...")
    df = pd.read_csv(CSV_PATH)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Set aesthetics
    sns.set_theme(style="whitegrid", context="talk")
    plt.rcParams["figure.figsize"] = (12, 8)

    # --- DATA NORMALIZATION (AUTO-DETECT SCALE) ---
    max_val = df["flip_rate"].max()
    
    if max_val > 1.0:
        print(f"Detected 0-100 scale (Max value: {max_val}). Using values as-is.")
        df["flip_rate_pct"] = df["flip_rate"]
    else:
        print(f"Detected 0.0-1.0 scale (Max value: {max_val}). Converting to percentage.")
        df["flip_rate_pct"] = df["flip_rate"] * 100

    # Aggregate Mean Flip Rate for curves
    summary = df.groupby(["context_size", "total_poisoned"]).agg(
        flip_rate_pct=("flip_rate_pct", "mean"),
        n_leaders=("n_leaders_in_context", "first"),
        poisoning_phase=("poisoning_phase", "first")
    ).reset_index()

    # =================================================================
    # PLOT 1: THE SYCOPHANCY CURVE (With Red X Transitions)
    # =================================================================
    plt.figure()
    
    sizes = sorted(summary["context_size"].unique())
    palette = sns.color_palette("viridis", len(sizes))
    
    for i, size in enumerate(sizes):
        subset = summary[summary["context_size"] == size]
        color = palette[i]
        
        plt.plot(
            subset["total_poisoned"], 
            subset["flip_rate_pct"], 
            marker='o', 
            linewidth=2.5, 
            color=color, 
            label=f"Size {size}"
        )
        
        # Mark Transition (End of Leaders)
        n_lead = int(subset["n_leaders"].iloc[0])
        transition_point = subset[subset["total_poisoned"] == n_lead]
        
        if not transition_point.empty:
            val = transition_point["flip_rate_pct"].values[0]
            plt.plot(n_lead, val, marker='X', markersize=14, color='red', markeredgecolor='white', zorder=10)

    # Legend
    from matplotlib.lines import Line2D
    handles, labels = plt.gca().get_legend_handles_labels()
    handles.append(Line2D([0], [0], marker='X', color='w', markerfacecolor='red', markersize=12, markeredgecolor='w'))
    labels.append("Transition (Leaders -> Influencers)")
    
    plt.title("The Sycophancy Curve: When does the model break?", fontsize=20, weight='bold')
    plt.xlabel("Number of Poisoned Context Samples", fontsize=14)
    plt.ylabel("Test Set Flip Rate (%)", fontsize=14)
    
    # Fix Y-Axis to 0-100 regardless of input noise
    plt.ylim(-5, 105)
    plt.axhline(50, color='gray', linestyle='--', alpha=0.5, label="50% Threshold")
    
    plt.legend(handles=handles, labels=labels, title="Context Size", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    
    filename1 = os.path.join(OUTPUT_DIR, "sycophancy_curve_fixed.png")
    plt.savefig(filename1, dpi=300)
    print(f"Saved Curve -> {filename1}")

    # =================================================================
    # PLOT 2: LEADERS VS INFLUENCERS IMPACT (Bar Chart)
    # =================================================================
    # Use max context size if 40 isn't available
    target_size = 40 if 40 in df["context_size"].unique() else df["context_size"].max()
    
    if target_size:
        plt.figure()
        phase_subset = summary[summary["context_size"] == target_size].copy()
        
        sns.barplot(
            data=phase_subset,
            x="total_poisoned",
            y="flip_rate_pct",
            hue="poisoning_phase",
            palette={"Poisoning Leaders": "#d62728", "Poisoning Influencers": "#1f77b4", "Baseline": "gray"},
            dodge=False
        )
        
        plt.title(f"Hierarchy Impact: Leaders vs. Influencers (Context Size {target_size})", fontsize=18)
        plt.xlabel("Total Poisoned Samples", fontsize=14)
        plt.ylabel("Flip Rate (%)", fontsize=14)
        plt.ylim(0, 105) 
        plt.legend(title="Target Group")
        plt.tight_layout()
        
        filename2 = os.path.join(OUTPUT_DIR, "hierarchy_impact_fixed.png")
        plt.savefig(filename2, dpi=300)
        print(f"Saved Hierarchy Bar Chart -> {filename2}")

    # =================================================================
    # PLOT 3: ROBUSTNESS HEATMAP
    # =================================================================
    heatmap_data = summary.pivot(index="context_size", columns="total_poisoned", values="flip_rate_pct")
    
    plt.figure(figsize=(14, 8))
    sns.heatmap(
        heatmap_data, 
        annot=True, 
        fmt=".0f", 
        cmap="Reds", 
        cbar_kws={'label': 'Flip Rate (%)'},
        vmin=0, vmax=100
    )
    
    plt.title("Robustness Heatmap: Flip Rate by Context & Poison Count", fontsize=18)
    plt.ylabel("Context Size (N)", fontsize=14)
    plt.xlabel("Number of Poisoned Samples", fontsize=14)
    plt.tight_layout()
    
    filename3 = os.path.join(OUTPUT_DIR, "robustness_heatmap_fixed.png")
    plt.savefig(filename3, dpi=300)
    print(f"Saved Heatmap -> {filename3}")

    print("\nAll visualizations saved correctly!")

if __name__ == "__main__":
    visualize_results()