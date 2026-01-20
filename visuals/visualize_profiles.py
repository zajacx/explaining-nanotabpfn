import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
import os

# --- CONFIGURATION ---
DATA_FILE = "visuals/hero_leaderboard.csv" 

def visualize_hero_profiles():
    if not os.path.exists(DATA_FILE):
        print(f"Error: {DATA_FILE} not found.")
        return

    print(f"Loading {DATA_FILE}...")
    df = pd.read_csv(DATA_FILE)

    # 1. PREPROCESSING
    metadata_cols = ['target', 'Global_Relevance']
    feature_cols = [c for c in df.columns if c not in metadata_cols]
    
    # Standardize based on WHOLE population
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df[feature_cols])
    
    df_scaled = pd.DataFrame(X_scaled, columns=feature_cols)
    df_scaled['target'] = df['target']
    df_scaled['Global_Relevance'] = df['Global_Relevance']
    
    # 2. SELECT TOP 10 PER CLASS
    # Malignant (0)
    top_malignant = df_scaled[df_scaled['target'] == 0].sort_values(
        by='Global_Relevance', ascending=False
    ).head(10)
    
    # Benign (1)
    top_benign = df_scaled[df_scaled['target'] == 1].sort_values(
        by='Global_Relevance', ascending=False
    ).head(10)
    
    # Combine for plotting
    all_heroes = pd.concat([top_malignant, top_benign])
    
    # Population Stats for Background
    pop_mean = df_scaled[feature_cols].mean()
    pop_q25 = df_scaled[feature_cols].quantile(0.25)
    pop_q75 = df_scaled[feature_cols].quantile(0.75)

    # 3. VISUALIZATION
    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(24, 12)) 
    
    x = np.arange(len(feature_cols))
    
    # A. Draw the "Crowd" (Background)
    plt.fill_between(x, pop_q25, pop_q75, color='gray', alpha=0.15, label='Population "Normal" Range (IQR)')
    plt.plot(x, pop_mean, color='gray', linestyle=':', alpha=0.5)

    # B. Draw the Heroes
    malignant_handle = None
    benign_handle = None
    
    for idx, row in all_heroes.iterrows():
        is_malignant = (row['target'] == 0)
        color = '#d62728' if is_malignant else '#1f77b4' # Red vs Blue
        
        # Highlight the #1 of each class slightly more
        # We check if this row is the absolute top of its specific class dataframe
        is_top_malignant = (idx == top_malignant.index[0])
        is_top_benign = (idx == top_benign.index[0])
        
        if is_top_malignant or is_top_benign:
            linewidth = 3.5
            alpha = 1.0
            zorder = 10
            marker_size = 6
        else:
            linewidth = 1.5
            alpha = 0.5
            zorder = 5
            marker_size = 0
            
        line, = plt.plot(
            x, 
            row[feature_cols], 
            color=color, 
            linewidth=linewidth, 
            alpha=alpha, 
            marker='o', 
            markersize=marker_size,
            zorder=zorder
        )
        
        if is_malignant and malignant_handle is None: malignant_handle = line
        if not is_malignant and benign_handle is None: benign_handle = line
        
        # Annotate just the two absolute leaders
        if is_top_malignant:
            # Annotate peak
            max_idx = np.argmax(row[feature_cols])
            plt.annotate(
                f"Top malignant\n(Score: {row['Global_Relevance']:.1f})",
                xy=(max_idx, row[feature_cols].iloc[max_idx]),
                xytext=(max_idx, row[feature_cols].iloc[max_idx] + 1.5),
                arrowprops=dict(facecolor=color, shrink=0.05),
                fontsize=11, color=color, fontweight='bold', ha='center'
            )
        
        if is_top_benign:
            min_idx = np.argmin(row[feature_cols]) # Benign usually negative Z
            plt.annotate(
                f"Top benign\n(Score: {row['Global_Relevance']:.1f})",
                xy=(min_idx, row[feature_cols].iloc[min_idx]),
                xytext=(min_idx, row[feature_cols].iloc[min_idx] - 1.5),
                arrowprops=dict(facecolor=color, shrink=0.05),
                fontsize=11, color=color, fontweight='bold', ha='center'
            )

    # C. Formatting
    plt.xticks(x, feature_cols, rotation=90, fontsize=11, fontweight='bold')
    plt.yticks(fontsize=12)
    plt.ylabel("Standard deviations from mean (z-score)", fontsize=14)
    plt.title("Geometric profiles: top 10 benign vs. top 10 malignant leaders", fontsize=20, weight='bold')
    
    plt.axhline(0, color='black', linewidth=1, linestyle='-', alpha=0.3)
    
    handles = [
        plt.Rectangle((0,0),1,1, color='gray', alpha=0.2),
        malignant_handle,
        benign_handle
    ]
    labels = [
        'Population norm (middle 50%)', 
        'Malignant leaders (class 0)', 
        'Benign leaders (class 1)'
    ]
    
    # Robust legend creation (handles NoneTypes if a class is missing)
    final_handles = [h for h in handles if h is not None]
    final_labels = [l for h, l in zip(handles, labels) if h is not None]

    plt.legend(final_handles, final_labels, loc='upper right', fontsize=12, frameon=True, shadow=True)
    plt.tight_layout()
    
    save_path = "visuals/hero_profiles.png"
    plt.savefig(save_path, dpi=300)
    print(f"Visualization saved to {save_path}")
    plt.show()

if __name__ == "__main__":
    visualize_hero_profiles()