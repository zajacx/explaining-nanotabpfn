# Hypothesis: choice of context doesn't matter for some particular rows;
# they are always misclassfied (or almost always)

import os
import torch
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from model import NanoTabPFNModel, NanoTabPFNClassifier
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# --- CONFIG ---
OUTPUT_DIR = "error_analysis_data"
WEIGHTS_PATH = "nanotabpfn_weights.pth"

def get_default_device():
    if torch.cuda.is_available(): return "cuda"
    if torch.backends.mps.is_available(): return "mps"
    return "cpu"

def save_lookup_tables(X_test, y_test, X_train, y_train, feature_names):
    """Saves test and train data to CSVs with explicit indices for cross-referencing."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # 1. Save Test Set (Rows in Heatmap)
    # We add an 'Experiment_Index' (0-56) to match the heatmap X-axis
    df_test = pd.DataFrame(X_test, columns=feature_names)
    df_test.insert(0, 'Heatmap_Index', np.arange(len(X_test)))
    df_test['True_Label'] = y_test
    # Map labels to names for readability
    df_test['Label_Name'] = df_test['True_Label'].map({0: 'Malignant', 1: 'Benign'}) 
    
    test_path = os.path.join(OUTPUT_DIR, "test_set_lookup.csv")
    df_test.to_csv(test_path, index=False)
    print(f"Saved Test Set Lookup -> {test_path}")

    # 2. Save Full Training Set (Source of Slices)
    # We add an 'Array_Index' so you know which rows corresponded to slice 20:60
    df_train = pd.DataFrame(X_train, columns=feature_names)
    df_train.insert(0, 'Array_Index', np.arange(len(X_train)))
    df_train['True_Label'] = y_train
    df_train['Label_Name'] = df_train['True_Label'].map({0: 'Malignant', 1: 'Benign'})
    
    train_path = os.path.join(OUTPUT_DIR, "training_data_full.csv")
    df_train.to_csv(train_path, index=False)
    print(f"Saved Training Data Lookup -> {train_path}")

def main():
    device = get_default_device()
    
    # 1. Setup Model
    model = NanoTabPFNModel(
        embedding_size=96,
        num_attention_heads=4,
        mlp_hidden_size=192,
        num_layers=3,
        num_outputs=2
    )
    
    try:
        model.load_state_dict(torch.load("nanotabpfn_weights.pth", map_location=device))
        print("Weights loaded successfully.")
    except FileNotFoundError:
        print("Error: weights file not found.")
        return

    model.to(device)
    model.eval()

    # 2. Setup Data
    data = load_breast_cancer()
    X, y = data.data, data.target
    feature_names = data.feature_names
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.08, random_state=42)
    save_lookup_tables(X_test, y_test, X_train, y_train, feature_names)
    print(f"Test set size: {len(X_test)}")
    
    # 3. Experiment Loop
    step = 50
    window_size = 50
    start_idxs = list(range(1, 470, step))
    
    # Matrix to store correctness: 0 = Correct, 1 = Error
    # Shape: (Num_Slices, Num_Test_Samples)
    error_matrix = []
    slice_labels = []

    print("Running sliding window inference...")
    
    for start_i in start_idxs:
        end_i = start_i + window_size
        X_train_cut = X_train[start_i:end_i]
        y_train_cut = y_train[start_i:end_i]

        # Safety: Skip if only 1 class is present
        if len(set(y_train_cut)) < 2:
            print(f"Skipping slice {start_i}-{end_i}: Lacks class diversity.")
            error_matrix.append([-1] * len(y_test)) # -1 for invalid/skipped
            slice_labels.append(f"{start_i}-{end_i} (Skip)")
            continue

        classifier = NanoTabPFNClassifier(model, device)
        classifier.fit(X_train_cut, y_train_cut)
        preds = classifier.predict(X_test)
        
        # Determine Errors (1 if Wrong, 0 if Correct)
        errors = (preds != y_test).astype(int)
        
        error_matrix.append(errors)
        slice_labels.append(f"Train slice {start_i}-{end_i}")
        
        acc = accuracy_score(y_test, preds)
        print(f"Slice {start_i}-{end_i}: Acc {acc:.4f}")

    # 4. Convert to DataFrame for Plotting
    error_df = pd.DataFrame(np.array(error_matrix), index=slice_labels)
    
    # Filter out skipped rows if any
    if -1 in error_df.values:
        error_df = error_df[error_df.iloc[:, 0] != -1]

    # 5. VISUALIZATION
    plt.figure(figsize=(20, 10))
    
    # Create Heatmap
    # Red = Error (1), White = Correct (0)
    # cbar=False to keep it clean, we know the binary nature
    sns.heatmap(error_df, cmap="Reds", cbar_kws={'label': 'Misclassified'}, linewidths=0.5, linecolor='gray')
    
    plt.title("Error overlap analysis: which test samples fail consistently?", fontsize=16)
    plt.xlabel("Test sample index", fontsize=14)
    plt.ylabel("Context window", fontsize=14)
    
    # Highlight "Hard" samples (columns that are mostly red)
    plt.tight_layout()
    plt.savefig(f"error_heatmap_step_{step}_window_{window_size}.png", dpi=300)
    print("\nSaved 'error_heatmap.png'.")
    
    # 6. Summary Stats
    failure_rates = error_df.mean(axis=0)
    hard_samples = failure_rates[failure_rates > 0.5].index.tolist()
    print("\n--- INSIGHTS ---")
    print(f"Total Test Samples: {len(X_test)}")
    print(f"Consistently 'Hard' Samples (Failed >50% of the time): {hard_samples}")
    print(f"These are the indices you should inspect with LRP!")

if __name__ == "__main__":
    main()