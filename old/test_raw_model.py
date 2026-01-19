import torch
import numpy as np
# Assuming the class definitions you provided are saved in 'model.py'
from model import NanoTabPFNModel 
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

def get_default_device():
    if torch.cuda.is_available(): return "cuda"
    if torch.backends.mps.is_available(): return "mps"
    return "cpu"

device = get_default_device()

# 1. Initialize Model
# We must match the hyperparameters used during training
model = NanoTabPFNModel(
    embedding_size=96,
    num_attention_heads=4,
    mlp_hidden_size=192,
    num_layers=3,
    num_outputs=2
)

print("Loading model weights...")
try:
    model.load_state_dict(torch.load("nanotabpfn_weights.pth", map_location=device))
    print("Success: Weights loaded.")
except FileNotFoundError:
    print("Error: 'nanotabpfn_weights.pth' not found. Please ensure weights exist.")
    exit()

model.to(device)
model.eval()

# 2. Prepare Data
X, y = load_breast_cancer(return_X_y=True)
# NanoTabPFN expects Float inputs and typically Float/Long targets depending on usage,
# but your TargetEncoder code takes y_train and computes mean, implying it treats labels as continuous/embedding.
X = X.astype(np.float32)
y = y.astype(np.float32) 

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

print(f"Test set size: {len(X_test)}")
print("Starting raw inference loop (Bypassing wrapper limits)...\n")

# 3. Loop through increasing context sizes
for i in range(20, 500, 20):
    
    # A. Define Context for this iteration
    X_context_np = X_train[:i]
    y_context_np = y_train[:i]
    
    # Calculate the Split Index: This tells the model "The first i rows are history"
    split_index = len(X_context_np)
    
    current_preds = []
    
    # B. Inference Loop: Process each test sample individually
    for j in range(len(X_test)):
        
        # 1. Get Query Sample
        X_query_sample = X_test[j].reshape(1, -1) # Shape (1, Features)
        
        # 2. Construct X Input (Context + Query)
        X_full = np.concatenate([X_context_np, X_query_sample], axis=0)
        
        # 3. Construct Y Input (Context ONLY)
        # CRITICAL CHANGE: We do NOT append a dummy '0' for the query.
        # The 'TargetEncoder' inside the model will detect the length mismatch 
        # between X_full (N+1) and y_context (N) and automatically pad the rest.
        y_input = y_context_np # Shape (N,)
        
        # 4. Convert to Torch Tensors & Add Batch Dimension
        # Model expects: 
        #   x: (Batch, Rows, Cols)
        #   y: (Batch, Rows, 1) or (Batch, Rows) -> unsqueezed inside
        X_tensor = torch.from_numpy(X_full).unsqueeze(0).to(device)
        y_tensor = torch.from_numpy(y_input).unsqueeze(0).to(device)
        
        # 5. Forward Pass
        with torch.no_grad():
            # Pass the split index explicitly as required by your new forward signature
            # Returns logits for the query rows (after split_index)
            output = model((X_tensor, y_tensor), train_test_split_index=split_index)
        
        # 6. Extract Prediction
        # Output shape from model.forward: (Batch, Num_Query_Rows, Num_Classes)
        # Since we passed 1 query row, shape is (1, 1, Num_Classes)
        prediction_logits = output[0, 0, :]
        
        # We assume binary classification (0 vs 1), so we take argmax
        predicted_class = prediction_logits.argmax().item()
        current_preds.append(predicted_class)

    # C. Calculate Accuracy
    acc = accuracy_score(y_test, current_preds)
    print(f"Context Size {i}: Accuracy = {acc:.4f}")