import torch
from model import NanoTabPFNModel, NanoTabPFNClassifier

def get_default_device():
    if torch.cuda.is_available(): return "cuda"
    if torch.backends.mps.is_available(): return "mps"
    return "cpu"

device = get_default_device()

model = NanoTabPFNModel(
    embedding_size=96,
    num_attention_heads=4,
    mlp_hidden_size=192,
    num_layers=3,
    num_outputs=2
)

print("Loading model weights...")
model.load_state_dict(torch.load("nanotabpfn_weights.pth", map_location=device))
model.to(device)
model.eval() # Crucial for inference

classifier = NanoTabPFNClassifier(model, device)

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

X, y = load_breast_cancer(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)

# TabPFN works by passing a small training set at inference time (in-context learning)
classifier.fit(X_train, y_train)
preds = classifier.predict(X_test)

print(f"Loaded model accuracy: {accuracy_score(y_test, preds)}")