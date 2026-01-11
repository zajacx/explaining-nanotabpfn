import torch
from model import NanoTabPFNModel, NanoTabPFNClassifier
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

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

X, y = load_breast_cancer(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

print("Test batch size:", len(X_test))

for i in range(20, 400, 20):
    X_train_cut = X_train[i:i+40]
    y_train_cut = y_train[i:i+40]
    classifier = NanoTabPFNClassifier(model, device)
    classifier.fit(X_train_cut, y_train_cut)
    preds = classifier.predict(X_test)
    print(f"Accuracy for train size {i}: {accuracy_score(y_test, preds)}")