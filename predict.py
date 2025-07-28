import torch
import os
import pickle
import torch.nn as nn
from torch.nn.functional import softmax

# ==== Model Definition ====
class SimpleClassifier(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(SimpleClassifier, self).__init__()
        self.fc = nn.Linear(input_dim, num_classes)
    
    def forward(self, x):
        return self.fc(x)

# ==== Load model and tokenizer ====
MODEL_DIR = os.path.join(os.path.dirname(__file__), "model")
MODEL_PATH = os.path.join(MODEL_DIR, "model.pt")
TOKENIZER_PATH = os.path.join(MODEL_DIR, "tokenizer.pkl")

model = SimpleClassifier(input_dim=768, num_classes=3)
model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
model.eval()

with open(TOKENIZER_PATH, "rb") as f:
    tokenizer = pickle.load(f)

LABELS = ["Termination", "License", "Liability"]

def classify_clauses(clauses):
    X = tokenizer.transform(clauses).toarray()
    X_tensor = torch.tensor(X, dtype=torch.float32)

    with torch.no_grad():
        logits = model(X_tensor)
        probs = softmax(logits, dim=1)

    preds = torch.argmax(probs, dim=1)
    confidences = torch.max(probs, dim=1).values

    return [
        {"label": LABELS[p.item()], "confidence": round(c.item(), 3)}
        for p, c in zip(preds, confidences)
    ]
