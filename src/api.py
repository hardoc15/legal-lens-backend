from flask import Flask, request, jsonify
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

model_path = "models/legalbert_clause_classifier"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSequenceClassification.from_pretrained(model_path)

@app.route("/predict", methods=["POST"])
def predict():
    data = request.json
    text = data.get("text", "")

    if not text.strip():
        return jsonify({"error": "Empty text"}), 400

    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=256)
    with torch.no_grad():
        logits = model(**inputs).logits
        predicted_class = torch.argmax(logits, dim=1).item()
        confidence = torch.softmax(logits, dim=1)[0][predicted_class].item()

    return jsonify({"label": predicted_class, "confidence": round(confidence, 4)})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
