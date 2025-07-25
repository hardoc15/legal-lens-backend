from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch

def predict(text, model_path="models/legalbert_clause_classifier"):
    HF_MODEL = "hardoc/legalbert-clause-classifier"

    tokenizer = AutoTokenizer.from_pretrained(HF_MODEL)
    model = AutoModelForSequenceClassification.from_pretrained(HF_MODEL)    

    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=256)

    with torch.no_grad():
        logits = model(**inputs).logits
        predicted_class = torch.argmax(logits, dim=1).item()
        confidence = torch.softmax(logits, dim=1).squeeze()[predicted_class].item()

    return predicted_class, confidence

if __name__ == "__main__":
    example = "The buyer must notify the seller 30 days before termination."
    label, confidence = predict(example)
    print(f"Prediction: {label} (confidence: {confidence:.2f})")
