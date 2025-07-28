from transformers import Trainer, TrainingArguments, AutoModelForSequenceClassification, AutoTokenizer
from datasets import Dataset
import json

def extract_samples_from_json(path):
    with open(path, "r") as f:
        raw_data = json.load(f)
    samples = []
    for doc in raw_data["data"]:
        for para in doc["paragraphs"]:
            for q in para["qas"]:
                question = q["question"]
                label = 1 if not q.get("is_impossible", False) else 0
                samples.append({"text": question, "label": label})
    return samples

def main():
    model_path = "models/legalbert_clause_classifier"
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    val_samples = extract_samples_from_json("data/test.json")
    val_dataset = Dataset.from_list(val_samples)

    def preprocess(example):
        return tokenizer(example["text"], truncation=True, padding="max_length", max_length=256)

    val_dataset = val_dataset.map(preprocess, batched=True, remove_columns=["text"])
    val_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])

    args = TrainingArguments(output_dir="./tmp", per_device_eval_batch_size=4)
    trainer = Trainer(model=model, args=args)

    results = trainer.evaluate(eval_dataset=val_dataset)
    print("Evaluation Results:")
    print(results)

if __name__ == "__main__":
    main()
