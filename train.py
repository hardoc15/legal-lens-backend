from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
)
from datasets import Dataset
import evaluate
import numpy as np
import json
import time


def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    preds = np.argmax(predictions, axis=1)
    accuracy = evaluate.load("accuracy")
    return accuracy.compute(predictions=preds, references=labels)

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
    start_time = time.time()

    model_name = "nlpaueb/legal-bert-base-uncased"
    print("Loading and processing JSON data...")

    train_samples = extract_samples_from_json("data/CUADV1.json")
    val_samples = extract_samples_from_json("data/test.json")

    train_dataset = Dataset.from_list(train_samples[:200])
    val_dataset = Dataset.from_list(val_samples[:50])

    print(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    def preprocess(example):
        return tokenizer(example["text"], truncation=True, padding="max_length", max_length=256)

    print("Tokenizing...")
    train_dataset = train_dataset.map(preprocess, batched=True, remove_columns=["text"])
    val_dataset = val_dataset.map(preprocess, batched=True, remove_columns=["text"])

    train_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])
    val_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])

    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

    training_args = TrainingArguments(
        output_dir="models/legalbert_clause_classifier",
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="epoch",
        logging_dir="./logs",
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        num_train_epochs=1,
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        report_to="none",
        dataloader_num_workers=0,
        disable_tqdm=False
    )

    print("Starting training...")
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
    )

    trainer.train()
    print("Training complete.")

    trainer.save_model("models/legalbert_clause_classifier")
    print("Model saved.")

    print(f"Total time: {time.time() - start_time:.2f} seconds")

if __name__ == "__main__":
    main()
