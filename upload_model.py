from huggingface_hub import upload_folder

upload_folder(
    repo_id="hardoc/legalbert-clause-classifier",  # use your actual model repo name
    folder_path="backend/models/legalbert_clause_classifier",
    repo_type="model",
    commit_message="Upload recovered model files"
)

print("✅ Model uploaded to Hugging Face.")
