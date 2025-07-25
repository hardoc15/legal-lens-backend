from huggingface_hub import HfApi, HfFolder, create_repo, upload_folder

# Replace with your Hugging Face username or org
model_repo = "hardoc/legalbert-clause-classifier"

# Upload the entire folder
upload_folder(
    repo_id=model_repo,
    folder_path="models/legalbert_clause_classifier",
    path_in_repo=".",  # Upload contents to root
)
