from huggingface_hub import upload_folder

upload_folder(
    folder_path="models/legalbert_clean",  # <-- update to your clean folder
    repo_id="hardoc/legalbert-clause-classifier",
    path_in_repo=".",  # uploads files to root of repo
    repo_type="model",
)
