#!/bin/bash

echo "🧹 Cleaning and restructuring LegalLens-AI..."

# Step 1: Delete unnecessary files and folders
rm -rf data.zip
rm -f contract_review.png
rm -f upload_to_hf.py
rm -rf legalbert_clean
rm -f .gitattributes
rm -f logo192.png logo512.png
rm -f evaluate.py train.py utils.py

# Step 2: Move frontend
mv legal-lens-ui frontend

# Step 3: Create backend structure
mkdir -p backend/src
mkdir -p backend/models

# Step 4: Move backend scripts
mv src/*.py backend/src/
mv models/* backend/models/
rm -rf src models

# Step 5: Move Docker and backend config files
mv Dockerfile backend/
mv render.yaml backend/
mv requirements.txt backend/
mv .dockerignore backend/

# Step 6: Clean notebooks/data → data/
mkdir -p data
mv notebooks/data/*.csv data/
rm -rf notebooks

# Step 7: Top-level organization
mv cuad_v1.csv data/ 2>/dev/null
mv test.json data/ 2>/dev/null
mv CUADV1.json data/ 2>/dev/null
mv train_separate_questions* data/ 2>/dev/null

echo "✅ Done. Your repo is now structured for top-tier polish!"
