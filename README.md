# Day 6 - Federated MLLM for Clinical AI

This folder contains the Day 6 progress on the Federated Multimodal Large Language Model (MLLM) project for clinical decision support.

## Structure

- `data/`: contains the `mednli_train_clean.jsonl` dataset (upload manually).
- `src/day6_federated_training.py`: script to preprocess and tokenize data using Bio_ClinicalBERT.
- `models/`: to store trained models.
- `notebooks/`: for Jupyter/Colab notebooks.
- `logs/`: for training logs.
- `plots/`: for visualizations.
- `configs/`: for YAML or JSON configuration files.

## How to Run

1. Upload your cleaned dataset (`mednli_train_clean.jsonl`) to the `data/` folder.
2. Run the script located in `src/day6_federated_training.py` on Google Colab or locally.
