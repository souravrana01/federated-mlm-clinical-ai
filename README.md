# Federated MLLM for Clinical Decision Support - MPR Practical Repository

This repository contains all practical implementation files for the MSc Midpoint Progress Review (MPR).

## 📁 Folder Structure

- `data/` – Contains cleaned dataset (`mednli_train_clean.jsonl`)
- `notebooks/` – Daily development notebooks (Colab-compatible)
- `models/` – Model checkpoints and configurations
- `scripts/` – Utility scripts for training and evaluation
- `results/` – Logs, plots, and performance metrics
- `docs/` – Literature tables, diagrams, and system architecture

## ✅ Summary of Implemented Components

- Federated learning simulation with Flower using BioClinicalBERT
- Dataset cleaning and tokenization
- Secure Aggregation and Differential Privacy integration
- Multimodal support under development

## ⚙️ Environment Requirements

```bash
pip install transformers datasets torch flower
```

## 📌 Dataset

Uses a preprocessed version of the MedNLI dataset stored in `data/mednli_train_clean.jsonl`.

## 🔄 To Run

Upload the code notebooks from `notebooks/` to Google Colab or run scripts locally via:

```bash
python scripts/train_federated.py
```

