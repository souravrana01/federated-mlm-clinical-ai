# Federated Clinical BERT - Day 1 Demo

This is a demonstration of how federated learning works using Flower, PyTorch, and a mock ClinicalBERT structure. It simulates training across 3 clients using fake sentence classification data, useful for understanding federated learning in healthcare AI.

## Files
- `train.py`: Main script to simulate local training and federated averaging.
- `requirements.txt`: Python libraries needed to run the code.
- `README.md`: This file.

## Run Instructions
1. Install requirements:
```bash
pip install -r requirements.txt
```

2. Run simulation:
```bash
python train.py
```

## Dataset
Today's simulation uses fake data to demonstrate working federated logic before applying real medical data (MedNLI / MedQA / etc.).
