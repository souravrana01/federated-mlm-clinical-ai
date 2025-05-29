# Federated MLLM for Clinical Decision Support

This project demonstrates a federated learning simulation using Flower, PyTorch, Hugging Face Transformers, and the MedNLI dataset. It focuses on privacy-preserving training of multimodal large language models (MLLMs) in a clinical setting.

## 🧠 Project Objectives

- Leverage **Bio_ClinicalBERT** for medical natural language inference (NLI)
- Simulate a federated learning setup across multiple clients
- Evaluate model performance with standard metrics
- Align with the MSc dissertation on privacy-preserving federated learning in healthcare

## 🚀 How to Run

### 1. Clone this repository

```bash
git clone https://github.com/YOUR_USERNAME/federated-mlm-clinical-ai.git
cd federated-mlm-clinical-ai
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Start the federated server

```bash
python server.py
```

### 4. Start one or more federated clients in separate terminals

```bash
python client.py
```

## 📦 Project Structure

- `client.py` - Federated client training on local MedNLI data
- `server.py` - Flower-based federated server logic
- `model.py` - ClinicalBERT classification model
- `dataset_utils.py` - Data preprocessing and loading functions
- `requirements.txt` - Dependencies list
- `README.md` - This file

## 🛡️ Ethics and Privacy

This project simulates federated training, aiming to ensure patient data privacy and comply with clinical data handling norms. No private data is used.

## 📜 License

This project is licensed under the MIT License. See `LICENSE` for details.
