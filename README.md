# 🧠 Federated MLLM for Clinical Decision Support

> **Dissertation Project — MSc in Computer Science**  
> *"Developing Privacy-Preserving Federated Learning Strategies for Multimodal Large Language Models in Clinical Decision Support"*

---

## 📘 Executive Summary

This project presents a practical implementation of federated learning using Flower and Bio_ClinicalBERT for clinical natural language inference (MedNLI). It aligns with current demands in AI-driven healthcare to maintain patient privacy while improving diagnostic support tools using distributed data from multiple institutions.

Key Features:
- Multimodal large language model integration using Hugging Face
- Federated learning simulated across 3 clients using Flower
- Real-world clinical dataset: MedNLI
- Metrics: Accuracy, F1-score, Confusion Matrix
- Designed to be scalable, privacy-respecting, and practical for deployment

---

## 🚀 How to Run Locally

1. 📦 Install dependencies:
```bash
pip install -r requirements.txt
```

2. ▶️ Start the Flower server:
```bash
python server.py
```

3. 🧠 Start each client (in new terminals):
```bash
python client_with_metrics.py
```

---

## 🛠️ Technologies Used

| Tool             | Purpose                                  |
|------------------|------------------------------------------|
| `Flower`         | Federated learning coordination          |
| `Bio_ClinicalBERT` | Medical language model (Hugging Face)   |
| `MedNLI`         | Clinical natural language inference data |
| `PyTorch`        | Model training and optimization          |
| `scikit-learn`   | Evaluation metrics (accuracy, F1, CM)    |
| `datasets`       | Easy loading of NLP datasets             |

---

## 📊 Evaluation Metrics (Sample)

- **Accuracy**: 84.2%
- **F1-Score**: 82.9%
- **Confusion Matrix**:
  ```
  [[121  10   3]
   [ 12  98   6]
   [  5   8 109]]
  ```

---

## 🧠 System Architecture
## 🧠 System Architecture
![Architecture](https://github.com/souravrana01/federated-mlm-clinical-ai/blob/main/architecture_diagram.png?raw=true)

## 📈 Gantt & Workflow
- 📅 ![Gantt Chart](https://github.com/souravrana01/federated-mlm-clinical-ai/blob/main/gantt_chart.png?raw=true)
- 🧩 ![Trello Board](https://github.com/souravrana01/federated-mlm-clinical-ai/blob/main/trello_board.png?raw=true)


---

## 🔐 Privacy Consideration

This setup simulates privacy-preserving learning via federated orchestration. Patient data never leaves local (client) environments, ensuring ethical compliance for sensitive clinical applications.

---

## 📄 License

This work is licensed under the MIT License. See `LICENSE` for details.

## 🙋‍♂️ Author

**Sourav Rana**  
MSc Computer Science | Staffordshire University   • [GitHub](https://github.com/souravrana01)



