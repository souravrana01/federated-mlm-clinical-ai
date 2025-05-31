# Day 2: Practical - Loading Local MedNLI Dataset
# Author: Sourav Rana | MSc Dissertation

import json
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer

# Configuration
MODEL_NAME = "emilyalsentzer/Bio_ClinicalBERT"
BATCH_SIZE = 4

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

# Custom Dataset Class
class MedNLIDataset(Dataset):
    def __init__(self, jsonl_path):
        with open(jsonl_path, 'r') as file:
            self.data = [json.loads(line) for line in file]

        self.samples = [
            tokenizer(
                item['sentence1'], item['sentence2'],
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            )
            for item in self.data
        ]
        self.labels = [item['gold_label'] for item in self.data]

        self.label_map = {'entailment': 0, 'neutral': 1, 'contradiction': 2}
        self.labels = [self.label_map.get(label, 1) for label in self.labels]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        return {
            'input_ids': sample['input_ids'].squeeze(),
            'attention_mask': sample['attention_mask'].squeeze(),
            'label': torch.tensor(self.labels[idx])
        }

# Dataset path
dataset_path = "mednli_train.jsonl"  # Ensure this file is uploaded in your Colab or local dir

# Usage Example
if __name__ == "__main__":
    dataset = MedNLIDataset(dataset_path)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    for batch in dataloader:
        print(batch)
        break
