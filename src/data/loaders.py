"""
TRUST-X Data Loaders Module
Defines PyTorch Dataset and DataLoader creation for multilingual code-mixed text.
"""

import os
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

class CodeMixedDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=128):
        self.texts = list(texts)
        self.labels = list(labels)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = int(self.labels[idx])

        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )

        item = {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'label': torch.tensor(label, dtype=torch.long)
        }

        # token_type_ids if available (e.g. for mBERT)
        if 'token_type_ids' in encoding:
            item['token_type_ids'] = encoding['token_type_ids'].flatten()

        return item

def get_dataloaders(data_dir, tokenizer, batch_size=16, max_length=128, num_workers=0):
    """
    Loads train, dev, and test CSVs from data_dir and creates PyTorch DataLoaders.
    """
    train_path = os.path.join(data_dir, "train.csv")
    dev_path = os.path.join(data_dir, "dev.csv")
    test_path = os.path.join(data_dir, "test.csv")

    train_df = pd.read_csv(train_path)
    dev_df = pd.read_csv(dev_path)
    test_df = pd.read_csv(test_path)

    train_dataset = CodeMixedDataset(train_df['text'], train_df['label'], tokenizer, max_length)
    dev_dataset = CodeMixedDataset(dev_df['text'], dev_df['label'], tokenizer, max_length)
    test_dataset = CodeMixedDataset(test_df['text'], test_df['label'], tokenizer, max_length)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    dev_loader = DataLoader(dev_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return {
        'train_loader': train_loader,
        'dev_loader': dev_loader,
        'test_loader': test_loader,
        'train_df': train_df,
        'dev_df': dev_df,
        'test_df': test_df
    }
