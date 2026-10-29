"""
Telugu processed dataset audit script for TRUST-X pilot audit.
DO NOT modify experiment outputs.
"""
import pandas as pd
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

train = pd.read_csv('Data/processed/telugu/train.csv')
dev   = pd.read_csv('Data/processed/telugu/dev.csv')
test  = pd.read_csv('Data/processed/telugu/test.csv')

print('=== TELUGU PROCESSED DATASET AUDIT ===')
print(f'Train: {len(train)} rows | Label 0: {(train.label==0).sum()} | Label 1: {(train.label==1).sum()}')
print(f'Dev:   {len(dev)} rows   | Label 0: {(dev.label==0).sum()} | Label 1: {(dev.label==1).sum()}')
print(f'Test:  {len(test)} rows  | Label 0: {(test.label==0).sum()} | Label 1: {(test.label==1).sum()}')
print(f'Total: {len(train)+len(dev)+len(test)}')

print()
print('Columns:', list(train.columns))
print('Label values in train:', sorted(train['label'].unique().tolist()))
print('Label values in dev:',   sorted(dev['label'].unique().tolist()))
print('Label values in test:',  sorted(test['label'].unique().tolist()))

# Cross-split deduplication check
train_texts = set(train['text'].str.strip())
dev_texts   = set(dev['text'].str.strip())
test_texts  = set(test['text'].str.strip())

train_dev_overlap  = train_texts & dev_texts
train_test_overlap = train_texts & test_texts
dev_test_overlap   = dev_texts & test_texts

print()
print('=== CROSS-SPLIT OVERLAP CHECK ===')
print(f'Train-Dev text overlap:  {len(train_dev_overlap)} samples')
print(f'Train-Test text overlap: {len(train_test_overlap)} samples')
print(f'Dev-Test text overlap:   {len(dev_test_overlap)} samples')

# Within-split duplicate check
print()
print('=== WITHIN-SPLIT DUPLICATE CHECK ===')
print(f'Train intra-duplicate texts: {train["text"].duplicated().sum()}')
print(f'Dev intra-duplicate texts:   {dev["text"].duplicated().sum()}')
print(f'Test intra-duplicate texts:  {test["text"].duplicated().sum()}')

# Label mapping examples
print()
print('=== LABEL MAPPING (unique original_label -> label) ===')
mapping_check = train.groupby(['original_label','label']).size().reset_index(name='count')
print(mapping_check.to_string())

# Class balance percentages
print()
print('=== CLASS BALANCE ===')
for split_name, df in [('train', train), ('dev', dev), ('test', test)]:
    n0 = (df['label']==0).sum()
    n1 = (df['label']==1).sum()
    total = len(df)
    print(f'{split_name}: {total} | non-hate(0)={n0} ({100*n0/total:.1f}%) | hate(1)={n1} ({100*n1/total:.1f}%)')

# Verify ID column format
print()
print('=== ID COLUMN FORMAT ===')
print('Train IDs sample:', train['id'].head(3).tolist())
print('Dev IDs sample:  ', dev['id'].head(3).tolist())
print('Test IDs sample: ', test['id'].head(3).tolist())
