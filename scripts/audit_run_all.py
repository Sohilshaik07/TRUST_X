"""
Audit helper: verify dataset sizes and config data_dir paths.
DO NOT modify any data or experiment outputs.
"""
import pandas as pd
import yaml
import os

datasets = {
    'telugu':    'Data/processed/telugu',
    'tamil':     'Data/processed/tamil',
    'malayalam': 'Data/processed/malayalam',
}

print('=== PROCESSED DATASET SIZES ===')
for lang, path in datasets.items():
    train = pd.read_csv(f'{path}/train.csv')
    dev   = pd.read_csv(f'{path}/dev.csv')
    test  = pd.read_csv(f'{path}/test.csv')
    n0 = (train['label']==0).sum()
    n1 = (train['label']==1).sum()
    pct_hate = 100 * n1 / len(train)
    print(f'{lang.upper():<12} train={len(train):6d} (0={n0}, 1={n1}, {pct_hate:.1f}% harmful) | dev={len(dev):5d} | test={len(test):5d}')

print()
print('=== FULL CONFIG COMPARISON TABLE ===')
configs = [
    'telugu_mbert','telugu_indicbert','telugu_xlmr',
    'tamil_mbert','tamil_indicbert','tamil_xlmr',
    'malayalam_mbert','malayalam_indicbert','malayalam_xlmr'
]
header = f"{'experiment_id':<35} {'data_dir':<30} {'model_key':<10} {'seed':<5} {'epochs':<7} {'batch':<6} {'lr':<10} {'max_len':<8} {'class_wt':<9} {'grad_acc'}"
print(header)
print('-' * len(header))
for cfg in configs:
    with open(f'configs/{cfg}.yaml') as f:
        c = yaml.safe_load(f)
    row = (f"{c['experiment_id']:<35} {c['data_dir']:<30} {c['model_key']:<10} "
           f"{c['seed']:<5} {c['epochs']:<7} {c['batch_size']:<6} "
           f"{c['learning_rate']:<10} {c['max_length']:<8} "
           f"{str(c.get('use_class_weights','?')):<9} {c.get('gradient_accumulation_steps','?')}")
    print(row)

print()
print('=== OUTPUT_DIR UNIQUENESS CHECK ===')
output_dirs = []
for cfg in configs:
    with open(f'configs/{cfg}.yaml') as f:
        c = yaml.safe_load(f)
    output_dirs.append(c['output_dir'])

if len(output_dirs) == len(set(output_dirs)):
    print('PASS: All 9 output_dirs are unique — no checkpoint collision.')
else:
    print('FAIL: Duplicate output_dirs detected!')
    seen = set()
    for d in output_dirs:
        if d in seen:
            print(f'  DUPLICATE: {d}')
        seen.add(d)
for d in output_dirs:
    print(f'  {d}')

print()
print('=== EXPERIMENT_ID UNIQUENESS CHECK ===')
exp_ids = []
for cfg in configs:
    with open(f'configs/{cfg}.yaml') as f:
        c = yaml.safe_load(f)
    exp_ids.append(c['experiment_id'])
if len(exp_ids) == len(set(exp_ids)):
    print('PASS: All 9 experiment_ids are unique.')
else:
    print('FAIL: Duplicate experiment_ids!')
for e in exp_ids:
    print(f'  {e}')

print()
print('=== RUN SEQUENCE (as executed by orchestrator) ===')
langs  = ['telugu','tamil','malayalam']
models = ['mbert','indicbert','xlmr']
print('Order | Config File               | Experiment ID')
print('-'*70)
seq = 1
for lang in langs:
    for model in models:
        cfg_name = f'{lang}_{model}.yaml'
        with open(f'configs/{cfg_name}') as f:
            c = yaml.safe_load(f)
        print(f'  {seq:2d}  | {cfg_name:<28} | {c["experiment_id"]}')
        seq += 1

print()
print('=== TELUGU PILOT CHECK: IS telugu_mbert_seed42 IN THE RUN? ===')
pilot_in_run = 'telugu_mbert_seed42' in exp_ids
print(f'telugu_mbert_seed42 in experiment list: {pilot_in_run}')
print(f'Position in run sequence: 1 (first experiment)')
