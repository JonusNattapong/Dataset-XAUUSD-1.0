from datasets import load_dataset

print('Loading local dataset script...')
# Use path to local script file
ds = load_dataset(r'd:\GitHub\Dataset-XAUUSD\xauusd_dataset.py')
print('Loaded splits:', list(ds.keys()))
print('Columns (first 10):', ds['train'].column_names[:10])
print('First example keys (first 10):', list(ds['train'][0].keys())[:10])
print('Number of examples in train split:', len(ds['train']))
