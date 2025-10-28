from datasets import load_dataset

CSV_URL = "https://huggingface.co/datasets/JonusNattapong/xauusd-dataset/resolve/main/XAUUSD_enhanced_ml_dataset_clean.csv"

print('Loading dataset using datasets.csv loader...')
dataset = load_dataset('csv', data_files=CSV_URL)
print('Splits:', list(dataset.keys()))
print('Columns (first 10):', dataset['train'].column_names[:10])
print('First example keys (first 10):', list(dataset['train'][0].keys())[:10])
print('Number of examples:', len(dataset['train']))
