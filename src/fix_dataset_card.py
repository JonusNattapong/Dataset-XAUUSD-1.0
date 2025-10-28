#!/usr/bin/env python3
"""
Create a proper dataset card for Hugging Face datasets compatibility
"""

from huggingface_hub import HfApi
import json

def create_proper_dataset_card():
    """Create a proper dataset card that works with Hugging Face datasets library"""

    repo_id = "JonusNattapong/xauusd-dataset"

    # Create a proper dataset card with YAML metadata
    dataset_card = """---
dataset_info:
  features:
    - name: "Date"
      dtype: "string"
    - name: "Open"
      dtype: "float64"
    - name: "High"
      dtype: "float64"
    - name: "Low"
      dtype: "float64"
    - name: "Close"
      dtype: "float64"
    - name: "Volume"
      dtype: "int64"
    - name: "Price_Change_1d"
      dtype: "float64"
    - name: "Price_Change_1d_Pct"
      dtype: "float64"
    - name: "Target_Binary"
      dtype: "int64"
  splits:
    - name: "train"
      num_bytes: 2007880
      num_examples: 708
  download_size: 2007880
  dataset_size: 2007880
---

# XAUUSD Price Prediction Dataset

Comprehensive dataset for XAUUSD (Gold vs US Dollar) price prediction using machine learning techniques.

## Dataset Description

This dataset contains comprehensive XAUUSD (Gold vs US Dollar) price data with advanced features for machine learning applications.

### Key Features:
- **Time Period**: 2000-2025 (full dataset), 2023-2025 (recent data)
- **Data Frequency**: Daily, Weekly, Hourly
- **Features**: 172+ technical indicators, economic variables, news sentiment
- **Target Variables**: Price changes, binary classification targets

### Data Sources:
- **Primary**: Yahoo Finance (GC=F ticker for Gold futures)
- **Economic Data**: US Dollar Index (DXY), Treasury yields, Oil prices, Silver prices
- **News Sentiment**: Financial news analysis with sentiment scores

### Machine Learning Performance:
- **Ensemble Model Accuracy**: 47.3% directional accuracy
- **Features Used**: 172 advanced features including technical indicators and economic variables
- **Cross-validation**: Time series split with expanding window

## Files

- `XAUUSD_enhanced_ml_dataset_clean.csv`: Cleaned ML dataset (172 features, 2023-2025)
- `XAUUSD_comprehensive_dataset.csv`: Full comprehensive dataset (2000-2025)
- `XAUUSD_2023_2025_dataset.csv`: Recent data optimized for ML (2023-2025)
- `XAUUSD_full_dataset.csv`: Basic historical data (2000-2025)

## Usage

### Load with Hugging Face Datasets

```python
from datasets import load_dataset

# Load the dataset
dataset = load_dataset("JonusNattapong/xauusd-dataset")

# Access the data
print(dataset.keys())
print(dataset['train'][0])
```

### Load CSV files directly

```python
import pandas as pd

# Load specific dataset
df = pd.read_csv("https://huggingface.co/datasets/JonusNattapong/xauusd-dataset/resolve/main/XAUUSD_enhanced_ml_dataset_clean.csv")
print(df.head())
```

## Citation

If you use this dataset in your research, please cite:

```bibtex
@misc{{tapachoom2025xauusd,
  title={{XAUUSD Price Prediction Dataset}},
  author={{Tapachoom, Nattapong}},
  year={{2025}},
  publisher={{Hugging Face}},
  url={{https://huggingface.co/datasets/JonusNattapong/xauusd-dataset}}
}}
```

## License

This dataset is available under the MIT License for educational and research purposes.
"""

    # Save locally first
    with open("dataset_card.md", "w", encoding="utf-8") as f:
        f.write(dataset_card)

    # Upload to Hugging Face
    api = HfApi()
    api.upload_file(
        path_or_fileobj="dataset_card.md",
        path_in_repo="README.md",
        repo_id=repo_id,
        repo_type="dataset"
    )

    print("✅ Updated dataset card for better compatibility")

    # Clean up
    import os
    os.remove("dataset_card.md")

if __name__ == "__main__":
    create_proper_dataset_card()