#!/usr/bin/env python3
"""
Upload XAUUSD Dataset to Hugging Face Hub
"""

import os
import pandas as pd
from huggingface_hub import HfApi, create_repo
from pathlib import Path
import json

def create_dataset_card(dataset_name, description, features_info):
    """Create a dataset card in markdown format"""
    card_content = f"""---
dataset_info:
  features:
{features_info}
  splits:
    - name: train
      num_bytes: 0
      num_examples: 0
  download_size: 0
  dataset_size: 0
---

# XAUUSD Dataset - {dataset_name}

{description}

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

### Usage Examples:

```python
from datasets import load_dataset

# Load the dataset
dataset = load_dataset("nattapong/xauusd-dataset")

# Access the data
print(dataset['train'].column_names)
print(dataset['train'][0])
```

### Citation:

If you use this dataset in your research, please cite:

```bibtex
@misc{{tapachoom2025xauusd,
  title={{XAUUSD Price Prediction Dataset}},
  author={{Tapachoom, Nattapong}},
  year={{2025}},
  publisher={{Hugging Face}},
  url={{https://huggingface.co/datasets/nattapong/xauusd-dataset}}
}}
```

### License:
This dataset is available under the MIT License for educational and research purposes.
"""

    return card_content

def upload_dataset_to_hf():
    """Upload XAUUSD datasets to Hugging Face Hub"""

    # Initialize Hugging Face API
    api = HfApi()

    # Dataset repository name
    repo_name = "xauusd-dataset"
    repo_id = f"nattapong/{repo_name}"

    print(f"Creating repository: {repo_id}")

    # Create the repository
    try:
        create_repo(
            repo_id=repo_id,
            repo_type="dataset",
            private=False,
            exist_ok=True
        )
        print(f"Repository created: https://huggingface.co/datasets/{repo_id}")
    except Exception as e:
        print(f"Repository might already exist: {e}")

    # Dataset directory
    dataset_dir = Path("dataset")
    if not dataset_dir.exists():
        print("Dataset directory not found!")
        return

    # Files to upload with descriptions
    datasets_to_upload = {
        "XAUUSD_enhanced_ml_dataset_clean.csv": {
            "description": "Cleaned machine learning dataset with 172 features for XAUUSD price prediction (2023-2025). This is the primary dataset for ML model training.",
            "features": """
    - name: Date
      dtype: string
    - name: Open
      dtype: float64
    - name: High
      dtype: float64
    - name: Low
      dtype: float64
    - name: Close
      dtype: float64
    - name: Volume
      dtype: int64
    - name: Price_Change_1d
      dtype: float64
    - name: Price_Change_1d_Pct
      dtype: float64
    - name: Target_Binary
      dtype: int64
    - name: RSI
      dtype: float64
    - name: MACD
      dtype: float64
    - name: Bollinger_Upper
      dtype: float64
    - name: Bollinger_Lower
      dtype: float64
    - name: SMA_20
      dtype: float64
    - name: EMA_20
      dtype: float64
    - name: DXY
      dtype: float64
    - name: US10Y
      dtype: float64
    - name: WTI_Oil
      dtype: float64
    - name: Silver
      dtype: float64
"""
        },
        "XAUUSD_comprehensive_dataset.csv": {
            "description": "Comprehensive XAUUSD dataset with all features including technical indicators, economic data, and news sentiment (2000-2025).",
            "features": """
    - name: Date
      dtype: string
    - name: Open
      dtype: float64
    - name: High
      dtype: float64
    - name: Low
      dtype: float64
    - name: Close
      dtype: float64
    - name: Volume
      dtype: int64
    - name: RSI_14
      dtype: float64
    - name: MACD_12_26_9
      dtype: float64
    - name: BB_Upper_20_2
      dtype: float64
    - name: BB_Lower_20_2
      dtype: float64
    - name: SMA_50
      dtype: float64
    - name: EMA_50
      dtype: float64
    - name: DXY
      dtype: float64
    - name: US10Y_Yield
      dtype: float64
    - name: WTI_Oil_Price
      dtype: float64
    - name: Silver_Price
      dtype: float64
    - name: News_Sentiment
      dtype: float64
"""
        },
        "XAUUSD_2023_2025_dataset.csv": {
            "description": "Recent XAUUSD dataset (2023-2025) with 119 features, optimized for current market conditions and ML model training.",
            "features": """
    - name: Date
      dtype: string
    - name: Open
      dtype: float64
    - name: High
      dtype: float64
    - name: Low
      dtype: float64
    - name: Close
      dtype: float64
    - name: Adj_Close
      dtype: float64
    - name: Volume
      dtype: int64
    - name: Price_Change_1d
      dtype: float64
    - name: Price_Change_1d_Pct
      dtype: float64
    - name: Target_Binary
      dtype: int64
    - name: RSI_14
      dtype: float64
    - name: MACD_12_26_9
      dtype: float64
    - name: Bollinger_Upper_20
      dtype: float64
    - name: Bollinger_Lower_20
      dtype: float64
"""
        },
        "XAUUSD_full_dataset.csv": {
            "description": "Complete historical XAUUSD dataset from 2000 to 2025 with basic OHLCV data.",
            "features": """
    - name: Date
      dtype: string
    - name: Open
      dtype: float64
    - name: High
      dtype: float64
    - name: Low
      dtype: float64
    - name: Close
      dtype: float64
    - name: Adj Close
      dtype: float64
    - name: Volume
      dtype: int64
"""
        }
    }

    # Upload each dataset
    for filename, info in datasets_to_upload.items():
        file_path = dataset_dir / filename
        if file_path.exists():
            print(f"Uploading {filename}...")

            # Upload the file
            api.upload_file(
                path_or_fileobj=str(file_path),
                path_in_repo=filename,
                repo_id=repo_id,
                repo_type="dataset"
            )
            print(f"✓ Uploaded {filename}")
        else:
            print(f"⚠ File not found: {filename}")

    # Create and upload README.md
    readme_content = create_dataset_card(
        "XAUUSD Price Prediction Dataset",
        "Comprehensive dataset for XAUUSD (Gold vs US Dollar) price prediction using machine learning techniques.",
        datasets_to_upload["XAUUSD_enhanced_ml_dataset_clean.csv"]["features"]
    )

    # Save README.md locally first
    readme_path = Path("README_HF.md")
    with open(readme_path, 'w', encoding='utf-8') as f:
        f.write(readme_content)

    # Upload README.md
    api.upload_file(
        path_or_fileobj=str(readme_path),
        path_in_repo="README.md",
        repo_id=repo_id,
        repo_type="dataset"
    )
    print("✓ Uploaded README.md")

    # Clean up temporary file
    readme_path.unlink()

    print("\n🎉 Dataset upload complete!")
    print(f"📊 View your dataset at: https://huggingface.co/datasets/{repo_id}")
    print("\n📈 Dataset includes:")
    for filename in datasets_to_upload.keys():
        print(f"   • {filename}")

if __name__ == "__main__":
    upload_dataset_to_hf()