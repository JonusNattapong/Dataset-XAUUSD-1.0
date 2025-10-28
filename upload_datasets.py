#!/usr/bin/env python3
"""
Upload XAUUSD Datasets to Hugging Face Hub
Run this after setting up authentication with setup_hf_auth.py
"""

import os
from pathlib import Path
from huggingface_hub import HfApi

def upload_datasets():
    """Upload XAUUSD datasets to Hugging Face"""

    repo_id = "JonusNattapong/xauusd-dataset"

    print(f"📤 Uploading datasets to: {repo_id}")
    print("=" * 50)

    # Initialize API
    api = HfApi()

    # Dataset directory
    dataset_dir = Path("dataset")

    # Files to upload
    datasets_to_upload = [
        "XAUUSD_enhanced_ml_dataset_clean.csv",
        "XAUUSD_comprehensive_dataset.csv",
        "XAUUSD_2023_2025_dataset.csv",
        "XAUUSD_full_dataset.csv"
    ]

    uploaded_files = []

    for filename in datasets_to_upload:
        file_path = dataset_dir / filename
        if file_path.exists():
            print(f"📁 Uploading {filename}...")

            try:
                # Upload file
                api.upload_file(
                    path_or_fileobj=str(file_path),
                    path_in_repo=filename,
                    repo_id=repo_id,
                    repo_type="dataset"
                )
                print(f"✅ Successfully uploaded {filename}")
                uploaded_files.append(filename)

            except Exception as e:
                print(f"❌ Failed to upload {filename}: {e}")
        else:
            print(f"⚠ File not found: {filename}")

    # Create a simple README
    readme_content = f"""# XAUUSD Dataset

Comprehensive dataset for XAUUSD (Gold vs US Dollar) price prediction using machine learning techniques.

## Files

{chr(10).join(f"- `{file}`" for file in uploaded_files)}

## Description

This dataset contains:
- **Time Period**: 2000-2025 (full dataset), 2023-2025 (recent data)
- **Features**: 172+ technical indicators, economic variables
- **ML Performance**: 47.3% directional accuracy with ensemble models

## Usage

```python
from datasets import load_dataset
dataset = load_dataset("{repo_id}")
```

## Citation

```bibtex
@misc{{tapachoom2025xauusd,
  title={{XAUUSD Price Prediction Dataset}},
  author={{Tapachoom, Nattapong}},
  year={{2025}},
  url={{https://huggingface.co/datasets/{repo_id}}}
}}
```
"""

    # Upload README
    readme_path = Path("README_temp.md")
    with open(readme_path, 'w', encoding='utf-8') as f:
        f.write(readme_content)

    try:
        api.upload_file(
            path_or_fileobj=str(readme_path),
            path_in_repo="README.md",
            repo_id=repo_id,
            repo_type="dataset"
        )
        print("✅ Uploaded README.md")
    except Exception as e:
        print(f"❌ Failed to upload README: {e}")

    # Clean up
    readme_path.unlink(missing_ok=True)

    print("\n🎉 Upload complete!")
    print(f"🌐 View your dataset at: https://huggingface.co/datasets/{repo_id}")
    print(f"📊 Files uploaded: {len(uploaded_files)}")
    for file in uploaded_files:
        print(f"   ✓ {file}")

if __name__ == "__main__":
    upload_datasets()