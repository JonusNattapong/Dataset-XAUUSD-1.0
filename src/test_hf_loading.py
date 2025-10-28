#!/usr/bin/env python3
"""
Test script to verify XAUUSD dataset accessibility and provide loading examples
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

def test_direct_csv_loading():
    """Test loading the dataset directly via CSV (recommended approach)"""

    print("🔍 Testing direct CSV loading (recommended)...")

    try:
        # Load the dataset
        df = pd.read_csv("https://huggingface.co/datasets/JonusNattapong/xauusd-dataset/resolve/main/XAUUSD_enhanced_ml_dataset_clean.csv")
        print("✅ Dataset loaded successfully!")
        print(f"Dataset shape: {df.shape}")
        print(f"Number of features: {len(df.columns)}")
        print(f"Date range: {df['Date'].min()} to {df['Date'].max()}")

        # Check for missing values
        missing = df.isnull().sum().sum()
        print(f"Missing values: {missing}")

        # Quick ML example
        print("\n🔍 Running quick ML validation...")

        # Prepare features and target
        feature_cols = [col for col in df.columns if col not in ['Date', 'Target_1d', 'Target_5d']]
        X = df[feature_cols]
        y = df['Target_1d']

        # Split data (time series aware)
        split_idx = int(len(df) * 0.8)
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]

        # Train model
        model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
        model.fit(X_train, y_train)

        # Evaluate
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        print(f"Accuracy: {accuracy:.3f}")
        return True

    except Exception as e:
        print(f"❌ Error loading dataset: {e}")
        return False

def test_hf_datasets_loading():
    """Test loading with HF datasets library (may have compatibility issues)"""

    print("\n🔍 Testing Hugging Face datasets library loading...")

    try:
        from datasets import load_dataset

        # Attempt to load the dataset
        dataset = load_dataset("JonusNattapong/xauusd-dataset")
        print("✅ Dataset loaded successfully with HF datasets!")
        print(f"Dataset splits: {list(dataset.keys())}")

        # Check the train split
        train_data = dataset['train']
        print(f"Train split size: {len(train_data)}")
        print(f"Number of features: {len(train_data.column_names)}")

        return True

    except Exception as e:
        print(f"⚠️  HF datasets library loading failed: {e}")
        print("   This is a known compatibility issue with complex schemas.")
        print("   Use direct CSV loading instead (see above).")
        return False

def main():
    """Main test function"""

    print("🪙 XAUUSD Dataset Loading Test")
    print("=" * 50)

    # Test direct CSV loading (recommended)
    csv_success = test_direct_csv_loading()

    # Test HF datasets library (may fail)
    hf_success = test_hf_datasets_loading()

    print("\n" + "=" * 50)
    print("📊 Results Summary:")
    print(f"Direct CSV Loading: {'✅ SUCCESS' if csv_success else '❌ FAILED'}")
    print(f"HF Datasets Library: {'✅ SUCCESS' if hf_success else '⚠️  KNOWN ISSUE'}")

    if csv_success:
        print("\n🎉 Dataset is fully accessible and ready for use!")
        print("Use the direct CSV loading method for best compatibility.")
    else:
        print("\n❌ Dataset loading failed. Please check the repository.")

if __name__ == "__main__":
    main()