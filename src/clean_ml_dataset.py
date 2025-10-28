import pandas as pd
import numpy as np

def clean_dataset(filepath='XAUUSD_enhanced_ml_dataset.csv'):
    """Clean the enhanced dataset by removing infinity and extreme values"""
    print("Cleaning enhanced dataset...")

    # Load data
    df = pd.read_csv(filepath, parse_dates=['Date'], index_col='Date')

    # Replace infinity with NaN
    df = df.replace([np.inf, -np.inf], np.nan)

    # Remove columns with too many NaN values (>50%)
    nan_threshold = len(df) * 0.5
    df = df.dropna(axis=1, thresh=nan_threshold)

    # Fill remaining NaN values
    df = df.fillna(method='bfill').fillna(method='ffill').fillna(0)

    # Remove extreme outliers (beyond 5 standard deviations)
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if col not in ['Target_1d', 'Target_5d']:  # Don't clean target variables
            mean_val = df[col].mean()
            std_val = df[col].std()
            if std_val > 0:  # Avoid division by zero
                df[col] = df[col].clip(mean_val - 5*std_val, mean_val + 5*std_val)

    # Ensure target variables are clean
    df = df.dropna(subset=['Target_1d', 'Price_Change_1d_Pct'])

    # Final check for any remaining infinity or NaN
    df = df.replace([np.inf, -np.inf], 0)
    df = df.fillna(0)

    # Save cleaned dataset
    cleaned_file = 'XAUUSD_enhanced_ml_dataset_clean.csv'
    df.to_csv(cleaned_file)

    print(f"✅ Dataset cleaned successfully!")
    print(f"📊 Shape: {df.shape}")
    print(f"📈 Features: {len(df.columns)}")
    print(f"💾 Saved as: {cleaned_file}")

    # Check for any remaining issues
    inf_count = np.isinf(df.select_dtypes(include=[np.number])).sum().sum()
    nan_count = df.isnull().sum().sum()

    print(f"🔍 Remaining infinity values: {inf_count}")
    print(f"🔍 Remaining NaN values: {nan_count}")

    return df

if __name__ == "__main__":
    clean_dataset()