import pandas as pd
import numpy as np
from ta import add_all_ta_features
from ta.utils import dropna
import yfinance as yf

def add_technical_indicators(df):
    """
    Add comprehensive technical indicators to the dataset
    """
    # Make a copy to avoid modifying original
    df_ta = df.copy()

    # Ensure we have the required columns
    required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
    if not all(col in df_ta.columns for col in required_cols):
        raise ValueError("DataFrame must contain Open, High, Low, Close, Volume columns")

    # Add all technical analysis features
    df_ta = add_all_ta_features(
        df_ta,
        open="Open",
        high="High",
        low="Low",
        close="Close",
        volume="Volume",
        fillna=True
    )

    # Additional custom indicators
    # Price changes
    df_ta['Price_Change'] = df_ta['Close'].pct_change()
    df_ta['Price_Change_5d'] = df_ta['Close'].pct_change(5)
    df_ta['Price_Change_20d'] = df_ta['Close'].pct_change(20)

    # Volatility measures
    df_ta['Volatility_5d'] = df_ta['Close'].rolling(5).std()
    df_ta['Volatility_20d'] = df_ta['Close'].rolling(20).std()

    # Volume indicators
    df_ta['Volume_Change'] = df_ta['Volume'].pct_change()
    df_ta['Volume_MA_5'] = df_ta['Volume'].rolling(5).mean()
    df_ta['Volume_MA_20'] = df_ta['Volume'].rolling(20).mean()

    # Price ranges
    df_ta['Daily_Range'] = df_ta['High'] - df_ta['Low']
    df_ta['Daily_Range_Pct'] = (df_ta['High'] - df_ta['Low']) / df_ta['Close'].shift(1)
    df_ta['Gap_Up'] = (df_ta['Open'] - df_ta['Close'].shift(1)) > 0
    df_ta['Gap_Down'] = (df_ta['Open'] - df_ta['Close'].shift(1)) < 0

    # Momentum indicators
    df_ta['ROC_5'] = ((df_ta['Close'] - df_ta['Close'].shift(5)) / df_ta['Close'].shift(5)) * 100
    df_ta['ROC_20'] = ((df_ta['Close'] - df_ta['Close'].shift(20)) / df_ta['Close'].shift(20)) * 100

    # Support/Resistance levels (simplified)
    df_ta['Rolling_Max_20'] = df_ta['High'].rolling(20).max()
    df_ta['Rolling_Min_20'] = df_ta['Low'].rolling(20).min()

    return df_ta

def add_economic_indicators(df):
    """
    Add economic indicators that affect gold prices
    """
    df_econ = df.copy()

    try:
        # US Dollar Index (DXY)
        dxy = yf.download("^DXY", start=df.index.min(), end=df.index.max(), interval="1d")
        dxy.columns = dxy.columns.droplevel(1)
        df_econ['DXY'] = dxy['Close']
        df_econ['DXY_Change'] = df_econ['DXY'].pct_change()

        # US Treasury yields (10-year)
        treasury_10y = yf.download("^TNX", start=df.index.min(), end=df.index.max(), interval="1d")
        treasury_10y.columns = treasury_10y.columns.droplevel(1)
        df_econ['US_10Y_Yield'] = treasury_10y['Close']
        df_econ['US_10Y_Change'] = df_econ['US_10Y_Yield'].pct_change()

        # Crude Oil (WTI)
        oil = yf.download("CL=F", start=df.index.min(), end=df.index.max(), interval="1d")
        oil.columns = oil.columns.droplevel(1)
        df_econ['WTI_Oil'] = oil['Close']
        df_econ['Oil_Change'] = df_econ['WTI_Oil'].pct_change()

        # Silver (correlated with gold)
        silver = yf.download("SI=F", start=df.index.min(), end=df.index.max(), interval="1d")
        silver.columns = silver.columns.droplevel(1)
        df_econ['Silver'] = silver['Close']
        df_econ['Gold_Silver_Ratio'] = df_econ['Close'] / df_econ['Silver']

    except Exception as e:
        print(f"Warning: Could not fetch some economic indicators: {e}")

    return df_econ

if __name__ == "__main__":
    # Load the main dataset
    print("Loading main dataset...")
    df = pd.read_csv('XAUUSD_full_dataset.csv', parse_dates=['Date'], index_col='Date')

    # Add technical indicators
    print("Adding technical indicators...")
    df_with_ta = add_technical_indicators(df)

    # Add economic indicators
    print("Adding economic indicators...")
    df_enhanced = add_economic_indicators(df_with_ta)

    # Save enhanced dataset
    output_file = 'XAUUSD_enhanced_dataset.csv'
    df_enhanced.to_csv(output_file)

    print(f"Enhanced dataset saved as {output_file}")
    print(f"Shape: {df_enhanced.shape}")
    print(f"New columns added: {len(df_enhanced.columns) - len(df.columns)}")
    print(f"Total columns: {len(df_enhanced.columns)}")

    # Show some statistics
    print("\nSample of new features:")
    new_cols = [col for col in df_enhanced.columns if col not in df.columns][:10]
    print(df_enhanced[new_cols].tail())