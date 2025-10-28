import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

def add_macroeconomic_indicators(df):
    """
    Add macroeconomic indicators that affect gold prices
    """
    df_macro = df.copy()

    # Federal Funds Rate (US interest rates)
    try:
        # Using FRED data through yfinance (limited availability)
        # Alternative: Use pre-downloaded economic data
        print("   Note: Macroeconomic data requires additional data sources")
        print("   Consider using FRED API or Quandl for comprehensive economic data")
    except:
        pass

    # Add synthetic economic indicators based on available data
    # These are simplified proxies - in production, use real economic data

    # US Dollar strength proxy (using DXY if available)
    if 'DXY' in df_macro.columns:
        df_macro['USD_Strength_Change'] = df_macro['DXY'].pct_change(5)

    # Interest rate expectations (simplified)
    df_macro['Real_Yield_Proxy'] = df_macro.get('US_10Y_Yield', 0) - df_macro['Close'].pct_change(252)  # Approximate inflation

    return df_macro

def add_commodity_correlations(df):
    """
    Add correlations with other commodities
    """
    df_comm = df.copy()

    # Gold to Silver ratio (already have)
    # Gold to Oil ratio
    if 'WTI_Oil' in df_comm.columns:
        df_comm['Gold_Oil_Ratio'] = df_comm['Close'] / df_comm['WTI_Oil']

    # Gold to Copper ratio (industrial metal)
    try:
        copper = yf.download("HG=F", start=df.index.min(), end=df.index.max(), interval="1d")
        copper.columns = copper.columns.droplevel(1)
        df_comm['Copper_Price'] = copper['Close']
        df_comm['Gold_Copper_Ratio'] = df_comm['Close'] / df_comm['Copper_Price']
    except:
        print("   Could not fetch copper data")

    # Gold to Bitcoin correlation (safe haven assets)
    try:
        btc = yf.download("BTC-USD", start=df.index.min(), end=df.index.max(), interval="1d")
        btc.columns = btc.columns.droplevel(1)
        df_comm['BTC_Price'] = btc['Close']
        df_comm['Gold_BTC_Ratio'] = df_comm['Close'] / df_comm['BTC_Price']
    except:
        print("   Could not fetch Bitcoin data")

    return df_comm

def add_advanced_features(df):
    """
    Add advanced technical and statistical features
    """
    df_adv = df.copy()

    # Lagged features (past prices as features)
    for lag in [1, 2, 3, 5, 10, 20]:
        df_adv[f'Close_Lag_{lag}'] = df_adv['Close'].shift(lag)
        df_adv[f'Return_Lag_{lag}'] = df_adv['Close'].pct_change(lag).shift(1)

    # Rolling statistics
    windows = [5, 10, 20, 50]
    for window in windows:
        df_adv[f'Rolling_Mean_{window}'] = df_adv['Close'].rolling(window=window).mean()
        df_adv[f'Rolling_Std_{window}'] = df_adv['Close'].rolling(window=window).std()
        df_adv[f'Rolling_Skew_{window}'] = df_adv['Close'].rolling(window=window).skew()
        df_adv[f'Rolling_Kurt_{window}'] = df_adv['Close'].rolling(window=window).kurt()

    # Momentum features
    df_adv['Momentum_1M'] = df_adv['Close'] / df_adv['Close'].shift(20) - 1
    df_adv['Momentum_3M'] = df_adv['Close'] / df_adv['Close'].shift(60) - 1
    df_adv['Momentum_6M'] = df_adv['Close'] / df_adv['Close'].shift(120) - 1

    # Volatility measures
    df_adv['Realized_Vol_5d'] = df_adv['Close'].pct_change().rolling(5).std() * np.sqrt(252)
    df_adv['Realized_Vol_20d'] = df_adv['Close'].pct_change().rolling(20).std() * np.sqrt(252)

    # Volume-based features
    if 'Volume' in df_adv.columns:
        df_adv['Volume_MA_Ratio'] = df_adv['Volume'] / df_adv['Volume'].rolling(20).mean()
        df_adv['Volume_Change_Rate'] = df_adv['Volume'].pct_change().rolling(5).mean()

    # Seasonal features
    df_adv['Day_of_Week'] = df_adv.index.dayofweek
    df_adv['Month'] = df_adv.index.month
    df_adv['Quarter'] = df_adv.index.quarter

    # Cyclical patterns (simplified)
    df_adv['Sin_Day'] = np.sin(2 * np.pi * df_adv.index.dayofyear / 365.25)
    df_adv['Cos_Day'] = np.cos(2 * np.pi * df_adv.index.dayofyear / 365.25)

    return df_adv

def add_risk_metrics(df):
    """
    Add risk and performance metrics
    """
    df_risk = df.copy()

    # Value at Risk (VaR) - historical simulation
    returns = df_risk['Close'].pct_change().dropna()
    df_risk['VaR_95'] = returns.rolling(252).quantile(0.05).shift(1)
    df_risk['VaR_99'] = returns.rolling(252).quantile(0.01).shift(1)

    # Expected Shortfall (CVaR)
    df_risk['CVaR_95'] = returns[returns <= returns.quantile(0.05)].rolling(252).mean().shift(1)

    # Sharpe ratio components
    risk_free_rate = 0.02  # Assume 2% risk-free rate
    df_risk['Excess_Return'] = df_risk['Close'].pct_change() - risk_free_rate/252
    df_risk['Rolling_Sharpe'] = df_risk['Excess_Return'].rolling(252).mean() / df_risk['Excess_Return'].rolling(252).std()

    # Maximum drawdown
    cumulative = (1 + df_risk['Close'].pct_change()).cumprod()
    running_max = cumulative.expanding().max()
    drawdown = (cumulative - running_max) / running_max
    df_risk['Max_Drawdown'] = drawdown.rolling(252).min().shift(1)

    return df_risk

def create_enhanced_dataset():
    """
    Create an enhanced dataset with all advanced features
    """
    print("Creating enhanced dataset with advanced features...")

    # Load the recent comprehensive dataset
    try:
        df = pd.read_csv('XAUUSD_2023_2025_dataset.csv', parse_dates=['Date'], index_col='Date')
        print(f"Loaded recent dataset: {len(df)} records")
    except FileNotFoundError:
        print("Recent dataset not found. Please run create_recent_dataset.py first.")
        return None

    # Add macroeconomic indicators
    print("1. Adding macroeconomic indicators...")
    df = add_macroeconomic_indicators(df)

    # Add commodity correlations
    print("2. Adding commodity correlations...")
    df = add_commodity_correlations(df)

    # Add advanced features
    print("3. Adding advanced technical features...")
    df = add_advanced_features(df)

    # Add risk metrics
    print("4. Adding risk metrics...")
    df = add_risk_metrics(df)

    # Final cleanup
    print("5. Final cleanup...")
    df = df.fillna(method='bfill').fillna(method='ffill').fillna(0)

    # Ensure proper data types
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = df[numeric_cols].astype(float)

    # Save enhanced dataset
    output_file = 'XAUUSD_enhanced_ml_dataset.csv'
    df.to_csv(output_file)

    print(f"\n✅ Enhanced ML dataset created successfully!")
    print(f"📊 Shape: {df.shape}")
    print(f"📅 Date range: {df.index.min()} to {df.index.max()}")
    print(f"📈 Features: {len(df.columns)}")
    print(f"💾 Saved as: {output_file}")

    # Show feature categories
    print("\n📋 Enhanced Feature Categories:")
    feature_info = {
        'Basic Price': ['Open', 'High', 'Low', 'Close', 'Volume'],
        'Technical Indicators': [col for col in df.columns if any(x in col.upper() for x in ['RSI', 'MACD', 'BB', 'SMA', 'EMA'])],
        'Economic Indicators': [col for col in df.columns if any(x in col.upper() for x in ['DXY', 'YIELD', 'OIL', 'SILVER', 'COPPER', 'BTC'])],
        'Lagged Features': [col for col in df.columns if 'LAG' in col.upper()],
        'Rolling Statistics': [col for col in df.columns if 'ROLLING' in col.upper()],
        'Momentum Features': [col for col in df.columns if 'MOMENTUM' in col.upper()],
        'Risk Metrics': [col for col in df.columns if any(x in col.upper() for x in ['VAR', 'CVAR', 'SHARPE', 'DRAWDOWN'])],
        'Seasonal Features': [col for col in df.columns if any(x in col.upper() for x in ['DAY', 'MONTH', 'QUARTER', 'SIN', 'COS'])]
    }

    for category, features in feature_info.items():
        actual_features = [f for f in features if f in df.columns]
        if actual_features:
            print(f"   {category}: {len(actual_features)} features")

    return df

if __name__ == "__main__":
    # Create enhanced ML dataset
    df_enhanced = create_enhanced_dataset()

    if df_enhanced is not None:
        print("\n🎉 Enhanced ML dataset ready for advanced modeling!")
        print("💡 Recommended next steps:")
        print("   - Use feature selection techniques (correlation, mutual information)")
        print("   - Try ensemble models (XGBoost, Random Forest)")
        print("   - Implement time series cross-validation")
        print("   - Consider LSTM/GRU networks for sequence prediction")