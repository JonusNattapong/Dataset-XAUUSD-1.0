import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Load the recent dataset
df = pd.read_csv('XAUUSD_2023_2025_dataset.csv', parse_dates=['Date'], index_col='Date')

print("XAUUSD Dataset Analysis (2023-2025)")
print("=" * 50)
print(f"Data shape: {df.shape}")
print(f"Date range: {df.index.min()} to {df.index.max()}")
print(".2f")
print(".2f")
print(".2f")

# Price trend analysis
plt.figure(figsize=(15, 10))

# Price chart
plt.subplot(2, 1, 1)
plt.plot(df.index, df['Close'], label='Close Price', color='gold', linewidth=2)
plt.plot(df.index, df['Close'].rolling(50).mean(), label='50-day MA', color='red', alpha=0.8)
plt.plot(df.index, df['Close'].rolling(200).mean(), label='200-day MA', color='blue', alpha=0.8)
plt.title('XAUUSD Price Trend (2023-2025)', fontsize=14, fontweight='bold')
plt.xlabel('Date')
plt.ylabel('Price (USD)')
plt.legend()
plt.grid(True, alpha=0.3)

# Volume chart
plt.subplot(2, 1, 2)
plt.bar(df.index, df['Volume'], color='purple', alpha=0.6, width=1)
plt.title('Trading Volume (2023-2025)', fontsize=14, fontweight='bold')
plt.xlabel('Date')
plt.ylabel('Volume')
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('XAUUSD_2023_2025_price_analysis.png', dpi=300, bbox_inches='tight')
plt.show()

# Key statistics for recent period
print("\nKey Statistics (2023-2025):")
print("-" * 30)
print(".2f")
print(".2f")
print(".2f")
print(".2f")
print(".2f")

# Best and worst performing periods
monthly_returns = df['Close'].resample('M').last().pct_change()
best_month = monthly_returns.idxmax().strftime('%Y-%m')
worst_month = monthly_returns.idxmin().strftime('%Y-%m')

print(f"\nBest performing month: {best_month} ({monthly_returns.max():.2%})")
print(f"Worst performing month: {worst_month} ({monthly_returns.min():.2%})")

# Volatility analysis
df['Daily_Return'] = df['Close'].pct_change()
volatility = df['Daily_Return'].std() * np.sqrt(252)  # Annualized volatility

print(".2%")

# Technical indicators summary
print("\nTechnical Indicators (Recent Values):")
print("-" * 40)
recent = df.iloc[-1]
print(".2f")
print(".2f")
print(".2f")
print(".2f")
print(".2f")

# Economic indicators correlation with gold price
economic_cols = [col for col in df.columns if any(x in col for x in ['DXY', 'YIELD', 'OIL', 'SILVER'])]
if economic_cols:
    print("\nEconomic Indicators Correlation with Gold Price:")
    print("-" * 50)
    correlations = df[economic_cols + ['Close']].corr()['Close'].drop('Close')
    for indicator, corr in correlations.items():
        print(".3f")

# Save summary statistics
summary_stats = {
    'Period': '2023-2025',
    'Total_Days': len(df),
    'Start_Price': df['Close'].iloc[0],
    'End_Price': df['Close'].iloc[-1],
    'Max_Price': df['Close'].max(),
    'Min_Price': df['Close'].min(),
    'Avg_Volume': df['Volume'].mean(),
    'Total_Return': (df['Close'].iloc[-1] / df['Close'].iloc[0] - 1) * 100,
    'Annualized_Volatility': volatility * 100,
    'Best_Month': f"{best_month} ({monthly_returns.max():.1%})",
    'Worst_Month': f"{worst_month} ({monthly_returns.min():.1%})"
}

pd.DataFrame([summary_stats]).to_csv('XAUUSD_2023_2025_summary.csv', index=False)

print("\n📊 Analysis complete! Charts and summary saved.")
print("📁 Files generated:")
print("   - XAUUSD_2023_2025_price_analysis.png")
print("   - XAUUSD_2023_2025_summary.csv")