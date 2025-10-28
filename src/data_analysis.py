import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Load the dataset
df = pd.read_csv('XAUUSD_full_dataset.csv', parse_dates=['Date'], index_col='Date')

print("Dataset Overview:")
print(df.head())
print(f"\nShape: {df.shape}")
print(f"\nData types:\n{df.dtypes}")
print(f"\nMissing values:\n{df.isnull().sum()}")

# Basic statistics
print("\nBasic Statistics:")
print(df.describe())

# Plot price trends
plt.figure(figsize=(15, 10))

# Price chart
plt.subplot(2, 1, 1)
plt.plot(df.index, df['Close'], label='Close Price', color='gold')
plt.plot(df.index, df['Open'], label='Open Price', alpha=0.7)
plt.title('XAUUSD Price Trends (2000-2025)')
plt.xlabel('Date')
plt.ylabel('Price (USD)')
plt.legend()
plt.grid(True, alpha=0.3)

# Volume chart
plt.subplot(2, 1, 2)
plt.bar(df.index, df['Volume'], color='blue', alpha=0.6)
plt.title('Trading Volume')
plt.xlabel('Date')
plt.ylabel('Volume')
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('price_analysis.png', dpi=300, bbox_inches='tight')
plt.show()

# Calculate moving averages
df['MA50'] = df['Close'].rolling(window=50).mean()
df['MA200'] = df['Close'].rolling(window=200).mean()

# Plot with moving averages
plt.figure(figsize=(15, 6))
plt.plot(df.index, df['Close'], label='Close Price', alpha=0.8)
plt.plot(df.index, df['MA50'], label='50-day MA', color='red')
plt.plot(df.index, df['MA200'], label='200-day MA', color='green')
plt.title('XAUUSD with Moving Averages')
plt.xlabel('Date')
plt.ylabel('Price (USD)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig('moving_averages.png', dpi=300, bbox_inches='tight')
plt.show()

# Calculate daily returns
df['Daily_Return'] = df['Close'].pct_change()

# Plot returns distribution
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.hist(df['Daily_Return'].dropna(), bins=50, alpha=0.7, color='purple')
plt.title('Daily Returns Distribution')
plt.xlabel('Daily Return')
plt.ylabel('Frequency')

plt.subplot(1, 2, 2)
plt.plot(df.index, df['Daily_Return'], alpha=0.7, color='purple')
plt.title('Daily Returns Over Time')
plt.xlabel('Date')
plt.ylabel('Daily Return')
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('returns_analysis.png', dpi=300, bbox_inches='tight')
plt.show()

# Correlation heatmap
plt.figure(figsize=(8, 6))
correlation_matrix = df[['Open', 'High', 'Low', 'Close', 'Volume']].corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0)
plt.title('Correlation Matrix')
plt.savefig('correlation_heatmap.png', dpi=300, bbox_inches='tight')
plt.show()

print("\nAnalysis complete. Charts saved as PNG files.")