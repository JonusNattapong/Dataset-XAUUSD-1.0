import yfinance as yf
import pandas as pd

# Define the ticker for XAUUSD (Gold vs US Dollar)
ticker = "GC=F"  # This is the Gold futures ticker; for spot XAUUSD, it might be "XAUUSD=X"

# Download historical data from 2000 to present, daily interval
data = yf.download(ticker, start="2000-01-01", end="2025-10-27", interval="1d")

# Save to CSV
data.columns = data.columns.droplevel(1)  # Remove ticker level
data.to_csv("XAUUSD_full_dataset.csv")

print("Dataset generated and saved as XAUUSD_full_dataset.csv")
print(f"Shape: {data.shape}")
print(data.head())