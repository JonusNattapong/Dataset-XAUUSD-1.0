import yfinance as yf
import pandas as pd

def generate_dataset(ticker="GC=F", start_date="2000-01-01", end_date="2025-10-27", interval="1d", filename=None):
    """
    Generate XAUUSD dataset for different timeframes

    Parameters:
    ticker: str - Yahoo Finance ticker symbol
    start_date: str - Start date in YYYY-MM-DD format
    end_date: str - End date in YYYY-MM-DD format
    interval: str - Data interval (1d, 1wk, 1mo, 1h, etc.)
    filename: str - Output filename (auto-generated if None)
    """
    print(f"Downloading {ticker} data from {start_date} to {end_date} with {interval} interval...")

    # Download data
    data = yf.download(ticker, start=start_date, end=end_date, interval=interval)

    # Flatten column names
    data.columns = data.columns.droplevel(1)

    # Generate filename if not provided
    if filename is None:
        filename = f"XAUUSD_{interval}_dataset.csv"

    # Save to CSV
    data.to_csv(filename)

    print(f"Dataset saved as {filename}")
    print(f"Shape: {data.shape}")
    print(f"Date range: {data.index.min()} to {data.index.max()}")

    return data

if __name__ == "__main__":
    # Generate datasets for different timeframes
    print("Generating XAUUSD datasets for different timeframes...\n")

    # Daily data (main dataset)
    daily_data = generate_dataset(interval="1d")

    # Weekly data
    weekly_data = generate_dataset(interval="1wk", filename="XAUUSD_weekly_dataset.csv")

    # Monthly data
    monthly_data = generate_dataset(interval="1mo", filename="XAUUSD_monthly_dataset.csv")

    # Hourly data (last 2 years for hourly)
    hourly_data = generate_dataset(
        start_date="2023-01-01",
        interval="1h",
        filename="XAUUSD_hourly_dataset.csv"
    )

    print("\nAll datasets generated successfully!")