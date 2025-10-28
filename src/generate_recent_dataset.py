import yfinance as yf
import pandas as pd

def generate_recent_dataset(ticker="GC=F", start_date="2023-01-01", end_date="2025-10-27", interval="1d", filename=None):
    """
    Generate fresh XAUUSD dataset for recent period (2023-2025)

    Parameters:
    ticker: str - Yahoo Finance ticker symbol
    start_date: str - Start date in YYYY-MM-DD format
    end_date: str - End date in YYYY-MM-DD format
    interval: str - Data interval (1d, 1wk, 1mo, 1h, etc.)
    filename: str - Output filename (auto-generated if None)
    """
    print(f"Downloading fresh {ticker} data from {start_date} to {end_date} with {interval} interval...")

    # Download data
    data = yf.download(ticker, start=start_date, end=end_date, interval=interval)

    # Flatten column names
    data.columns = data.columns.droplevel(1)

    # Generate filename if not provided
    if filename is None:
        filename = f"XAUUSD_{start_date.split('-')[0]}_{end_date.split('-')[0]}_fresh.csv"

    # Save to CSV
    data.to_csv(filename)

    print(f"Fresh dataset saved as {filename}")
    print(f"Shape: {data.shape}")
    print(f"Date range: {data.index.min()} to {data.index.max()}")

    return data

if __name__ == "__main__":
    # Generate fresh datasets for recent period
    print("Generating fresh XAUUSD datasets for 2023-2025 period...\n")

    # Daily data (main recent dataset)
    daily_data = generate_recent_dataset(interval="1d")

    # Weekly data for recent period
    weekly_data = generate_recent_dataset(interval="1wk", filename="XAUUSD_2023_2025_weekly.csv")

    # Hourly data (last 6 months for hourly)
    hourly_data = generate_recent_dataset(
        start_date="2025-04-01",
        interval="1h",
        filename="XAUUSD_recent_hourly.csv"
    )

    print("\nFresh datasets for 2023-2025 period generated successfully!")