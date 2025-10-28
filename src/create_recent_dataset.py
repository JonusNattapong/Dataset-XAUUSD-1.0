import pandas as pd
import numpy as np
import yfinance as yf
from src.feature_engineering import add_technical_indicators, add_economic_indicators
from src.news_scraper import GoldNewsScraper
import warnings
warnings.filterwarnings('ignore')

def create_recent_dataset():
    """
    Create XAUUSD dataset for 2023-2025 period with all features
    """
    print("Creating XAUUSD dataset for 2023-2025 period...")

    # Step 1: Load or download recent price data
    print("1. Loading/downloading recent price data (2023-2025)...")

    try:
        # Try to load fresh recent data first
        df = pd.read_csv('XAUUSD_2023_2025_fresh.csv', parse_dates=['Date'], index_col='Date')
        print(f"   Loaded fresh data: {len(df)} records from 2023-2025")

        # Add technical indicators to fresh data
        print("2. Adding technical indicators...")
        df = add_technical_indicators(df)

        # Add economic indicators
        print("3. Adding economic indicators...")
        df = add_economic_indicators(df)

    except FileNotFoundError:
        try:
            # Fallback to filtering existing comprehensive dataset
            df_full = pd.read_csv('XAUUSD_comprehensive_dataset.csv', parse_dates=['Date'], index_col='Date')
            df = df_full[df_full.index >= '2023-01-01']
            print(f"   Filtered existing data: {len(df)} records from 2023-2025")
        except FileNotFoundError:
            # If no existing data, download fresh
            print("   No existing data found, downloading fresh data...")
            ticker = "GC=F"
            df_raw = yf.download(ticker, start="2023-01-01", end="2025-10-27", interval="1d")
            df_raw.columns = df_raw.columns.droplevel(1)
            df = df_raw.copy()
            print(f"   Downloaded {len(df)} records")

            # Add technical indicators
            print("2. Adding technical indicators...")
            df = add_technical_indicators(df)

            # Add economic indicators
            print("3. Adding economic indicators...")
            df = add_economic_indicators(df)

        # Add technical indicators
        print("2. Adding technical indicators...")
        df = add_technical_indicators(df)

        # Add economic indicators
        print("3. Adding economic indicators...")
        df = add_economic_indicators(df)

    # Step 4: Add news sentiment (more relevant for recent period)
    print("4. Adding news sentiment data...")
    try:
        scraper = GoldNewsScraper()
        news_df = scraper.scrape_gold_news(days_back=365)  # Last year for more news

        if not news_df.empty:
            # Aggregate daily sentiment
            daily_sentiment = news_df.groupby(news_df['Date'].dt.date).agg({
                'Sentiment_Polarity': 'mean',
                'Sentiment_Subjectivity': 'mean',
                'Title': 'count'
            }).rename(columns={
                'Sentiment_Polarity': 'News_Sentiment_Avg',
                'Sentiment_Subjectivity': 'News_Subjectivity_Avg',
                'Title': 'News_Volume'
            })

            daily_sentiment.index = pd.to_datetime(daily_sentiment.index)

            # Merge with main dataset
            df = df.merge(
                daily_sentiment,
                left_index=True,
                right_index=True,
                how='left'
            )

            # Fill missing news data
            df['News_Sentiment_Avg'] = df['News_Sentiment_Avg'].fillna(0)
            df['News_Subjectivity_Avg'] = df['News_Subjectivity_Avg'].fillna(0.5)
            df['News_Volume'] = df['News_Volume'].fillna(0)

            print(f"   Added news sentiment data for {len(daily_sentiment)} days")
        else:
            print("   No recent news data available")

    except Exception as e:
        print(f"   Error adding news data: {e}")

    # Step 5: Add target variables
    print("5. Adding target variables...")

    df['Target_1d'] = (df['Close'].shift(-1) > df['Close']).astype(int)
    df['Target_5d'] = (df['Close'].shift(-5) > df['Close']).astype(int)
    df['Price_Change_1d_Pct'] = df['Close'].pct_change().shift(-1)
    df['Price_Change_5d_Pct'] = df['Close'].pct_change(5).shift(-5)

    # Remove rows with NaN targets
    df = df.dropna(subset=['Target_1d'])

    # Step 6: Final cleanup
    print("6. Final cleanup...")
    df = df.fillna(method='bfill').fillna(method='ffill').fillna(0)

    # Ensure proper data types
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = df[numeric_cols].astype(float)

    # Step 7: Save recent dataset
    output_file = 'XAUUSD_2023_2025_dataset.csv'
    df.to_csv(output_file)

    print(f"\n✅ Recent dataset created successfully!")
    print(f"📊 Shape: {df.shape}")
    print(f"📅 Date range: {df.index.min()} to {df.index.max()}")
    print(f"📈 Features: {len(df.columns)}")
    print(f"💾 Saved as: {output_file}")

    return df

if __name__ == "__main__":
    # Create recent dataset
    df_recent = create_recent_dataset()

    print("\n🎉 Dataset for 2023-2025 period created!")
    print("📁 Generated file: XAUUSD_2023_2025_dataset.csv")