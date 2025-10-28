import pandas as pd
import numpy as np
from src.feature_engineering import add_technical_indicators, add_economic_indicators
from src.news_scraper import GoldNewsScraper
import warnings
warnings.filterwarnings('ignore')

def create_comprehensive_dataset():
    """
    Create a comprehensive XAUUSD dataset with all features
    """
    print("Creating comprehensive XAUUSD dataset...")

    # Step 1: Load base price data
    print("1. Loading base price data...")
    try:
        df = pd.read_csv('XAUUSD_full_dataset.csv', parse_dates=['Date'], index_col='Date')
        print(f"   Loaded {len(df)} price records")
    except FileNotFoundError:
        print("   Error: XAUUSD_full_dataset.csv not found. Please run generate_dataset.py first.")
        return None

    # Step 2: Add technical indicators
    print("2. Adding technical indicators...")
    df_with_ta = add_technical_indicators(df)
    print(f"   Added {len(df_with_ta.columns) - len(df.columns)} technical indicators")

    # Step 3: Add economic indicators
    print("3. Adding economic indicators...")
    df_with_econ = add_economic_indicators(df_with_ta)
    print(f"   Added economic indicators. Total columns: {len(df_with_econ.columns)}")

    # Step 4: Add news sentiment (recent news)
    print("4. Adding news sentiment data...")
    try:
        scraper = GoldNewsScraper()
        news_df = scraper.scrape_gold_news(days_back=90)  # Last 90 days

        if not news_df.empty:
            # Aggregate daily sentiment
            daily_sentiment = news_df.groupby(news_df['Date'].dt.date).agg({
                'Sentiment_Polarity': 'mean',
                'Sentiment_Subjectivity': 'mean',
                'Title': 'count'  # Number of news articles
            }).rename(columns={
                'Sentiment_Polarity': 'News_Sentiment_Avg',
                'Sentiment_Subjectivity': 'News_Subjectivity_Avg',
                'Title': 'News_Volume'
            })

            # Convert index to datetime
            daily_sentiment.index = pd.to_datetime(daily_sentiment.index)

            # Merge with main dataset
            df_final = df_with_econ.merge(
                daily_sentiment,
                left_index=True,
                right_index=True,
                how='left'
            )

            # Fill missing news data with 0 or neutral values
            df_final['News_Sentiment_Avg'] = df_final['News_Sentiment_Avg'].fillna(0)
            df_final['News_Subjectivity_Avg'] = df_final['News_Subjectivity_Avg'].fillna(0.5)
            df_final['News_Volume'] = df_final['News_Volume'].fillna(0)

            print(f"   Added news sentiment data for {len(daily_sentiment)} days")
        else:
            df_final = df_with_econ
            print("   No recent news data available")

    except Exception as e:
        print(f"   Error adding news data: {e}")
        df_final = df_with_econ

    # Step 5: Add target variables for prediction
    print("5. Adding target variables...")

    # Future price movements (for supervised learning)
    df_final['Target_1d'] = (df_final['Close'].shift(-1) > df_final['Close']).astype(int)  # 1 if price goes up tomorrow
    df_final['Target_5d'] = (df_final['Close'].shift(-5) > df_final['Close']).astype(int)  # 1 if price goes up in 5 days
    df_final['Price_Change_1d_Pct'] = df_final['Close'].pct_change().shift(-1)  # Actual percentage change
    df_final['Price_Change_5d_Pct'] = df_final['Close'].pct_change(5).shift(-5)

    # Remove rows with NaN targets
    df_final = df_final.dropna(subset=['Target_1d'])

    # Step 6: Final cleanup
    print("6. Final cleanup...")

    # Remove any remaining NaN values (from indicators)
    df_final = df_final.fillna(method='bfill').fillna(method='ffill').fillna(0)

    # Ensure proper data types
    numeric_cols = df_final.select_dtypes(include=[np.number]).columns
    df_final[numeric_cols] = df_final[numeric_cols].astype(float)

    # Step 7: Save comprehensive dataset
    output_file = 'XAUUSD_comprehensive_dataset.csv'
    df_final.to_csv(output_file)

    print(f"\n✅ Comprehensive dataset created successfully!")
    print(f"📊 Shape: {df_final.shape}")
    print(f"📅 Date range: {df_final.index.min()} to {df_final.index.max()}")
    print(f"📈 Features: {len(df_final.columns)}")
    print(f"💾 Saved as: {output_file}")

    # Show feature categories
    print("\n📋 Feature Categories:")
    feature_info = {
        'Price Data': ['Open', 'High', 'Low', 'Close', 'Volume'],
        'Technical Indicators': [col for col in df_final.columns if any(x in col.upper() for x in ['SMA', 'EMA', 'RSI', 'MACD', 'BB', 'STOCH', 'WILLiams'])],
        'Economic Indicators': [col for col in df_final.columns if any(x in col.upper() for x in ['DXY', 'YIELD', 'OIL', 'SILVER'])],
        'News Sentiment': [col for col in df_final.columns if 'NEWS' in col.upper()],
        'Custom Features': [col for col in df_final.columns if any(x in col for x in ['Price_Change', 'Volatility', 'Volume_', 'Daily_Range', 'Gap_', 'ROC_'])],
        'Target Variables': [col for col in df_final.columns if 'TARGET' in col.upper() or 'PCT' in col.upper()]
    }

    for category, features in feature_info.items():
        actual_features = [f for f in features if f in df_final.columns]
        if actual_features:
            print(f"   {category}: {len(actual_features)} features")

    return df_final

def create_feature_importance_analysis(df):
    """
    Create a basic feature importance analysis
    """
    print("\n🔍 Creating feature importance analysis...")

    # Correlation with target
    target_cols = [col for col in df.columns if 'TARGET' in col.upper()]
    if target_cols:
        target = target_cols[0]
        correlations = df.corr()[target].abs().sort_values(ascending=False)

        # Save top 20 most correlated features
        top_features = correlations.head(20)
        top_features.to_csv('feature_importance_correlation.csv')

        print("   Top 10 features correlated with target:")
        for i, (feature, corr) in enumerate(top_features.head(10).items()):
            print(".3f")

if __name__ == "__main__":
    # Create comprehensive dataset
    df_comprehensive = create_comprehensive_dataset()

    if df_comprehensive is not None:
        # Create feature importance analysis
        create_feature_importance_analysis(df_comprehensive)

        print("\n🎉 All datasets created successfully!")
        print("\n📁 Generated files:")
        print("   - XAUUSD_comprehensive_dataset.csv (main dataset)")
        print("   - gold_news_dataset.csv (news data)")
        print("   - gold_news_with_impact.csv (news with price impact)")
        print("   - feature_importance_correlation.csv (feature analysis)")