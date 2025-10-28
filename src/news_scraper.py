import requests
from bs4 import BeautifulSoup
import pandas as pd
from textblob import TextBlob
import time
from datetime import datetime, timedelta
import re

class GoldNewsScraper:
    def __init__(self):
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }

    def get_sentiment(self, text):
        """Calculate sentiment polarity and subjectivity"""
        blob = TextBlob(text)
        return blob.sentiment.polarity, blob.sentiment.subjectivity

    def scrape_gold_news(self, days_back=30):
        """
        Scrape gold-related news from multiple sources
        """
        news_data = []
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days_back)

        print(f"Scraping gold news from {start_date.date()} to {end_date.date()}...")

        # Kitco News (primary source for gold news)
        try:
            kitco_news = self.scrape_kitco_news()
            news_data.extend(kitco_news)
        except Exception as e:
            print(f"Error scraping Kitco: {e}")

        # Investing.com Gold News
        try:
            investing_news = self.scrape_investing_news()
            news_data.extend(investing_news)
        except Exception as e:
            print(f"Error scraping Investing.com: {e}")

        # Convert to DataFrame
        if news_data:
            df_news = pd.DataFrame(news_data)
            df_news['Date'] = pd.to_datetime(df_news['Date'])
            df_news = df_news.sort_values('Date', ascending=False)

            # Calculate sentiment
            df_news['Sentiment_Polarity'], df_news['Sentiment_Subjectivity'] = zip(
                *df_news['Title'].apply(self.get_sentiment)
            )

            return df_news
        else:
            return pd.DataFrame()

    def scrape_kitco_news(self):
        """Scrape news from Kitco"""
        url = "https://www.kitco.com/news"
        response = requests.get(url, headers=self.headers)
        soup = BeautifulSoup(response.content, 'html.parser')

        news_items = []

        # Find news articles
        articles = soup.find_all('div', class_='news-item')[:20]  # Limit to recent 20

        for article in articles:
            try:
                title_elem = article.find('h3') or article.find('h2') or article.find('a')
                title = title_elem.text.strip() if title_elem else "No title"

                link_elem = article.find('a')
                link = "https://www.kitco.com" + link_elem['href'] if link_elem and link_elem.get('href') else ""

                date_elem = article.find('time') or article.find('span', class_='date')
                date_str = date_elem.text.strip() if date_elem else datetime.now().strftime('%Y-%m-%d')

                # Try to parse date
                try:
                    date = pd.to_datetime(date_str)
                except:
                    date = datetime.now()

                if 'gold' in title.lower() or 'xau' in title.lower():
                    news_items.append({
                        'Date': date,
                        'Title': title,
                        'Source': 'Kitco',
                        'Link': link,
                        'Category': 'Gold News'
                    })

            except Exception as e:
                continue

        return news_items

    def scrape_investing_news(self):
        """Scrape news from Investing.com"""
        url = "https://www.investing.com/news/commodities-news/gold"
        response = requests.get(url, headers=self.headers)
        soup = BeautifulSoup(response.content, 'html.parser')

        news_items = []

        # Find news articles
        articles = soup.find_all('article')[:15]  # Limit to recent 15

        for article in articles:
            try:
                title_elem = article.find('a', class_='title')
                if not title_elem:
                    continue

                title = title_elem.text.strip()
                link = "https://www.investing.com" + title_elem['href']

                time_elem = article.find('time')
                date_str = time_elem['datetime'] if time_elem and time_elem.get('datetime') else datetime.now().strftime('%Y-%m-%d %H:%M:%S')

                try:
                    date = pd.to_datetime(date_str)
                except:
                    date = datetime.now()

                if 'gold' in title.lower():
                    news_items.append({
                        'Date': date,
                        'Title': title,
                        'Source': 'Investing.com',
                        'Link': link,
                        'Category': 'Gold News'
                    })

            except Exception as e:
                continue

        return news_items

    def get_news_impact_score(self, news_df, price_df):
        """
        Calculate news impact on gold prices
        """
        if news_df.empty or price_df.empty:
            return news_df

        # Merge news with price data
        news_with_price = pd.merge_asof(
            news_df.sort_values('Date'),
            price_df[['Close']].reset_index().rename(columns={'Date': 'Price_Date'}),
            left_on='Date',
            right_on='Price_Date',
            direction='backward'
        )

        # Calculate price movement after news
        news_with_price['Price_1d_After'] = news_with_price['Price_Date'].apply(
            lambda x: price_df.loc[price_df.index >= x].head(2)['Close'].iloc[-1] if len(price_df.loc[price_df.index >= x]) > 1 else None
        )

        news_with_price['Price_Change_1d'] = (
            (news_with_price['Price_1d_After'] - news_with_price['Close']) / news_with_price['Close'] * 100
        )

        # Impact score based on sentiment and price movement
        news_with_price['Impact_Score'] = (
            news_with_price['Sentiment_Polarity'] * news_with_price['Price_Change_1d'].abs()
        )

        return news_with_price

def main():
    scraper = GoldNewsScraper()

    # Scrape recent news (last 30 days)
    news_df = scraper.scrape_gold_news(days_back=30)

    if not news_df.empty:
        # Save news data
        news_df.to_csv('gold_news_dataset.csv', index=False)

        # Load price data for impact analysis
        try:
            price_df = pd.read_csv('XAUUSD_full_dataset.csv', parse_dates=['Date'], index_col='Date')

            # Calculate news impact
            news_with_impact = scraper.get_news_impact_score(news_df, price_df)
            news_with_impact.to_csv('gold_news_with_impact.csv', index=False)

            print(f"News data saved. Found {len(news_df)} articles.")
            print(f"News with impact analysis saved. Shape: {news_with_impact.shape}")

        except FileNotFoundError:
            print("Price dataset not found. Saving news data only.")
            print(f"News data saved. Found {len(news_df)} articles.")
    else:
        print("No news data collected.")

if __name__ == "__main__":
    main()