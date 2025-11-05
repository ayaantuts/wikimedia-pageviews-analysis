import requests
import pandas as pd
import numpy as np
import joblib
import json
from datetime import datetime, timedelta

# --- 1. Configuration ---

# Load the corpus we trained on
with open('models/article_corpus.json', 'r') as f:
    ARTICLE_CORPUS = json.load(f)

# Load the saved model and mappings
MODEL = joblib.load('models/domain_event_model.joblib')
ARTICLE_MAPPINGS = joblib.load('models/article_mappings.joblib')

# Define the Domain Alert Threshold
# As per your doc, we'll use 20%
DOMAIN_ALERT_THRESHOLD = 0.20  # [cite: 43]

# API and headers
HEADERS = {
    'User-Agent': 'BTechEventDetector/1.0 (B.Tech Final Year Project; contact@example.com)'
}
URL_TEMPLATE = (
    "https://wikimedia.org/api/rest_v1/metrics/pageviews/per-article/"
    "en.wikipedia/all-access/all-agents/{article}/daily/{start}/{end}"
)

# --- 2. Helper Functions ---

def fetch_recent_data(article_list, days_history=40):
    """
    Fetches the last N days of pageview data for all articles.
    We fetch 40 days to safely calculate 30-day rolling features.
    """
    print(f"Fetching data for {len(article_list)} articles...")
    
    # Define date range
    end_date = (datetime.today() - timedelta(days=1))
    start_date = (end_date - timedelta(days=days_history))
    
    end_str = end_date.strftime("%Y%m%d00")
    start_str = start_date.strftime("%Y%m%d00")
    
    all_data = []
    
    for article in article_list:
        article_url = article.replace(' ', '_')
        url = URL_TEMPLATE.format(article=article_url, start=start_str, end=end_str)
        
        try:
            response = requests.get(url, headers=HEADERS)
            response.raise_for_status()
            items = response.json().get('items', [])
            
            for item in items:
                all_data.append({
                    'time_stamp_str': item['timestamp'],
                    'article_name': item['article'].replace('_', ' '),
                    'pageviews': item['views']
                })
        except Exception as e:
            print(f"  Warning: Could not fetch data for '{article}'. Reason: {e}")
            
    df = pd.DataFrame(all_data)
    df['time_stamp'] = pd.to_datetime(df['time_stamp_str'], format='%Y%m%d%H')
    df = df.set_index('time_stamp').sort_index()
    return df

def engineer_features_for_inference(df, article_mappings):
    """
    Re-creates the exact feature set our model was trained on.
    """
    print("Engineering features for the most recent day...")
    
    # Use the loaded mappings to create a lookup dictionary
    article_map_dict = {name: i for i, name in enumerate(article_mappings)}
    
    # --- Feature Engineering (must match training script) ---
    ROLLING_WINDOW_30D = 30
    ROLLING_WINDOW_7D = 7
    
    all_processed_dfs = []

    for article in df['article_name'].unique():
        df_article = df[df['article_name'] == article].copy()
        
        # 1. Rolling Features
        df_article['rolling_mean_30d'] = df_article['pageviews'].rolling(window=ROLLING_WINDOW_30D, min_periods=1).mean()
        df_article['rolling_mean_7d'] = df_article['pageviews'].rolling(window=ROLLING_WINDOW_7D, min_periods=1).mean()
        
        # 2. Ratio Features
        df_article['ratio_to_30d_mean'] = df_article['pageviews'] / (df_article['rolling_mean_30d'] + 1)
        df_article['ratio_to_7d_mean'] = df_article['pageviews'] / (df_article['rolling_mean_7d'] + 1)
        
        all_processed_dfs.append(df_article)
        
    df_features = pd.concat(all_processed_dfs).sort_index().fillna(0)
    
    # --- Filter for *only* the last day ---
    last_day = df_features.index.max()
    df_today = df_features[df_features.index == last_day].copy()
    
    print(f"Processing data for: {last_day.date()}")
    
    # 3. Time-based features
    df_today['day_of_year'] = df_today.index.dayofyear
    df_today['month'] = df_today.index.month
    df_today['day_of_week'] = df_today.index.dayofweek
    
    # 4. Article code mapping
    df_today['article_code'] = df_today['article_name'].map(article_map_dict)
    
    # Handle any articles that might be new or un-mapped
    df_today['article_code'] = df_today['article_code'].fillna(-1).astype(int)
    
    # 5. Define final feature columns (must match training)
    feature_cols = [
        'pageviews',
        'ratio_to_30d_mean',
        'ratio_to_7d_mean',
        'day_of_year',
        'month',
        'day_of_week',
        'article_code'
    ]
    
    # Re-order columns to match model's expectation
    df_final_features = df_today[feature_cols]
    
    # Also return the article names for the report
    return df_final_features, df_today['article_name']

# --- 3. Main Execution Function ---

def run_daily_alerting():
    """
    Main function to run the daily inference and alerting pipeline.
    """
    # 1. Daily Ingestion
    df_recent = fetch_recent_data(ARTICLE_CORPUS)
    
    if df_recent.empty:
        print("No data fetched. Exiting.")
        return

    # 2. Feature Engineering
    X_today, article_names = engineer_features_for_inference(df_recent, ARTICLE_MAPPINGS)
    
    # [cite_start]3. Article-Level Prediction [cite: 41]
    predictions = MODEL.predict(X_today)
    
    # [cite_start]4. Domain-Level Aggregation [cite: 42]
    abnormal_count = np.sum(predictions)
    total_articles = len(predictions)
    abnormal_percentage = abnormal_count / total_articles
    
    # [cite_start]5. Event Alerting [cite: 43]
    print("\n--- Daily Domain Alert Report ---")
    print(f"Domain: Influenza")
    print(f"Date: {(datetime.today() - timedelta(days=1)).date()}")
    print(f"Analyzed {total_articles} articles.")
    print(f"Anomalous Articles Detected: {abnormal_count}")
    print(f"Abnormal Percentage: {abnormal_percentage * 100:.2f}%")
    print(f"Alert Threshold: {DOMAIN_ALERT_THRESHOLD * 100:.2f}%")
    
    if abnormal_percentage >= DOMAIN_ALERT_THRESHOLD:
        print("\n*************************************************")
        print("!!! DOMAIN EVENT ALERT TRIGGERED !!!")
        print("Widespread anomalous interest detected in the domain.")
        print("*************************************************")
        
        # Print the specific articles that were flagged
        print("\nFlagged articles:")
        for name, pred in zip(article_names, predictions):
            if pred == 1:
                print(f"- {name}")
    else:
        print("\n--- Status: Normal ---")
        print("No widespread domain event detected.")

# --- Run the pipeline ---
if __name__ == "__main__":
    run_daily_alerting()