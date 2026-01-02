import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import joblib
import os
import json
import sys

# Import custom modules
# We use try/except to handle running this script directly vs from server.py
try:
    from src.data_loader import WikiResearchFetcher
    from src.feature_engineering import FeatureEngineer
    from src.model_trainer import EventDetectionModel
    from src.traffic_forensics import ForensicAnalyzer
except ImportError:
    from data_loader import WikiResearchFetcher
    from feature_engineering import FeatureEngineer
    from model_trainer import EventDetectionModel
    from traffic_forensics import ForensicAnalyzer

class DomainMonitor:
    """
    The Real-Time Research Pipeline.
    Orchestrates Data Collection -> Forensics -> Inference -> Alerting.
    """
    
    def __init__(self, domain_name="Influenza_Research"):
        self.domain = domain_name
        
        # [CHANGED] Load the article corpus list from the models folder
        corpus_path = os.path.join("models", "article_corpus.json")
        if os.path.exists(corpus_path):
            with open(corpus_path, 'r') as f:
                self.articles = json.load(f)
            print(f"✅ Loaded {len(self.articles)} articles from corpus.")
        else:
            print("⚠️ Corpus not found. Using defaults.")
            self.articles = ["Influenza", "Common_cold", "Oseltamivir", "H1N1", "Fever"]
        
        # Initialize Modules
        self.fetcher = WikiResearchFetcher()
        self.engineer = FeatureEngineer()
        self.forensics = ForensicAnalyzer()
        self.model_loader = EventDetectionModel()
        
        # [CHANGED] Point to the correct .joblib model path
        model_path = os.path.join("models", "domain_event_model.joblib")
        
        if os.path.exists(model_path):
            self.model = joblib.load(model_path)
            print(f"✅ Pre-trained Research Model Loaded from {model_path}.")
        else:
            print(f"⚠️ No model found at {model_path}. Please run 'init_models.py'.")
            self.model = None

    def _get_sufficient_history(self, target_date):
        """
        Research Requirement: To calculate a 30-day moving average for 'Today',
        we technically need the last 40-60 days of data.
        """
        end_date = target_date
        start_date = target_date - timedelta(days=60) 
        return start_date, end_date

    def analyze_domain_today(self, target_date=None):
        """
        The Master Loop.
        1. Fetches data for all articles.
        2. Calculates Forensic Scores.
        3. Predicts Anomalies.
        4. Aggregates into a Domain Report.
        """
        if target_date is None:
            target_date = datetime.now() - timedelta(days=1) 

        print(f"\n🔬 Starting Analysis for: {target_date.date()}")
        start_date, end_date = self._get_sufficient_history(target_date)
        
        results = []
        
        for article in self.articles:
            try:
                # 1. Fetch Raw Data (Traffic + Edits + Spiders)
                dk = self.fetcher.fetch_cluster_data(article, start_date, end_date)
                df_raw = dk.get(article)
                
                if df_raw is None or df_raw.empty:
                    print(f"  x No data for {article}")
                    continue

                # 2. Run Forensic Analysis (The "Quality Check")
                # We analyze the last 14 days of this article's traffic
                recent_traffic = df_raw['views_user'].tail(14)
                veracity = self.forensics.calculate_veracity_score(recent_traffic)
                
                # 3. Engineer Features (Ratios, Z-scores)
                # [CHANGED] Ensure we define dummy cluster if not found to avoid errors
                df_features = self.engineer.process_article(df_raw, dk)
                
                # Extract ONLY the target row (the specific day we are analyzing)
                target_row = df_features.iloc[[-1]].copy()
                
                # 4. Model Inference
                if self.model:
                    # Select only the columns the model expects
                    # This ensures feature alignment with the trained model
                    X_input = target_row[self.model_loader.feature_cols]
                    prediction = self.model.predict(X_input)[0]
                    proba = self.model.predict_proba(X_input)[0][1]
                else:
                    prediction = 0
                    proba = 0.0

                # 5. Log Result
                result = {
                    'article': article,
                    'z_score': round(target_row['z_score_30d'].values[0], 2),
                    'is_anomaly': int(prediction),
                    'model_confidence': round(proba, 2),
                    'forensic_score': veracity['veracity_score'],
                }
                results.append(result)
                print(f"  > {article[:15]}... Z={result['z_score']}, Trust={result['forensic_score']}, Anomaly={result['is_anomaly']}")

            except Exception as e:
                print(f"  x Error processing {article}: {e}")

        # 6. Generate Domain Report
        self._generate_alert_report(results, target_date)

    def _generate_alert_report(self, results, date):
        if not results:
            print("No data available.")
            return

        df_res = pd.DataFrame(results)
        
        # Metric 1: Anomaly Density
        anomaly_count = df_res['is_anomaly'].sum()
        total_articles = len(df_res)
        anomaly_pct = (anomaly_count / total_articles) * 100
        
        # Metric 2: Forensic Trust
        anomalous_articles = df_res[df_res['is_anomaly'] == 1]
        if not anomalous_articles.empty:
            avg_trust = anomalous_articles['forensic_score'].mean()
        else:
            avg_trust = 1.0 
            
        print("\n" + "="*40)
        print(f"📊 FINAL RESEARCH REPORT: {self.domain} [{date.date()}]")
        print("="*40)
        print(f"Anomaly Density: {anomaly_pct:.1f}% ({anomaly_count}/{total_articles} articles)")
        print(f"Forensic Trust:  {avg_trust:.2f} / 1.0")
        
        THRESHOLD = 20.0 
        
        if anomaly_pct >= THRESHOLD:
            print("\n🚨🚨 CRITICAL ALERT: CONFIRMED DOMAIN EVENT 🚨🚨")
            if avg_trust < 0.5:
                print("(Note: Low Trust Score indicates potential bot activity)")
        else:
            print("\n✅ Status: Normal Baseline Activity.")
        print("="*40)

if __name__ == "__main__":
    monitor = DomainMonitor()
    monitor.analyze_domain_today()