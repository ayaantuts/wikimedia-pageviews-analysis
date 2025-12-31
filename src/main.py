import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import joblib
import os

# Import our custom research modules
from data_loader import WikiResearchFetcher
from feature_engineering import FeatureEngineer
from model_trainer import EventDetectionModel
from traffic_forensics import ForensicAnalyzer

class DomainMonitor:
    """
    The Real-Time Research Pipeline.
    Orchestrates Data Collection -> Forensics -> Inference -> Alerting.
    """
    
    def __init__(self, domain_name="Influenza_Research", articles=None):
        self.domain = domain_name
        self.articles = articles or ["Influenza", "Common_cold", "Oseltamivir", "H1N1", "Fever"]
        
        # Initialize Modules
        self.fetcher = WikiResearchFetcher()
        self.engineer = FeatureEngineer()
        self.forensics = ForensicAnalyzer()
        self.model_loader = EventDetectionModel()
        
        # Load Model (or trigger training if missing)
        if os.path.exists("model_v1.pkl"):
            self.model = joblib.load("model_v1.pkl")
            print("✅ Pre-trained Research Model Loaded.")
        else:
            print("⚠️ No model found. Please run 'model_trainer.py' first.")
            self.model = None

    def _get_sufficient_history(self, target_date):
        """
        Research Requirement: To calculate a 30-day moving average for 'Today',
        we technically need the last 40 days of data to be safe.
        """
        end_date = target_date
        start_date = target_date - timedelta(days=60) # Buffer for rolling windows
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
            target_date = datetime.now() - timedelta(days=1) # Analyze Yesterday (latest complete data)

        print(f"\n🔬 Starting Analysis for: {target_date.date()}")
        start_date, end_date = self._get_sufficient_history(target_date)
        
        results = []
        
        for article in self.articles:
            try:
                # 1. Fetch Raw Data (Traffic + Edits + Spiders)
                df_raw = self.fetcher.get_research_dataset(article, start_date, end_date)
                
                if df_raw.empty:
                    continue

                # 2. Run Forensic Analysis (The "Quality Check")
                # We analyze the last 14 days of this article's traffic
                recent_traffic = df_raw['views_user'].tail(14)
                veracity = self.forensics.calculate_veracity_score(recent_traffic)
                
                # 3. Engineer Features (Ratios, Z-scores)
                df_features = self.engineer.process_article(df_raw)
                
                # Extract ONLY the target row (the specific day we are analyzing)
                # We use .iloc[-1] because the dataframe ends on 'target_date'
                target_row = df_features.iloc[[-1]].copy()
                
                # 4. Model Inference
                if self.model:
                    # Select only the columns the model expects
                    X_input = target_row[self.model_loader.feature_cols]
                    prediction = self.model.predict(X_input)[0]
                    proba = self.model.predict_proba(X_input)[0][1] # Confidence score
                else:
                    prediction = 0
                    proba = 0.0

                # 5. Log Result
                result = {
                    'article': article,
                    'views': target_row['views_user'].values[0],
                    'editors': target_row['unique_editors'].values[0],
                    'z_score': round(target_row['z_score_30d'].values[0], 2),
                    'is_anomaly': prediction,
                    'model_confidence': round(proba, 2),
                    'forensic_score': veracity['veracity_score'], # 0.0 to 1.0
                    'forensic_details': veracity['details']
                }
                results.append(result)
                print(f"  > {article}: Z={result['z_score']}, Conf={result['model_confidence']}, Trust={result['forensic_score']}")

            except Exception as e:
                print(f"  x Error processing {article}: {e}")

        # 6. Generate Domain Report
        self._generate_alert_report(results, target_date)

    def _generate_alert_report(self, results, date):
        """
        Decides if a 'Domain Event' has occurred based on Weighted Logic.
        """
        if not results:
            print("No data available.")
            return

        df_res = pd.DataFrame(results)
        
        # --- The Research Logic ---
        
        # Metric 1: Anomaly Density (% of basket that is spiking)
        anomaly_count = df_res['is_anomaly'].sum()
        total_articles = len(df_res)
        anomaly_pct = (anomaly_count / total_articles) * 100
        
        # Metric 2: Forensic Trust (Average trust of the spiking articles)
        # If the anomalies have Low Trust, it's likely a bot attack.
        anomalous_articles = df_res[df_res['is_anomaly'] == 1]
        if not anomalous_articles.empty:
            avg_trust = anomalous_articles['forensic_score'].mean()
        else:
            avg_trust = 1.0 # Default high trust if no anomalies
            
        # --- Alert Decision Tree ---
        print("\n" + "="*40)
        print(f"📊 FINAL RESEARCH REPORT: {self.domain} [{date.date()}]")
        print("="*40)
        print(f"Anomaly Density: {anomaly_pct:.1f}% ({anomaly_count}/{total_articles} articles)")
        print(f"Forensic Trust:  {avg_trust:.2f} / 1.0")
        
        THRESHOLD = 20.0 # 20% of articles must spike
        TRUST_THRESHOLD = 0.40 # Must be at least 40% organic-looking
        
        if anomaly_pct >= THRESHOLD:
            if avg_trust >= TRUST_THRESHOLD:
                print("\n🚨🚨 CRITICAL ALERT: CONFIRMED DOMAIN EVENT 🚨🚨")
                print("Likelihood: HIGH (Supported by Organic Traffic Patterns)")
                print("Top Contributors:")
                print(anomalous_articles[['article', 'z_score', 'editors']].to_string(index=False))
            else:
                print("\n⚠️ WARNING: ANOMALY DETECTED, BUT TRUST IS LOW ⚠️")
                print("Diagnosis: SUSPECTED BOT ATTACK / CRAWLER ACTIVITY")
                print("Reason: Spectral Analysis failed to find human heartbeats.")
        else:
            print("\n✅ Status: Normal Baseline Activity.")
            
        print("="*40)

# --- Execution ---
if __name__ == "__main__":
    # Example: Monitor the "Public Health" domain
    monitor = DomainMonitor(
        domain_name="Public_Health_Watch",
        articles=["Influenza", "Pandemic", "Vaccine", "Oseltamivir", "Centers_for_Disease_Control_and_Prevention"]
    )
    
    # Run analysis for "Yesterday"
    monitor.analyze_domain_today()