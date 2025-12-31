import json
import pandas as pd
from pathlib import Path
from traffic_forensics import ForensicAnalyzer
from data_loader import WikiResearchFetcher


def analyze_csv_data(csv_file="influenza_research_data.csv"):
    print(f"--- Running Traffic Forensics on CSV '{csv_file}' ---")
    
    file_path = Path(csv_file)
    if not file_path.exists():
        print(f"Error: File {csv_file} not found.")
        return

    analyzer = ForensicAnalyzer()
    
    print(f"{'Article Name':<40} | {'Score':<5} | {'FFT':<6} | {'Benford P':<10} | {'Status'}")
    print("-" * 85)

    try:
        df = pd.read_csv(file_path)
        
        # Ensure timestamp is datetime and set as index
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df.set_index('timestamp', inplace=True)
            
        # We investigate 'views_user' as the organic traffic signal
        if 'views_user' not in df.columns:
            print(f"Error: 'views_user' column not found in {csv_file}")
            return
            
        traffic_series = df['views_user'].fillna(0)
        
        # Use filename as article name/identifier for now
        article_name = file_path.stem

        # Run Analysis
        veracity = analyzer.calculate_veracity_score(traffic_series)
        
        score = veracity['veracity_score']
        fft_strength = veracity['details']['fft']
        benford_p = veracity['details']['benford_p']
        
        status = "OK"
        if score < 0.40:
            status = "SUSPICIOUS"
        elif score > 0.70:
            status = "TRUSTED"
        else:
            status = "NEUTRAL"

        print(f"{article_name[:40]:<40} | {score:<5} | {fft_strength:<6.4f} | {benford_p:<10.4f} | {status}")
        
    except Exception as e:
        print(f"Error processing {csv_file}: {e}")

if __name__ == "__main__":
    analyze_csv_data()
