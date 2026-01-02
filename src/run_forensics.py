import json
import pandas as pd
from pathlib import Path
from traffic_forensics import ForensicAnalyzer
from data_loader import WikiResearchFetcher
from visualization import ResearchDashboard

def analyze_all_csvs(data_dir="data"):
    print(f"--- Running Traffic Forensics on CSVs in '{data_dir}' ---")
    
    path = Path(data_dir)
    if not path.exists():
        print(f"Error: Directory {data_dir} not found.")
        return

    csv_files = list(path.glob("*_research_data.csv"))
    
    if not csv_files:
        print(f"No matching CSV files found in {data_dir}")
        return

    analyzer = ForensicAnalyzer()
    dashboard = ResearchDashboard() 
    
    print(f"{'Article Name':<30} | {'Score':<5} | {'FFT':<6} | {'Benford':<8} | {'Max Z':<6} | {'AutoC':<6} | {'Status'}")
    print("-" * 95)

    results = []

    for csv_file in csv_files:
        try:
            df = pd.read_csv(csv_file)
            
            # Ensure timestamp is datetime and set as index
            if 'timestamp' in df.columns:
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                df.set_index('timestamp', inplace=True)
                
            # We investigate 'views_user' as the organic traffic signal
            if 'views_user' not in df.columns:
                print(f"Error: 'views_user' column not found in {csv_file.name}")
                continue
                
            traffic_series = df['views_user'].fillna(0)
            
            # Use filename as article name/identifier
            article_name = csv_file.stem.replace("_research_data", "")

            # Run Analysis
            veracity = analyzer.calculate_veracity_score(traffic_series)
            
            # Generate Verification Plot
            dashboard.plot_traffic_volume(traffic_series, article_name)
            
            score = veracity['veracity_score']
            fft_strength = veracity['details']['fft']
            benford_p = veracity['details']['benford_p']
            max_z = veracity['details']['max_z']
            autocorr = veracity['details']['autocorr']
            
            status = "OK"
            if score < 0.50:
                status = "SUSPICIOUS"
            elif score > 0.70:
                status = "TRUSTED"
            else:
                status = "NEUTRAL"

            print(f"{article_name[:30]:<30} | {score:<5} | {fft_strength:<6.4f} | {benford_p:<8.4f} | {max_z:<6.2f} | {autocorr:<6.2f} | {status}")
            
            results.append(status)
            
        except Exception as e:
            print(f"Error processing {csv_file.name}: {e}")

    # Summary
    print("-" * 85)
    suspicious_count = results.count("SUSPICIOUS")
    print(f"\nAnalysis Complete. Found {suspicious_count} suspicious articles out of {len(results)}.")

if __name__ == "__main__":
    analyze_all_csvs()
