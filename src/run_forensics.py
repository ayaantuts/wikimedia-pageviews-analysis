import json
import pandas as pd
from pathlib import Path
from traffic_forensics import ForensicAnalyzer

def analyze_all_data(data_dir="./data"):
    print(f"--- Running Traffic Forensics on Data in '{data_dir}' ---")
    
    data_path = Path(data_dir)
    json_files = list(data_path.glob("*.json"))
    
    if not json_files:
        print("No JSON files found.")
        return

    analyzer = ForensicAnalyzer()
    results = []

    print(f"{'Article Name':<40} | {'Score':<5} | {'FFT':<6} | {'Benford P':<10} | {'Status'}")
    print("-" * 85)

    for file in json_files:
        try:
            with open(file, 'r') as f:
                data = json.load(f)
            
            if 'items' not in data or not data['items']:
                continue

            # Create DataFrame for this article
            items = data['items']
            df = pd.DataFrame(items)
            df['pageviews'] = df['views'] # Rename for consistency if needed, though Analyzer uses arguments
            
            # Extract traffic series
            traffic_series = df['views']
            article_name = items[0]['article'].replace('_', ' ')

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
            
            results.append({
                'article': article_name,
                'score': score,
                'fft': fft_strength,
                'benford_p': benford_p,
                'status': status
            })

        except Exception as e:
            print(f"Error processing {file.name}: {e}")

    # Summary
    print("-" * 85)
    suspicious_count = sum(1 for r in results if r['status'] == "SUSPICIOUS")
    print(f"\nAnalysis Complete. Found {suspicious_count} suspicious articles out of {len(results)}.")

if __name__ == "__main__":
    analyze_all_data()
