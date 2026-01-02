import schedule
import time
import sys
import os
import logging

# [CHANGED] Add 'src' to system path so we can import modules from it
sys.path.append(os.path.join(os.getcwd(), 'src'))

# [CHANGED] Import the class from our corrected src/main.py
from src.main import DomainMonitor

logging.basicConfig(
    filename='server_log.txt',
    level=logging.INFO,
    format='%(asctime)s - %(message)s'
)

def job():
    print("\n[Server] Starting Daily Analysis Job...")
    logging.info("Starting Daily Analysis Job")
    
    try:
        # [CHANGED] Initialize the Monitor and run the analysis
        monitor = DomainMonitor()
        monitor.analyze_domain_today()
        
        print("[Server] Job Completed Successfully.")
        logging.info("Job Completed Successfully")
        
    except Exception as e:
        print(f"[Server] Job Failed: {e}")
        logging.error(f"Job Failed: {e}")

# Schedule (e.g., every day at 09:00)
schedule.every().day.at("19:35").do(job)

# Run once immediately on startup
print("[Server] Running immediate startup check...")
job()

# Server Loop
print("[Server] Server is running. Waiting for scheduled jobs...")
while True:
    schedule.run_pending()
    time.sleep(60)