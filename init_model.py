import os
import sys
sys.path.append(os.path.join(os.getcwd(), 'src'))
import json
import joblib
import pandas as pd
import numpy as np
from src.feature_engineering import FeatureEngineer
from src.model_trainer import EventDetectionModel

# 1. Create 'models' directory if it doesn't exist
if not os.path.exists('models'):
    os.makedirs('models')
    print("Created 'models' directory.")

print("Generating model artifacts...")

# 2. Generate Dummy Training Data (Simulating the pipeline)
# We need this to train the initial model structure
dates = pd.date_range(start="2024-01-01", periods=200)
data = {
    'views_user': np.random.randint(100, 1000, 200),
    'views_spider': np.random.randint(10, 50, 200),
    'edit_count': np.random.randint(0, 5, 200),
    'unique_editors': np.random.randint(0, 3, 200),
    'volatility_bytes': np.random.randint(0, 500, 200)
}
df_raw = pd.DataFrame(data, index=dates)

# Inject a fake event so the model learns something
df_raw.loc[dates[150]:dates[155], 'views_user'] += 8000
df_raw.loc[dates[150]:dates[155], 'unique_editors'] += 10

# 3. Process Data
engineer = FeatureEngineer()
# We pass a dummy cluster dict as required by your updated code
dummy_cluster = {'Main': df_raw} 
df_ready = engineer.process_article(df_raw, dummy_cluster)

# 4. Train Model
# We update the model path to save to the 'models' folder
model_save_path = "models/domain_event_model.joblib"
trainer = EventDetectionModel(model_path=model_save_path)
trained_model = trainer.train_time_series_split(df_ready)

print(f"✅ Model saved to: {model_save_path}")

# 5. Save Article Mappings (Categories)
# The FeatureEngineer adds 'article_name' (though our dummy data might not have it, 
# we will simulate it for the mapping file)
df_ready['article_name'] = 'Simulated_Article'
df_ready['article_name'] = df_ready['article_name'].astype('category')
article_categories = df_ready['article_name'].cat.categories

mappings_path = "models/article_mappings.joblib"
joblib.dump(article_categories, mappings_path)
print(f"✅ Mappings saved to: {mappings_path}")

# 6. Save Article Corpus (The list of articles to monitor)
# This list is used by daily_automate.py to know WHAT to fetch.
# You can add your real articles here.
article_corpus = ["Influenza", "Common cold", "H1N1", "Fever", "Oseltamivir"]
corpus_path = "models/article_corpus.json"

with open(corpus_path, 'w') as f:
    json.dump(article_corpus, f)
print(f"✅ Corpus list saved to: {corpus_path}")

print("\nInitialization Complete! You can now run 'server.py'.")