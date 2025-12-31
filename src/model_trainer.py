import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import classification_report, precision_score, recall_score
import joblib
import matplotlib.pyplot as plt

class EventDetectionModel:
    """
    The Research Model.
    Uses a Random Forest Classifier trained on 'Pseudo-Labeled' data.
    
    Why Random Forest? 
    1. Handles non-linear relationships (e.g., High Views + Low Edits = Noise).
    2. Provides Feature Importance (Explainability).
    3. Robust to outliers/noise compared to Neural Networks on small datasets.
    """
    
    def __init__(self, model_path="model_v1.pkl"):
        self.model = RandomForestClassifier(
            n_estimators=100,
            class_weight='balanced', # Critical: Events are rare (1% of data), this forces the model to care.
            random_state=42,
            n_jobs=-1
        )
        self.model_path = model_path
        self.feature_cols = [
            'ratio_to_30d_mean', 'ratio_to_7d_mean', 'z_score_30d', # Signal Strength
            'view_to_edit_ratio', 'edit_ratio_7d', 'editor_diversity', # Knowledge Dynamics
            'spider_ratio', 'trust_flag', # Forensic Filters
            'day_of_week', 'is_weekend' # Temporal Context
        ]

    def generate_pseudo_labels(self, df, z_threshold=2.5, min_editors=2):
        """
        Creates the 'Ground Truth' (y) for training.
        
        Research Logic:
        An event is ONLY valid if:
        1. Traffic is statistically anomalous (Z-Score > 2.5).
        2. It is NOT dominated by bots (Trust Flag == 1).
        3. At least 'min_editors' distinct humans touched the page (Crowd Validation).
        """
        df = df.copy()
        
        # Vectorized condition
        condition = (
            (df['z_score_30d'] > z_threshold) & 
            (df['trust_flag'] == 1) & 
            (df['unique_editors'] >= min_editors)
        )
        
        df['is_event'] = np.where(condition, 1, 0)
        return df

    def train_time_series_split(self, df):
        """
        Performs Walk-Forward Validation (The Gold Standard for Time Series).
        We cannot use random shuffle! We must train on past, test on future.
        """
        # 1. Prepare Data
        df_labeled = self.generate_pseudo_labels(df)
        X = df_labeled[self.feature_cols]
        y = df_labeled['is_event']
        
        # 2. Time Series Split (5 splits)
        tscv = TimeSeriesSplit(n_splits=5)
        
        print(f"--- Starting Time-Series Validation (N={len(df)}) ---")
        
        fold = 1
        for train_index, test_index in tscv.split(X):
            X_train, X_test = X.iloc[train_index], X.iloc[test_index]
            y_train, y_test = y.iloc[train_index], y.iloc[test_index]
            
            # Train
            self.model.fit(X_train, y_train)
            
            # Predict
            y_pred = self.model.predict(X_test)
            
            # Evaluate
            prec = precision_score(y_test, y_pred, zero_division=0)
            rec = recall_score(y_test, y_pred, zero_division=0)
            print(f"Fold {fold}: Precision={prec:.2f}, Recall={rec:.2f} (Train Size: {len(X_train)})")
            fold += 1
            
        # 3. Final Training on Full Dataset
        print("--- Final Model Training ---")
        self.model.fit(X, y)
        joblib.dump(self.model, self.model_path)
        print(f"Model saved to {self.model_path}")
        
        return self.model

    def get_feature_importance(self):
        """
        Generates the 'Explainability' Report for your paper.
        """
        importances = self.model.feature_importances_
        indices = np.argsort(importances)[::-1]
        
        print("\n--- Feature Importance Report (Gini) ---")
        for f in range(len(self.feature_cols)):
            print(f"{f+1}. {self.feature_cols[indices[f]]}: {importances[indices[f]]:.4f}")
            
        return indices, importances

# --- Usage Simulation ---
if __name__ == "__main__":
    # Load the processed data from previous step
    # For now, we simulate the dataframe structure
    from feature_engineering import FeatureEngineer
    
    # ... (Reuse the dummy data generation from previous step) ...
    dates = pd.date_range(start="2024-01-01", periods=200) # Increased size
    data = {
        'views_user': np.random.randint(100, 1000, 200),
        'views_spider': np.random.randint(10, 50, 200),
        'edit_count': np.random.randint(0, 5, 200),
        'unique_editors': np.random.randint(0, 3, 200),
        'volatility_bytes': np.random.randint(0, 500, 200)
    }
    df_raw = pd.DataFrame(data, index=dates)
    
    # Inject synthetic event
    df_raw.loc[dates[150]:dates[155], 'views_user'] += 8000
    df_raw.loc[dates[150]:dates[155], 'unique_editors'] += 10 # High crowd
    
    # Feature Engineering
    engineer = FeatureEngineer()
    df_ready = engineer.process_article(df_raw)
    
    # Modeling
    detector = EventDetectionModel()
    detector.train_time_series_split(df_ready)
    detector.get_feature_importance()