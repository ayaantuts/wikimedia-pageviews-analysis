import pandas as pd
import numpy as np
from traffic_forensics import ForensicAnalyzer

class FeatureEngineer:
	"""
	Transforms raw time-series data into a Machine Learning dataset.
	Implements:
	1. Temporal Ratios (The 'Spike' detectors)
	2. Knowledge Dynamics (The 'Context' detectors)
	3. Forensic Signals (The 'Quality' filters)
	"""

	def __init__(self):
		self.forensic_analyzer = ForensicAnalyzer()

	def _calculate_rolling_stats(self, df, window_days, column):
		"""
		Helper to calculate rolling mean and std deviation.
		Crucial: uses shift(1) to prevent Data Leakage (using today's data to predict today).
		"""
		# We compute stats for the *previous* N days to predict the *current* day
		shifted_col = df[column].shift(1) 
		
		rolling_mean = shifted_col.rolling(window=window_days).mean()
		rolling_std = shifted_col.rolling(window=window_days).std()
		
		return rolling_mean, rolling_std

	def add_temporal_features(self, df):
		"""
		Adds Time-Series features (Seasonality & Trends).
		"""
		# 1. Calendar Features (Seasonality)
		df['day_of_week'] = df.index.dayofweek
		df['is_weekend'] = df['day_of_week'].apply(lambda x: 1 if x >= 5 else 0)
		df['month'] = df.index.month
		
		# 2. Lag Features (Autocorrelation)
		# "Was traffic high yesterday?"
		df['lag_1d'] = df['views_user'].shift(1)
		df['lag_7d'] = df['views_user'].shift(7) # Weekly cycle comparison
		
		return df

	def add_statistical_features(self, df):
		"""
		Adds Z-Scores and Ratios (Anomaly Intensity).
		Ref: Your successful 'Attempt 3' in the project overview.
		"""
		# Calculate Baselines (30-day and 7-day)
		mean_30d, std_30d = self._calculate_rolling_stats(df, 30, 'views_user')
		mean_7d, _ = self._calculate_rolling_stats(df, 7, 'views_user')
		
		# 1. The "Signal Strength" (Z-Score)
		# Avoid division by zero with a small epsilon
		df['z_score_30d'] = (df['views_user'] - mean_30d) / (std_30d + 1e-6)
		
		# 2. The "Growth Ratios" (The features that worked best for you)
		df['ratio_to_30d_mean'] = df['views_user'] / (mean_30d + 1e-6)
		df['ratio_to_7d_mean'] = df['views_user'] / (mean_7d + 1e-6)
		
		# 3. Velocity (Daily Growth Rate)
		df['growth_rate_1d'] = df['views_user'].pct_change()
		
		return df

	def add_knowledge_graph_features(self, df):
		"""
		Adds Contextual features from the Edit History.
		This distinguishes 'Viral Trivia' from 'Breaking News'.
		"""
		# 1. Edit Velocity: Are edits accelerating?
		edit_mean_7d, _ = self._calculate_rolling_stats(df, 7, 'edit_count')
		df['edit_ratio_7d'] = df['edit_count'] / (edit_mean_7d + 1e-6)
		
		# 2. The "Curiosity-to-Action" Ratio
		# High Views / Low Edits = Passive Curiosity
		# High Views / High Edits = Active Knowledge Construction
		df['view_to_edit_ratio'] = df['views_user'] / (df['edit_count'] + 1)
		
		# 3. Crowd Consensus (Unique Editors)
		# High views but only 1 editor = Suspicious
		df['editor_diversity'] = df['unique_editors'] / (df['edit_count'] + 1)
		
		# 4. Content Volatility (Bytes Changed)
		# Massive byte change = Major Rewrite (Event)
		df['content_shock'] = df['volatility_bytes'] > df['volatility_bytes'].rolling(30).mean() * 3
		df['content_shock'] = df['content_shock'].astype(int)
		
		return df

	def add_forensic_features(self, df):
		"""
		Adds the 'Bot vs Human' signals.
		"""
		# 1. The Spider Ratio
		# If Spider views spike alongside User views, it's likely a crawler swarm, not news.
		df['spider_ratio'] = df['views_spider'] / (df['views_user'] + 1)
		
		# 2. Rolling Veracity Score (Simplified for efficiency)
		# We calculate the FFT/Benford score for the *previous 14 days* window
		# Note: This is computationally expensive, so we apply it only to rows with Z-score > 2
		# For the prototype, we create a placeholder or a simplified flag
		
		# Simple heuristic: If spider_ratio > 0.5, trust is low
		df['trust_flag'] = np.where(df['spider_ratio'] < 0.2, 1, 0)
		
		return df

	def add_cluster_features(self, df_main, cluster_dict):
		"""
		New Feature: Compare 'Influenza' traffic to 'Average Related Page' traffic.
		If Main spikes but Cluster is flat -> Likely Bot/Spam.
		If Main spikes AND Cluster spikes -> Real Event.
		"""
		# Calculate "Cluster Average" traffic
		related_dfs = [df for name, df in cluster_dict.items() if name != 'Main']
		
		if not related_dfs:
			df_main['cluster_correlation'] = 0
			return df_main
			
		# Combine all related views into one average series
		cluster_views = pd.concat([df['views_user'] for df in related_dfs], axis=1).mean(axis=1)
		
		# Feature 1: Cluster Correlation (Rolling 7-day correlation)
		df_main['cluster_correlation'] = df_main['views_user'].rolling(7).corr(cluster_views)
		
		# Feature 2: Divergence (Is Main growing much faster than Cluster?)
		# A massive divergence (e.g. > 10x) is suspicious (bot attack)
		df_main['cluster_divergence'] = df_main['views_user'] / (cluster_views + 1)
		
		return df_main

	def process_article(self, df_merged, cluster_dict):
		"""
		Pipeline Master Function.
		"""
		# 1. Clean Data
		df = df_merged.copy()
		df.fillna(0, inplace=True)
		
		# 2. Apply Modules
		df = self.add_temporal_features(df)
		df = self.add_statistical_features(df)
		df = self.add_knowledge_graph_features(df)
		df = self.add_forensic_features(df)
		df = self.add_cluster_features(df, cluster_dict)
		
		# 3. Drop initial rows (NaNs from rolling windows)
		df.dropna(inplace=True)

		df.to_csv("./Influenza_processed_data.csv")

		return df

# --- Usage Simulation ---
if __name__ == "__main__":
	# Create Dummy Data to test the pipeline
	dates = pd.date_range(start="2025-01-01", periods=100)
	data = {
		'views_user': np.random.randint(100, 1000, 100),
		'views_spider': np.random.randint(10, 50, 100),
		'edit_count': np.random.randint(0, 5, 100),
		'unique_editors': np.random.randint(0, 3, 100),
		'volatility_bytes': np.random.randint(0, 500, 100)
	}
	df_raw = pd.DataFrame(data, index=dates)
		
	# Introduce a "fake event" at index 80
	df_raw.loc[dates[80]:dates[85], 'views_user'] += 5000  # Massive spike
	df_raw.loc[dates[80]:dates[85], 'edit_count'] += 50	# High edits
		
	# Run Pipeline
	engineer = FeatureEngineer()
	df_ready = engineer.process_article(df_raw)
		
	print("Features Generated Successfully:")
	print(df_ready[['views_user', 'z_score_30d', 'ratio_to_30d_mean', 'view_to_edit_ratio']].tail(20))