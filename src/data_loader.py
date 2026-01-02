import requests
import pandas as pd
from datetime import datetime, timedelta
import time

import json
import numpy as np
from pathlib import Path

class WikiResearchFetcher:
	"""
	A research-grade data fetcher for Wikimedia projects.
	Designed to support Forensic Traffic Analysis and Knowledge Graph construction.
	"""

	@staticmethod
	def load_traffic_from_file(file_path):
		"""
		Loads traffic data from a JSON file into a standardized pandas Series.
		
		Args:
			file_path (str or Path): Path to the JSON file.
			
		Returns:
			tuple: (article_name, traffic_series)
				- article_name (str): Name of the article.
				- traffic_series (pd.Series): Daily view counts indexed by datetime.
		"""
		try:
			with open(file_path, 'r') as f:
				data = json.load(f)
			
			if 'items' not in data or not data['items']:
				return None, None

			items = data['items']
			df = pd.DataFrame(items)
			
			# Ensure we have the right columns
			if 'views' not in df.columns or 'timestamp' not in df.columns:
				return None, None
				
			# Convert timestamp
			df['timestamp'] = pd.to_datetime(df['timestamp'], format='%Y%m%d00')
			df.set_index('timestamp', inplace=True)
			
			# Get article name
			article_name = items[0]['article'].replace('_', ' ')
			
			return article_name, df['views']
			
		except Exception as e:
			print(f"Error loading {file_path}: {e}")
			return None, None

	def __init__(self, project="en.wikipedia.org", user_agent="ResearchBot/1.0 (fyp_project@google.com)"):
		"""
		Args:
			project (str): The project domain (e.g., 'en.wikipedia').
			user_agent (str): Required by Wiki policy to prevent blocking.
		"""
		self.project = project
		self.headers = {"User-Agent": user_agent}
		self.base_url_rest = f"https://wikimedia.org/api/rest_v1/metrics/pageviews/per-article/{project}"
		self.base_url_action = f"https://{project}/w/api.php"

	def get_related_pages(self, article, limit=5):
		"""
		Fetches 'Related Pages' using the 'morelike' search algorithm.
		This is the engine behind the 'Read More' footer.
		"""
		params = {
				"action": "query",
				"format": "json",
				"list": "search",
				"srsearch": f"morelike:{article}", # The "Related" magic
				"srlimit": limit,
				"srprop": "" # We only need titles, not snippets
		}
		
		try:
				response = requests.get(self.base_url_action, headers=self.headers, params=params)
				data = response.json()
				
				if 'query' in data and 'search' in data['query']:
					# Extract clean titles
					related_titles = [item['title'] for item in data['query']['search']]
					return related_titles
				return []
				
		except Exception as e:
				print(f"Error fetching related pages for {article}: {e}")
				return []

	def fetch_pageviews_daily(self, article, start_date, end_date, agent_type='user'):
		"""
		Fetches daily traffic with strict 'agent' filtering for Traffic Forensics.

		Ref: 'Data Purity' in your guidelines.
		Args:
			agent_type (str): 'user' (for organic), 'spider' (for bot tracking), or 'all-agents'.
		"""
		# Format dates for API (YYYYMMDD)
		start_str = start_date.strftime("%Y%m%d")
		end_str = end_date.strftime("%Y%m%d")

		# Construct URL per your "Traffic Forensics" requirement
		# granularily is fixed to 'daily' for FFT analysis
		endpoint = f"/all-access/{agent_type}/{article}/daily/{start_str}/{end_str}"
		url = self.base_url_rest + endpoint

		try:
			response = requests.get(url, headers=self.headers)
			response.raise_for_status()
			data = response.json()
	
			# Transform into Research-Ready DataFrame
			items = data.get('items', [])
			df = pd.DataFrame(items)
	
			if not df.empty:
				df['timestamp'] = pd.to_datetime(df['timestamp'], format='%Y%m%d00')
				df = df[['timestamp', 'views']]
				df.rename(columns={'views': f'views_{agent_type}'}, inplace=True)
				df.set_index('timestamp', inplace=True)
		
			return df
	
		except requests.exceptions.RequestException as e:
			print(f"Error fetching pageviews for {article}: {e}")
			return pd.DataFrame()

	def fetch_revision_history(self, article, start_date, end_date):
		"""
		Fetches detailed edit history to build the 'Knowledge Graph' & 'Co-Edit' layers.

		This enables:
		1. Revision Count (Velocity)
		2. Bytes Changed (Vandalism/Content Shift)
		3. Unique Editors (Bot vs Human Crowd)
		"""
		params = {
			"action": "query",
			"format": "json",
			"prop": "revisions",
			"titles": article,
			"rvprop": "timestamp|user|size|ids", # Fetch user (for bot check) and size (for content shift)
			"rvlimit": "max",
			"rvstart": start_date.strftime("%Y-%m-%dT00:00:00Z"),
			"rvend": end_date.strftime("%Y-%m-%dT23:59:59Z"),
			"rvdir": "newer" # Oldest to newest
		}

		revisions = []

		while True:
			try:
				response = requests.get(self.base_url_action, headers=self.headers, params=params)
				data = response.json()
		
				pages = data['query']['pages']
				page_id = next(iter(pages))
		
				if 'revisions' in pages[page_id]:
					revisions.extend(pages[page_id]['revisions'])
		
				# Handle Pagination (if >500 edits)
				if 'continue' in data:
					params.update(data['continue'])
				else:
					break
			
			except Exception as e:
				print(f"Error fetching revisions: {e}")
				break

		# Process into Daily Aggregates for the Model
		if revisions:
			df = pd.DataFrame(revisions)
			df['timestamp'] = pd.to_datetime(df['timestamp'])
	
			# Feature Engineering for Research
			df['timestamp'] = df['timestamp'].dt.date
			df['size_change'] = df['size'].diff().abs() # Magnitude of change
	
			daily_stats = df.groupby('timestamp').agg({
				'revid': 'count',			# Total Edits
				'user': 'nunique',		# Unique Editors (Crowd signal)
				'size_change': 'sum'		# Total Content Shift
			}).rename(columns={
				'revid': 'edit_count', 
				'user': 'unique_editors',
				'size_change': 'volatility_bytes'
			})
	
			daily_stats.index = pd.to_datetime(daily_stats.index)
			return daily_stats
	
		return pd.DataFrame()

	def get_research_dataset(self, article, start_date, end_date):
		"""
		The Master Function: Fuses Traffic + Forensics + Knowledge Context
		"""
		print(f"Fetching data for '{article}'...")

		# 1. Fetch Organic Human Traffic (The Signal)
		df_organic = self.fetch_pageviews_daily(article, start_date, end_date, agent_type='user')

		# 2. Fetch Bot/Spider Traffic (For Forensic Comparison)
		df_bot = self.fetch_pageviews_daily(article, start_date, end_date, agent_type='spider')

		# 3. Fetch Knowledge Context (Edits)
		df_edits = self.fetch_revision_history(article, start_date, end_date)

		# 4. Fuse Datasets
		# We use an outer join to ensure we keep days with Views but 0 Edits (common)
		full_df = df_organic.join(df_bot, how='outer').join(df_edits, how='outer')

		# Fill NaN (0 edits/views on a day is a valid data point)
		full_df.fillna(0, inplace=True)

		return full_df
	
	def fetch_cluster_data(self, main_article, start_date, end_date):
		"""
		Fetches the 'Cluster' (Main + 5 Related) to detect localized Bot Attacks.
		"""
		cluster_data = {}
		
		# 1. Fetch Main
		print(f"Fetching Main: {main_article}")
		cluster_data[main_article] = self.get_research_dataset(main_article, start_date, end_date)
		
		# 2. Fetch Related
		related = self.get_related_pages(main_article)
		print(f"Found Context: {related}")
		
		for rel_title in related:
			# We add a small delay or check to be polite to the API
			print(f"  Fetching Related: {rel_title}")
			df_rel = self.get_research_dataset(rel_title, start_date, end_date)
			if not df_rel.empty:
				cluster_data[rel_title] = df_rel
				
		return cluster_data

# --- Usage Example ---
if __name__ == "__main__":
	fetcher = WikiResearchFetcher()

	start = datetime(2015, 1, 1)
	end = datetime(2025, 12, 30) # Using your example date range

	# Get the complete forensic dataset
	cluster = fetcher.fetch_cluster_data("Influenza", start, end)

	for name, df in cluster.items():
		print(f"\nDataset for '{name}':")
		print(df.head())
		print(f"\nDataset Shape: {df.shape}")
	
		# Save for Phase 2 (Analysis)
		df.to_csv(f"data/{name}_research_data.csv")