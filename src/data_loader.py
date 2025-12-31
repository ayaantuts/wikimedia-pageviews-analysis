import requests
import pandas as pd
from datetime import datetime, timedelta
import time

class WikiResearchFetcher:
	"""
	A research-grade data fetcher for Wikimedia projects.
	Designed to support Forensic Traffic Analysis and Knowledge Graph construction.
	"""

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
				'revid': 'count',		   # Total Edits
				'user': 'nunique',		  # Unique Editors (Crowd signal)
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

# --- Usage Example ---
if __name__ == "__main__":
	fetcher = WikiResearchFetcher()

	start = datetime(2025, 1, 1)
	end = datetime(2025, 10, 3) # Using your example date range

	# Get the complete forensic dataset
	df = fetcher.get_research_dataset("Influenza", start, end)

	print(df.head())
	print(f"\nDataset Shape: {df.shape}")

	# Save for Phase 2 (Analysis)
	df.to_csv("influenza_research_data.csv")