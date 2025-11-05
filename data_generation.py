import wikipediaapi,re
import requests, os, json
from datetime import datetime, timedelta
from pathlib import Path

class Config():
	def __init__(self, mode, data_dir):
		self.BASE_LINK = "https://wikimedia.org/api/rest_v1/metrics/pageviews/per-article/en.wikipedia/all-access/all-agents/{topic_name}/daily/{start_time}/{end_time}"
		# Yesterday's timestamp
		if mode=="training":
			self.start_date = "2015070100"
			self.end_date = (datetime.today() - timedelta(days=1)).strftime("%Y%m%d00")
		else:
			self.start_date = (datetime.today() - timedelta(days=7)).strftime("%Y%m%d00")
			self.end_date = (datetime.today() - timedelta(days=1)).strftime("%Y%m%d00")
		self.BASE_LINK = self.BASE_LINK
		os.makedirs(data_dir, exist_ok=True)
		self.headers = {
			"User-Agent": "BTechEventDetector/1.0 (B.Tech Final Year Project; contact@example.com)"
		}
		self.data_dir = Path(data_dir)

class FetchData:
	def __init__(self, mode="training", data_dir="./data", category="Influenza"):  # Happiness for a team member
		self.category = category
		self.wiki = wikipediaapi.Wikipedia(
			user_agent="BTechEventDetector/1.0 (B.Tech Final Year Project; contact@example.com)",
			language="en"
		)
		self.config = Config(mode=mode, data_dir=data_dir)
		self.DATA_DIR = Path(data_dir)
	
	def _call_api(self, topic):
		print("Getting data for", topic)
		page_link = self.config.BASE_LINK.format(topic_name=topic, start_time=self.config.start_date, end_time=self.config.end_date)
		file_name = "_".join(topic.split(" "))
		response = requests.get(page_link, headers=self.config.headers)
		response.raise_for_status()
		print(f"Pageviews for {topic} on {self.config.end_date}:", response.json().get("items", [{}])[-1].get("views", 0))
		with open(self.config.data_dir / f"{file_name}_pageviews_{self.config.end_date}.json", "w") as f:
			json.dump(response.json(), f, indent=2)


	def get_pageviews(self, topics=[]):
		for topic in topics:
			self._call_api(topic=topic)
		print("Fetching completed.")

	def get_topics_from_cat(self, limit=50, max_depth=1, min_length=2000, min_backlinks=10):
			visited_categories = set()
			topics_data = []

			def recurse_category(cat_page, current_depth):
				if current_depth > max_depth or cat_page.title in visited_categories:
					return

				visited_categories.add(cat_page.title)

				for member in cat_page.categorymembers.values():
					# Only process main articles and subcategories
					if member.ns == wikipediaapi.Namespace.MAIN:
						page = self.wiki.page(member.title)
						content = page.text
						length = len(content)
						backlinks_count = len(page.backlinks)
						has_infobox = bool(re.search(r"\{\{Infobox", content, re.IGNORECASE))

						# Apply relevance filters
						if length < min_length or backlinks_count < min_backlinks:
							continue

						relevance_score = length + backlinks_count * 100
						if has_infobox:
							relevance_score *= 1.2

						topics_data.append((member.title, relevance_score))
						print(f"{member.title} | len={length} | backlinks={backlinks_count} | infobox={has_infobox}")

					elif member.ns == wikipediaapi.Namespace.CATEGORY:
						# Recursively explore subcategories
						recurse_category(member, current_depth + 1)

			start_category = self.wiki.page(f"Category:{self.category}")
			if not start_category.exists():
				print(f"Category '{self.category}' not found.")
				return []

			print(f"Scanning category '{self.category}' (max_depth={max_depth})...")
			recurse_category(start_category, 0)

			if not topics_data:
				print(f"No relevant topics found under '{self.category}'.")
				return []

			# Sort by relevance and limit results
			topics_data.sort(key=lambda x: x[1], reverse=True)
			final_topics = [title for title, _ in topics_data]

			print(f"\nFetched {len(final_topics)} relevant topics from '{self.category}'.")
			return final_topics[:limit]

if __name__ == "__main__":
	fetcher = FetchData(category="Influenza")
	topics = fetcher.get_topics_from_cat(limit=20)
	fetcher.get_pageviews(topics=topics)