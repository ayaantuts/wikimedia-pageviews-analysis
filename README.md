# Event Detection System for Specific Domains Using Wikipedia Pageview Anomalies

> An automated machine learning system that detects significant real-world events by monitoring anomalous patterns in Wikipedia pageview traffic across domain-specific article collections.

---

## Project Overview

This system leverages the power of public information-seeking behavior to identify emerging events before they reach mainstream media coverage. By analyzing Wikipedia pageview data—a signal amplified in the AI era through LLM citations—the system provides early warnings for significant domain-specific events.

### Key Features

- **Early Event Detection**: Identify emerging events 1-3 days before mainstream media coverage
- **Domain-Agnostic**: Easily adaptable to any field (health, finance, environment, technology)
- **Machine Learning Powered**: Uses advanced ML models (Random Forest, XGBoost) for robust predictions
- **Real-Time Monitoring**: Automated daily analysis and alerting pipeline
- **Data-Driven**: Leverages 10+ years of historical Wikipedia pageview data

---

## Use Cases
### Public Health
Monitor disease outbreaks and health concerns through anomalous interest in medical topics.

**Example:** Early detection of influenza outbreaks by tracking pageviews for flu symptoms, vaccines, and pandemic-related articles.

### Financial Markets
Detect market sentiment shifts and emerging financial crises.

**Example:** Identify economic instability through spikes in articles about recession, unemployment, and monetary policy.

### Environmental Monitoring
Track natural disasters and climate events.

**Example:** Early warning of natural disasters through heightened interest in earthquake, hurricane, or flood-related topics.

### Technology & Cybersecurity
Identify emerging threats and technological disruptions.

**Example:** Detect cybersecurity incidents through anomalous traffic to vulnerability, malware, and security protocol articles.

---

## Methodology
### Three-Phase Approach

1. **Data Acquisition & Corpus Curation**
	- Select domain (e.g., "Influenza")
	- Curate ~50 relevant Wikipedia articles
	- Extract 10+ years of historical pageview data via Wikimedia API

2. **Modeling & Anomaly Detection**
	- Generate pseudo-labels using statistical methods
	- Engineer temporal and article-specific features
	- Train classification models (Random Forest, XGBoost)
	- Optimize for precision and recall

3. **Real-Time Inference & Alerting**
	- Daily ingestion of pageview data
	- Article-level anomaly predictions
	- Domain-level aggregation
	- Alert when >20% of articles are abnormal