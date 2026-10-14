# Early Job Detection ML

Detect emerging job postings early using Spark + machine learning.

## Problem

Job boards move fast. The most interesting roles — a new team hiring, a skill
suddenly in demand, a posting that will close quickly — are easy to miss if
you only check once a day. This project builds a pipeline that ingests job
postings as they appear, cleans and featurizes the text, and scores each
posting for two things:

1. **Early detection:** is this a genuinely new / emerging role vs. a repost?
2. **Urgency:** is this posting likely to be filled quickly?

It is meant as a portfolio demo, not a production scraper. The scraper uses
a placeholder API and falls back to sample data, and it respects robots.txt
and rate limits.

## Architecture

```
Job Board API / RSS  ->  scraper.py  ->  raw JSON (data/)
        |
        v
preprocess.py (Spark: clean text, tokenize, TF-IDF)
        |
        v
train.py (Spark ML: LogisticRegression + RandomForest)
        |
        v
streaming_detect.py (Structured Streaming: score new postings live)
        |
        v
Scored postings -> console / Delta / Parquet
```

## Tech Stack

- **Python 3.10+**
- **PySpark 3.5** — Structured Streaming + MLlib
- **scikit-learn** — quick offline baselines and metric checks
- **NLTK** — stopwords / token helpers (optional, Spark does most of the work)
- **Pandas / NumPy** — small local exploration

## Project Structure

```
early-job-detection-ml/
├── README.md
├── requirements.txt
├── .gitignore
├── data/
│   └── sample_jobs.json       # 6 example postings for local runs
├── notebooks/
│   └── job_trends.ipynb       # EDA: trends, skills, posting velocity
└── src/
    ├── config.py              # Paths, Spark settings, model params
    ├── scraper.py             # Fetch postings (placeholder API + fallback)
    ├── preprocess.py          # Spark text cleaning + TF-IDF
    ├── train.py               # Train + evaluate classifiers
    └── streaming_detect.py    # Real-time scoring job
```

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Spark needs Java 11 or 17
java -version
```

If you do not have Spark installed locally, `pip install pyspark` is enough
for the demo — it ships a local Spark distribution.

## How to Run

### 1. Fetch postings (uses sample data if no API is configured)

```bash
python src/scraper.py --out data/fetched_jobs.json --limit 50
```

### 2. Preprocess + featurize

```bash
python src/preprocess.py --input data/sample_jobs.json --output data/features
```

### 3. Train the classifiers

```bash
python src/train.py --input data/sample_jobs.json --model-output models/
```

This prints accuracy, F1, and AUC for both models, and saves the best
model to `models/`.

### 4. Stream new postings

```bash
python src/streaming_detect.py --input data/stream_in --model models/best_model
```

Drop JSON files into `data/stream_in/` and watch scored output in the
console. In production you would point this at Kafka instead of files.

## Example Output

```
+--------------------------------+----------------+----------+----------+
| title                          | is_emerging    | urgency  | score    |
+--------------------------------+----------------+----------+----------+
| Senior Data Engineer, Streaming| true           | high     | 0.91     |
| Data Analyst, Reporting        | false          | low      | 0.23     |
| ML Platform Engineer, GenAI    | true           | high     | 0.87     |
+--------------------------------+----------------+----------+----------+
```

## Models

Two classifiers are trained on the same TF-IDF features:

- **Logistic Regression** — strong baseline, easy to interpret (which words
  push a posting toward "emerging").
- **Random Forest** — captures non-linear interactions (e.g. title + skills
  + location together).

The training script compares them on a holdout split and keeps the better
one by F1. For this small demo dataset, expect optimistic metrics — the
point is the pipeline, not beating a benchmark.

### Features

- TF-IDF over title + description (top 5,000 terms)
- Posting length, number of listed skills
- Days since first seen (for repost detection)
- One-hot / indexed: job category, remote flag, seniority

### Labels (demo)

- `is_emerging`: 1 if the role mentions a skill that appeared in the last
  30 days in the sample, or is the first posting for that title + company.
- `is_urgent`: 1 if the posting contains urgency signals ("immediate",
  "ASAP", "urgent", short application window) in the demo labels.

In a real deployment you would label from outcomes (time-to-fill, number
of applicants) rather than keywords.

## Notebook

`notebooks/job_trends.ipynb` shows:

- Postings per week by category
- Fastest-growing skills
- Repost rate and time-to-fill proxies

## Ethics Note on Job Data

- Only use data you have permission to collect. Respect each board's
  Terms of Service and `robots.txt`.
- Do not scrape personal data from applicant profiles.
- Do not use the model to filter people — it scores postings, not candidates.
- Sample data in this repo is synthetic and contains no real personal info.

## Limitations

- The demo dataset is tiny (6 rows) — it exists so the pipeline runs
  end-to-end without an API key.
- The "emerging" label is a heuristic. Real labels need historical outcomes.
- Streaming demo uses the file source. Swap it for Kafka / Kinesis for
  a production setup (see the companion repo
  `realtime-kafka-msk-spark-pipeline` for a Kafka example).

## Possible Extensions

- Add a Kafka source + Delta Lake sink
- Add skill embeddings (Word2Vec / sentence transformers)
- Track skill velocity: week-over-week growth per skill
- Add a simple Streamlit dashboard for scored postings

## License

MIT
