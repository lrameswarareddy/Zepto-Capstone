# Zepto Capstone Project

Masai AI/ML course submission with three independent modules: a web data pipeline, Titanic analytics, and a Zepto policy support assistant.

## Repository Layout

- `data_pipeline/`: scrapes Books to Scrape, cleans the data, converts GBP to INR, and writes `books.db`.
- `analytics/`: performs Titanic EDA, missing-value handling, visualizations, model comparison, tuning, and model persistence.
- `support_assistant/`: serves a FastAPI policy assistant backed by the eight documents in `support_assistant/docs/`.

## Setup

Create and activate a virtual environment, then install the module requirements:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r data_pipeline\requirements.txt -r analytics\requirements.txt -r support_assistant\requirements.txt
```

## Run the Modules

Run the scraper and SQLite query demonstrations from the repository root:

```powershell
python data_pipeline\build_pipeline.py
```

Run the Titanic analysis and model workflow:

```powershell
python analytics\analytics_pipeline.py
```

This creates `analytics/titanic.csv`, plots, `best_pipeline.joblib`, and `decision_tree.png`.

Start the support assistant in its default mock mode:

```powershell
cd support_assistant
$env:MOCK_LLM = "1"
python -m uvicorn main:app --reload
```

The API is available at `http://127.0.0.1:8000`. Use `GET /health` and `POST /ask`.

## Docker

```powershell
docker build -t zepto-support-assistant .\support_assistant
docker run --rm -p 8000:8000 zepto-support-assistant
```

The assistant currently uses deterministic mock responses when `MOCK_LLM=1`; this keeps local and container runs independent of an external LLM credential.
