# AI Powered Smart Email Classifier

This project classifies incoming emails by:

1. Category (for example: spam, updates, promotions, social media, verify code)
2. Urgency (high, medium, low)

It is designed for live Gmail ingestion and dashboard monitoring.

## Project Overview

The solution has three parts:

1. FastAPI backend API in `integration_layer_api.py`
2. Streamlit dashboard in `streamlit_app.py`
3. Google Apps Script trigger that reads Gmail and sends emails to the API

Data flow:

1. Gmail trigger runs on schedule
2. Script sends unread email content to `POST /ingest`
3. API predicts category and urgency
4. API stores records and exposes them through `GET /predictions`
5. Dashboard reads from `GET /predictions` and displays analytics

## API Endpoints

1. `GET /health`: service and model status
2. `GET /predictions`: latest classified records for dashboard
3. `POST /ingest`: classify one email payload

## Local Run

From project root:

```powershell
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python integration_layer_api.py
```

In another terminal:

```powershell
.\.venv\Scripts\Activate.ps1
streamlit run streamlit_app.py
```

Local URLs:

1. API docs: `http://127.0.0.1:8000/docs`
2. API health: `http://127.0.0.1:8000/health`
3. Dashboard: `http://localhost:8501`

## Deployment Setup

### API (Render)

Runtime and startup are configured for Python 3.11 and Uvicorn.

Important for data persistence:

1. Add a persistent disk in Render
2. Mount it at `/var/data`
3. Set environment variable `STORE_DIR=/var/data`

Without a persistent disk, counts can reset after restarts or redeploys.

### Dashboard (Streamlit Cloud)

The dashboard is deployed separately and reads data from the API endpoint.

## Google Apps Script Trigger

Recommended pattern:

1. Use one time-driven trigger every 5 minutes
2. Do not keep running the script manually
3. Label processed or failed messages to avoid retry loops

After code updates in Apps Script:

1. Save script
2. Run `setupTrigger()` once
3. Confirm trigger is active in the Triggers panel

## Troubleshooting

If dashboard count is not increasing:

1. Check API health endpoint
2. Check `record_count` and `store_path` in `/health`
3. Confirm Apps Script trigger is running in Executions
4. Verify failures are not repeating on the same emails

If Apps Script shows `HTTP 403` with HTML response:

1. It is usually content blocking, not JSON parsing
2. Keep sanitized/truncated fallback logic in script
3. Mark failed threads with labels to prevent infinite retries

If count suddenly resets:

1. Confirm API is using persistent storage path (`/var/data`)
2. Confirm persistent disk is attached and mounted in Render

## Repository Structure

1. `integration_layer_api.py`: FastAPI service and model inference
2. `streamlit_app.py`: dashboard and analytics
3. `module2_trained_models/`: category model artifacts
4. `module3_trained_models/`: urgency model and vectorizer artifacts
5. `requirements.txt`: runtime dependencies
6. `render.yaml`, `Procfile`, `runtime.txt`: deployment/runtime configuration

## Notes

1. There is no unlimited free quota in Google Apps Script.
2. Free usage always has daily limits.
3. Reduce retries and batch size if execution errors increase.
