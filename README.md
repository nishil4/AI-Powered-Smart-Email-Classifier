# AI Smart Email Classifier

Production-ready email classification system with:
- FastAPI ingestion API (`integration_layer_api.py`)
- Streamlit dashboard (`streamlit_app.py`)
- Category + urgency ML models (`module2_trained_models`, `module3_trained_models`)

## 1) Local Validation

```powershell
# from project root
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python integration_layer_api.py
```

In a new terminal:
```powershell
.\.venv\Scripts\Activate.ps1
streamlit run streamlit_app.py
```

Check:
- API docs: `http://127.0.0.1:8000/docs`
- API health: `http://127.0.0.1:8000/health`
- Dashboard: `http://localhost:8501`

## 2) Deploy API (recommended: Azure App Service)

```powershell
az login
az group create --name email-classifier-rg --location eastus
az appservice plan create --name email-classifier-plan --resource-group email-classifier-rg --sku B1 --is-linux
az webapp create --resource-group email-classifier-rg --plan email-classifier-plan --name <your-unique-app-name> --runtime "PYTHON:3.11"
```

Create a startup command in App Service (Configuration > General settings):
```bash
gunicorn -w 2 -k uvicorn.workers.UvicornWorker -b 0.0.0.0:8000 integration_layer_api:app
```

Deploy code (zip deploy):
```powershell
Compress-Archive -Path * -DestinationPath app.zip -Force
az webapp deployment source config-zip --resource-group email-classifier-rg --name <your-unique-app-name> --src app.zip
```

Your production API URL becomes:
`https://<your-unique-app-name>.azurewebsites.net`

Test:
- `https://<your-unique-app-name>.azurewebsites.net/health`
- `https://<your-unique-app-name>.azurewebsites.net/docs`

## 3) Update Gmail Apps Script

In Apps Script, set:
```javascript
const API_ENDPOINT = "https://<your-unique-app-name>.azurewebsites.net/ingest";
```

Then:
1. Save script
2. Run `setupTriggerEvery5Min()` once
3. Authorize permissions
4. Run `ingestUnreadEmails()` once for smoke test
5. Confirm logs show successful 2xx responses

## 4) Verify End-to-End Flow

1. Send a test unread email to Gmail inbox.
2. Wait for trigger (or run `ingestUnreadEmails()` manually).
3. Confirm API receives records.
4. Open dashboard and verify new predictions appear.
5. Use download button in dashboard to export classified results.

## 5) Notes

- Cloudflare/ngrok is only for local testing; production should call your cloud URL directly.
- Keep model files in deployment package:
  - `module2_trained_models/best_email_classifier.pkl`
  - `module3_trained_models/best_urgency_classifier.pkl`
  - `module3_trained_models/tfidf_vectorizer.pkl`
  - `module3_trained_models/urgency_rule_config.json`
