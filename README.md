# ERP Document AI — RAG System
## Tech Solutions Pvt Ltd

---

## Project Structure
```
rag_project/
├── app.py                ← UI version (test UI, port 5000)
├── app_api.py            ← Clean API version + Drive webhook (port 5001)
├── drive_webhook.py      ← Google Drive push notification handler
├── setup_drive_watch.py  ← Run ONCE to register webhook with Google Drive
├── service_account.json  ← 🔑 Replace with your Google Cloud service account
├── requirements.txt      ← Python dependencies
├── uploads/              ← Uploaded files stored here
└── static/
    └── index.html        ← Test UI (used with app.py only)
```

---

## Quick Start

### Step 1 — Install dependencies
```bash
pip install -r requirements.txt
```

### Step 2 — Add your API keys
Open app.py and app_api.py and replace:
```python
OPENAI_API_KEY = "ysk-proj-wB4Rp6Cko2gDo4eDJF69QdJlG6fCOaM_xukhFAJFFgv8RPuTi-m0JwBepr6epnBmgVohAbJEPgT3BlbkFJVlspcr_IXPGZf3yITEW8sZljjNoxjh8nQ6AFtWOwqR1URDM4fIOiA1RVUnKaWArN0QulsjbHAA"   # sk-...
```

### Step 3 — Start Ollama
```bash
ollama serve
ollama pull qwen2.5:3b
```

### Step 4 — Run the UI version (for testing)
```bash
python app.py
# Open http://localhost:5000
```

### Step 5 — Run the API version (for chatbot integration)
```bash
python app_api.py
# API available at http://localhost:5001
```

---

## API Endpoints (app_api.py)

| Method | Endpoint              | Description                        |
|--------|-----------------------|------------------------------------|
| POST   | /upload               | Upload + index a document          |
| POST   | /query                | Ask a question across all docs     |
| GET    | /files                | List all indexed files             |
| POST   | /drive-webhook        | Google Drive push notification     |
| GET    | /drive-webhook/status | Check webhook status               |

### /query Request Body
```json
{
  "message": "What is the profit margin?",
  "history": [
    { "role": "user",      "content": "How many employees?" },
    { "role": "assistant", "content": "There are 120 employees." }
  ]
}
```

---

## Google Drive Auto-Sync Setup

### Step 1 — Google Cloud Console
1. Go to console.cloud.google.com
2. Create project → Enable "Google Drive API"
3. Create Service Account → Download JSON → save as service_account.json
4. Share your Drive folder with the service account email

### Step 2 — Fill in placeholders in drive_webhook.py and setup_drive_watch.py
```python
FOLDER_ID      = "your_folder_id_from_drive_url"
WEBHOOK_URL    = "https://your-server.com/drive-webhook"
WEBHOOK_SECRET = "any_random_secret_string"
```

### Step 3 — Register the webhook (run once)
```bash
python setup_drive_watch.py setup
```

### Step 4 — Renew every 6 days (Google Drive max is 7 days)
```bash
python setup_drive_watch.py renew
# Or add to crontab: 0 9 */6 * * cd /your/project && python setup_drive_watch.py renew
```

---

## Pinecone Index Settings
- Index Name  : rag-bot
- Dimensions  : 1536  (text-embedding-3-small)
- Metric      : cosine

---

## Notes
- For local testing of webhook, use ngrok: ngrok http 5001
- Re-index files after any chunking config changes
- uploads/ folder is shared between app.py and app_api.py
# RAG-Powered-Chatbot
