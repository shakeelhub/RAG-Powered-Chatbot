<div align="center">

```
██████╗  █████╗  ██████╗    ██████╗  ██████╗ ████████╗
██╔══██╗██╔══██╗██╔════╝    ██╔══██╗██╔═══██╗╚══██╔══╝
██████╔╝███████║██║  ███╗   ██████╔╝██║   ██║   ██║   
██╔══██╗██╔══██║██║   ██║   ██╔══██╗██║   ██║   ██║   
██║  ██║██║  ██║╚██████╔╝   ██████╔╝╚██████╔╝   ██║   
╚═╝  ╚═╝╚═╝  ╚═╝ ╚═════╝    ╚═════╝  ╚═════╝   ╚═╝   
```

### 🧠 Stop ctrl+F-ing your documents. Start asking questions.

<p>
  <img src="https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white"/>
  <img src="https://img.shields.io/badge/FastAPI-009688?style=for-the-badge&logo=fastapi&logoColor=white"/>
  <img src="https://img.shields.io/badge/Pinecone-000000?style=for-the-badge&logo=pinecone&logoColor=white"/>
  <img src="https://img.shields.io/badge/OpenAI-412991?style=for-the-badge&logo=openai&logoColor=white"/>
</p>
<p>
  <img src="https://img.shields.io/badge/Ollama-qwen2.5:3b-FF6B35?style=for-the-badge"/>
  <img src="https://img.shields.io/badge/Google_Drive-Auto_Sync-4285F4?style=for-the-badge&logo=googledrive&logoColor=white"/>
  <img src="https://img.shields.io/badge/RAG-Powered-8B5CF6?style=for-the-badge"/>
  <img src="https://img.shields.io/badge/License-MIT-yellow?style=for-the-badge"/>
</p>

</div>

---

<div align="center">

## 〔 WHAT IS THIS? 〕

</div>

An **ERP Document AI** that turns your company's pile of PDFs, reports, and invoices into a **conversational knowledge base**.

Upload a document. Ask anything about it.  
It reads, chunks, embeds, indexes, and retrieves — so you don't have to.

Drop a folder in **Google Drive** → it auto-syncs, auto-indexes, and makes your new doc queryable in seconds.  
No manual uploads. No re-indexing. It just works.

---

## ⚡ How It Works

```
┌─────────────────────────────────────────────────────────────────┐
│                                                                 │
│   You drop a PDF         →   /upload hits Pinecone             │
│   You ask a question     →   embeddings find the right chunk   │
│   Ollama reads context   →   you get a smart answer            │
│   Google Drive updates   →   webhook auto-indexes the file     │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

```
YOUR DOCUMENT
      │
      ▼
  CHUNKING          ← split into digestible pieces
      │
      ▼
  EMBEDDING         ← text-embedding-3-small (1536 dims)
      │
      ▼
  PINECONE          ← stored as vectors (cosine similarity)
      │
      ▼
  USER ASKS         ← "What's the profit margin for Q3?"
      │
      ▼
  TOP-K RETRIEVAL   ← most relevant chunks pulled instantly
      │
      ▼
  QWEN 2.5 (LOCAL)  ← reads chunks + chat history → answers
      │
      ▼
  CLEAN RESPONSE    ← grounded, accurate, document-aware ✅
```

---

## 🗂️ Project Structure

```
rag_project/
│
├── 🎯 app.py                 ← UI version for testing        (port 5000)
├── ⚙️  app_api.py             ← Clean API + Drive webhook     (port 5001)
├── 🔔 drive_webhook.py       ← Google Drive push handler
├── 🔧 setup_drive_watch.py   ← Run ONCE to register webhook
├── 🔑 service_account.json   ← Replace with your GCP service account
├── 📦 requirements.txt       ← pip install -r this
├── 📁 uploads/               ← Uploaded files live here
└── 🌐 static/
    └── index.html            ← Test UI (app.py only)
```

---

## 🚀 Quick Start

### 1️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

### 2️⃣ Add Your API Keys

Open `app.py` and `app_api.py` and replace:

```python
OPENAI_API_KEY = "sk-..."   # your OpenAI key for embeddings
```

### 3️⃣ Start Ollama

```bash
ollama serve
ollama pull qwen2.5:3b
```

### 4️⃣ Run UI Version (for testing)

```bash
python app.py
# → http://localhost:5000
```

### 5️⃣ Run API Version (for chatbot integration)

```bash
python app_api.py
# → http://localhost:5001
```

---

## 📡 API Reference

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/upload` | Upload + index a document |
| `POST` | `/query` | Ask a question across all docs |
| `GET` | `/files` | List all indexed files |
| `POST` | `/drive-webhook` | Google Drive push notification |
| `GET` | `/drive-webhook/status` | Check webhook status |

### `/query` — Request Body

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

## 🔄 Google Drive Auto-Sync Setup

### Step 1 — Google Cloud Console

```
1. console.cloud.google.com
2. Create project → Enable "Google Drive API"
3. Create Service Account → Download JSON → save as service_account.json
4. Share your Drive folder with the service account email
```

### Step 2 — Fill in Placeholders

```python
# in drive_webhook.py and setup_drive_watch.py:

FOLDER_ID      = "your_folder_id_from_drive_url"
WEBHOOK_URL    = "https://your-server.com/drive-webhook"
WEBHOOK_SECRET = "any_random_secret_string"
```

### Step 3 — Register Webhook (run once)

```bash
python setup_drive_watch.py setup
```

### Step 4 — Auto-Renew Every 6 Days

```bash
# Manual renew
python setup_drive_watch.py renew

# Or set it and forget it via cron:
0 9 */6 * * cd /your/project && python setup_drive_watch.py renew
```

> ⚠️ Google Drive webhooks expire after **7 days max**. Renew before then or your sync breaks silently.

---

## 🗄️ Pinecone Index Settings

```
Index Name   →   rag-bot
Dimensions   →   1536   (text-embedding-3-small)
Metric       →   cosine
```

---

## 📝 Notes

```
🔧  Local webhook testing?   →  ngrok http 5001
🔄  Changed chunking config? →  Re-index all files
📁  Shared folder?           →  uploads/ is shared between app.py + app_api.py
```

---

## 🤝 Contributing

```bash
git checkout -b feat/your-feature
git commit -m "feat: something smart"
git push origin feat/your-feature
# open a PR 🚀
```

---

<div align="center">

```
"Your documents have all the answers.
 You just needed to ask the right way."
```

**⭐ Star this if it saved you from reading a 200-page PDF manually.**

</div>
