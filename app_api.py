# ═══════════════════════════════════════════════════════════════════
# app_api.py  —  Pure REST API + Google Drive Webhook
# ═══════════════════════════════════════════════════════════════════

from flask import Flask, request, jsonify
from flask_cors import CORS
from drive_webhook import drive_bp          # ✅ Google Drive webhook
import os, re, requests

# ─────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────
PINECONE_API_KEY  = YOUR_PINECONE_API_KEY
OPENAI_API_KEY    = YOUR_OPENAI_API_KEY
PINECONE_INDEX    = "rag-bot"
OLLAMA_URL        = "http://localhost:11434/api/generate"
LLM_MODEL         = "qwen2.5:3b"
EMBED_MODEL       = "text-embedding-3-small"
EMBED_DIMENSION   = 1536
CHUNK_SIZE        = 500
CHUNK_OVERLAP     = 1
TOP_K_RETRIEVE    = 20
TOP_K_FINAL       = 5
HISTORY_TURNS     = 3
UPLOAD_FOLDER     = "uploads"
OPENAI_BATCH_SIZE = 100

app = Flask(__name__)
CORS(app)
app.register_blueprint(drive_bp)           # ✅ Plugs in /drive-webhook route
os.makedirs(UPLOAD_FOLDER, exist_ok=True)


# ─────────────────────────────────────────────────────────────
# RERANKER
# ─────────────────────────────────────────────────────────────
reranker = None

def get_reranker():
    global reranker
    if reranker is None:
        try:
            from sentence_transformers import CrossEncoder
            reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
            print("✅ Reranker loaded.")
        except Exception as e:
            print(f"⚠️  Reranker unavailable ({e}).")
            reranker = "unavailable"
    return reranker


# ─────────────────────────────────────────────────────────────
# PINECONE HOST CACHE
# ─────────────────────────────────────────────────────────────
_pinecone_host_cache = None

def get_pinecone_headers():
    return {"Api-Key": PINECONE_API_KEY, "Content-Type": "application/json"}

def get_pinecone_host():
    global _pinecone_host_cache
    if _pinecone_host_cache:
        return _pinecone_host_cache
    url      = f"https://api.pinecone.io/indexes/{PINECONE_INDEX}"
    response = requests.get(url, headers={"Api-Key": PINECONE_API_KEY})
    if not response.ok:
        raise Exception(f"Pinecone index '{PINECONE_INDEX}' not found.\n{response.text}")
    _pinecone_host_cache = "https://" + response.json()["host"]
    return _pinecone_host_cache


# ─────────────────────────────────────────────────────────────
# FILE EXTRACTION
# ─────────────────────────────────────────────────────────────

def extract_file_content(file_path):
    ext = os.path.splitext(file_path)[1].lower()
    if ext == ".pdf":
        import fitz
        doc  = fitz.open(file_path)
        text = "".join([page.get_text() for page in doc])
        doc.close()
        return text.strip()
    elif ext in [".xlsx", ".xls"]:
        import pandas as pd
        return pd.read_excel(file_path).to_string(index=False)
    elif ext == ".csv":
        import pandas as pd
        return pd.read_csv(file_path).to_string(index=False)
    elif ext == ".docx":
        from docx import Document
        doc = Document(file_path)
        return "\n".join([p.text for p in doc.paragraphs if p.text.strip()])
    elif ext == ".doc":
        raise ValueError("Old .doc format not supported. Please resave as .docx.")
    return ""


# ─────────────────────────────────────────────────────────────
# SENTENCE-AWARE CHUNKING
# ─────────────────────────────────────────────────────────────

def chunk_text(text):
    sentences = re.split(r'(?<=[.!?])\s+|\n+', text)
    sentences = [s.strip() for s in sentences if s.strip()]
    chunks, current_sentences, current_length = [], [], 0
    for sentence in sentences:
        slen = len(sentence)
        if current_length + slen > CHUNK_SIZE and current_sentences:
            chunks.append(" ".join(current_sentences))
            current_sentences = current_sentences[-CHUNK_OVERLAP:] if CHUNK_OVERLAP else []
            current_length    = sum(len(s) for s in current_sentences)
        current_sentences.append(sentence)
        current_length += slen
    if current_sentences:
        chunks.append(" ".join(current_sentences))
    return chunks


# ─────────────────────────────────────────────────────────────
# EMBEDDINGS
# ─────────────────────────────────────────────────────────────

def get_embeddings(texts, input_type="passage"):
    all_embeddings = []
    for i in range(0, len(texts), OPENAI_BATCH_SIZE):
        batch    = texts[i:i + OPENAI_BATCH_SIZE]
        response = requests.post(
            "https://api.openai.com/v1/embeddings",
            headers={"Authorization": f"Bearer {OPENAI_API_KEY}", "Content-Type": "application/json"},
            json={"model": EMBED_MODEL, "input": batch}
        )
        if not response.ok:
            raise Exception(f"OpenAI Embedding failed: {response.text}")
        data = response.json()["data"]
        data.sort(key=lambda x: x["index"])
        all_embeddings.extend([item["embedding"] for item in data])
    return all_embeddings


# ─────────────────────────────────────────────────────────────
# INDEXING
# ─────────────────────────────────────────────────────────────

def is_file_indexed(file_name, host):
    try:
        r = requests.post(
            f"{host}/query", headers=get_pinecone_headers(),
            json={"vector": [0.0]*EMBED_DIMENSION, "topK": 1,
                  "filter": {"file_name": {"$eq": file_name}}, "includeMetadata": True}
        )
        return len(r.json().get("matches", [])) > 0
    except Exception:
        return False

def index_file(file_name, file_path, host):
    if is_file_indexed(file_name, host):
        return True
    content = extract_file_content(file_path)
    if not content or not content.strip():
        raise ValueError("Could not extract text. File may be empty or scanned.")
    chunks     = chunk_text(content)
    embeddings = get_embeddings(chunks)
    vectors    = [
        {"id": f"{file_name}_chunk_{i}", "values": emb,
         "metadata": {"file_name": file_name, "chunk_index": i, "text": chunk}}
        for i, (chunk, emb) in enumerate(zip(chunks, embeddings))
    ]
    for i in range(0, len(vectors), 100):
        resp = requests.post(f"{host}/vectors/upsert", headers=get_pinecone_headers(),
                             json={"vectors": vectors[i:i+100]})
        if not resp.ok:
            raise Exception(f"Pinecone upsert failed: {resp.text}")
    return True


# ─────────────────────────────────────────────────────────────
# RETRIEVE + RERANK
# ─────────────────────────────────────────────────────────────

def retrieve_chunks_global(question, host):
    q_emb    = get_embeddings([question], input_type="query")[0]
    response = requests.post(
        f"{host}/query", headers=get_pinecone_headers(),
        json={"vector": q_emb, "topK": TOP_K_RETRIEVE, "includeMetadata": True}
    )
    if not response.ok:
        raise Exception(f"Pinecone query failed: {response.text}")
    return [
        {"text": m["metadata"]["text"],
         "file_name": m["metadata"].get("file_name", "Unknown"),
         "score": m.get("score", 0)}
        for m in response.json().get("matches", []) if "metadata" in m
    ]

def rerank_chunks(question, candidates):
    if not candidates:
        return []
    model = get_reranker()
    if model == "unavailable":
        return candidates[:TOP_K_FINAL]
    scores = model.predict([[question, c["text"]] for c in candidates])
    ranked = sorted(zip(scores, candidates), key=lambda x: x[0], reverse=True)
    return [c for _, c in ranked[:TOP_K_FINAL]]


# ─────────────────────────────────────────────────────────────
# CONVERSATION MEMORY
# ─────────────────────────────────────────────────────────────

def build_search_query(question, history):
    if not history:
        return question
    recent = " ".join(t.get("content", "") for t in history[-2:])
    return (recent + " " + question).strip()

def generate_answer(chunks, question, history=None):
    if history is None:
        history = []
    context = "\n\n---\n\n".join(
        f"[Source: {c['file_name']}]\n{c['text']}" for c in chunks
    )
    history_text = ""
    if history:
        history_text = "\nCONVERSATION HISTORY:\n"
        for turn in history[-(HISTORY_TURNS * 2):]:
            role = "User" if turn["role"] == "user" else "Assistant"
            history_text += f"{role}: {turn['content']}\n"
        history_text += "\n"

    prompt = f"""You are an AI Assistant helping a business user analyze documents.

STRICT RULES:
- Answer ONLY from the DOCUMENT CONTEXT provided below
- Use CONVERSATION HISTORY only to resolve references like "it", "that", "this"
- Do NOT mention chunks, vectors, embeddings, or any technical terms
- Answer clearly and professionally
- Mention which document the info came from if multiple documents are involved
- If the answer is not in the context say: "This information is not available in the uploaded documents"
{history_text}
DOCUMENT CONTEXT:
{context}

CURRENT QUESTION: {question}
"""
    try:
        response = requests.post(
            OLLAMA_URL,
            json={"model": LLM_MODEL, "prompt": prompt, "stream": False},
            timeout=120
        )
        response.raise_for_status()
        return response.json().get("response", "").strip()
    except requests.exceptions.ConnectionError:
        raise Exception("Cannot connect to Ollama. Run 'ollama serve' and try again.")
    except requests.exceptions.Timeout:
        raise Exception("Ollama timed out. Try a simpler question or restart Ollama.")


# ═══════════════════════════════════════════════════════════════════
# API ENDPOINTS
# ═══════════════════════════════════════════════════════════════════

@app.route("/upload", methods=["POST"])
def upload_file():
    if "file" not in request.files:
        return jsonify({"error": "No file provided"}), 400
    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No file selected"}), 400
    allowed = {".pdf", ".xlsx", ".xls", ".csv", ".docx", ".doc"}
    ext     = os.path.splitext(file.filename)[1].lower()
    if ext not in allowed:
        return jsonify({"error": f"Unsupported file type: {ext}"}), 400
    file_path = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(file_path)
    try:
        host = get_pinecone_host()
        index_file(file.filename, file_path, host)
        return jsonify({"success": True, "file_name": file.filename,
                        "message": f"'{file.filename}' uploaded and indexed successfully!"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/query", methods=["POST"])
def query():
    data    = request.json or {}
    message = data.get("message", "").strip()
    history = data.get("history", [])
    if not message:
        return jsonify({"error": "No message provided"}), 400
    try:
        host         = get_pinecone_host()
        search_query = build_search_query(message, history)
        candidates   = retrieve_chunks_global(search_query, host)
        chunks       = rerank_chunks(message, candidates)
        if not chunks:
            return jsonify({"answer": "No relevant information found in the uploaded documents."})
        answer = generate_answer(chunks, message, history)
        return jsonify({"answer": answer})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/files", methods=["GET"])
def list_files():
    allowed = {".pdf", ".xlsx", ".xls", ".csv", ".docx", ".doc"}
    files   = [f for f in os.listdir(UPLOAD_FOLDER)
               if os.path.splitext(f)[1].lower() in allowed]
    return jsonify({"files": sorted(files)})


# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    get_reranker()
    app.run(debug=False, port=5001)
