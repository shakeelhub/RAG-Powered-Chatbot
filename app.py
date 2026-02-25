from flask import Flask, request, jsonify, send_from_directory
import os
import re
import requests

# ─────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────
PINECONE_API_KEY  = "pcsk_4cgDK_DQsFitGvScBaRfG68qKMKHhpQjJfrjhQ5R6kJuQJZUWWBDKEkhJ4iZAirjyGiez"
OPENAI_API_KEY    = "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"   
PINECONE_INDEX    = "rag-bot"
OLLAMA_URL        = "http://localhost:11434/api/generate"
LLM_MODEL         = "qwen2.5:3b"
EMBED_MODEL       = "text-embedding-3-small"
EMBED_DIMENSION   = 1536
CHUNK_SIZE        = 500        # Max chars per chunk (sentence-aware)
CHUNK_OVERLAP     = 1          # Number of sentences to overlap between chunks
TOP_K_RETRIEVE    = 20         # Retrieve wide net from Pinecone
TOP_K_FINAL       = 5          # Keep only top-5 after reranking
HISTORY_TURNS     = 3          # Number of past Q&A turns to remember
UPLOAD_FOLDER     = "uploads"
OPENAI_BATCH_SIZE = 100

app = Flask(__name__, static_folder="static")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs("static", exist_ok=True)

# ─────────────────────────────────────────────────────────────
# RERANKER — Load once at startup
# ─────────────────────────────────────────────────────────────
reranker = None

def get_reranker():
    """Load cross-encoder reranker model once and cache it."""
    global reranker
    if reranker is None:
        try:
            from sentence_transformers import CrossEncoder
            reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
            print("✅ Reranker loaded: cross-encoder/ms-marco-MiniLM-L-6-v2")
        except Exception as e:
            print(f"⚠️  Reranker not available ({e}). Falling back to vector similarity order.")
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
    url = f"https://api.pinecone.io/indexes/{PINECONE_INDEX}"
    response = requests.get(url, headers={"Api-Key": PINECONE_API_KEY})
    if not response.ok:
        raise Exception(f"Pinecone index '{PINECONE_INDEX}' not found.\n{response.text}")
    _pinecone_host_cache = "https://" + response.json()["host"]
    return _pinecone_host_cache


# ─────────────────────────────────────────────────────────────
# FILE CONTENT EXTRACTION
# ─────────────────────────────────────────────────────────────

def extract_file_content(file_path):
    ext = os.path.splitext(file_path)[1].lower()
    if ext == ".pdf":
        import fitz
        doc = fitz.open(file_path)
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
        raise ValueError("Old .doc format not supported. Please resave as .docx and try again.")
    return ""


# ─────────────────────────────────────────────────────────────
# ✅ IMPROVEMENT #1 — SENTENCE-AWARE CHUNKING
# Instead of blindly slicing every 500 chars (which breaks words,
# sentences, and table rows mid-way), we now split on sentence
# boundaries first, then group complete sentences into chunks.
# This ensures every chunk = a complete, meaningful thought,
# which produces much better embedding vectors.
# ─────────────────────────────────────────────────────────────

def chunk_text(text):
    # Split on sentence-ending punctuation followed by whitespace
    # Also split on newlines (important for tables, bullet points)
    sentences = re.split(r'(?<=[.!?])\s+|\n+', text)
    sentences = [s.strip() for s in sentences if s.strip()]

    chunks = []
    current_sentences = []
    current_length = 0

    for sentence in sentences:
        sentence_len = len(sentence)

        # If adding this sentence exceeds chunk size, save current chunk
        if current_length + sentence_len > CHUNK_SIZE and current_sentences:
            chunks.append(" ".join(current_sentences))

            # OVERLAP: keep last N sentences in the next chunk for context continuity
            current_sentences = current_sentences[-CHUNK_OVERLAP:] if CHUNK_OVERLAP > 0 else []
            current_length = sum(len(s) for s in current_sentences)

        current_sentences.append(sentence)
        current_length += sentence_len

    # Don't forget the last chunk
    if current_sentences:
        chunks.append(" ".join(current_sentences))

    return chunks


# ─────────────────────────────────────────────────────────────
# OPENAI EMBEDDINGS (batched)
# ─────────────────────────────────────────────────────────────

def get_embeddings(texts, input_type="passage"):
    all_embeddings = []
    for i in range(0, len(texts), OPENAI_BATCH_SIZE):
        batch = texts[i:i + OPENAI_BATCH_SIZE]
        response = requests.post(
            "https://api.openai.com/v1/embeddings",
            headers={
                "Authorization": f"Bearer {OPENAI_API_KEY}",
                "Content-Type": "application/json"
            },
            json={"model": EMBED_MODEL, "input": batch}
        )
        if not response.ok:
            raise Exception(f"OpenAI Embedding failed: {response.text}")
        data = response.json()["data"]
        data.sort(key=lambda x: x["index"])
        all_embeddings.extend([item["embedding"] for item in data])
    return all_embeddings


# ─────────────────────────────────────────────────────────────
# CHECK IF ALREADY INDEXED
# ─────────────────────────────────────────────────────────────

def is_file_indexed(file_name, host):
    try:
        response = requests.post(
            f"{host}/query",
            headers=get_pinecone_headers(),
            json={
                "vector": [0.0] * EMBED_DIMENSION,
                "topK": 1,
                "filter": {"file_name": {"$eq": file_name}},
                "includeMetadata": True
            }
        )
        return len(response.json().get("matches", [])) > 0
    except Exception:
        return False


# ─────────────────────────────────────────────────────────────
# INDEX FILE TO PINECONE
# ─────────────────────────────────────────────────────────────

def index_file(file_name, file_path, host):
    if is_file_indexed(file_name, host):
        return True
    content = extract_file_content(file_path)
    if not content or not content.strip():
        raise ValueError("Could not extract text from file. It may be empty or a scanned image.")
    chunks = chunk_text(content)
    embeddings = get_embeddings(chunks, input_type="passage")
    vectors = [
        {
            "id": f"{file_name}_chunk_{i}",
            "values": emb,
            "metadata": {"file_name": file_name, "chunk_index": i, "text": chunk}
        }
        for i, (chunk, emb) in enumerate(zip(chunks, embeddings))
    ]
    for i in range(0, len(vectors), 100):
        resp = requests.post(
            f"{host}/vectors/upsert",
            headers=get_pinecone_headers(),
            json={"vectors": vectors[i:i + 100]}
        )
        if not resp.ok:
            raise Exception(f"Pinecone upsert failed: {resp.text}")
    return True


# ─────────────────────────────────────────────────────────────
# ✅ IMPROVEMENT #2 — RETRIEVE + RERANK
# Old approach: Retrieve top-5 from Pinecone by cosine similarity
# and send directly to LLM. Problem: cosine similarity is a blunt
# tool — it finds "semantically nearby" but not "most relevant".
#
# New approach:
# Step 1 → Retrieve top-20 candidates from Pinecone (wide net)
# Step 2 → Cross-encoder scores each [question + chunk] PAIR together
#           (it sees both at once, so it reasons about relevance)
# Step 3 → Keep only the best 5 by cross-encoder score
#
# The cross-encoder is far more precise than cosine similarity.
# It can improve answer accuracy by 30-40% on retrieval tasks.
# Falls back gracefully to Pinecone ordering if model unavailable.
# ─────────────────────────────────────────────────────────────

def retrieve_chunks_global(question, host):
    """
    Step 1: Embed the question and retrieve TOP_K_RETRIEVE=20 candidates
    from Pinecone across all documents (no file filter).
    """
    q_embedding = get_embeddings([question], input_type="query")[0]
    response = requests.post(
        f"{host}/query",
        headers=get_pinecone_headers(),
        json={
            "vector": q_embedding,
            "topK": TOP_K_RETRIEVE,   # Wide net: 20 candidates
            "includeMetadata": True
        }
    )
    if not response.ok:
        raise Exception(f"Pinecone query failed: {response.text}")
    matches = response.json().get("matches", [])
    return [
        {
            "text":      m["metadata"]["text"],
            "file_name": m["metadata"].get("file_name", "Unknown"),
            "score":     m.get("score", 0)
        }
        for m in matches if "metadata" in m
    ]


def rerank_chunks(question, candidates):
    """
    Step 2 & 3: Cross-encoder reranking.
    The cross-encoder sees [question, chunk] together and scores relevance.
    Returns top TOP_K_FINAL=5 most relevant chunks.
    """
    if not candidates:
        return []

    model = get_reranker()

    # If sentence-transformers is not installed, fall back to Pinecone ordering
    if model == "unavailable":
        print("ℹ️  Using Pinecone similarity order (install sentence-transformers for reranking)")
        return candidates[:TOP_K_FINAL]

    # Cross-encoder scores each [question, chunk] pair
    pairs  = [[question, c["text"]] for c in candidates]
    scores = model.predict(pairs)

    # Sort by cross-encoder score descending, take top 5
    ranked = sorted(zip(scores, candidates), key=lambda x: x[0], reverse=True)
    top    = [c for _, c in ranked[:TOP_K_FINAL]]
    return top


# ─────────────────────────────────────────────────────────────
# ✅ IMPROVEMENT #3 — CONVERSATION MEMORY
# Old approach: Every /query call was stateless — each question
# was treated as if it was the very first message ever.
# Follow-up questions like "What about that department?" or
# "Compare it with last quarter" completely failed because
# "that" and "it" had no reference in the current message.
#
# New approach:
# 1. Frontend sends the full conversation history with every request
# 2. Backend builds a "context-aware search query" by combining
#    the last assistant answer + current question — so Pinecone
#    gets a meaningful, reference-resolved query vector
# 3. The LLM receives the last N turns of conversation so it can
#    resolve pronouns, compare, and do follow-up reasoning
# ─────────────────────────────────────────────────────────────

def build_search_query(question, history):
    """
    Combine recent conversation context with the current question
    to create a better Pinecone search query.
    Example:
      history[-1] = "The profit margin is 29.2%"
      question    = "Who is responsible for that?"
      search query = "The profit margin is 29.2% Who is responsible for that?"
    This produces a much richer, reference-resolved embedding vector.
    """
    if not history:
        return question

    # Take last assistant response as context for the search query
    recent_context = ""
    for turn in history[-2:]:  # Last Q + last A
        recent_context += turn.get("content", "") + " "

    return (recent_context + question).strip()


def generate_answer(chunks, question, history=None):
    """
    Build prompt with:
    - Conversation history (last N turns) for context continuity
    - Retrieved document chunks as grounding context
    - Current question
    """
    if history is None:
        history = []

    # Build context from retrieved chunks with source labels
    context_parts = []
    for c in chunks:
        context_parts.append(f"[Source: {c['file_name']}]\n{c['text']}")
    context = "\n\n---\n\n".join(context_parts)

    # Format conversation history for the prompt (last N turns)
    history_text = ""
    if history:
        history_text = "\nCONVERSATION HISTORY (for context only):\n"
        for turn in history[-(HISTORY_TURNS * 2):]:  # N turns = N*2 messages
            role = "User" if turn["role"] == "user" else "Assistant"
            history_text += f"{role}: {turn['content']}\n"
        history_text += "\n"

    prompt = f"""You are an AI Assistant helping a business user analyze documents.

STRICT RULES:
- Answer ONLY from the DOCUMENT CONTEXT provided below
- Use CONVERSATION HISTORY only to understand what "it", "that", "this" refers to
- Do NOT mention chunks, vectors, embeddings, or any technical terms
- Answer clearly and professionally
- If multiple documents are referenced, mention which document the info came from
- If the answer is not in the context, say: "This information is not available in the uploaded documents"
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
        raise Exception("Cannot connect to Ollama. Run 'ollama serve' in your terminal and try again.")
    except requests.exceptions.Timeout:
        raise Exception("Ollama took too long to respond. Try a simpler question or restart Ollama.")


# ─────────────────────────────────────────────────────────────
# ROUTES
# ─────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return send_from_directory("static", "index.html")


@app.route("/upload", methods=["POST"])
def upload_file():
    if "file" not in request.files:
        return jsonify({"error": "No file provided"}), 400
    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No file selected"}), 400
    allowed = {".pdf", ".xlsx", ".xls", ".csv", ".docx", ".doc"}
    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in allowed:
        return jsonify({"error": f"Unsupported file type: {ext}"}), 400
    file_path = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(file_path)
    try:
        host = get_pinecone_host()
        index_file(file.filename, file_path, host)
        return jsonify({
            "success": True,
            "file_name": file.filename,
            "message": f"'{file.filename}' uploaded and indexed successfully!"
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/query", methods=["POST"])
def query():
    """
    Ask a question across ALL uploaded documents.
    Accepts optional conversation history for multi-turn memory.

    Request body:
    {
        "message": "What is the profit margin?",
        "history": [                              ← optional, sent by frontend
            {"role": "user",      "content": "How many employees?"},
            {"role": "assistant", "content": "There are 120 employees."}
        ]
    }
    """
    data    = request.json
    message = data.get("message", "").strip()
    history = data.get("history", [])    # ← conversation history from frontend

    if not message:
        return jsonify({"error": "No message provided"}), 400

    try:
        host = get_pinecone_host()

        # ✅ Improvement #3: Build context-aware search query using history
        search_query = build_search_query(message, history)

        # ✅ Improvement #2: Retrieve 20, rerank to top 5
        candidates = retrieve_chunks_global(search_query, host)
        chunks     = rerank_chunks(message, candidates)

        if not chunks:
            return jsonify({"answer": "No relevant information found in any of the uploaded documents."})

        # ✅ Improvement #3: Pass history to LLM for multi-turn reasoning
        answer = generate_answer(chunks, message, history)
        return jsonify({"answer": answer})

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/files", methods=["GET"])
def list_files():
    files = os.listdir(UPLOAD_FOLDER)
    allowed = {".pdf", ".xlsx", ".xls", ".csv", ".docx", ".doc"}
    files = [f for f in files if os.path.splitext(f)[1].lower() in allowed]
    return jsonify({"files": sorted(files)})


if __name__ == "__main__":
    get_reranker()   # Pre-load reranker model at startup (not on first query)
    app.run(debug=True, port=5002)
