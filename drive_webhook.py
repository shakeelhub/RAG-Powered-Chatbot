from flask import Blueprint, request, jsonify
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload
from google.oauth2 import service_account
import os
import io
import requests as req

# ─────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────
SERVICE_ACCOUNT_FILE = "service_account.json"
SCOPES               = ["https://www.googleapis.com/auth/drive.readonly"]
UPLOAD_FOLDER        = "uploads"
WEBHOOK_SECRET       = "myragSecret123"       # 🔑 Same as setup_drive_watch.py
FOLDER_ID = "1PxERO0LtjITMhBv07PBaX4DxCy4gckik"  # your folder id
ALLOWED_EXTENSIONS   = {".pdf", ".xlsx", ".xls", ".csv", ".docx"}

drive_bp = Blueprint("drive_webhook", __name__)
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Track already processed files to avoid duplicates
_processed_files = set()


# ─────────────────────────────────────────────────────────────
# GOOGLE DRIVE HELPERS
# ─────────────────────────────────────────────────────────────

def get_drive_service():
    creds = service_account.Credentials.from_service_account_file(
        SERVICE_ACCOUNT_FILE, scopes=SCOPES
    )
    return build("drive", "v3", credentials=creds)


def list_new_files_in_folder():
    """
    Lists all supported files in the watched folder.
    Returns list of {id, name} dicts.
    """
    service = get_drive_service()
    query   = f"'{FOLDER_ID}' in parents and trashed = false"
    result  = service.files().list(
        q=query,
        fields="files(id, name, mimeType)",
        orderBy="createdTime desc",
        pageSize=10
    ).execute()
    files = result.get("files", [])
    # Filter to supported file types only
    return [
        f for f in files
        if os.path.splitext(f["name"])[1].lower() in ALLOWED_EXTENSIONS
    ]


def download_file_from_drive(file_id, file_name):
    service   = get_drive_service()
    request_d = service.files().get_media(fileId=file_id)
    file_path = os.path.join(UPLOAD_FOLDER, file_name)
    with io.FileIO(file_path, "wb") as fh:
        downloader = MediaIoBaseDownload(fh, request_d)
        done = False
        while not done:
            status, done = downloader.next_chunk()
            print(f"  ↳ Downloading {file_name}: {int(status.progress() * 100)}%")
    return file_path


# ─────────────────────────────────────────────────────────────
# WEBHOOK ENDPOINT
# ─────────────────────────────────────────────────────────────

@drive_bp.route("/drive-webhook", methods=["POST"])
def drive_webhook():

    # Step 1 — Verify secret token
    token = request.headers.get("X-Goog-Channel-Token", "")
    if token != WEBHOOK_SECRET:
        print(f"❌ Webhook rejected — invalid token: {token}")
        return jsonify({"error": "Unauthorized"}), 401

    resource_state = request.headers.get("X-Goog-Resource-State", "")
    channel_id     = request.headers.get("X-Goog-Channel-ID", "")

    print(f"\n📨 Drive Notification Received")
    print(f"   State     : {resource_state}")
    print(f"   Channel ID: {channel_id}")

    # Step 2 — Ignore sync and non-update states
    if resource_state == "sync":
        print("   ↳ Sync handshake — ignoring.")
        return "", 200

    if resource_state not in ("update", "add", "change"):
        print(f"   ↳ State '{resource_state}' — ignoring.")
        return "", 200

    # Step 3 — List files in the folder and process new ones
    try:
        files = list_new_files_in_folder()
        if not files:
            print("   ↳ No supported files found in folder.")
            return "", 200

        for file in files:
            file_id   = file["id"]
            file_name = file["name"]

            # Skip already processed files
            if file_id in _processed_files:
                print(f"   ↳ Already indexed: {file_name} — skipping.")
                continue

            print(f"   ↳ New file detected: {file_name}")

            # Download file
            file_path = download_file_from_drive(file_id, file_name)
            print(f"   ↳ Downloaded to: {file_path}")

            # Index into Pinecone via /upload endpoint
            with open(file_path, "rb") as f:
                response = req.post(
                    "http://localhost:5001/upload",
                    files={"file": (file_name, f)}
                )

            result = response.json()
            if result.get("success"):
                print(f"   ✅ Indexed successfully: {file_name}")
                _processed_files.add(file_id)
            else:
                print(f"   ❌ Indexing failed: {result.get('error')}")

        return "", 200

    except Exception as e:
        print(f"   ❌ Webhook error: {e}")
        return "", 200


# ─────────────────────────────────────────────────────────────
# STATUS ENDPOINT
# ─────────────────────────────────────────────────────────────

@drive_bp.route("/drive-webhook/status", methods=["GET"])
def webhook_status():
    return jsonify({
        "status":  "active",
        "message": "Google Drive webhook is running and listening for file uploads."
    })