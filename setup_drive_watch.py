# ═══════════════════════════════════════════════════════════════════
# setup_drive_watch.py
#
# Run this script ONCE to register your Flask server as a
# Google Drive push notification receiver.
#
# What it does:
#   1. Authenticates with Google Drive API
#   2. Tells Google: "Watch this folder, and POST to my URL
#      whenever a file is added or changed"
#   3. Google Drive will then push notifications instantly
#
# Run: python setup_drive_watch.py
# ═══════════════════════════════════════════════════════════════════

from google.oauth2 import service_account
from googleapiclient.discovery import build
import uuid
import json
import datetime

# ─────────────────────────────────────────────────────────────
# CONFIG — Fill these in before running
# ─────────────────────────────────────────────────────────────
SERVICE_ACCOUNT_FILE = "service_account.json"   # 🔑 From Google Cloud Console
SCOPES               = ["https://www.googleapis.com/auth/drive.readonly"]

# The Google Drive folder ID to watch
# Get it from the URL: drive.google.com/drive/folders/THIS_PART
FOLDER_ID = "1PxERO0LtjITMhBv07PBaX4DxCy4gckik"   # 🔑 Replace this

# Your public Flask server URL (must be HTTPS for Google Drive webhooks)
# For local testing use: ngrok http 5001  → get the https URL
WEBHOOK_URL = "https://glandlike-bao-unmordantly.ngrok-free.dev/drive-webhook"   # 🔑 Replace this

# Must match WEBHOOK_SECRET in drive_webhook.py exactly
WEBHOOK_SECRET = "myragSecret123"   # 🔑 Replace this

# ─────────────────────────────────────────────────────────────
# REGISTER THE WATCH
# ─────────────────────────────────────────────────────────────

def setup_drive_watch():
    print("🔧 Setting up Google Drive push notification watch...")
    print(f"   Folder ID  : {FOLDER_ID}")
    print(f"   Webhook URL: {WEBHOOK_URL}")

    # Authenticate
    creds   = service_account.Credentials.from_service_account_file(
        SERVICE_ACCOUNT_FILE, scopes=SCOPES
    )
    service = build("drive", "v3", credentials=creds)

    # Generate a unique channel ID for this watch
    channel_id = str(uuid.uuid4())

    # Watch expiry — Google Drive max is 7 days
    # We set it to 6 days; you need to re-run this script before it expires
    # OR set up a cron job to auto-renew (see bottom of this file)
    expiry_ms = int(
        (datetime.datetime.utcnow() + datetime.timedelta(days=6)).timestamp() * 1000
    )

    # Register the watch on the folder
    watch_body = {
        "id":         channel_id,        # unique channel identifier
        "type":       "web_hook",
        "address":    WEBHOOK_URL,        # Google will POST here
        "token":      WEBHOOK_SECRET,     # sent as X-Goog-Channel-Token header
        "expiration": expiry_ms
    }

    try:
        response = service.files().watch(
            fileId=FOLDER_ID,
            body=watch_body
        ).execute()

        print("\n✅ Watch registered successfully!")
        print(f"   Channel ID  : {response.get('id')}")
        print(f"   Resource ID : {response.get('resourceId')}")
        print(f"   Expiry      : {response.get('expiration')} ms (unix)")
        print(f"   Expires in  : ~6 days from now")

        # Save the channel info so you can stop/renew it later
        watch_info = {
            "channel_id":   response.get("id"),
            "resource_id":  response.get("resourceId"),
            "expiration":   response.get("expiration"),
            "webhook_url":  WEBHOOK_URL,
            "folder_id":    FOLDER_ID,
            "created_at":   datetime.datetime.utcnow().isoformat()
        }
        with open("drive_watch_info.json", "w") as f:
            json.dump(watch_info, f, indent=2)

        print("\n💾 Watch info saved to: drive_watch_info.json")
        print("   (Keep this file — you need it to stop or renew the watch)")
        print("\n🚀 Google Drive will now instantly notify your server")
        print("   when any file is added to the watched folder!")

    except Exception as e:
        print(f"\n❌ Failed to register watch: {e}")
        print("\nCommon reasons:")
        print("  1. WEBHOOK_URL must be HTTPS (use ngrok for local testing)")
        print("  2. Service account must have access to the folder")
        print("  3. Google Drive API must be enabled in Cloud Console")


# ─────────────────────────────────────────────────────────────
# STOP AN EXISTING WATCH (run if you want to cancel)
# ─────────────────────────────────────────────────────────────

def stop_drive_watch():
    print("🛑 Stopping existing Google Drive watch...")
    try:
        with open("drive_watch_info.json") as f:
            watch_info = json.load(f)
    except FileNotFoundError:
        print("❌ drive_watch_info.json not found. Nothing to stop.")
        return

    creds   = service_account.Credentials.from_service_account_file(
        SERVICE_ACCOUNT_FILE, scopes=SCOPES
    )
    service = build("drive", "v3", credentials=creds)

    try:
        service.channels().stop(body={
            "id":         watch_info["channel_id"],
            "resourceId": watch_info["resource_id"]
        }).execute()
        print("✅ Watch stopped successfully.")
        os.remove("drive_watch_info.json")
    except Exception as e:
        print(f"❌ Failed to stop watch: {e}")


# ─────────────────────────────────────────────────────────────
# RENEW WATCH (Google Drive watch expires every 7 days max)
# Run this every 6 days, or set up a cron job:
#   crontab -e
#   0 9 */6 * * cd /your/project && python setup_drive_watch.py renew
# ─────────────────────────────────────────────────────────────

def renew_drive_watch():
    print("🔄 Renewing Google Drive watch...")
    stop_drive_watch()
    setup_drive_watch()
    print("✅ Watch renewed for another 6 days.")


# ─────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────
import sys
import os

if __name__ == "__main__":
    action = sys.argv[1] if len(sys.argv) > 1 else "setup"

    if action == "setup":
        setup_drive_watch()
    elif action == "stop":
        stop_drive_watch()
    elif action == "renew":
        renew_drive_watch()
    else:
        print(f"Unknown action: {action}")
        print("Usage: python setup_drive_watch.py [setup|stop|renew]")
