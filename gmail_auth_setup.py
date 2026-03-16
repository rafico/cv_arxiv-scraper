"""One-time OAuth2 setup for Gmail API access (CLI fallback).

For most users, the web UI at /settings handles Gmail authorization
automatically via browser redirect — no need to run this script.

This script is a fallback for headless / CLI-only environments. It will:
  1. Read your ``credentials.json`` (downloaded from Google Cloud Console).
  2. Open a browser for you to grant the ``gmail.send`` scope.
  3. Save a ``token.json`` file with restrictive (600) permissions.

Prerequisites:
  - Create a Google Cloud project: https://console.cloud.google.com/
  - Enable the Gmail API for the project.
  - Create OAuth Client ID credentials (Web application type).
  - Download the JSON file and save it as ``credentials.json`` in the project root.
  - Set the OAuth consent screen publishing status to "In production"
    so the refresh token does not expire after 7 days.
"""

import os
import sys
from pathlib import Path

SCOPES = ["https://www.googleapis.com/auth/gmail.send"]

PROJECT_ROOT = Path(__file__).resolve().parent
CREDENTIALS_PATH = PROJECT_ROOT / "credentials.json"
TOKEN_PATH = PROJECT_ROOT / "token.json"


def main() -> None:
    try:
        from google_auth_oauthlib.flow import InstalledAppFlow
    except ImportError:
        print(
            "ERROR: Required packages not installed. Run:\n"
            "  pip install google-auth google-auth-oauthlib google-api-python-client",
            file=sys.stderr,
        )
        sys.exit(1)

    if not CREDENTIALS_PATH.exists():
        print(
            f"ERROR: {CREDENTIALS_PATH} not found.\n\n"
            "To create it:\n"
            "  1. Go to https://console.cloud.google.com/apis/credentials\n"
            "  2. Create OAuth Client ID (Desktop application)\n"
            "  3. Download the JSON and save it as 'credentials.json' here.",
            file=sys.stderr,
        )
        sys.exit(1)

    print("Starting OAuth2 authorization flow...")
    print(f"  Scope: gmail.send (send-only — cannot read your inbox)")
    print(f"  Credentials: {CREDENTIALS_PATH}")
    print()

    flow = InstalledAppFlow.from_client_secrets_file(str(CREDENTIALS_PATH), SCOPES)
    creds = flow.run_local_server(port=0)

    TOKEN_PATH.write_text(creds.to_json())
    os.chmod(TOKEN_PATH, 0o600)

    print(f"\nToken saved to: {TOKEN_PATH}")
    print(f"  Permissions: 600 (owner-only read/write)")
    print(f"\nSetup complete! You can now run: python email_digest.py")


if __name__ == "__main__":
    main()
