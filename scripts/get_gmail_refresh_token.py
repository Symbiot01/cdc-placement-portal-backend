#!/usr/bin/env python3
"""One-time helper: mint a Gmail API refresh token for the CDC poller.

Prerequisites (Google Cloud Console):
  1. Enable the Gmail API for your project.
  2. Create an OAuth client of type "Desktop app".
  3. Set GMAIL_OAUTH_CLIENT_ID and GMAIL_OAUTH_CLIENT_SECRET in your shell.

Then run:
    python scripts/get_gmail_refresh_token.py

A browser opens; approve the gmail.readonly scope. The script prints the refresh
token exactly once. Paste it into Coolify as GMAIL_OAUTH_REFRESH_TOKEN, then throw
it away — it is a long-lived credential that can read your mailbox.

Scope is read-only and limited to Gmail; it cannot send, delete, or modify mail.
"""

from __future__ import annotations

import http.server
import json
import os
import secrets
import socket
import sys
import urllib.parse
import urllib.request
import webbrowser

AUTH_ENDPOINT = "https://accounts.google.com/o/oauth2/v2/auth"
TOKEN_ENDPOINT = "https://oauth2.googleapis.com/token"
SCOPE = "https://www.googleapis.com/auth/gmail.readonly"


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


class _CallbackHandler(http.server.BaseHTTPRequestHandler):
    code: str | None = None
    error: str | None = None
    expected_state: str = ""

    def do_GET(self) -> None:  # noqa: N802 - stdlib naming
        parsed = urllib.parse.urlparse(self.path)
        params = urllib.parse.parse_qs(parsed.query)

        if params.get("state", [""])[0] != self.expected_state:
            self._respond("State mismatch — aborting for safety.")
            type(self).error = "state mismatch"
            return

        if "error" in params:
            self._respond(f"Authorization failed: {params['error'][0]}")
            type(self).error = params["error"][0]
            return

        code = params.get("code", [""])[0]
        type(self).code = code
        self._respond("Authorization received. You can close this tab.")

    def _respond(self, message: str) -> None:
        body = f"<html><body><h3>{message}</h3></body></html>".encode()
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args) -> None:  # silence request logging
        return


def main() -> int:
    client_id = (os.environ.get("GMAIL_OAUTH_CLIENT_ID") or "").strip()
    client_secret = (os.environ.get("GMAIL_OAUTH_CLIENT_SECRET") or "").strip()
    if not client_id or not client_secret:
        print(
            "Missing credentials.\n"
            "Set GMAIL_OAUTH_CLIENT_ID and GMAIL_OAUTH_CLIENT_SECRET first:\n"
            "  export GMAIL_OAUTH_CLIENT_ID='...'\n"
            "  export GMAIL_OAUTH_CLIENT_SECRET='...'",
            file=sys.stderr,
        )
        return 1

    port = _free_port()
    redirect_uri = f"http://127.0.0.1:{port}/"
    state = secrets.token_urlsafe(24)

    _CallbackHandler.expected_state = state

    query = urllib.parse.urlencode(
        {
            "client_id": client_id,
            "redirect_uri": redirect_uri,
            "response_type": "code",
            "scope": SCOPE,
            "access_type": "offline",
            "prompt": "consent",  # force refresh_token issuance
            "state": state,
        }
    )
    auth_url = f"{AUTH_ENDPOINT}?{query}"

    server = http.server.HTTPServer(("127.0.0.1", port), _CallbackHandler)
    print("Opening your browser to approve Gmail read-only access…")
    print(f"If it does not open automatically, visit:\n{auth_url}\n")
    try:
        webbrowser.open(auth_url)
    except Exception:
        pass

    server.handle_request()
    server.server_close()

    handler = _CallbackHandler
    if handler.error or not handler.code:
        print(f"Authorization did not complete: {handler.error or 'no code returned'}", file=sys.stderr)
        return 1

    payload = urllib.parse.urlencode(
        {
            "client_id": client_id,
            "client_secret": client_secret,
            "code": handler.code,
            "grant_type": "authorization_code",
            "redirect_uri": redirect_uri,
        }
    ).encode()

    request = urllib.request.Request(TOKEN_ENDPOINT, data=payload, method="POST")
    request.add_header("Content-Type", "application/x-www-form-urlencoded")
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            tokens = json.loads(response.read().decode())
    except Exception as exc:  # noqa: BLE001
        print(f"Token exchange failed: {exc}", file=sys.stderr)
        return 1

    refresh_token = tokens.get("refresh_token")
    if not refresh_token:
        print(
            "No refresh_token in the response.\n"
            "Revoke prior access at https://myaccount.google.com/permissions and retry.",
            file=sys.stderr,
        )
        return 1

    print("\n" + "=" * 68)
    print("GMAIL_OAUTH_REFRESH_TOKEN (paste into Coolify, then delete this output):")
    print("=" * 68)
    print(refresh_token)
    print("=" * 68)
    print("\nAlso set GMAIL_OAUTH_CLIENT_ID and GMAIL_OAUTH_CLIENT_SECRET.")
    print("Scope granted: gmail.readonly (cannot send, delete, or modify mail).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
