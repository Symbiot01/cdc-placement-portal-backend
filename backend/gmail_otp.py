"""Gmail-based OTP reader for unattended ERP logins.

Uses the Gmail REST API with a long-lived refresh token, exchanging it for a
short-lived access token at call time. Only `gmail.readonly` is required, and the
token is read from the environment / gitignored `.enverp` — never persisted to DB,
never logged, and never exposed in HTTP responses.

Reliability & security guarantees:
  * Snapshots pre-existing OTP message IDs BEFORE `getEmilOTP.htm` is triggered so
    an unread OTP from an earlier attempt can never be mistaken for the new one.
  * Polls adaptively for up to 55 seconds (ERP mail delivery typically takes 3–20s).
  * Verifies `internalDate`, strips HTML/CSS/scripts, inspects Subject/From headers,
    and tracks consumed Gmail message IDs in memory to prevent OTP replay.
"""

from __future__ import annotations

import base64
import html
import json
import logging
import os
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Callable
from pathlib import Path
from typing import Any

from backend.erp_login import load_dotenv_file

logger = logging.getLogger("cdc_poller.gmail_otp")

TOKEN_ENDPOINT = "https://oauth2.googleapis.com/token"
GMAIL_SEARCH_ENDPOINT = "https://gmail.googleapis.com/gmail/v1/users/me/messages"

ERP_OTP_GMAIL_QUERY = (
    'in:anywhere -in:trash newer_than:1d '
    '(otp OR "one-time password" OR "one time password" OR "verification code") '
    '(iitkgp OR kharagpur OR erp)'
)

OTP_AFTER_LABEL_RE = re.compile(
    r"(?:otp|one[\s-]*time(?:\s+password)?|verification\s+code)\D{0,64}(?<!\d)(\d{6})(?!\d)",
    re.IGNORECASE,
)
OTP_BEFORE_LABEL_RE = re.compile(
    r"(?<!\d)(\d{6})(?!\d)\D{0,64}(?:otp|one[\s-]*time(?:\s+password)?|verification\s+code)",
    re.IGNORECASE,
)
OTP_PATTERN = re.compile(r"\b(\d{6})\b")
ERP_SENDER_HINTS = ("erp", "iitkgp", "kharagpur", "kgp")
MESSAGE_ID_RE = re.compile(r"^[A-Za-z0-9]{1,64}$")
ADAPTIVE_POLL_DELAYS_SECONDS = (1.5, 2.0, 2.0, 2.5, 3.0, 3.5)
CLOCK_SKEW_WITH_BASELINE_MS = 5 * 60 * 1000
CLOCK_SKEW_NO_BASELINE_MS = 15 * 1000


class GmailOtpError(RuntimeError):
    """Raised when OTP retrieval fails. Never contains token material."""


class GmailOtpProvider:
    """Fetches the freshly issued ERP OTP from Gmail using a read-only refresh token."""

    def __init__(
        self,
        client_id: str,
        client_secret: str,
        refresh_token: str,
        *,
        timeout: float = 20.0,
        search_window_seconds: int = 600,
        poll_timeout_seconds: float = 55.0,
    ) -> None:
        if not (client_id and client_secret and refresh_token):
            raise GmailOtpError("Gmail OAuth credentials are incomplete")
        self._client_id = client_id
        self._client_secret = client_secret
        self._refresh_token = refresh_token
        self._timeout = timeout
        self._search_window_seconds = search_window_seconds
        self._poll_timeout_seconds = max(0.0, float(poll_timeout_seconds))
        self._cached_access_token: str | None = None
        self._access_token_expiry: float = 0.0
        self._consumed_ids: set[str] = set()

    @classmethod
    def from_env(cls, root_dir: Path | None = None) -> "GmailOtpProvider | None":
        base = root_dir or Path(__file__).resolve().parent.parent
        merged = {
            **load_dotenv_file(base / ".env"),
            **load_dotenv_file(base / ".enverp"),
        }

        def _get(name: str) -> str:
            return (os.environ.get(name) or merged.get(name) or "").strip()

        client_id = _get("GMAIL_OAUTH_CLIENT_ID")
        client_secret = _get("GMAIL_OAUTH_CLIENT_SECRET")
        refresh_token = _get("GMAIL_OAUTH_REFRESH_TOKEN")
        if not (client_id and client_secret and refresh_token):
            return None

        raw_poll_timeout = _get("GMAIL_OTP_POLL_TIMEOUT_SECONDS")
        try:
            poll_timeout = float(raw_poll_timeout) if raw_poll_timeout else 55.0
        except ValueError:
            poll_timeout = 55.0

        return cls(
            client_id,
            client_secret,
            refresh_token,
            poll_timeout_seconds=poll_timeout,
        )

    # ------------------------------------------------------------------ oauth

    def _access_token(self) -> str:
        now = time.time()
        if self._cached_access_token and now < self._access_token_expiry - 60:
            return self._cached_access_token

        payload = urllib.parse.urlencode(
            {
                "client_id": self._client_id,
                "client_secret": self._client_secret,
                "refresh_token": self._refresh_token,
                "grant_type": "refresh_token",
            }
        ).encode()
        req = urllib.request.Request(TOKEN_ENDPOINT, data=payload, method="POST")
        req.add_header("Content-Type", "application/x-www-form-urlencoded")
        try:
            with urllib.request.urlopen(req, timeout=self._timeout) as resp:
                data = json.loads(resp.read().decode())
        except urllib.error.HTTPError as exc:
            # Never surface the response body: it can echo credential fragments.
            raise GmailOtpError(f"Gmail token exchange failed with HTTP {exc.code}") from exc
        except urllib.error.URLError as exc:
            raise GmailOtpError(f"Gmail token endpoint unreachable: {exc.reason}") from exc

        token = str(data.get("access_token") or "")
        if not token:
            raise GmailOtpError("Gmail token exchange returned no access token")
        self._cached_access_token = token
        self._access_token_expiry = now + float(data.get("expires_in") or 3600)
        return token

    def _api_get(self, url: str) -> dict[str, Any]:
        req = urllib.request.Request(url, method="GET")
        req.add_header("Authorization", f"Bearer {self._access_token()}")
        req.add_header("Accept", "application/json")
        try:
            with urllib.request.urlopen(req, timeout=self._timeout) as resp:
                return json.loads(resp.read().decode())
        except urllib.error.HTTPError as exc:
            if exc.code == 401:
                self._cached_access_token = None
            raise GmailOtpError(f"Gmail API request failed with HTTP {exc.code}") from exc
        except urllib.error.URLError as exc:
            raise GmailOtpError(f"Gmail API unreachable: {exc.reason}") from exc

    # -------------------------------------------------------------------- otp

    def snapshot_existing_ids(self) -> set[str]:
        """Snapshot recent OTP email IDs BEFORE `getEmilOTP.htm` is called."""
        try:
            query = urllib.parse.quote(ERP_OTP_GMAIL_QUERY)
            listing = self._api_get(f"{GMAIL_SEARCH_ENDPOINT}?q={query}&maxResults=10&includeSpamTrash=true")
            ids: set[str] = set()
            for item in listing.get("messages") or []:
                mid = str(item.get("id") or "").strip()
                if MESSAGE_ID_RE.match(mid):
                    ids.add(mid)
            return ids
        except Exception as exc:  # noqa: BLE001
            logger.warning("Pre-OTP Gmail snapshot skipped: %s", exc)
            return set()

    def _find_otp_once(
        self,
        *,
        not_before_ms: float | None,
        excluded_ids: set[str],
    ) -> str | None:
        query = urllib.parse.quote(ERP_OTP_GMAIL_QUERY)
        listing = self._api_get(f"{GMAIL_SEARCH_ENDPOINT}?q={query}&maxResults=8&includeSpamTrash=true")
        candidates: list[tuple[float, str, str]] = []

        for message in listing.get("messages") or []:
            message_id = str(message.get("id") or "").strip()
            if not MESSAGE_ID_RE.match(message_id) or message_id in excluded_ids:
                continue
            detail = self._api_get(f"{GMAIL_SEARCH_ENDPOINT}/{message_id}?format=full")
            internal_ms = _read_internal_date_ms(detail)
            if not_before_ms is not None and internal_ms > 0 and internal_ms < not_before_ms:
                continue

            text = _extract_message_text(detail)
            if not _looks_like_erp_mail(text):
                continue
            otp = extract_erp_otp(text)
            if otp:
                candidates.append((internal_ms, message_id, otp))

        if not candidates:
            return None

        # Pick newest email first
        candidates.sort(key=lambda item: item[0], reverse=True)
        _, chosen_id, chosen_otp = candidates[0]
        self._consumed_ids.add(chosen_id)
        if len(self._consumed_ids) > 50:
            # Keep memory bounded
            self._consumed_ids = set(list(self._consumed_ids)[-30:])
        logger.info("Found fresh ERP OTP in Gmail message %s", chosen_id)
        return chosen_otp

    def fetch_otp(
        self,
        not_before: float | None = None,
        *,
        exclude_ids: set[str] | None = None,
        poll_timeout_seconds: float | None = None,
        sleep_fn: Callable[[float], None] = time.sleep,
    ) -> str | None:
        """Poll Gmail until the freshly requested ERP OTP arrives, or return None on timeout."""
        timeout = self._poll_timeout_seconds if poll_timeout_seconds is None else max(0.0, poll_timeout_seconds)
        deadline = time.monotonic() + timeout

        has_baseline = exclude_ids is not None
        not_before_ms: float | None = None
        if not_before is not None:
            skew = CLOCK_SKEW_WITH_BASELINE_MS if has_baseline else CLOCK_SKEW_NO_BASELINE_MS
            not_before_ms = (not_before * 1000.0) - skew

        attempt = 0
        while True:
            combined_excluded = set(self._consumed_ids)
            if exclude_ids:
                combined_excluded.update(exclude_ids)

            try:
                otp = self._find_otp_once(
                    not_before_ms=not_before_ms,
                    excluded_ids=combined_excluded,
                )
                if otp:
                    return otp
            except GmailOtpError:
                if time.monotonic() >= deadline:
                    raise
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return None

            delay = ADAPTIVE_POLL_DELAYS_SECONDS[
                min(attempt, len(ADAPTIVE_POLL_DELAYS_SECONDS) - 1)
            ]
            attempt += 1
            sleep_fn(min(delay, remaining))


def _read_internal_date_ms(message: dict[str, Any]) -> float:
    raw = message.get("internalDate")
    try:
        return float(raw) if raw is not None else 0.0
    except (TypeError, ValueError):
        return 0.0


def _b64url_decode(data: str) -> str:
    padded = data.replace("-", "+").replace("_", "/")
    padded += "=" * (-len(padded) % 4)
    try:
        return base64.b64decode(padded).decode("utf-8", errors="replace")
    except Exception:
        return ""


def _walk_parts(payload: dict[str, Any], depth: int = 0) -> list[dict[str, Any]]:
    if depth > 6:
        return []
    parts = [payload]
    for part in payload.get("parts") or []:
        if isinstance(part, dict) and not part.get("filename"):
            parts.extend(_walk_parts(part, depth + 1))
    return parts


def _headers_context(payload: dict[str, Any]) -> str:
    headers = payload.get("headers")
    if not isinstance(headers, list):
        return ""
    out: list[str] = []
    for h in headers:
        if not isinstance(h, dict):
            continue
        name = str(h.get("name") or "").lower()
        if name in {"subject", "from"}:
            val = str(h.get("value") or "").strip()
            if val:
                out.append(val[:400])
    return "\n".join(out)


def _to_plain_text(raw: str) -> str:
    cleaned = re.sub(r"<style[\s\S]*?</style>", " ", raw or "", flags=re.IGNORECASE)
    cleaned = re.sub(r"<script[\s\S]*?</script>", " ", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"<[^>]+>", " ", cleaned)
    cleaned = html.unescape(cleaned)
    return re.sub(r"\s+", " ", cleaned).strip()


def _extract_message_text(message: dict[str, Any]) -> str:
    payload = message.get("payload") or {}
    chunks: list[str] = []
    header_ctx = _headers_context(payload)
    if header_ctx:
        chunks.append(header_ctx)
    for part in _walk_parts(payload):
        body = part.get("body") or {}
        data = body.get("data")
        if data and len(str(data)) <= 120_000:
            chunks.append(_b64url_decode(str(data)))
    if len(chunks) <= (1 if header_ctx else 0):
        snippet = message.get("snippet")
        if snippet:
            chunks.append(str(snippet))
    return _to_plain_text("\n".join(chunks))[:20_000]


def extract_erp_otp(text: str) -> str | None:
    plain = _to_plain_text(text)
    if not plain or not _looks_like_erp_mail(plain):
        return None
    for pattern in (OTP_AFTER_LABEL_RE, OTP_BEFORE_LABEL_RE, OTP_PATTERN):
        match = pattern.search(plain)
        if match:
            return match.group(1)
    return None


def _looks_like_erp_mail(text: str) -> bool:
    """Bias toward the ERP mail so we never harvest a code from another sender."""
    low = (text or "").lower()
    if not any(kw in low for kw in ("otp", "one-time password", "one time password", "verification code")):
        return False
    return any(hint in low for hint in ERP_SENDER_HINTS)
