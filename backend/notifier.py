"""WhatsApp Cloud API notifier for new CDC notices & secret invite-link broadcast group.

Instead of hardcoding individual phone numbers in `.env`, members self-join via
a secret `wa.me/<BUSINESS_NUMBER>?text=JOIN%20<WA_JOIN_CODE>` invite link
(unlimited members, no Official Business Account required). Static
`WA_RECIPIENTS` remains optional and is seamlessly merged if present.

Sending modes supported:
  1. Standard Utility Template (`WA_TEMPLATE_NAME`, e.g. `cdc_new_notice`) with
     3 body parameters: {{1}} company, {{2}} category, {{3}} notice excerpt.
     When a PDF attachment exists, uploads media once and sends a follow-up
     `document` message (works inside an active 24h window).
  2. Document-Header Utility Template (`WA_DOC_TEMPLATE_NAME`, optional):
     When configured and a notice has a PDF attachment, sends a single combined
     template with `header: document(media_id)` + `body: {{1}}, {{2}}, {{3}}` —
     delivering BOTH text and PDF 24/7 even outside the 24h window.
  3. Direct replies (`send_text_message`) inside the 24-hour service window to
     confirm `JOIN <CODE>` / `STOP` actions automatically on WhatsApp.

Security & privacy guarantees:
  * Access tokens and join codes are never logged or included in exceptions.
  * Phone numbers are masked (`+91******3210`) in logs and failure reports, and
    encrypted at rest in SQLite.
  * Attachments are magic-byte checked and capped at 20 MB.
"""

from __future__ import annotations

import json
import logging
import mimetypes
import os
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path

from backend.erp_login import load_dotenv_file

logger = logging.getLogger("cdc_poller.notifier")

GRAPH_BASE = "https://graph.facebook.com/v21.0"
E164_RE = re.compile(r"^\+[1-9]\d{6,14}$")
MAX_WA_MEDIA_BYTES = 20 * 1024 * 1024

CIRCUIT_BREAKER_THRESHOLD = 3
CIRCUIT_BREAKER_COOLDOWN_SECONDS = 15 * 60


class NotifierError(RuntimeError):
    """Raised when the WhatsApp channel is misconfigured or permanently failing."""


@dataclass
class DeliveryResult:
    attempted: int = 0
    delivered: int = 0
    failed: list[str] = field(default_factory=list)
    attachment_sent: bool = False

    @property
    def ok(self) -> bool:
        return self.attempted > 0 and self.delivered == self.attempted


def mask_phone(phone: str) -> str:
    clean = (phone or "").strip()
    if len(clean) <= 6:
        return "***"
    return f"{clean[:3]}******{clean[-4:]}"


def parse_recipients(raw: str) -> list[str]:
    """Parse and validate a comma/space/semicolon separated E.164 recipient list."""
    candidates = [p.strip() for p in re.split(r"[,\s;]+", raw or "") if p.strip()]
    valid: list[str] = []
    for number in candidates:
        normalized = number.replace(" ", "").replace("-", "")
        if not normalized.startswith("+"):
            # Accept a bare 10-digit Indian mobile number for convenience; never guess others.
            if re.fullmatch(r"\d{10}", normalized):
                normalized = f"+91{normalized}"
        if E164_RE.match(normalized) and normalized not in valid:
            valid.append(normalized)
        else:
            logger.warning("Ignoring malformed WhatsApp recipient entry")
    return valid


def _clean_param(value: str, limit: int = 500) -> str:
    """WhatsApp template params reject newlines, tabs, 4+ consecutive spaces, and control chars."""
    cleaned = re.sub(r"[\r\n\t]+", " ", value or "").strip()
    cleaned = re.sub(r"\s{2,}", " ", cleaned)
    if not cleaned:
        return "-"
    if len(cleaned) <= limit:
        return cleaned
    return cleaned[: limit - 1].rstrip() + "…"


class WhatsAppNotifier:
    def __init__(
        self,
        access_token: str,
        phone_number_id: str,
        recipients: list[str],
        template_name: str,
        template_lang: str = "en_US",
        *,
        doc_template_name: str = "",
        business_phone_e164: str = "",
        allow_empty_recipients: bool = False,
        timeout: float = 30.0,
    ) -> None:
        if not access_token or not phone_number_id:
            raise NotifierError("WhatsApp Cloud API credentials are incomplete")
        if not recipients and not allow_empty_recipients:
            raise NotifierError("No valid WhatsApp recipients configured")
        if not template_name:
            raise NotifierError("WA_TEMPLATE_NAME is required for business-initiated messages")
        self._token = access_token
        self._phone_number_id = phone_number_id
        self._recipients = list(recipients)
        self._template_name = template_name
        self._template_lang = template_lang
        self._doc_template_name = doc_template_name.strip()
        self._business_phone_e164 = re.sub(r"[^\d]", "", business_phone_e164 or "")
        self._timeout = timeout
        self._consecutive_failures = 0
        self._open_until = 0.0

    @property
    def recipients_count(self) -> int:
        return len(self._recipients)

    @property
    def raw_recipients(self) -> list[str]:
        return list(self._recipients)

    @property
    def masked_recipients(self) -> list[str]:
        return [mask_phone(r) for r in self._recipients]

    @property
    def template_name(self) -> str:
        return self._template_name

    @property
    def doc_template_name(self) -> str:
        return self._doc_template_name

    @property
    def business_phone_digits(self) -> str:
        return self._business_phone_e164

    def build_invite_link(self, join_code: str) -> str | None:
        """Build `https://wa.me/<digits>?text=JOIN%20<code>` when business number + join code are set."""
        clean_code = (join_code or "").strip()
        if not self._business_phone_e164 or not clean_code:
            return None
        encoded_text = urllib.parse.quote(f"JOIN {clean_code}")
        return f"https://wa.me/{self._business_phone_e164}?text={encoded_text}"

    @classmethod
    def from_env(
        cls,
        root_dir: Path | None = None,
        *,
        extra_recipients: list[str] | None = None,
        allow_empty_recipients: bool = False,
    ) -> "WhatsAppNotifier | None":
        base = root_dir or Path(__file__).resolve().parent.parent
        merged = {
            **load_dotenv_file(base / ".env"),
            **load_dotenv_file(base / ".enverp"),
        }

        def _get(name: str, default: str = "") -> str:
            return (os.environ.get(name) or merged.get(name) or default).strip()

        token = _get("WA_ACCESS_TOKEN")
        phone_id = _get("WA_PHONE_NUMBER_ID")
        static_recipients = parse_recipients(_get("WA_RECIPIENTS"))
        combined_recipients: list[str] = list(static_recipients)
        for extra in extra_recipients or []:
            for parsed_num in parse_recipients(extra):
                if parsed_num not in combined_recipients:
                    combined_recipients.append(parsed_num)

        template = _get("WA_TEMPLATE_NAME")
        lang = _get("WA_TEMPLATE_LANG", "en_US") or "en_US"
        doc_template = _get("WA_DOC_TEMPLATE_NAME")
        business_phone = _get("WA_BUSINESS_PHONE_NUMBER")
        join_code = _get("WA_JOIN_CODE")

        # When join-link broadcast mode is configured (`WA_JOIN_CODE` set) or caller
        # explicitly allows empty recipients, initialize notifier even before the first member joins.
        can_be_empty = allow_empty_recipients or bool(join_code)
        if not (token and phone_id and template):
            return None
        if not combined_recipients and not can_be_empty:
            return None
        return cls(
            token,
            phone_id,
            combined_recipients,
            template,
            lang,
            doc_template_name=doc_template,
            business_phone_e164=business_phone,
            allow_empty_recipients=can_be_empty,
        )

    # ------------------------------------------------------------- transports

    def _post(
        self,
        url: str,
        *,
        json_body: dict | None = None,
        data: bytes | None = None,
        headers: dict[str, str] | None = None,
    ) -> tuple[int, dict]:
        body = json.dumps(json_body).encode("utf-8") if json_body is not None else data
        req = urllib.request.Request(url, data=body, method="POST")
        req.add_header("Authorization", f"Bearer {self._token}")
        if json_body is not None:
            req.add_header("Content-Type", "application/json")
        for key, value in (headers or {}).items():
            req.add_header(key, value)
        try:
            with urllib.request.urlopen(req, timeout=self._timeout) as resp:
                return resp.status, json.loads(resp.read().decode("utf-8") or "{}")
        except urllib.error.HTTPError as exc:
            raw = exc.read().decode("utf-8", errors="replace") if exc.fp else ""
            try:
                payload = json.loads(raw)
            except json.JSONDecodeError:
                payload = {"error": {"message": raw[:300]}}
            return exc.code, payload
        except urllib.error.URLError as exc:
            return 0, {"error": {"message": f"unreachable: {exc.reason}"}}

    def _circuit_open(self) -> bool:
        return time.time() < self._open_until

    def _record(self, success: bool) -> None:
        if success:
            self._consecutive_failures = 0
            return
        self._consecutive_failures += 1
        if self._consecutive_failures >= CIRCUIT_BREAKER_THRESHOLD:
            self._open_until = time.time() + CIRCUIT_BREAKER_COOLDOWN_SECONDS
            logger.error(
                "WhatsApp circuit breaker opened for %ss after %s consecutive failures",
                CIRCUIT_BREAKER_COOLDOWN_SECONDS,
                self._consecutive_failures,
            )

    # ---------------------------------------------------------------- sending

    def send_notice_template(
        self, *, company: str, category: str, excerpt: str
    ) -> DeliveryResult:
        """Fan out one notice to every recipient as a utility template message."""
        result = DeliveryResult()
        if self._circuit_open():
            result.failed = [mask_phone(r) for r in self._recipients]
            result.attempted = len(self._recipients)
            logger.warning("Skipping WhatsApp fan-out: circuit breaker is open")
            return result

        url = f"{GRAPH_BASE}/{self._phone_number_id}/messages"
        body = {
            "messaging_product": "whatsapp",
            "type": "template",
            "template": {
                "name": self._template_name,
                "language": {"code": self._template_lang},
                "components": [
                    {
                        "type": "body",
                        "parameters": [
                            {"type": "text", "text": _clean_param(company, 120)},
                            {"type": "text", "text": _clean_param(category, 120)},
                            {"type": "text", "text": _clean_param(excerpt, 500)},
                        ],
                    }
                ],
            },
        }

        for recipient in self._recipients:
            result.attempted += 1
            payload = {**body, "to": recipient}
            status, response = self._post(url, json_body=payload)
            if 200 <= status < 300:
                result.delivered += 1
                self._record(True)
            else:
                message = _error_message(response)
                masked = mask_phone(recipient)
                result.failed.append(f"{masked}: {message}")
                self._record(False)
                logger.warning("WhatsApp send failed for %s: %s", masked, message)
        return result

    def send_notice_with_document_template(
        self,
        *,
        company: str,
        category: str,
        excerpt: str,
        media_id: str,
        filename: str,
    ) -> DeliveryResult:
        """Send a Document-Header Utility Template (works outside 24h window with PDF attached)."""
        if not self._doc_template_name:
            raise NotifierError("WA_DOC_TEMPLATE_NAME is not configured")

        result = DeliveryResult()
        if self._circuit_open():
            result.failed = [mask_phone(r) for r in self._recipients]
            result.attempted = len(self._recipients)
            return result

        safe_name = re.sub(r"[^A-Za-z0-9._-]+", "_", filename)[:90] or "notice.pdf"
        url = f"{GRAPH_BASE}/{self._phone_number_id}/messages"
        body = {
            "messaging_product": "whatsapp",
            "type": "template",
            "template": {
                "name": self._doc_template_name,
                "language": {"code": self._template_lang},
                "components": [
                    {
                        "type": "header",
                        "parameters": [
                            {
                                "type": "document",
                                "document": {
                                    "id": media_id,
                                    "filename": safe_name,
                                },
                            }
                        ],
                    },
                    {
                        "type": "body",
                        "parameters": [
                            {"type": "text", "text": _clean_param(company, 120)},
                            {"type": "text", "text": _clean_param(category, 120)},
                            {"type": "text", "text": _clean_param(excerpt, 500)},
                        ],
                    },
                ],
            },
        }

        for recipient in self._recipients:
            result.attempted += 1
            status, response = self._post(url, json_body={**body, "to": recipient})
            if 200 <= status < 300:
                result.delivered += 1
                self._record(True)
            else:
                message = _error_message(response)
                masked = mask_phone(recipient)
                result.failed.append(f"{masked}: {message}")
                self._record(False)
                logger.warning("WhatsApp doc-template send failed for %s: %s", masked, message)
        result.attachment_sent = result.delivered > 0
        return result

    def upload_media(self, content: bytes, filename: str, mime_type: str) -> str | None:
        """Upload a verified binary attachment and return its media id, or None on failure."""
        if self._circuit_open():
            return None
        if not content or len(content) > MAX_WA_MEDIA_BYTES:
            logger.warning("Skipping WhatsApp media upload: invalid size (%d bytes)", len(content or b""))
            return None

        boundary = f"----cdc{int(time.time() * 1000)}"
        safe_name = re.sub(r"[^A-Za-z0-9._-]+", "_", filename)[:90] or "notice.pdf"
        parts = [
            f"--{boundary}\r\n"
            'Content-Disposition: form-data; name="messaging_product"\r\n\r\n'
            "whatsapp\r\n",
            f"--{boundary}\r\n"
            'Content-Disposition: form-data; name="type"\r\n\r\n'
            f"{mime_type}\r\n",
            f"--{boundary}\r\n"
            f'Content-Disposition: form-data; name="file"; filename="{safe_name}"\r\n'
            f"Content-Type: {mime_type}\r\n\r\n",
        ]
        body = b"".join(p.encode("utf-8") for p in parts) + content + f"\r\n--{boundary}--\r\n".encode("utf-8")
        url = f"{GRAPH_BASE}/{self._phone_number_id}/media"
        status, response = self._post(
            url,
            data=body,
            headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
        )
        if 200 <= status < 300 and response.get("id"):
            self._record(True)
            return str(response["id"])
        self._record(False)
        logger.warning("WhatsApp media upload failed: %s", _error_message(response))
        return None

    def send_document(
        self, *, media_id: str, filename: str, caption: str
    ) -> DeliveryResult:
        """Send an already-uploaded document to every recipient (active 24h window)."""
        result = DeliveryResult()
        if self._circuit_open():
            result.attempted = len(self._recipients)
            result.failed = [mask_phone(r) for r in self._recipients]
            return result

        url = f"{GRAPH_BASE}/{self._phone_number_id}/messages"
        safe_name = re.sub(r"[^A-Za-z0-9._-]+", "_", filename)[:90] or "notice.pdf"
        body = {
            "messaging_product": "whatsapp",
            "type": "document",
            "document": {
                "id": media_id,
                "filename": safe_name,
                "caption": _clean_param(caption, 900),
            },
        }
        for recipient in self._recipients:
            result.attempted += 1
            status, response = self._post(url, json_body={**body, "to": recipient})
            if 200 <= status < 300:
                result.delivered += 1
                self._record(True)
            else:
                masked = mask_phone(recipient)
                result.failed.append(f"{masked}: {_error_message(response)}")
                self._record(False)
        result.attachment_sent = result.delivered > 0
        return result

    def send_text_message(self, *, to_e164: str, text: str) -> bool:
        """Send a direct free-form WhatsApp reply inside an active 24h window (e.g., JOIN/STOP ack)."""
        if self._circuit_open():
            return False
        parsed = parse_recipients(to_e164)
        if not parsed:
            return False
        body_text = (text or "").strip()[:1500]
        if not body_text:
            return False
        url = f"{GRAPH_BASE}/{self._phone_number_id}/messages"
        payload = {
            "messaging_product": "whatsapp",
            "to": parsed[0],
            "type": "text",
            "text": {"preview_url": False, "body": body_text},
        }
        status, response = self._post(url, json_body=payload)
        if 200 <= status < 300:
            self._record(True)
            return True
        logger.warning(
            "WhatsApp text reply failed for %s: %s",
            mask_phone(parsed[0]),
            _error_message(response),
        )
        return False


def _error_message(response: dict) -> str:
    error = response.get("error") if isinstance(response, dict) else None
    if isinstance(error, dict):
        message = str(error.get("message") or "")[:200]
        code = error.get("code")
        return f"{message} (code {code})" if code else message
    return "unknown WhatsApp API error"


def guess_mime_type(filename: str, content: bytes) -> str | None:
    """Sniff magic bytes directly; ERP sometimes returns an HTML error page as a PDF."""
    if not content:
        return None
    if content[:4] == b"%PDF":
        return "application/pdf"
    if content[:4] == b"PK\x03\x04":
        return "application/zip"
    if content[:3] == b"\xff\xd8\xff":
        return "image/jpeg"
    if content[:4] == b"\x89PNG":
        return "image/png"
    head = content[:256].lstrip()
    if head.startswith(b"<"):
        return None  # HTML/XML error page — never forward this
    guessed, _ = mimetypes.guess_type(filename)
    return guessed
