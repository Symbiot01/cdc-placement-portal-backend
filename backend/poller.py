"""Scheduled ERP notice poller & in-backend cron scheduler.

Runs automatically inside the FastAPI backend lifespan (every 15 or 30 minutes,
configurable via `CDC_POLL_INTERVAL_SECONDS`, default `900` = 15 min) OR via
`python -m backend.poller`, and performs:

  1. Reuse the warm ERP session via `holdSession.htm` (or complete a full
     password + security-question + Gmail OTP login only when ERP drops it).
  2. Read the jqGrid notice feed (`jqqueryid=54`).
  3. On first boot (when `notice_notifications` is empty and bootstrap seeding
     is enabled), baseline-seed existing historical notices so 100 old notices
     are never blasted to WhatsApp.
  4. For every genuinely new notice: fan out the approved WhatsApp template to
     configured member phone numbers, plus attach the verified PDF document.
  5. Record delivery in `notice_notifications` and execution telemetry in
     `poller_runs` so restarts and repeat polls are 100% idempotent.
"""

from __future__ import annotations

import argparse
import fcntl
import html
import logging
import os
import re
import sys
import threading
import time
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from backend.cache_store import CdcCacheStore
from backend.erp_login import (
    ErpCredentials,
    ErpSession,
    ErpSessionExpired,
    is_poller_enabled,
    load_credentials,
    load_dotenv_file,
)
from backend.gmail_otp import GmailOtpProvider
from backend.notifier import (
    NotifierError,
    WhatsAppNotifier,
    guess_mime_type,
)

logger = logging.getLogger("cdc_poller")

DEFAULT_DB_PATH = Path(
    os.environ.get(
        "CDC_CACHE_DB_PATH",
        str(Path(__file__).resolve().parent / "data" / "cdc_cache.sqlite3"),
    )
)
DEFAULT_POLL_INTERVAL_SECONDS = 900  # 15 minutes (set 1800 for 30 minutes)
MIN_POLL_INTERVAL_SECONDS = 300      # 5-minute safety floor to protect ERP
MAX_POLL_INTERVAL_SECONDS = 86400    # 24-hour ceiling
MIN_MANUAL_TRIGGER_COOLDOWN_SECONDS = 15.0
EXCERPT_CHARS = 480

_TAG_RE = re.compile(r"<[^>]+>")
_DOWNLOAD_RE = re.compile(
    r">\s*Download\s*<|title\s*=\s*['\"]Download['\"]|AdmFilePDF\.htm",
    re.IGNORECASE,
)
_TP_NOTICE_RE = re.compile(
    r"TPNotice\(\s*['\"]([^'\"]+)['\"]\s*,\s*['\"](\d+)['\"]\s*\)", re.IGNORECASE
)
_LEGACY_NOTICE_RE = re.compile(
    r"NoticeView\(\s*['\"](\d+)['\"]\s*,\s*['\"]([^'\"]+)['\"]\s*\)", re.IGNORECASE
)


@dataclass
class ParsedNotice:
    id: str
    type: str
    category: str
    company: str
    body: str
    when: str
    has_attachment: bool
    attachment_year: str

    def to_store_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "category": self.category,
            "company": self.company,
            "notice": self.body,
            "when": self.when,
        }


def _strip_html(value: str) -> str:
    with_breaks = re.sub(r"</?(?:br|p|div|li|tr)\b[^>]*>", " ", value or "", flags=re.IGNORECASE)
    text = _TAG_RE.sub(" ", with_breaks)
    text = html.unescape(text)
    return re.sub(r"\s+", " ", text).strip()


def parse_jqgrid_rows(payload: str) -> list[list[str]]:
    """Parse jqGrid XML `<cell><![CDATA[...]]></cell>` rows."""
    if not payload or not payload.strip():
        return []
    trimmed = payload.strip()
    start = trimmed.find("<rows")
    xml_body = trimmed[start:] if start >= 0 else trimmed
    try:
        root = ET.fromstring(xml_body)
    except ET.ParseError:
        # ERP occasionally truncates the feed; salvage complete rows instead of failing.
        rows: list[list[str]] = []
        for row_match in re.finditer(r"<row\b[^>]*>(.*?)</row>", xml_body, re.DOTALL):
            cells = re.findall(r"<cell>(.*?)</cell>", row_match.group(1), re.DOTALL)
            cells = [
                re.sub(r"^<!\[CDATA\[(.*?)\]\]>$", r"\1", c, flags=re.DOTALL) for c in cells
            ]
            if cells:
                rows.append(cells)
        return rows
    rows = []
    for row in root.findall("row"):
        cells = [cell.text or "" for cell in row.findall("cell")]
        rows.append(cells)
    return rows


def parse_notice_cells(index: int, cells: list[str]) -> ParsedNotice | None:
    """Mirror the frontend's notice parsing, including real TPNotice download detection."""
    padded = list(cells) + [""] * max(0, 8 - len(cells))
    raw_id = _strip_html(padded[0])
    notice_type = _strip_html(padded[1]).upper()
    category = _strip_html(padded[2])
    company = _strip_html(padded[3])
    notice_html = padded[4]
    when = _strip_html(padded[6])
    view_html = padded[7]

    body = _strip_html(notice_html)
    title_match = re.search(r'title=["\']([^"\']{20,})["\']', notice_html)
    if title_match:
        title_text = _strip_html(title_match.group(1))
        if len(title_text) > len(body):
            body = title_text

    tp_call = _TP_NOTICE_RE.search(view_html)
    legacy_call = _LEGACY_NOTICE_RE.search(view_html)
    has_attachment = bool(_DOWNLOAD_RE.search(view_html))
    attachment_year = (
        (tp_call.group(1) if tp_call else "")
        or (legacy_call.group(2) if legacy_call else "")
        or "2026-2027"
    )
    notice_id = (
        (tp_call.group(2) if tp_call else "")
        or (legacy_call.group(1) if legacy_call else "")
        or raw_id
    )
    if not notice_id:
        return None

    return ParsedNotice(
        id=notice_id.strip()[:32],
        type=notice_type,
        category=category or "General",
        company=company or "CDC General",
        body=body,
        when=when,
        has_attachment=has_attachment,
        attachment_year=attachment_year,
    )


def filter_placement_notices(
    notices: list[ParsedNotice], placement_only: bool = True
) -> list[ParsedNotice]:
    if not placement_only:
        return notices
    return [n for n in notices if n.type == "PLACEMENT"]


def build_excerpt(body: str, limit: int = EXCERPT_CHARS) -> str:
    text = re.sub(r"\s+", " ", body or "").strip()
    if not text:
        return "(no text in notice — see attached document)"
    if len(text) <= limit:
        return text
    return text[: limit - 1].rstrip() + "…"


def _fetch_grid_with_recovery(session: ErpSession, otp_provider: Any, rows: int = 100) -> str:
    """Fetch notice grid; if CDC sub-session expired, force one fresh login and retry."""
    session.ensure(otp_provider)
    try:
        session.open_placement()
        return session.fetch_notice_grid(rows=rows)
    except ErpSessionExpired:
        if not hasattr(session, "login"):
            raise
        logger.info("CDC placement sub-session expired; performing fresh ERP login and retrying")
        session.login(otp_provider)
        session.open_placement()
        return session.fetch_notice_grid(rows=rows)


def run_once(
    *,
    store: CdcCacheStore,
    session: ErpSession | None,
    notifier: WhatsAppNotifier | None,
    otp_provider: Any = None,
    placement_only: bool = True,
    send_attachments: bool = True,
    dry_run: bool = False,
    bootstrap_if_empty: bool = False,
    seed_only: bool = False,
    secret_values: list[str] | None = None,
) -> dict[str, Any]:
    """One poll cycle. Returns a sanitized summary dict; safe to call repeatedly."""
    summary: dict[str, Any] = {
        "polled_at": time.time(),
        "status": "ok",
        "total_notices": 0,
        "new_notices": 0,
        "seeded_notices": 0,
        "delivered": 0,
        "attachments_sent": 0,
        "errors": [],
    }

    def _add_error(msg: str) -> None:
        cleaned = _redact_secrets(msg, secret_values)
        summary["errors"].append(cleaned[:280])

    if session is None:
        summary["status"] = "unconfigured"
        _add_error("ERP session unavailable (CDC_ERP_ROLLNO / CDC_ERP_PASSWORD not configured)")
        return summary

    try:
        payload = _fetch_grid_with_recovery(session, otp_provider, rows=100)
    except Exception as exc:  # noqa: BLE001 - the cron must never crash on ERP flakiness
        logger.warning("ERP fetch failed: %s", _redact_secrets(str(exc), secret_values))
        summary["status"] = "erp_error"
        _add_error(f"ERP fetch failed: {exc}")
        return summary

    rows = parse_jqgrid_rows(payload)
    parsed = [n for n in (parse_notice_cells(i + 1, c) for i, c in enumerate(rows)) if n]
    # ERP sometimes repeats a row in a single feed; collapse by id before diffing.
    deduped: list[ParsedNotice] = []
    seen_ids: set[str] = set()
    for notice in parsed:
        if notice.id in seen_ids:
            continue
        seen_ids.add(notice.id)
        deduped.append(notice)
    notices = filter_placement_notices(deduped, placement_only)
    summary["total_notices"] = len(notices)

    # Keep the shared `notices` table in sync, then diff against `notice_notifications`.
    store.upsert_notices([n.to_store_dict() for n in notices])

    # Cold-start protection: on the very first run of a newly deployed cron, seed existing
    # historical notices instead of spamming ~100 old notices to WhatsApp.
    if notices and (seed_only or (bootstrap_if_empty and len(store.notified_notice_ids()) == 0)):
        seeded = store.mark_notices_notified(
            [n.to_store_dict() for n in notices],
            channel="bootstrap_seed",
            recipients=0,
            attachment_sent=False,
        )
        summary["seeded_notices"] = seeded
        summary["status"] = "bootstrapped"
        logger.info(
            "Bootstrapped %s existing CDC notices into ledger; future notices will alert WhatsApp",
            seeded,
        )
        return summary

    fresh = store.select_unnotified_notices([n.to_store_dict() for n in notices])
    fresh_ids = {item["id"] for item in fresh}
    # ERP orders newest-first; reverse so multiple new notices arrive chronologically on WhatsApp.
    fresh_parsed = list(reversed([n for n in notices if n.id in fresh_ids]))
    summary["new_notices"] = len(fresh_parsed)

    if not fresh_parsed:
        logger.info("No new CDC notices (%s scanned)", len(notices))
        return summary

    if dry_run or notifier is None:
        summary["delivered"] = 0
        summary["status"] = "dry_run" if dry_run else "notifier_unconfigured"
        _add_error("Notifier not configured" if notifier is None else "dry run")
        return summary

    if getattr(notifier, "recipients_count", 1) == 0:
        summary["delivered"] = 0
        summary["status"] = "waiting_for_subscribers"
        _add_error("No members have joined the WhatsApp broadcast link yet; notices queued until first member joins.")
        return summary

    for notice in fresh_parsed:
        try:
            excerpt_body = (
                f"[{notice.when}] {notice.body}" if notice.when and notice.body else notice.body
            )
            excerpt = build_excerpt(excerpt_body)

            content = b""
            mime: str | None = None
            safe_filename = f"{notice.company}_{notice.category}_{notice.id}.pdf"

            if send_attachments and notice.has_attachment:
                year = notice.attachment_year or "2026-2027"
                try:
                    content = session.fetch_attachment(notice.id, year)
                except Exception as exc:  # noqa: BLE001
                    _add_error(f"attachment {notice.id} download failed: {exc}")
                    content = b""

                mime = guess_mime_type(safe_filename, content) if content else None
                if content and mime != "application/pdf":
                    logger.warning("Discarding unusable attachment for notice %s", notice.id)
                    _add_error(f"attachment {notice.id} was not a valid PDF")
                    content = b""
                    mime = None

            use_doc_template = bool(
                content
                and mime == "application/pdf"
                and getattr(notifier, "doc_template_name", "")
                and hasattr(notifier, "send_notice_with_document_template")
            )

            attachment_was_sent = False
            if use_doc_template:
                media_id = notifier.upload_media(content, safe_filename, "application/pdf")
                if media_id:
                    result = notifier.send_notice_with_document_template(
                        company=notice.company,
                        category=notice.category,
                        excerpt=excerpt,
                        media_id=media_id,
                        filename=safe_filename,
                    )
                    attachment_was_sent = result.attachment_sent
                    if attachment_was_sent:
                        summary["attachments_sent"] += 1
                else:
                    # Fallback to standard text template so the notice text is still delivered
                    result = notifier.send_notice_template(
                        company=notice.company,
                        category=notice.category,
                        excerpt=excerpt,
                    )
            else:
                result = notifier.send_notice_template(
                    company=notice.company,
                    category=notice.category,
                    excerpt=excerpt,
                )
                if content and mime == "application/pdf" and result.delivered > 0:
                    media_id = notifier.upload_media(content, safe_filename, "application/pdf")
                    if media_id:
                        doc_result = notifier.send_document(
                            media_id=media_id,
                            filename=safe_filename,
                            caption=f"CDC notice #{notice.id} — {notice.company}",
                        )
                        if doc_result.attachment_sent:
                            attachment_was_sent = True
                            summary["attachments_sent"] += 1
                        if doc_result.failed:
                            for err_item in doc_result.failed[:2]:
                                _add_error(err_item)

            if result.delivered:
                summary["delivered"] += result.delivered
            if result.failed:
                for err_item in result.failed[:3]:
                    _add_error(err_item)

            # CRITICAL: Only mark notified when at least 1 recipient succeeded.
            # If 0 succeeded (e.g., temporary outage/expired token), keep unnotified to retry next tick!
            if result.delivered > 0:
                store.mark_notices_notified(
                    [notice.to_store_dict()],
                    channel="whatsapp",
                    recipients=result.delivered,
                    attachment_sent=attachment_was_sent,
                )
        except NotifierError as exc:
            _add_error(str(exc))
            break
        except Exception as exc:  # noqa: BLE001
            logger.exception("Unhandled error while notifying notice %s", notice.id)
            _add_error(f"notice {notice.id}: {exc}")

    if summary["errors"] and summary["delivered"] == 0 and summary["new_notices"] > 0:
        summary["status"] = "delivery_failed"
    elif summary["errors"]:
        summary["status"] = "partial"
    return summary


def _redact_secrets(text: str, secret_values: list[str] | None = None) -> str:
    out = str(text or "")
    for secret in secret_values or []:
        clean = (secret or "").strip()
        if len(clean) >= 4 and clean in out:
            out = out.replace(clean, "[REDACTED]")
    return out


def resolve_poll_interval_seconds(root_dir: Path | None = None) -> int:
    base = root_dir or Path(__file__).resolve().parent.parent
    merged = {
        **load_dotenv_file(base / ".env"),
        **load_dotenv_file(base / ".enverp"),
    }
    raw = (
        os.environ.get("CDC_POLL_INTERVAL_SECONDS")
        or merged.get("CDC_POLL_INTERVAL_SECONDS")
        or str(DEFAULT_POLL_INTERVAL_SECONDS)
    ).strip()
    try:
        parsed = int(raw)
    except ValueError:
        parsed = DEFAULT_POLL_INTERVAL_SECONDS
    return max(MIN_POLL_INTERVAL_SECONDS, min(MAX_POLL_INTERVAL_SECONDS, parsed))


def resolve_bootstrap_seed(root_dir: Path | None = None) -> bool:
    base = root_dir or Path(__file__).resolve().parent.parent
    merged = {
        **load_dotenv_file(base / ".env"),
        **load_dotenv_file(base / ".enverp"),
    }
    raw = os.environ.get("CDC_POLLER_BOOTSTRAP_SEED")
    if raw is None:
        raw = merged.get("CDC_POLLER_BOOTSTRAP_SEED", "1")
    return str(raw).strip().lower() not in {"0", "false", "no", "off"}


class PollerScheduler:
    """Thread-safe, cross-worker file-locked background cron scheduler for FastAPI."""

    def __init__(
        self,
        store: CdcCacheStore,
        *,
        root_dir: Path | None = None,
        interval_seconds: int | None = None,
    ) -> None:
        self._store = store
        self._root_dir = root_dir or Path(__file__).resolve().parent.parent
        self._interval_seconds = (
            max(MIN_POLL_INTERVAL_SECONDS, min(MAX_POLL_INTERVAL_SECONDS, int(interval_seconds)))
            if interval_seconds is not None
            else resolve_poll_interval_seconds(self._root_dir)
        )
        self._cookie_jar_path = store.db_path.parent / ".erp_cookies.lwp"
        self._lock_file_path = store.db_path.parent / ".poller.lock"
        self._lock_fd: int | None = None

        self._run_lock = threading.Lock()
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._session: ErpSession | None = None
        self._session_rollno: str = ""
        self._last_manual_trigger_mono: float = 0.0

    @property
    def is_running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    @property
    def interval_seconds(self) -> int:
        return self._interval_seconds

    def _acquire_leader_lock(self) -> bool:
        """Ensure only ONE worker process starts the background cron thread."""
        if self._lock_fd is not None:
            return True
        try:
            self._lock_file_path.parent.mkdir(parents=True, exist_ok=True)
            fd = os.open(str(self._lock_file_path), os.O_CREAT | os.O_RDWR, 0o600)
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            self._lock_fd = fd
            return True
        except OSError:
            try:
                os.close(fd)  # type: ignore[possibly-undefined]
            except Exception:  # noqa: BLE001
                pass
            return False

    def _release_leader_lock(self) -> None:
        if self._lock_fd is None:
            return
        try:
            fcntl.flock(self._lock_fd, fcntl.LOCK_UN)
        except OSError:
            pass
        try:
            os.close(self._lock_fd)
        except OSError:
            pass
        self._lock_fd = None

    def _build_runtime_components(
        self,
    ) -> tuple[ErpCredentials | None, ErpSession | None, GmailOtpProvider | None, WhatsAppNotifier | None, list[str]]:
        creds = load_credentials(self._root_dir)
        otp_provider = GmailOtpProvider.from_env(self._root_dir)
        dynamic_subscribers = self._store.list_active_subscriber_phones()
        notifier = WhatsAppNotifier.from_env(
            self._root_dir,
            extra_recipients=dynamic_subscribers,
        )

        secrets_to_redact: list[str] = []
        if creds is not None:
            secrets_to_redact.append(creds.password)
            secrets_to_redact.extend(creds.security_answers.values())
            if self._session is None or self._session_rollno != creds.rollno:
                self._session = ErpSession(creds, cookie_jar_path=self._cookie_jar_path)
                self._session_rollno = creds.rollno
        else:
            self._session = None
            self._session_rollno = ""

        return creds, self._session, otp_provider, notifier, secrets_to_redact

    def start_if_enabled(self) -> bool:
        """Start the background daemon loop if `CDC_ERP_ENABLED=1` and this worker wins leader lock."""
        if not is_poller_enabled(self._root_dir):
            logger.info("CDC background notice cron disabled (set CDC_ERP_ENABLED=1 to enable)")
            return False
        if self.is_running:
            return True
        if not self._acquire_leader_lock():
            logger.info("Another Uvicorn worker owns the CDC notice cron lock; skipping duplicate scheduler")
            return False

        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._loop,
            name="cdc-whatsapp-poller",
            daemon=True,
        )
        self._thread.start()
        logger.info(
            "Started CDC -> WhatsApp background cron job (every %ss / %.1f min)",
            self._interval_seconds,
            self._interval_seconds / 60.0,
        )
        return True

    def stop(self, timeout: float = 5.0) -> None:
        self._stop_event.set()
        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=timeout)
        self._thread = None
        self._release_leader_lock()

    def _loop(self) -> None:
        while not self._stop_event.is_set():
            if is_poller_enabled(self._root_dir):
                try:
                    self.execute_cycle(trigger_source="cron")
                except Exception:  # noqa: BLE001
                    logger.exception("Scheduled CDC poll cycle encountered an unexpected error")
            # Sleep in interruptible wait so shutdown is immediate
            if self._stop_event.wait(timeout=float(self._interval_seconds)):
                break

    def execute_cycle(
        self,
        *,
        trigger_source: str = "cron",
        dry_run: bool = False,
        seed_only: bool = False,
    ) -> dict[str, Any]:
        """Run one single-flight poll cycle and record sanitized telemetry."""
        acquired = self._run_lock.acquire(blocking=False)
        if not acquired:
            return {
                "polled_at": time.time(),
                "trigger_source": trigger_source,
                "status": "busy",
                "total_notices": 0,
                "new_notices": 0,
                "seeded_notices": 0,
                "delivered": 0,
                "attachments_sent": 0,
                "errors": ["A poll cycle is already in progress."],
            }
        try:
            if trigger_source == "manual_api":
                now_mono = time.monotonic()
                if (
                    self._last_manual_trigger_mono > 0
                    and (now_mono - self._last_manual_trigger_mono) < MIN_MANUAL_TRIGGER_COOLDOWN_SECONDS
                ):
                    wait_s = int(
                        MIN_MANUAL_TRIGGER_COOLDOWN_SECONDS - (now_mono - self._last_manual_trigger_mono)
                    ) + 1
                    return {
                        "polled_at": time.time(),
                        "trigger_source": trigger_source,
                        "status": "cooldown",
                        "total_notices": 0,
                        "new_notices": 0,
                        "seeded_notices": 0,
                        "delivered": 0,
                        "attachments_sent": 0,
                        "errors": [f"Manual trigger cooldown active; retry in {wait_s}s."],
                    }
                self._last_manual_trigger_mono = now_mono

            _, session, otp_provider, notifier, secrets_to_redact = self._build_runtime_components()
            bootstrap = resolve_bootstrap_seed(self._root_dir)
            summary = run_once(
                store=self._store,
                session=session,
                notifier=notifier,
                otp_provider=otp_provider,
                placement_only=True,
                send_attachments=True,
                dry_run=dry_run,
                bootstrap_if_empty=bootstrap,
                seed_only=seed_only,
                secret_values=secrets_to_redact,
            )
            return self._store.record_poller_run(summary, trigger_source=trigger_source)
        finally:
            self._run_lock.release()

    def status_snapshot(self) -> dict[str, Any]:
        """Build a zero-secret operational snapshot for the protected admin endpoint."""
        creds = load_credentials(self._root_dir)
        otp_provider = GmailOtpProvider.from_env(self._root_dir)
        dynamic_subscribers = self._store.list_active_subscriber_phones()
        notifier = WhatsAppNotifier.from_env(
            self._root_dir,
            extra_recipients=dynamic_subscribers,
        )
        merged_env = {
            **load_dotenv_file(self._root_dir / ".env"),
            **load_dotenv_file(self._root_dir / ".enverp"),
        }
        join_code_set = bool(
            (os.environ.get("WA_JOIN_CODE") or merged_env.get("WA_JOIN_CODE") or "").strip()
        )
        business_phone_set = bool(
            (
                os.environ.get("WA_BUSINESS_PHONE_NUMBER")
                or merged_env.get("WA_BUSINESS_PHONE_NUMBER")
                or ""
            ).strip()
        )
        subscribers_summary = self._store.list_whatsapp_subscribers_summary()
        active_subscribers_count = sum(1 for s in subscribers_summary if s["subscribed"])

        optin_windows = (
            self._store.check_recipients_optin_windows(notifier.raw_recipients)
            if notifier is not None
            else []
        )

        return {
            "enabled": is_poller_enabled(self._root_dir),
            "scheduler_running": self.is_running,
            "interval_seconds": self._interval_seconds,
            "interval_minutes": round(self._interval_seconds / 60.0, 1),
            "bootstrap_seed_on_empty": resolve_bootstrap_seed(self._root_dir),
            "erp": {
                "configured": creds is not None,
                "masked_rollno": creds.masked_rollno() if creds is not None else None,
                "security_answers_count": len(creds.security_answers) if creds is not None else 0,
                "warm_cookie_jar_exists": self._cookie_jar_path.is_file(),
            },
            "gmail_otp": {
                "configured": otp_provider is not None,
            },
            "whatsapp": {
                "configured": notifier is not None,
                "mode": "join_link_broadcast" if join_code_set else "recipient_fanout",
                "join_link_configured": join_code_set and business_phone_set,
                "join_code_configured": join_code_set,
                "template_name": notifier.template_name if notifier is not None else None,
                "doc_template_name": (
                    notifier.doc_template_name if notifier is not None and notifier.doc_template_name else None
                ),
                "subscribers_active_count": active_subscribers_count,
                "recipients_count": notifier.recipients_count if notifier is not None else 0,
                "recipients": optin_windows,
                "subscribers": subscribers_summary,
            },
            "ledger": {
                "notified_notices_total": len(self._store.notified_notice_ids()),
                "cached_notices_total": len(self._store.existing_notice_ids()),
            },
            "recent_runs": self._store.get_recent_poller_runs(limit=10),
        }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Poll IIT KGP ERP for new CDC notices and alert WhatsApp."
    )
    parser.add_argument("--once", action="store_true", help="Run a single poll cycle and exit")
    parser.add_argument("--dry-run", action="store_true", help="Parse and diff without sending")
    parser.add_argument(
        "--seed-only",
        action="store_true",
        help="Baseline-seed all current ERP notices as notified without sending WhatsApp alerts",
    )
    parser.add_argument(
        "--no-bootstrap-seed",
        action="store_true",
        help="Disable automatic cold-start seeding when notice_notifications is empty",
    )
    parser.add_argument(
        "--all-notice-types", action="store_true", help="Include non-placement notices"
    )
    parser.add_argument("--no-attachments", action="store_true", help="Skip PDF attachments")
    parser.add_argument(
        "--interval",
        type=int,
        default=resolve_poll_interval_seconds(),
        help="Seconds between cycles (default from CDC_POLL_INTERVAL_SECONDS or 900 = 15m)",
    )
    parser.add_argument("--db", default=str(DEFAULT_DB_PATH), help="SQLite cache path")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        stream=sys.stdout,
    )

    if not is_poller_enabled() and not args.dry_run and not args.seed_only:
        logger.info("CDC_ERP_ENABLED is not set — poller is disabled")
        return 0

    credentials: ErpCredentials | None = load_credentials()
    if credentials is None:
        logger.error("ERP credentials are missing; set CDC_ERP_ROLLNO and CDC_ERP_PASSWORD")
        return 1
    if not credentials.security_answers:
        logger.warning("No ERP security answers configured; a login will fail if ERP asks")

    otp_provider = GmailOtpProvider.from_env()
    if otp_provider is None:
        logger.warning("Gmail OAuth is not configured; only keepalive logins are possible")

    db_path = Path(args.db)
    store = CdcCacheStore(db_path)

    def _make_cli_notifier() -> WhatsAppNotifier | None:
        return WhatsAppNotifier.from_env(extra_recipients=store.list_active_subscriber_phones())

    notifier = _make_cli_notifier()
    if notifier is None and not args.dry_run and not args.seed_only:
        logger.error(
            "WhatsApp is not configured; set WA_ACCESS_TOKEN/WA_PHONE_NUMBER_ID/WA_TEMPLATE_NAME "
            "plus WA_JOIN_CODE (or WA_RECIPIENTS)"
        )
        return 1

    cookie_jar_path = db_path.parent / ".erp_cookies.lwp"
    session = ErpSession(credentials, cookie_jar_path=cookie_jar_path)
    secrets_to_redact = [credentials.password, *credentials.security_answers.values()]

    def cycle() -> dict[str, Any]:
        bootstrap = (not args.no_bootstrap_seed) and resolve_bootstrap_seed()
        live_notifier = _make_cli_notifier()
        summary = run_once(
            store=store,
            session=session,
            notifier=live_notifier,
            otp_provider=otp_provider,
            placement_only=not args.all_notice_types,
            send_attachments=not args.no_attachments,
            dry_run=args.dry_run,
            bootstrap_if_empty=bootstrap,
            seed_only=args.seed_only,
            secret_values=secrets_to_redact,
        )
        store.record_poller_run(summary, trigger_source="cli")
        return summary

    if args.once or args.seed_only:
        summary = cycle()
        logger.info("Poll complete: %s", summary)
        return 0 if not summary["errors"] or args.dry_run else 1

    interval = max(MIN_POLL_INTERVAL_SECONDS, min(MAX_POLL_INTERVAL_SECONDS, int(args.interval)))
    logger.info("Starting CDC notice poller every %ss", interval)
    while True:
        try:
            summary = cycle()
            logger.info(
                "Poll: status=%s scanned=%s new=%s seeded=%s delivered=%s attachments=%s errors=%s",
                summary["status"],
                summary["total_notices"],
                summary["new_notices"],
                summary["seeded_notices"],
                summary["delivered"],
                summary["attachments_sent"],
                len(summary["errors"]),
            )
        except KeyboardInterrupt:
            logger.info("Poller stopped")
            return 0
        except Exception:  # noqa: BLE001 - a long-running loop must survive any single failure
            logger.exception("Poll cycle crashed; continuing")
        time.sleep(interval)


if __name__ == "__main__":
    raise SystemExit(main())
