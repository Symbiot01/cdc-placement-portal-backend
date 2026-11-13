"""Tests for the ERP notice poller, WhatsApp notifier, and credential handling.

Everything here runs fully offline: ERP, Gmail, and the WhatsApp Graph API are
all mocked. No real credentials are used or required.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from backend.cache_store import CdcCacheStore
from backend.erp_login import ErpCredentials, ErpSession, load_credentials
from backend.gmail_otp import _extract_message_text, _looks_like_erp_mail
from backend.notifier import (
    DeliveryResult,
    WhatsAppNotifier,
    guess_mime_type,
    parse_recipients,
)
from backend.poller import (
    build_excerpt,
    parse_jqgrid_rows,
    parse_notice_cells,
    run_once,
)

# ---------------------------------------------------------------- fixtures


def _notice_xml(*rows: str) -> str:
    body = "".join(rows)
    return f'<?xml version="1.0" encoding="UTF-8"?><rows><page>1</page>{body}</rows>'


def _notice_row(
    *,
    notice_id: str,
    company: str,
    category: str,
    body: str,
    with_download: bool,
    notice_type: str = "PLACEMENT",
    year: str = "2026-2027",
) -> str:
    view = (
        f"<a title='Download' href='javascript:void(0);' "
        f"onclick='TPNotice(\"{year}\",\"{notice_id}\");'>Download</a>"
        if with_download
        else f"<a title='' href='javascript:void(0);' onclick='TPNotice(\"{year}\",\"{notice_id}\");'></a>"
    )
    cells = [
        notice_id,
        notice_type,
        category,
        company,
        f"<a>{body}</a>",
        "CDC",
        "11-11-2026 09:00",
        view,
    ]
    return "<row id='0'>" + "".join(f"<cell><![CDATA[{c}]]></cell>" for c in cells) + "</row>"


class FakeSession:
    """Stands in for ErpSession with a controllable feed and attachment bytes."""

    def __init__(self, payload: str, attachment: bytes = b"%PDF-1.7\n%%EOF"):
        self.payload = payload
        self.attachment = attachment
        self.ensured = 0
        self.opened = 0
        self.attachment_calls = 0

    def ensure(self, otp_provider: Any = None) -> None:
        self.ensured += 1

    def open_placement(self) -> None:
        self.opened += 1

    def fetch_notice_grid(self, rows: int = 100) -> str:
        return self.payload

    def fetch_attachment(self, notice_id: str, year: str) -> bytes:
        self.attachment_calls += 1
        return self.attachment


class FakeNotifier:
    def __init__(self):
        self.templates: list[dict[str, Any]] = []
        self.documents: list[dict[str, Any]] = []
        self.uploads: list[bytes] = []

    def send_notice_template(self, *, company: str, category: str, excerpt: str) -> DeliveryResult:
        self.templates.append({"company": company, "category": category, "excerpt": excerpt})
        return DeliveryResult(attempted=1, delivered=1)

    def upload_media(self, content: bytes, filename: str, mime_type: str) -> str:
        self.uploads.append(content)
        return f"media-{len(self.uploads)}"

    def send_document(self, *, media_id: str, filename: str, caption: str) -> DeliveryResult:
        self.documents.append({"media_id": media_id, "filename": filename, "caption": caption})
        return DeliveryResult(attempted=1, delivered=1, attachment_sent=True)


@pytest.fixture()
def store(tmp_path: Path) -> CdcCacheStore:
    return CdcCacheStore(tmp_path / "cache.sqlite3")


# ------------------------------------------------------------ parsing tests


def test_parses_real_jqgrid_notice_feed_and_attachment_flag() -> None:
    xml = _notice_xml(
        _notice_row(
            notice_id="1569",
            company="Airbus India Private Limited",
            category="Shortlist",
            body="Shortlisted candidates for the Associate Engineer role",
            with_download=True,
        ),
        _notice_row(
            notice_id="1567",
            company="IQVIA",
            category="Announcement",
            body="A notice with no uploaded file",
            with_download=False,
        ),
        _notice_row(
            notice_id="1555",
            company="Google",
            category="Internship",
            body="Internship drive",
            with_download=False,
            notice_type="INTERNSHIP",
        ),
    )
    rows = parse_jqgrid_rows(xml)
    assert len(rows) == 3

    first = parse_notice_cells(1, rows[0])
    assert first is not None
    assert first.id == "1569"
    assert first.company == "Airbus India Private Limited"
    assert first.has_attachment is True
    assert first.attachment_year == "2026-2027"

    second = parse_notice_cells(2, rows[1])
    assert second is not None
    # Empty <a></a> must never be reported as an attachment.
    assert second.has_attachment is False


def test_parse_jqgrid_salvages_truncated_feed() -> None:
    truncated = (
        '<?xml version="1.0"?><rows><page>1</page>'
        "<row id='0'><cell><![CDATA[1]]></cell><cell><![CDATA[PLACEMENT]]></cell>"
        "<cell><![CDATA[x]]></cell><cell><![CDATA[Y]]></cell><cell><![CDATA[z]]></cell>"
        "<cell><![CDATA[CDC]]></cell><cell><![CDATA[now]]></cell><cell><![CDATA[]]></cell></row>"
    )
    rows = parse_jqgrid_rows(truncated)
    assert len(rows) == 1
    assert parse_notice_cells(1, rows[0]).id == "1"


def test_excerpt_is_clamped_and_single_line() -> None:
    excerpt = build_excerpt("Line one\n\nLine two   with   spaces" * 80)
    assert "\n" not in excerpt
    assert len(excerpt) <= 480
    assert excerpt.endswith("…")


# -------------------------------------------------- idempotency (critical)


def test_no_new_notices_produces_no_send(store: CdcCacheStore) -> None:
    xml = _notice_xml(
        _notice_row(
            notice_id="100",
            company="Amazon",
            category="Shortlist",
            body="Existing notice",
            with_download=False,
        )
    )
    notifier = FakeNotifier()
    session = FakeSession(xml)

    first = run_once(store=store, session=session, notifier=notifier)
    assert first["new_notices"] == 1
    assert len(notifier.templates) == 1

    # Second cycle with the identical feed must stay silent.
    second = run_once(store=store, session=session, notifier=notifier)
    assert second["new_notices"] == 0
    assert len(notifier.templates) == 1


def test_only_the_brand_new_notice_is_announced(store: CdcCacheStore) -> None:
    session = FakeSession(
        _notice_xml(
            _notice_row(
                notice_id="200",
                company="Airbus",
                category="Shortlist",
                body="Old notice",
                with_download=False,
            )
        )
    )
    notifier = FakeNotifier()
    run_once(store=store, session=session, notifier=notifier)
    assert len(notifier.templates) == 1

    session.payload = _notice_xml(
        _notice_row(
            notice_id="200",
            company="Airbus",
            category="Shortlist",
            body="Old notice",
            with_download=False,
        ),
        _notice_row(
            notice_id="201",
            company="Deloitte",
            category="PPT",
            body="Pre-placement talk rescheduled",
            with_download=False,
        ),
    )
    summary = run_once(store=store, session=session, notifier=notifier)
    assert summary["new_notices"] == 1
    assert len(notifier.templates) == 2
    assert notifier.templates[-1]["company"] == "Deloitte"


def test_duplicate_ids_within_one_feed_are_sent_once(store: CdcCacheStore) -> None:
    row = _notice_row(
        notice_id="300",
        company="IQVIA",
        category="Test",
        body="Written test link",
        with_download=False,
    )
    session = FakeSession(_notice_xml(row, row))
    notifier = FakeNotifier()
    summary = run_once(store=store, session=session, notifier=notifier)
    assert summary["new_notices"] == 1
    assert len(notifier.templates) == 1


def test_placement_only_filter_is_respected(store: CdcCacheStore) -> None:
    xml = _notice_xml(
        _notice_row(
            notice_id="400",
            company="P",
            category="x",
            body="placement",
            with_download=False,
            notice_type="PLACEMENT",
        ),
        _notice_row(
            notice_id="401",
            company="I",
            category="y",
            body="internship",
            with_download=False,
            notice_type="INTERNSHIP",
        ),
    )
    notifier = FakeNotifier()
    run_once(store=store, session=FakeSession(xml), notifier=notifier, placement_only=True)
    assert len(notifier.templates) == 1
    assert notifier.templates[0]["company"] == "P"


# ------------------------------------------------------ attachment handling


def test_attachment_is_sent_for_notice_with_download_link(store: CdcCacheStore) -> None:
    xml = _notice_xml(
        _notice_row(
            notice_id="500",
            company="Airbus",
            category="Shortlist",
            body="See attached shortlist",
            with_download=True,
        )
    )
    notifier = FakeNotifier()
    session = FakeSession(xml, attachment=b"%PDF-1.7\nshortlist\n%%EOF")
    summary = run_once(store=store, session=session, notifier=notifier)

    assert session.attachment_calls == 1
    assert len(notifier.uploads) == 1
    assert notifier.documents[0]["filename"].endswith(".pdf")
    assert summary["attachments_sent"] == 1


def test_corrupt_attachment_is_never_forwarded(store: CdcCacheStore) -> None:
    xml = _notice_xml(
        _notice_row(
            notice_id="600",
            company="Airbus",
            category="Shortlist",
            body="Attachment is an ERP error page",
            with_download=True,
        )
    )
    notifier = FakeNotifier()
    # ERP returns an HTML 500 page instead of a PDF.
    session = FakeSession(xml, attachment=b"<html><body>500 Internal Server</body></html>")
    summary = run_once(store=store, session=session, notifier=notifier)

    assert notifier.uploads == []
    assert notifier.documents == []
    assert summary["attachments_sent"] == 0
    assert any("not a valid PDF" in e for e in summary["errors"])


def test_no_attachment_fetch_without_download_link(store: CdcCacheStore) -> None:
    xml = _notice_xml(
        _notice_row(
            notice_id="700",
            company="Airbus",
            category="Announcement",
            body="no file",
            with_download=False,
        )
    )
    session = FakeSession(xml)
    run_once(store=store, session=session, notifier=FakeNotifier())
    assert session.attachment_calls == 0


# ------------------------------------------------------------ notifier rules


def test_parse_recipients_validates_e164_and_dedupes() -> None:
    parsed = parse_recipients("+919876543210, 9876543210, +91 98765 43210, +14155552671, nonsense")
    assert parsed == ["+919876543210", "+14155552671"]


def test_notifier_requires_template_name() -> None:
    from backend.notifier import NotifierError

    with pytest.raises(NotifierError):
        WhatsAppNotifier("token", "phoneid", ["+919876543210"], "")


def test_notifier_rejects_empty_recipients() -> None:
    from backend.notifier import NotifierError

    with pytest.raises(NotifierError):
        WhatsAppNotifier("token", "phoneid", [], "cdc_new_notice")


def test_template_params_are_single_line_and_clamped() -> None:
    notifier = WhatsAppNotifier("token", "phoneid", ["+919876543210"], "cdc_new_notice")
    captured: dict[str, Any] = {}

    def fake_post(url: str, *, json_body: dict | None = None, data: bytes | None = None,
                  headers: dict | None = None) -> tuple[int, dict]:
        captured["body"] = json_body
        return 200, {"messages": [{"id": "wamid.1"}]}

    notifier._post = fake_post  # type: ignore[assignment]
    result = notifier.send_notice_template(
        company="Airbus India\nPrivate Limited",
        category="Shortlist\n",
        excerpt="x" * 2000,
    )
    assert result.delivered == 1
    params = captured["body"]["template"]["components"][0]["parameters"]
    assert "\n" not in params[0]["text"]
    assert len(params[2]["text"]) <= 500


def test_fanout_failure_does_not_block_other_recipients() -> None:
    notifier = WhatsAppNotifier(
        "token", "phoneid", ["+919876543210", "+919876543211"], "cdc_new_notice"
    )
    calls: list[str] = []

    def fake_post(url: str, *, json_body: dict | None = None, data: bytes | None = None,
                  headers: dict | None = None) -> tuple[int, dict]:
        calls.append(json_body["to"])
        if json_body["to"].endswith("210"):
            return 400, {"error": {"message": "recipient not opted in", "code": 131047}}
        return 200, {"messages": [{"id": "wamid"}]}

    notifier._post = fake_post  # type: ignore[assignment]
    result = notifier.send_notice_template(company="A", category="B", excerpt="C")
    assert len(calls) == 2
    assert result.attempted == 2
    assert result.delivered == 1
    assert len(result.failed) == 1


def test_circuit_breaker_opens_after_repeated_failures() -> None:
    notifier = WhatsAppNotifier("token", "phoneid", ["+919876543210"], "cdc_new_notice")

    def always_fail(url: str, **kwargs: Any) -> tuple[int, dict]:
        return 500, {"error": {"message": "boom", "code": 1}}

    notifier._post = always_fail  # type: ignore[assignment]
    for _ in range(3):
        notifier.send_notice_template(company="A", category="B", excerpt="C")
    assert notifier._circuit_open() is True

    # While open, no HTTP call should be attempted.
    called = False

    def should_not_run(url: str, **kwargs: Any) -> tuple[int, dict]:
        nonlocal called
        called = True
        return 200, {}

    notifier._post = should_not_run  # type: ignore[assignment]
    result = notifier.send_notice_template(company="A", category="B", excerpt="C")
    assert called is False
    assert result.delivered == 0


def test_guess_mime_rejects_html_error_pages() -> None:
    assert guess_mime_type("n.pdf", b"%PDF-1.7") == "application/pdf"
    assert guess_mime_type("n.pdf", b"<html>500</html>") is None
    assert guess_mime_type("n.png", b"\x89PNG\r\n") == "image/png"


# ------------------------------------------------------ credential isolation


def test_load_credentials_returns_none_without_secrets(monkeypatch: pytest.MonkeyPatch,
                                                       tmp_path: Path) -> None:
    for key in ("CDC_ERP_ROLLNO", "CDC_ERP_PASSWORD", "ROLLNO", "PASSWORD"):
        monkeypatch.delenv(key, raising=False)
    assert load_credentials(tmp_path) is None


def test_load_credentials_prefers_env_json_answers(monkeypatch: pytest.MonkeyPatch,
                                                   tmp_path: Path) -> None:
    monkeypatch.setenv("CDC_ERP_ROLLNO", "22CS10001")
    monkeypatch.setenv("CDC_ERP_PASSWORD", "secret-value")
    monkeypatch.setenv(
        "CDC_ERP_SECURITY_ANSWERS",
        json.dumps({"What is your best anime movie?": "Spirited Away"}),
    )
    creds = load_credentials(tmp_path)
    assert creds is not None
    assert creds.rollno == "22CS10001"
    # Answer lookup tolerates punctuation/whitespace drift from ERP.
    assert creds.lookup_answer("what is your BEST anime movie") == "Spirited Away"
    assert creds.lookup_answer("unrelated question") is None


def test_credentials_never_appear_in_error_messages(monkeypatch: pytest.MonkeyPatch) -> None:
    creds = ErpCredentials(rollno="22CS10001", password="hunter2-very-secret", security_answers={})
    session = ErpSession(creds)

    def boom(*args: Any, **kwargs: Any) -> tuple[int, str, str]:
        return 200, "https://erp.iitkgp.ac.in/IIT_ERP3/home.htm", "FALSE"

    session._request = boom  # type: ignore[assignment]
    with pytest.raises(Exception) as excinfo:
        session._fetch_matching_question()
    assert "hunter2" not in str(excinfo.value)


def test_gmail_mail_heuristic_ignores_non_erp_codes() -> None:
    assert _looks_like_erp_mail("Your ERP OTP is 123456") is True
    assert _looks_like_erp_mail("Your Amazon OTP is 123456") is False


def test_gmail_body_text_extraction_handles_nested_parts() -> None:
    import base64 as b64

    def enc(text: str) -> str:
        return b64.urlsafe_b64encode(text.encode()).decode().rstrip("=")

    message = {
        "payload": {
            "parts": [
                {"body": {"data": enc("Your ERP OTP is ")}},
                {"parts": [{"body": {"data": enc("654321")}}]},
            ]
        }
    }
    text = _extract_message_text(message)
    assert "654321" in text


# ------------------------------------ cron, bootstrap, OTP & API security tests


def test_bootstrap_if_empty_seeds_history_without_spamming_and_alerts_next_notice(
    store: CdcCacheStore,
) -> None:
    initial_xml = _notice_xml(
        _notice_row(
            notice_id="901",
            company="OldCorp1",
            category="Shortlist",
            body="Historical notice 1",
            with_download=False,
        ),
        _notice_row(
            notice_id="900",
            company="OldCorp0",
            category="PPT",
            body="Historical notice 0",
            with_download=False,
        ),
    )
    session = FakeSession(initial_xml)
    notifier = FakeNotifier()

    boot = run_once(
        store=store,
        session=session,
        notifier=notifier,
        bootstrap_if_empty=True,
    )
    assert boot["status"] == "bootstrapped"
    assert boot["seeded_notices"] == 2
    assert boot["delivered"] == 0
    assert notifier.templates == []

    # Next cron cycle 15 minutes later: 2 new notices arrive (903 newest at top, 902 older below)
    session.payload = _notice_xml(
        _notice_row(
            notice_id="903",
            company="NewestCorp",
            category="Shortlist",
            body="Second new notice",
            with_download=False,
        ),
        _notice_row(
            notice_id="902",
            company="EarlierNewCorp",
            category="Announcement",
            body="First new notice",
            with_download=False,
        ),
        _notice_row(
            notice_id="901",
            company="OldCorp1",
            category="Shortlist",
            body="Historical notice 1",
            with_download=False,
        ),
    )
    nxt = run_once(
        store=store,
        session=session,
        notifier=notifier,
        bootstrap_if_empty=True,
    )
    assert nxt["new_notices"] == 2
    assert nxt["delivered"] == 2
    # Sent chronologically: 902 first, 903 second
    assert [t["company"] for t in notifier.templates] == ["EarlierNewCorp", "NewestCorp"]


def test_zero_delivery_does_not_swallow_notice_so_next_tick_retries(
    store: CdcCacheStore,
) -> None:
    xml = _notice_xml(
        _notice_row(
            notice_id="950",
            company="RetryCorp",
            category="Shortlist",
            body="Transient failure test",
            with_download=False,
        )
    )
    session = FakeSession(xml)

    class FailingThenWorkingNotifier(FakeNotifier):
        def __init__(self) -> None:
            super().__init__()
            self.fail_now = True

        def send_notice_template(self, *, company: str, category: str, excerpt: str) -> DeliveryResult:
            if self.fail_now:
                return DeliveryResult(attempted=2, delivered=0, failed=["+91******0001: timeout"])
            return super().send_notice_template(company=company, category=category, excerpt=excerpt)

    notifier = FailingThenWorkingNotifier()
    first = run_once(store=store, session=session, notifier=notifier)
    assert first["status"] == "delivery_failed"
    assert first["delivered"] == 0
    assert "950" not in store.notified_notice_ids()

    notifier.fail_now = False
    second = run_once(store=store, session=session, notifier=notifier)
    assert second["status"] == "ok"
    assert second["delivered"] == 1
    assert "950" in store.notified_notice_ids()


def test_document_header_template_sends_combined_message_outside_24h_window(
    store: CdcCacheStore,
) -> None:
    xml = _notice_xml(
        _notice_row(
            notice_id="960",
            company="Rubrik",
            category="Shortlist",
            body="Interview shortlist PDF",
            with_download=True,
        )
    )
    notifier = WhatsAppNotifier(
        "token",
        "phoneid",
        ["+919876543210"],
        "cdc_new_notice",
        doc_template_name="cdc_new_notice_pdf",
    )
    posts: list[tuple[str, dict | None]] = []

    def fake_post(
        url: str,
        *,
        json_body: dict | None = None,
        data: bytes | None = None,
        headers: dict | None = None,
    ) -> tuple[int, dict]:
        posts.append((url, json_body))
        if url.endswith("/media"):
            return 200, {"id": "media-doc-960"}
        return 200, {"messages": [{"id": "wamid.960"}]}

    notifier._post = fake_post  # type: ignore[assignment]
    summary = run_once(
        store=store,
        session=FakeSession(xml, attachment=b"%PDF-1.7\nshortlist\n%%EOF"),
        notifier=notifier,
    )
    assert summary["delivered"] == 1
    assert summary["attachments_sent"] == 1
    assert len(posts) == 2  # 1 media upload + 1 combined doc-header template
    sent_tpl = posts[1][1]["template"]
    assert sent_tpl["name"] == "cdc_new_notice_pdf"
    assert sent_tpl["components"][0]["type"] == "header"
    assert sent_tpl["components"][0]["parameters"][0]["document"]["id"] == "media-doc-960"


def test_enverp_slug_security_questions_match_live_erp_questions(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    for key in ("CDC_ERP_ROLLNO", "CDC_ERP_PASSWORD", "CDC_ERP_SECURITY_ANSWERS"):
        monkeypatch.delenv(key, raising=False)
    (tmp_path / ".enverp").write_text(
        "ROLLNO=22CS10001\n"
        "PASSWORD=super-secret-pass\n"
        "ERP_SECURITY_Q_FAVORITE_SPORT=Cricket\n"
        "ERP_SECURITY_Q_6TH_FAILED_SUBJECT=Math\n",
        encoding="utf-8",
    )
    creds = load_credentials(tmp_path)
    assert creds is not None
    assert creds.lookup_answer("What is your favorite sport?") == "Cricket"
    assert creds.lookup_answer("Which was your 6th sem failed subject?") == "Math"
    assert creds.lookup_answer("What is your mother's maiden name?") is None


def test_gmail_otp_adaptive_polling_skips_preexisting_and_consumes_fresh() -> None:
    import base64 as b64
    import time as _time
    from backend.gmail_otp import GmailOtpProvider

    provider = GmailOtpProvider("cid", "csecret", "rtoken", poll_timeout_seconds=5.0)
    now_ms = int(_time.time() * 1000)

    def enc(text: str) -> str:
        return b64.urlsafe_b64encode(text.encode()).decode().rstrip("=")

    calls = {"list": 0}

    def fake_api_get(url: str) -> dict[str, Any]:
        if "/messages?" in url:
            calls["list"] += 1
            if calls["list"] == 1:
                # Snapshot before sendOtp: stale message "old1" is already in inbox
                return {"messages": [{"id": "old1"}]}
            if calls["list"] == 2:
                # First poll attempt: new email hasn't arrived yet!
                return {"messages": [{"id": "old1"}]}
            # Second poll attempt: fresh email "new2" arrives!
            return {"messages": [{"id": "new2"}, {"id": "old1"}]}
        if "/old1?" in url:
            return {
                "internalDate": str(now_ms),
                "payload": {"body": {"data": enc("IITKGP ERP OTP is 111111")}},
            }
        return {
            "internalDate": str(now_ms + 2000),
            "payload": {"body": {"data": enc("IITKGP ERP verification code is 842917")}},
        }

    provider._api_get = fake_api_get  # type: ignore[assignment]
    pre_ids = provider.snapshot_existing_ids()
    assert pre_ids == {"old1"}

    sleeps: list[float] = []
    otp = provider.fetch_otp(
        not_before=now_ms / 1000.0,
        exclude_ids=pre_ids,
        sleep_fn=lambda s: sleeps.append(s),
    )
    assert otp == "842917"
    assert len(sleeps) == 1
    # Re-polling cannot replay consumed message "new2"
    otp_replay = provider.fetch_otp(
        not_before=now_ms / 1000.0,
        exclude_ids=pre_ids,
        poll_timeout_seconds=0.0,
    )
    assert otp_replay is None


def test_poller_admin_and_whatsapp_webhook_endpoints_security(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    import hashlib
    import hmac as _hmac
    from fastapi.testclient import TestClient
    from backend.app import create_app

    db_file = tmp_path / "api_poller.sqlite3"
    app = create_app(db_path=db_file, start_poller_daemon=False)
    client = TestClient(app)

    # 1. Fail-closed (503) when CDC_ADMIN_SECRET is unset or too short
    monkeypatch.delenv("CDC_ADMIN_SECRET", raising=False)
    assert client.get("/api/v1/poller/status").status_code == 503

    monkeypatch.setenv("CDC_ADMIN_SECRET", "too-short")
    assert client.get("/api/v1/poller/status").status_code == 503

    admin_secret = "prod-admin-secret-at-least-24-chars-long!"
    monkeypatch.setenv("CDC_ADMIN_SECRET", admin_secret)

    # 2. Rejects unauthenticated or wrong secret with 401
    assert client.get("/api/v1/poller/status").status_code == 401
    assert (
        client.get("/api/v1/poller/status", headers={"X-CDC-Admin-Secret": "wrong-secret"})
        .status_code
        == 401
    )

    # 3. Zero-Credential Boundary still blocks credentials even on admin endpoint
    cred_attempt = client.post(
        "/api/v1/poller/trigger",
        headers={"X-CDC-Admin-Secret": admin_secret},
        json={"password": "secret"},
    )
    assert cred_attempt.status_code == 400
    assert "Privacy boundary violation" in cred_attempt.json()["detail"]

    # 4. Authenticated status & dry-run trigger succeed with zero secret leakage
    monkeypatch.setenv("WA_ACCESS_TOKEN", "EAABsbCS-secret-meta-token")
    monkeypatch.setenv("WA_PHONE_NUMBER_ID", "123456789012345")
    monkeypatch.setenv("WA_RECIPIENTS", "+919876543210,+919876543211")
    monkeypatch.setenv("WA_TEMPLATE_NAME", "cdc_new_notice")

    status_resp = client.get(
        "/api/v1/poller/status",
        headers={"Authorization": f"Bearer {admin_secret}"},
    )
    assert status_resp.status_code == 200
    status_body = status_resp.json()
    assert status_body["whatsapp"]["configured"] is True
    assert status_body["whatsapp"]["recipients_count"] == 2
    serialized = json.dumps(status_body)
    assert "EAABsbCS" not in serialized
    assert "9876543210" not in serialized  # raw phone never leaked!
    assert "+91******3210" in serialized

    # 5. WhatsApp Webhook GET verification & XSS challenge rejection
    monkeypatch.setenv("WA_WEBHOOK_VERIFY_TOKEN", "verify-token-cdc-2026")
    bad_verify = client.get(
        "/api/v1/whatsapp/webhook",
        params={"hub.mode": "subscribe", "hub.verify_token": "wrong", "hub.challenge": "12345"},
    )
    assert bad_verify.status_code == 403

    xss_verify = client.get(
        "/api/v1/whatsapp/webhook",
        params={
            "hub.mode": "subscribe",
            "hub.verify_token": "verify-token-cdc-2026",
            "hub.challenge": "<script>alert(1)</script>",
        },
    )
    assert xss_verify.status_code == 400

    ok_verify = client.get(
        "/api/v1/whatsapp/webhook",
        params={
            "hub.mode": "subscribe",
            "hub.verify_token": "verify-token-cdc-2026",
            "hub.challenge": "987654321",
        },
    )
    assert ok_verify.status_code == 200
    assert ok_verify.text == "987654321"

    # 6. WhatsApp Webhook POST HMAC-SHA256 verification + hashed 24h opt-in tracking
    import time as _time

    wa_app_secret = "meta-app-secret-signing-key-2026"
    monkeypatch.setenv("WA_APP_SECRET", wa_app_secret)

    webhook_event = {
        "object": "whatsapp_business_account",
        "entry": [
            {
                "id": "111",
                "changes": [
                    {
                        "field": "messages",
                        "value": {
                            "messages": [
                                {
                                    "from": "919876543210",
                                    "user_id": "meta-bsuid-123",
                                    "timestamp": str(int(_time.time())),
                                    "type": "text",
                                    "text": {"body": "Hi"},
                                }
                            ],
                            "statuses": [
                                {
                                    "recipient_id": "919876543211",
                                    "status": "delivered",
                                }
                            ],
                        },
                    }
                ],
            }
        ],
    }
    raw_payload = json.dumps(webhook_event).encode("utf-8")

    # Forged signature -> 401
    forged = client.post(
        "/api/v1/whatsapp/webhook",
        content=raw_payload,
        headers={
            "Content-Type": "application/json",
            "X-Hub-Signature-256": "sha256=" + "0" * 64,
        },
    )
    assert forged.status_code == 401

    # Valid HMAC-SHA256 signature -> 200 and updates 24h opt-in window without storing raw phone in DB
    valid_sig = _hmac.new(wa_app_secret.encode("utf-8"), raw_payload, hashlib.sha256).hexdigest()
    valid_post = client.post(
        "/api/v1/whatsapp/webhook",
        content=raw_payload,
        headers={
            "Content-Type": "application/json",
            "X-Hub-Signature-256": f"sha256={valid_sig}",
        },
    )
    assert valid_post.status_code == 200
    assert valid_post.json() == {
        "ok": True,
        "inbound_recorded": 1,
        "statuses_recorded": 1,
        "joined_count": 0,
        "unsubscribed_count": 0,
    }

    # Verify status now reports +91******3210 as opted_in_ever=True & window_24h_open=True
    after_status = client.get(
        "/api/v1/poller/status",
        headers={"X-CDC-Admin-Secret": admin_secret},
    ).json()
    rec_map = {r["masked_phone"]: r for r in after_status["whatsapp"]["recipients"]}
    assert rec_map["+91******3210"]["opted_in_ever"] is True
    assert rec_map["+91******3210"]["window_24h_open"] is True
    assert rec_map["+91******3211"]["opted_in_ever"] is False

    # Verify raw phone number is NOT stored anywhere in SQLite file bytes
    raw_sqlite_bytes = db_file.read_bytes()
    assert b"9876543210" not in raw_sqlite_bytes
    assert b"9876543211" not in raw_sqlite_bytes


def test_join_link_broadcast_without_static_wa_recipients(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    import hashlib
    import hmac as _hmac
    import time as _time
    from fastapi.testclient import TestClient
    from backend.app import create_app

    db_file = tmp_path / "broadcast_group.sqlite3"
    app = create_app(db_path=db_file, start_poller_daemon=False)
    client = TestClient(app)

    admin_secret = "prod-admin-secret-at-least-24-chars-long!"
    wa_app_secret = "meta-app-secret-signing-key-2026"
    join_code = "kgp-cdc-2026-secret"

    monkeypatch.setenv("CDC_ADMIN_SECRET", admin_secret)
    monkeypatch.setenv("WA_APP_SECRET", wa_app_secret)
    monkeypatch.setenv("WA_ACCESS_TOKEN", "EAABsbCS-secret-meta-token")
    monkeypatch.setenv("WA_PHONE_NUMBER_ID", "1412592381940914")
    monkeypatch.setenv("WA_BUSINESS_PHONE_NUMBER", "+15551541295")
    monkeypatch.setenv("WA_JOIN_CODE", join_code)
    monkeypatch.setenv("WA_TEMPLATE_NAME", "cdc_new_notice")
    monkeypatch.delenv("WA_RECIPIENTS", raising=False)

    # Intercept outbound WhatsApp messages (JOIN welcome replies + notice templates)
    sent_payloads: list[dict[str, Any]] = []

    def fake_post(
        self: WhatsAppNotifier,
        url: str,
        *,
        json_body: dict | None = None,
        data: bytes | None = None,
        headers: dict | None = None,
    ) -> tuple[int, dict]:
        if json_body is not None:
            sent_payloads.append(json_body)
        return 200, {"messages": [{"id": f"wamid.{len(sent_payloads)}"}]}

    monkeypatch.setattr(WhatsAppNotifier, "_post", fake_post)

    # 1. Admin fetches secret wa.me invite link — no individual numbers configured yet
    invite_resp = client.get(
        "/api/v1/whatsapp/group-invite",
        headers={"X-CDC-Admin-Secret": admin_secret},
    )
    assert invite_resp.status_code == 200
    invite_data = invite_resp.json()
    assert invite_data["invite_link"] == f"https://wa.me/15551541295?text=JOIN%20{join_code}"
    assert invite_data["subscribers_active_count"] == 0

    # 2. Before any member joins, a new notice is NOT lost — status is waiting_for_subscribers
    store = app.state.store
    xml_notice = _notice_xml(
        _notice_row(
            notice_id="1801",
            company="Goldman Sachs",
            category="Shortlist",
            body="Final interview shortlist released",
            with_download=False,
        )
    )
    session = FakeSession(xml_notice)
    pre_join_notifier = WhatsAppNotifier.from_env(tmp_path, extra_recipients=store.list_active_subscriber_phones())
    assert pre_join_notifier is not None
    assert pre_join_notifier.recipients_count == 0
    pre_summary = run_once(store=store, session=session, notifier=pre_join_notifier)
    assert pre_summary["status"] == "waiting_for_subscribers"
    assert "1801" not in store.notified_notice_ids()

    def _post_webhook(sender: str, text_body: str, profile_name: str = "Student") -> dict[str, Any]:
        ev = {
            "object": "whatsapp_business_account",
            "entry": [
                {
                    "id": "1493620782616154",
                    "changes": [
                        {
                            "field": "messages",
                            "value": {
                                "contacts": [{"wa_id": sender, "profile": {"name": profile_name}}],
                                "messages": [
                                    {
                                        "from": sender,
                                        "timestamp": str(int(_time.time())),
                                        "type": "text",
                                        "text": {"body": text_body},
                                    }
                                ],
                            },
                        }
                    ],
                }
            ],
        }
        raw = json.dumps(ev).encode("utf-8")
        sig = _hmac.new(wa_app_secret.encode("utf-8"), raw, hashlib.sha256).hexdigest()
        resp = client.post(
            "/api/v1/whatsapp/webhook",
            content=raw,
            headers={
                "Content-Type": "application/json",
                "X-Hub-Signature-256": f"sha256={sig}",
            },
        )
        assert resp.status_code == 200
        return resp.json()

    # 3. Wrong join code is rejected (0 joined)
    bad_join = _post_webhook("919999900000", "JOIN wrong-code", "Intruder")
    assert bad_join["joined_count"] == 0
    assert store.list_active_subscriber_phones() == []

    # 4. Two members tap the wa.me invite link and send `JOIN <WA_JOIN_CODE>`
    j1 = _post_webhook("919876500001", f"JOIN {join_code}", "Sahil")
    j2 = _post_webhook("919876500002", f"join {join_code}", "Aarav")
    assert j1["joined_count"] == 1
    assert j2["joined_count"] == 1
    assert len(sent_payloads) == 2  # Two automatic welcome confirmation messages sent

    # Verify raw phone numbers are encrypted at rest (never stored in plaintext in SQLite)
    raw_bytes = db_file.read_bytes()
    assert b"9876500001" not in raw_bytes
    assert b"9876500002" not in raw_bytes
    assert store.list_active_subscriber_phones() == ["+919876500001", "+919876500002"]

    # 5. Next cron tick broadcasts notice #1801 to both joined subscribers!
    sent_payloads.clear()
    live_notifier = WhatsAppNotifier.from_env(
        tmp_path, extra_recipients=store.list_active_subscriber_phones()
    )
    assert live_notifier is not None
    assert live_notifier.recipients_count == 2
    post_summary = run_once(store=store, session=session, notifier=live_notifier)
    assert post_summary["status"] == "ok"
    assert post_summary["delivered"] == 2
    assert {p["to"] for p in sent_payloads} == {"+919876500001", "+919876500002"}
    assert "1801" in store.notified_notice_ids()

    # 6. Member 1 replies `STOP` -> unsubscribed immediately; Admin removes Member 2 by subscriber_id
    unsub = _post_webhook("919876500001", "STOP", "Sahil")
    assert unsub["unsubscribed_count"] == 1
    assert store.list_active_subscriber_phones() == ["+919876500002"]

    status_json = client.get(
        "/api/v1/poller/status",
        headers={"X-CDC-Admin-Secret": admin_secret},
    ).json()
    assert status_json["whatsapp"]["mode"] == "join_link_broadcast"
    assert status_json["whatsapp"]["subscribers_active_count"] == 1
    active_sub_id = [
        s["subscriber_id"] for s in status_json["whatsapp"]["subscribers"] if s["subscribed"]
    ][0]

    rem_resp = client.post(
        "/api/v1/whatsapp/subscribers/remove",
        headers={"X-CDC-Admin-Secret": admin_secret},
        json={"subscriber_id": active_sub_id},
    )
    assert rem_resp.status_code == 200
    assert rem_resp.json()["removed"] is True
    assert store.list_active_subscriber_phones() == []


