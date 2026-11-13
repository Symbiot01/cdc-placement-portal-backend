# IIT KGP CDC Placement Portal — AI CV Matcher & Cache Backend

Privacy-preserving, security-hardened FastAPI backend for the IIT Kharagpur CDC Placement Portal.

## Security & Privacy Guarantees
- **Zero-Credential Boundary:** Rejects (`HTTP 400`) any payload containing ERP credentials or mailbox tokens (`rollno`, `password`, `security_answers`, `otp`, `ssoToken`, `cookies`, `access_token`, `refresh_token`).
- **Cryptographic CV Set Binding:** Verifies `sha256(CV1|CV2|CV3) == cvset_hash` and issues HMAC-SHA256 `X-CDC-Session-Token` headers.
- **In-RAM + SQLite WAL Cache:** Serves cached `cv_diffs` and role recommendations in sub-millisecond time without re-uploading CVs or re-calling the LLM.
- **Single-Flight Concurrency & Rate Limiting:** Deduplicates concurrent requests per `cvset_hash` and enforces per-IP sliding-window rate limits + daily per-student LLM quotas.

## Quick Start

```bash
uv venv .venv
uv pip install --python .venv/bin/python fastapi uvicorn pydantic pypdf httpx pytest

# Configure .env (never committed)
# GEMINI_API_KEY=your_key_here

.venv/bin/python -m uvicorn backend.app:app --host 127.0.0.1 --port 8765 --reload --reload-dir backend
```

## Running Tests

```bash
.venv/bin/python -m pytest -q
```

---

## CDC Notice → WhatsApp Cron Scheduler & Webhook API

An in-backend scheduled cron daemon (`PollerScheduler` wired into FastAPI `lifespan`)
that logs into ERP headlessly every **15 or 30 minutes**, fetches `jqqueryid=54`
placement notices, and fans out every brand-new notice (plus verified PDF
attachment) to configured WhatsApp member numbers.

### Why a keepalive instead of logging in every run

ERP login requires password + rotating security question + a **6-digit email OTP**.
A stateless 15-minute job would fire ~96 OTP emails a day and get rate-limited.
Instead, `ErpSession` persists its session cookies in an owner-only `0600`
`LWPCookieJar` (`backend/data/.erp_cookies.lwp`) and calls `holdSession.htm` each
tick, performing a full Gmail-OTP login only when ERP drops the session.

### Configuration (all via Coolify Environment Variables / `.enverp`)

| Variable | Purpose |
|---|---|
| `CDC_ERP_ENABLED` | `1` to auto-start the background cron inside FastAPI, `0` to disable |
| `CDC_POLL_INTERVAL_SECONDS` | `900` (15 min, default) or `1800` (30 min); clamped `[300, 86400]` |
| `CDC_POLLER_BOOTSTRAP_SEED` | `1` (default) baseline-seeds old notices on first boot to prevent spamming history |
| `CDC_ERP_ROLLNO` | Your IIT KGP ERP roll number |
| `CDC_ERP_PASSWORD` | Your IIT KGP ERP password |
| `CDC_ERP_SECURITY_ANSWERS` | JSON map of all 3 ERP security Q&As: `{"Exact question 1": "ans1", ...}` |
| `GMAIL_OAUTH_CLIENT_ID` / `_SECRET` / `GMAIL_OAUTH_REFRESH_TOKEN` | Desktop OAuth client + refresh token (`gmail.readonly`) for unattended OTP polling |
| `WA_ACCESS_TOKEN` | Meta System User permanent token (`whatsapp_business_messaging`) |
| `WA_PHONE_NUMBER_ID` | Meta WhatsApp Cloud API **Phone Number ID** (numeric ID, not `+91...`) |
| `WA_BUSINESS_PHONE_NUMBER` | Your Cloud API WhatsApp phone number in E.164 format (e.g. `+15551541295` or `+91...`) used to generate the `wa.me` invite link |
| `WA_JOIN_CODE` | Min 8-char secret invite token — members tap `https://wa.me/<NUMBER>?text=JOIN%20<WA_JOIN_CODE>` to self-subscribe (no `WA_RECIPIENTS` list needed!) |
| `WA_RECIPIENTS` | *(Optional)* Static comma-separated member numbers merged with invite-link subscribers |
| `WA_TEMPLATE_NAME` / `WA_TEMPLATE_LANG` | Approved **Utility** text template, e.g. `cdc_new_notice` / `en_US` |
| `WA_DOC_TEMPLATE_NAME` | *(Optional, recommended)* Approved **Utility** template with a **Document** header (`cdc_new_notice_pdf`) for 24/7 PDF delivery outside the 24h window |
| `CDC_ADMIN_SECRET` | Min 24-char random secret protecting `/api/v1/poller/status` & `/api/v1/poller/trigger` |
| `WA_WEBHOOK_VERIFY_TOKEN` | Secret string entered in Meta Console → WhatsApp → Webhook Verify Token |
| `WA_APP_SECRET` | Meta App Secret used to verify `X-Hub-Signature-256` on webhook POSTs |

Mint the Gmail refresh token once on your laptop:

```bash
export GMAIL_OAUTH_CLIENT_ID='...'
export GMAIL_OAUTH_CLIENT_SECRET='...'
python scripts/get_gmail_refresh_token.py
```

### WhatsApp Templates (Meta Business Manager)

1. **Text Utility Template (`WA_TEMPLATE_NAME=cdc_new_notice`)** — *Required*:
   ```
   New CDC notice: {{1}} — {{2}}
   {{3}}
   ```
2. **Document-Header Utility Template (`WA_DOC_TEMPLATE_NAME=cdc_new_notice_pdf`)** — *Recommended for PDFs*:
   - **Header**: `Document`
   - **Body**: same `{{1}}`, `{{2}}`, `{{3}}` as above.
   - Delivers BOTH the notice text and official PDF attachment in a single message **even outside Meta's 24-hour customer service window**.

### API Endpoints & Meta Webhook Endpoints

| Method & Path | Auth | Purpose |
|---|---|---|
| `GET /api/v1/poller/status` | `X-CDC-Admin-Secret` or `Bearer <CDC_ADMIN_SECRET>` | Inspect cron health, masked rollno, active broadcast subscribers, 24h windows, and recent run logs |
| `POST /api/v1/poller/trigger` | `X-CDC-Admin-Secret` or `Bearer <CDC_ADMIN_SECRET>` | Trigger an immediate single-flight poll (`{"dry_run": false, "seed_only": false}`) |
| `GET /api/v1/whatsapp/group-invite` | `X-CDC-Admin-Secret` or `Bearer <CDC_ADMIN_SECRET>` | Returns the shareable `https://wa.me/...?text=JOIN%20...` invite link and masked subscriber list |
| `POST /api/v1/whatsapp/subscribers/remove` | `X-CDC-Admin-Secret` or `Bearer <CDC_ADMIN_SECRET>` | Revoke a subscriber by hashed `subscriber_id` |
| `GET /api/v1/whatsapp/webhook` | `hub.verify_token == WA_WEBHOOK_VERIFY_TOKEN` | Meta Cloud API webhook verification handshake |
| `POST /api/v1/whatsapp/webhook` | `X-Hub-Signature-256` (`WA_APP_SECRET` HMAC) | Handles `JOIN <WA_JOIN_CODE>` self-subscriptions, `STOP` unsubscribes, 24h windows, and delivery receipts |

### CLI Usage (Optional / Manual Testing)

```bash
# Baseline-seed current ERP notices without sending WhatsApp messages
python -m backend.poller --seed-only

# One cycle, dry-run — verifies ERP login + Gmail OTP + jqGrid parsing
python -m backend.poller --once --dry-run

# Standalone loop every 15 minutes
python -m backend.poller --interval 900
```

### Security & Delivery Guarantees

- **Zero-Credential API Boundary:** ERP passwords, OTPs, OAuth tokens, and cookies are strictly rejected (`HTTP 400`) on all API payloads and never appear in responses or logs.
- **Cold-Start Flood Protection:** First boot automatically baseline-seeds existing ERP notices instead of spamming ~100 historical notices to members.
- **Idempotent & Retry-Safe:** Notices are marked delivered in `notice_notifications` only after `delivered > 0`, so transient outages automatically retry on the next tick without losing or duplicating alerts.
- **Hashed Opt-In Ledger:** Member phone numbers are stored in SQLite solely as salted SHA-256 hashes + masked previews (`+91******3210`) — zero raw phone PII at rest.
