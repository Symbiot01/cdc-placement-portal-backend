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
