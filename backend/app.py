"""Privacy-preserving, security-hardened FastAPI AI CV Matcher & In-RAM/SQLite Cache Server.

Accepts ONLY user-consented non-credential CDC data:
- CV1, CV2, CV3 PDF blobs (base64) — verified against SHA-256 `cvset_hash` and sent ONCE
- Placement company/role rows & JDs
- Placement notices

Security guarantees:
1. Strictly rejects any payload containing ERP credentials (rollno, password, security answers, OTP, cookies).
2. Verifies cryptographic binding (`sha256(cv1|cv2|cv3) == cvset_hash`) and PDF header/size limits.
3. Issues and verifies HMAC-SHA256 `session_token` bound to `cvset_hash` + `student_hash`.
4. Enforces per-IP sliding-window rate limits and per-student daily LLM call quotas to protect Gemini billing.
5. Serves cached `cv_diffs` and `recommendations` directly from Python RAM in sub-millisecond time.
"""

from __future__ import annotations

import base64
import collections
import hashlib
import hmac
import ipaddress
import json
import logging
import os
import re
import secrets
import threading
import time
from pathlib import Path
from typing import Any

logger = logging.getLogger("cdc_portal.api")

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field

from backend.cache_store import CdcCacheStore, CvVerificationError
from backend.llm_matcher import (
    LlmUnavailableError,
    analyze_cvs_and_roles_with_llm,
    get_llm_config,
    load_dotenv_file,
    match_roles_with_llm,
)

ROOT_DIR = Path(__file__).resolve().parent.parent
DEFAULT_DB_PATH = Path(
    os.environ.get("CDC_CACHE_DB_PATH", str(ROOT_DIR / "backend" / "data" / "cdc_cache.sqlite3"))
)
TRUSTED_PROXY_PEERS = {"127.0.0.1", "::1", "localhost"}
MAX_CONCURRENT_OUTBOUND_LLM = int(os.environ.get("CDC_MAX_CONCURRENT_LLM", "10"))
_LLM_SEMAPHORE = threading.BoundedSemaphore(max(1, MAX_CONCURRENT_OUTBOUND_LLM))
_CVSET_LOCKS_GUARD = threading.Lock()
_CVSET_SINGLEFLIGHT_LOCKS: dict[str, threading.RLock] = {}


def _get_cvset_lock(cvset_hash: str) -> threading.RLock:
    with _CVSET_LOCKS_GUARD:
        lock = _CVSET_SINGLEFLIGHT_LOCKS.get(cvset_hash)
        if lock is None:
            if len(_CVSET_SINGLEFLIGHT_LOCKS) > 4096:
                _CVSET_SINGLEFLIGHT_LOCKS.clear()
            lock = threading.RLock()
            _CVSET_SINGLEFLIGHT_LOCKS[cvset_hash] = lock
        return lock


def _load_or_create_token_secret(db_file: Path) -> bytes:
    """Ensure all Uvicorn workers share the same 256-bit HMAC secret across processes and restarts."""
    env_secret = os.environ.get("CDC_TOKEN_SECRET", "").strip()
    if not env_secret:
        dotenv_vals = load_dotenv_file(ROOT_DIR / ".env")
        env_secret = dotenv_vals.get("CDC_TOKEN_SECRET", "").strip()
    if env_secret:
        return env_secret.encode("utf-8")

    secret_file = db_file.parent / ".cdc_token_secret"
    secret_file.parent.mkdir(parents=True, exist_ok=True)
    try:
        if secret_file.is_file():
            raw = secret_file.read_text(encoding="utf-8").strip()
            if len(raw) >= 32:
                return raw.encode("utf-8")
        generated = secrets.token_hex(32)
        fd = os.open(str(secret_file), os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(generated)
        return generated.encode("utf-8")
    except OSError:
        return secrets.token_bytes(32)


def _is_trusted_proxy_peer(peer_ip: str) -> bool:
    if peer_ip in TRUSTED_PROXY_PEERS:
        return True
    try:
        addr = ipaddress.ip_address(peer_ip)
        return addr.is_loopback or addr.is_private
    except ValueError:
        return False


def _extract_client_ip(request: Request) -> str:
    """Extract client IP safely, trusting proxy headers ONLY when peer is local/Docker reverse proxy (e.g., Coolify/Traefik/Nginx)."""
    peer_ip = (request.client.host if request.client else None) or "local"
    if _is_trusted_proxy_peer(peer_ip):
        cf_ip = request.headers.get("CF-Connecting-IP", "").strip()
        if cf_ip:
            return cf_ip[:64]
        real_ip = request.headers.get("X-Real-IP", "").strip()
        if real_ip:
            return real_ip[:64]
        xff = request.headers.get("X-Forwarded-For", "").strip()
        if xff:
            first_hop = xff.split(",")[0].strip()
            if first_hop:
                return first_hop[:64]
    return peer_ip

FORBIDDEN_CREDENTIAL_KEYS = {
    "rollno",
    "user_id",
    "password",
    "security_answers",
    "answer",
    "email_otp",
    "otp",
    "ssotoken",
    "cookies",
}

HEX_HASH_RE = re.compile(r"^[A-Fa-f0-9_\-]{16,128}$")
MAX_REQUEST_BODY_BYTES = 8 * 1024 * 1024  # 8 MB max payload
SESSION_TOKEN_TTL_SECONDS = 24 * 3600


class CvPdfsPayload(BaseModel):
    model_config = ConfigDict(extra="forbid")

    cv1: str = Field(..., min_length=4, max_length=3_000_000)
    cv2: str = Field(..., min_length=4, max_length=3_000_000)
    cv3: str = Field(..., min_length=4, max_length=3_000_000)


class RoleSnapshotItem(BaseModel):
    model_config = ConfigDict(extra="forbid")

    com_id: str = Field(..., min_length=1, max_length=32)
    jnf_id: str = Field(..., min_length=1, max_length=32)
    yop: str = Field(default="2026-2027", max_length=16)
    company: str = Field(..., max_length=200)
    role: str = Field(..., max_length=200)
    ctc: str = Field(default="", max_length=120)
    deadline: str = Field(default="", max_length=64)
    jd_text: str = Field(default="", max_length=8000)
    is_open: bool = True


class NoticeSnapshotItem(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str = Field(..., min_length=1, max_length=32)
    category: str = Field(default="General", max_length=120)
    company: str = Field(default="CDC General", max_length=200)
    notice: str = Field(default="", max_length=8000)
    when: str = Field(default="", max_length=64)


class SyncCdcSnapshotRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    cvset_hash: str = Field(..., min_length=16, max_length=128)
    cv_pdfs: CvPdfsPayload | None = None
    roles: list[RoleSnapshotItem] = Field(default_factory=list, max_length=250)
    notices: list[NoticeSnapshotItem] = Field(default_factory=list, max_length=200)
    precompute_matches: bool = True


class MatchCvsRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    cvset_hash: str = Field(..., min_length=16, max_length=128)
    cv_pdfs: CvPdfsPayload | None = None
    roles: list[RoleSnapshotItem] = Field(..., min_length=1, max_length=100)


def _contains_forbidden_keys(obj: Any) -> str | None:
    if isinstance(obj, dict):
        for k, v in obj.items():
            if str(k).lower() in FORBIDDEN_CREDENTIAL_KEYS:
                return str(k)
            nested = _contains_forbidden_keys(v)
            if nested:
                return nested
    elif isinstance(obj, list):
        for item in obj:
            nested = _contains_forbidden_keys(item)
            if nested:
                return nested
    return None


def issue_session_token(
    secret: bytes,
    cvset_hash: str,
    student_hash: str = "",
    ttl_seconds: int = SESSION_TOKEN_TTL_SECONDS,
) -> str:
    payload = {
        "cvset_hash": cvset_hash,
        "sub": student_hash or cvset_hash[:24],
        "exp": int(time.time()) + ttl_seconds,
    }
    raw = base64.urlsafe_b64encode(json.dumps(payload, separators=(",", ":")).encode("utf-8")).rstrip(b"=")
    sig = hmac.new(secret, raw, hashlib.sha256).hexdigest()
    return f"{raw.decode('ascii')}.{sig}"


def verify_session_token(
    secret: bytes,
    token: str,
    expected_cvset_hash: str,
) -> dict[str, Any]:
    if not token or "." not in token:
        raise HTTPException(status_code=401, detail="Invalid or missing X-CDC-Session-Token")
    b64_part, sig_part = token.rsplit(".", 1)
    expected_sig = hmac.new(secret, b64_part.encode("ascii"), hashlib.sha256).hexdigest()
    if not hmac.compare_digest(expected_sig, sig_part):
        raise HTTPException(status_code=401, detail="Invalid X-CDC-Session-Token signature")
    pad = "=" * ((4 - len(b64_part) % 4) % 4)
    try:
        payload = json.loads(base64.urlsafe_b64decode(b64_part + pad).decode("utf-8"))
    except Exception as exc:
        raise HTTPException(status_code=401, detail="Malformed X-CDC-Session-Token") from exc
    if int(payload.get("exp", 0)) < int(time.time()):
        raise HTTPException(status_code=401, detail="Expired X-CDC-Session-Token")
    if payload.get("cvset_hash") != expected_cvset_hash:
        raise HTTPException(
            status_code=403,
            detail="X-CDC-Session-Token does not match requested cvset_hash",
        )
    return payload


class SlidingWindowRateLimiter:
    """Thread-safe in-RAM sliding-window rate limiter per client IP."""

    def __init__(self, max_requests: int = 90, window_seconds: float = 60.0) -> None:
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self._lock = threading.Lock()
        self._hits: dict[str, collections.deque[float]] = {}

    def allow(self, client_key: str) -> bool:
        now = time.monotonic()
        cutoff = now - self.window_seconds
        with self._lock:
            dq = self._hits.get(client_key)
            if dq is None:
                dq = collections.deque()
                self._hits[client_key] = dq
            while dq and dq[0] < cutoff:
                dq.popleft()
            if len(dq) >= self.max_requests:
                return False
            dq.append(now)
            return True


def _run_llm_for_roles_and_diffs(
    store: CdcCacheStore,
    cvset_hash: str,
    cv_set: dict[str, str],
    missing_roles: list[dict[str, Any]],
    need_diffs: bool,
    max_daily_llm_calls: int,
) -> tuple[dict[str, dict[str, Any]], dict[str, str] | None, str]:
    """Call LLM once for any uncached roles and/or uncached CV diffs, enforcing single-flight, concurrency gate, and daily quota."""
    with _get_cvset_lock(cvset_hash):
        # Re-check cache inside per-student single-flight lock in case a concurrent tab just finished
        still_missing = [
            r
            for r in missing_roles
            if store.get_cached_recommendation(cvset_hash, r["com_id"], r["jnf_id"]) is None
        ]
        current_diffs = store.get_cv_diffs(cvset_hash)
        still_need_diffs = need_diffs and (current_diffs is None)
        if not still_missing and not still_need_diffs:
            cached_map = {
                f"{r['com_id']}_{r['jnf_id']}": store.get_cached_recommendation(
                    cvset_hash, r["com_id"], r["jnf_id"]
                )
                for r in missing_roles
            }
            return (
                {k: v for k, v in cached_map.items() if v is not None},
                current_diffs,
                "cache",
            )

        subject_key = cv_set.get("student_hash") or cvset_hash
        if not store.check_and_increment_llm_quota(subject_key, max_daily_calls=max_daily_llm_calls):
            raise HTTPException(
                status_code=429,
                detail="Daily AI CV matching quota reached for this profile; cached recommendations remain available.",
            )

        acquired = _LLM_SEMAPHORE.acquire(timeout=25.0)
        if not acquired:
            raise LlmUnavailableError(
                "AI matcher queue is temporarily saturated due to high placement traffic; please retry in a few seconds."
            )
        try:
            if not still_need_diffs:
                llm_matches, model_name = match_roles_with_llm(
                    cv_set["cv1_text"],
                    cv_set["cv2_text"],
                    cv_set["cv3_text"],
                    still_missing,
                )
            else:
                llm_matches, new_diffs, model_name = analyze_cvs_and_roles_with_llm(
                    cv_set["cv1_text"],
                    cv_set["cv2_text"],
                    cv_set["cv3_text"],
                    still_missing,
                    include_cv_diffs=True,
                )
                if new_diffs:
                    store.save_cv_diffs(
                        cvset_hash,
                        new_diffs["cv1"],
                        new_diffs["cv2"],
                        new_diffs["cv3"],
                    )

            # Persist all newly matched recommendations inside the single-flight lock
            # so any concurrent request waiting on this cvset_hash wakes up to a warm cache
            resolved_by_key: dict[str, dict[str, Any]] = {}
            for r in missing_roles:
                com_id = r["com_id"]
                jnf_id = r["jnf_id"]
                key = f"{com_id}_{jnf_id}"
                existing_rec = store.get_cached_recommendation(cvset_hash, com_id, jnf_id)
                if existing_rec is not None:
                    resolved_by_key[key] = existing_rec
                    continue
                matched = llm_matches.get(key)
                if matched:
                    resolved_by_key[key] = store.save_recommendation(
                        cvset_hash=cvset_hash,
                        com_id=com_id,
                        jnf_id=jnf_id,
                        cv=matched["cv"],
                        confidence=matched["confidence"],
                        reason=matched["reason"],
                        model=model_name,
                    )
            return resolved_by_key, store.get_cv_diffs(cvset_hash), model_name
        finally:
            _LLM_SEMAPHORE.release()


def create_app(
    db_path: Path | None = None,
    *,
    require_kgp_roll: bool | None = None,
    max_requests_per_minute: int = 90,
    max_daily_llm_calls: int = 30,
) -> FastAPI:
    resolved_db_path = db_path or DEFAULT_DB_PATH
    store = CdcCacheStore(resolved_db_path)
    enforce_roll = (
        require_kgp_roll
        if require_kgp_roll is not None
        else os.environ.get("CDC_REQUIRE_KGP_ROLL", "0") == "1"
    )
    token_secret = _load_or_create_token_secret(resolved_db_path)
    rate_limiter = SlidingWindowRateLimiter(
        max_requests=max_requests_per_minute, window_seconds=60.0
    )

    app = FastAPI(
        title="IIT KGP CDC Privacy-Preserving AI Matcher & Cache",
        version="2.2.0",
        docs_url=None,
        redoc_url=None,
    )
    app.state.store = store
    app.state.token_secret = token_secret

    allowed_origins = [
        origin.strip()
        for origin in os.environ.get(
            "CDC_CORS_ORIGINS",
            "http://127.0.0.1:5173,http://localhost:5173,http://127.0.0.1:8765,http://localhost:8765",
        ).split(",")
        if origin.strip()
    ]

    app.add_middleware(
        CORSMiddleware,
        allow_origins=allowed_origins,
        allow_methods=["GET", "POST", "OPTIONS"],
        allow_headers=["Content-Type", "X-CDC-Session-Token"],
    )

    @app.middleware("http")
    async def reject_credentials_rate_limit_and_harden_headers(request: Request, call_next):
        client_ip = _extract_client_ip(request)
        if request.url.path.startswith("/api/") and not rate_limiter.allow(client_ip):
            return JSONResponse(
                status_code=429,
                content={"detail": "Rate limit exceeded. Please wait a moment and retry."},
            )

        if request.method == "POST" and request.url.path.startswith("/api/"):
            body_bytes = await request.body()
            if len(body_bytes) > MAX_REQUEST_BODY_BYTES:
                return JSONResponse(
                    status_code=413,
                    content={"detail": "Request payload exceeds maximum allowed size (8 MB)."},
                )
            try:
                raw_json = json.loads(body_bytes.decode("utf-8"))
            except Exception:
                raw_json = None
            bad_key = _contains_forbidden_keys(raw_json)
            if bad_key:
                return JSONResponse(
                    status_code=400,
                    content={
                        "detail": f"Privacy boundary violation: credential field '{bad_key}' is never accepted by the backend."
                    },
                )

        response = await call_next(request)
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["Referrer-Policy"] = "no-referrer"
        response.headers["Content-Security-Policy"] = "default-src 'none'; frame-ancestors 'none'"
        response.headers["Cache-Control"] = "no-store"
        return response

    @app.get("/api/v1/health")
    def health() -> dict[str, Any]:
        cfg = get_llm_config()
        llm_configured = bool(cfg["gemini_api_key"] or cfg["openrouter_api_key"])
        return {
            "ok": True,
            "llm_configured": llm_configured,
        }

    @app.get("/api/v1/cvset-status/{cvset_hash}")
    def cvset_status(cvset_hash: str, request: Request) -> dict[str, Any]:
        """Check if a cvset_hash is cached and caller holds a valid X-CDC-Session-Token for it."""
        if not HEX_HASH_RE.match(cvset_hash):
            raise HTTPException(status_code=400, detail="Invalid cvset_hash")
        cv_set = store.get_cv_set(cvset_hash)
        header_token = request.headers.get("X-CDC-Session-Token", "").strip()
        if cv_set is None or not header_token:
            return {
                "cvset_hash": cvset_hash,
                "cv_cached": False,
                "cv_diffs": None,
                "recommendations": [],
            }
        try:
            verify_session_token(token_secret, header_token, cvset_hash)
        except HTTPException:
            return {
                "cvset_hash": cvset_hash,
                "cv_cached": False,
                "cv_diffs": None,
                "recommendations": [],
            }
        cached_recs = list(store.list_cached_recommendations(cvset_hash).values())
        token = issue_session_token(
            token_secret,
            cvset_hash=cvset_hash,
            student_hash=cv_set.get("student_hash", ""),
        )
        return {
            "cvset_hash": cvset_hash,
            "cv_cached": True,
            "session_token": token,
            "cv_diffs": store.get_cv_diffs(cvset_hash),
            "recommendations": cached_recs,
        }

    @app.post("/api/v1/sync-cdc-snapshot")
    def sync_cdc_snapshot(req: SyncCdcSnapshotRequest, request: Request) -> dict[str, Any]:
        if not HEX_HASH_RE.match(req.cvset_hash):
            raise HTTPException(status_code=400, detail="Invalid cvset_hash")

        header_token = request.headers.get("X-CDC-Session-Token", "").strip()
        cv_set: dict[str, str] | None = None

        if req.cv_pdfs is not None:
            try:
                cv_set = store.upsert_cv_set_from_pdfs(
                    req.cvset_hash,
                    req.cv_pdfs.cv1,
                    req.cv_pdfs.cv2,
                    req.cv_pdfs.cv3,
                    require_kgp_roll=enforce_roll,
                )
            except CvVerificationError as exc:
                raise HTTPException(status_code=400, detail=str(exc)) from exc
            if header_token:
                verify_session_token(token_secret, header_token, req.cvset_hash)
        else:
            existing = store.get_cv_set(req.cvset_hash)
            if existing is not None:
                verify_session_token(token_secret, header_token, req.cvset_hash)
                cv_set = existing

        role_dicts = [r.model_dump() for r in req.roles]
        stored_roles = store.upsert_roles(role_dicts)
        stored_notices = store.upsert_notices([n.model_dump() for n in req.notices])

        cached_by_key = store.list_cached_recommendations(req.cvset_hash)
        cv_diffs = store.get_cv_diffs(req.cvset_hash)

        if cv_set is not None and req.precompute_matches:
            missing_roles: list[dict[str, Any]] = []
            for r in role_dicts:
                if not r.get("is_open", True):
                    continue
                key = f"{r['com_id']}_{r['jnf_id']}"
                if key not in cached_by_key:
                    if not r.get("jd_text"):
                        r["jd_text"] = store.get_role_jd(r["com_id"], r["jnf_id"])
                    missing_roles.append(r)

            need_diffs = cv_diffs is None
            if missing_roles or need_diffs:
                try:
                    llm_matches, cv_diffs, _model_name = _run_llm_for_roles_and_diffs(
                        store=store,
                        cvset_hash=req.cvset_hash,
                        cv_set=cv_set,
                        missing_roles=missing_roles,
                        need_diffs=need_diffs,
                        max_daily_llm_calls=max_daily_llm_calls,
                    )
                    for key, saved_rec in llm_matches.items():
                        cached_by_key[key] = saved_rec
                except (LlmUnavailableError, HTTPException) as exc:
                    logger.warning("Background sync LLM precompute skipped: %s", exc)

        session_token = (
            issue_session_token(
                token_secret,
                cvset_hash=req.cvset_hash,
                student_hash=cv_set.get("student_hash", ""),
            )
            if cv_set is not None
            else None
        )

        return {
            "ok": True,
            "cv_cached": cv_set is not None,
            "session_token": session_token,
            "stored_roles": stored_roles,
            "stored_notices": stored_notices,
            "cv_diffs": cv_diffs,
            "recommendations": list(cached_by_key.values()),
        }

    @app.post("/api/v1/match-cvs")
    def match_cvs(req: MatchCvsRequest, request: Request) -> dict[str, Any]:
        if not HEX_HASH_RE.match(req.cvset_hash):
            raise HTTPException(status_code=400, detail="Invalid cvset_hash")

        header_token = request.headers.get("X-CDC-Session-Token", "").strip()

        if req.cv_pdfs is not None:
            try:
                cv_set = store.upsert_cv_set_from_pdfs(
                    req.cvset_hash,
                    req.cv_pdfs.cv1,
                    req.cv_pdfs.cv2,
                    req.cv_pdfs.cv3,
                    require_kgp_roll=enforce_roll,
                )
            except CvVerificationError as exc:
                raise HTTPException(status_code=400, detail=str(exc)) from exc
            if header_token:
                verify_session_token(token_secret, header_token, req.cvset_hash)
        else:
            cv_set = store.get_cv_set(req.cvset_hash)
            if cv_set is None:
                raise HTTPException(
                    status_code=400,
                    detail="CV PDFs for this cvset_hash are not cached yet; please include cv_pdfs.",
                )
            verify_session_token(token_secret, header_token, req.cvset_hash)

        role_dicts = [r.model_dump() for r in req.roles]
        store.upsert_roles(role_dicts)

        results_by_key: dict[str, dict[str, Any]] = {}
        missing_roles: list[dict[str, Any]] = []

        for r in role_dicts:
            com_id = r["com_id"]
            jnf_id = r["jnf_id"]
            key = f"{com_id}_{jnf_id}"
            cached = store.get_cached_recommendation(req.cvset_hash, com_id, jnf_id)
            if cached is not None:
                results_by_key[key] = cached
            else:
                if not r.get("jd_text"):
                    r["jd_text"] = store.get_role_jd(com_id, jnf_id)
                missing_roles.append(r)

        cv_diffs = store.get_cv_diffs(req.cvset_hash)

        if missing_roles:
            try:
                llm_matches, cv_diffs, _model_name = _run_llm_for_roles_and_diffs(
                    store=store,
                    cvset_hash=req.cvset_hash,
                    cv_set=cv_set,
                    missing_roles=missing_roles,
                    need_diffs=cv_diffs is None,
                    max_daily_llm_calls=max_daily_llm_calls,
                )
            except LlmUnavailableError as exc:
                logger.warning("match-cvs failed: %s", exc)
                # Strict no-heuristic-fallback contract: return 503 when LLM is unavailable
                raise HTTPException(
                    status_code=503,
                    detail=f"AI CV Matcher unavailable: {exc}",
                ) from exc

            for r in missing_roles:
                com_id = r["com_id"]
                jnf_id = r["jnf_id"]
                key = f"{com_id}_{jnf_id}"
                saved = llm_matches.get(key)
                if not saved:
                    raise HTTPException(
                        status_code=503,
                        detail=f"LLM response did not return a recommendation for role {key}",
                    )
                results_by_key[key] = saved

        ordered = [
            results_by_key[f"{r['com_id']}_{r['jnf_id']}"]
            for r in role_dicts
            if f"{r['com_id']}_{r['jnf_id']}" in results_by_key
        ]
        session_token = issue_session_token(
            token_secret,
            cvset_hash=req.cvset_hash,
            student_hash=cv_set.get("student_hash", ""),
        )
        return {
            "cvset_hash": req.cvset_hash,
            "session_token": session_token,
            "cv_diffs": cv_diffs,
            "recommendations": ordered,
        }

    return app


app = create_app()
