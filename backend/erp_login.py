"""Headless ERP session manager for the CDC notice poller.

This module is the ONLY place ERP credentials are used. It is deliberately
isolated from request payloads so that:

  * the public API surface still hard-rejects credential fields
    (`FORBIDDEN_CREDENTIAL_KEYS`), and
  * ERP credentials live exclusively in server environment variables / gitignored
    `.enverp` — never accepted from HTTP requests, never stored in SQLite, and
    never placed in logs or error responses.

Design notes
------------
ERP login is password + rotating security question + a 6-digit email OTP. Logging
in on every 15/30-minute cron tick would fire 48–96 OTP emails a day and get the
account rate-limited by ERP. So we persist the session cookies in a `0600`
owner-only `LWPCookieJar` (with `ignore_discard=True` so servlet session cookies
survive restarts) and keep the session warm with `holdSession.htm`. A full
re-login only happens when ERP actually drops the session.
"""

from __future__ import annotations

import http.cookiejar
import inspect
import json
import logging
import os
import random
import re
import tempfile
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger("cdc_poller.erp_login")

ERP_HOST = "https://erp.iitkgp.ac.in"
USER_AGENT = "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36"

SLUG_PREFIX = "__slug__:"
SAFE_NOTICE_ID_RE = re.compile(r"^\d{1,12}$")
SAFE_YEAR_RE = re.compile(r"^\d{4}-\d{4}$")


class ErpAuthError(RuntimeError):
    """Raised when ERP refuses a login attempt. Never embeds the password/OTP."""


class ErpSessionExpired(RuntimeError):
    """Raised when the cached session is no longer valid and must be rebuilt."""


def _is_auth_landing_url(url: str) -> bool:
    low = (url or "").lower()
    return "login.htm" in low or "auth.htm" in low or "sessionexpired" in low


@dataclass
class ErpCredentials:
    rollno: str
    password: str
    # ERP question text (or __slug__:<tokens>) -> answer.
    security_answers: dict[str, str] = field(default_factory=dict)

    def lookup_answer(self, live_question: str) -> str | None:
        target = _normalize_question(live_question)
        if not target:
            return None
        target_words = set(target.split())

        # 1. Exact normalized match
        for question, answer in self.security_answers.items():
            if question.startswith(SLUG_PREFIX):
                continue
            if _normalize_question(question) == target:
                return answer

        # 2. Substring match on normalized full question (mirrors frontend erpParsers.ts)
        for question, answer in self.security_answers.items():
            if question.startswith(SLUG_PREFIX):
                continue
            norm_q = _normalize_question(question)
            if len(norm_q) >= 8 and (norm_q in target or target in norm_q):
                return answer

        # 3. Slug token subset match for `.enverp` ERP_SECURITY_Q_<SLUG> entries
        for question, answer in self.security_answers.items():
            if not question.startswith(SLUG_PREFIX):
                continue
            slug_tokens = [t for t in _normalize_question(question[len(SLUG_PREFIX) :]).split() if t]
            if len(slug_tokens) >= 2 and all(tok in target_words for tok in slug_tokens):
                return answer

        return None

    def masked_rollno(self) -> str:
        roll = (self.rollno or "").strip()
        return roll[:4] + "***" if len(roll) > 4 else "***"


def _normalize_question(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", (text or "").lower()).strip()


def load_dotenv_file(env_path: Path) -> dict[str, str]:
    """Minimal .env/.enverp parser (mirrors backend.llm_matcher's behaviour)."""
    values: dict[str, str] = {}
    if not env_path.is_file():
        return values
    for raw_line in env_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        if key:
            values[key] = value.strip().strip("'").strip('"')
    return values


def load_credentials(root_dir: Path | None = None) -> ErpCredentials | None:
    """Build credentials from env vars, falling back to `.env` and `.enverp`."""
    base = root_dir or Path(__file__).resolve().parent.parent
    file_env = {
        **load_dotenv_file(base / ".env"),
        **load_dotenv_file(base / ".enverp"),
    }

    def _get(*names: str) -> str:
        for name in names:
            value = os.environ.get(name) or file_env.get(name)
            if value:
                return value.strip()
        return ""

    rollno = _get("CDC_ERP_ROLLNO", "ROLLNO")
    password = _get("CDC_ERP_PASSWORD", "PASSWORD")
    if not rollno or not password:
        return None

    answers: dict[str, str] = {}

    # Preferred form: a single JSON object of {question: answer}.
    raw_json = _get("CDC_ERP_SECURITY_ANSWERS")
    if raw_json:
        try:
            parsed = json.loads(raw_json)
        except json.JSONDecodeError:
            parsed = None
        if isinstance(parsed, dict):
            answers.update({str(k): str(v) for k, v in parsed.items() if str(v).strip()})

    # Legacy .enverp form: ERP_SECURITY_Q_<SLUG>=<answer>, paired with optional
    # ERP_SECURITY_QUESTION_<SLUG>=<question text>, or matched via slug keywords.
    if not answers:
        merged_all = {**file_env, **os.environ}
        for key, value in merged_all.items():
            if not key.startswith("ERP_SECURITY_Q_") or not value or not str(value).strip():
                continue
            slug = key[len("ERP_SECURITY_Q_") :]
            question = (
                os.environ.get(f"ERP_SECURITY_QUESTION_{slug}")
                or file_env.get(f"ERP_SECURITY_QUESTION_{slug}")
                or ""
            ).strip()
            if question:
                answers[question] = str(value).strip()
            elif slug:
                answers[f"{SLUG_PREFIX}{slug}"] = str(value).strip()

    # Questions can also be supplied as a parallel JSON array.
    raw_questions = _get("CDC_ERP_SECURITY_QUESTIONS")
    if raw_questions and answers:
        try:
            questions = json.loads(raw_questions)
        except json.JSONDecodeError:
            questions = None
        if isinstance(questions, list) and len(questions) == len(answers):
            answers = dict(zip((str(q) for q in questions), answers.values()))

    return ErpCredentials(rollno=rollno, password=password, security_answers=answers)


def is_poller_enabled(root_dir: Path | None = None) -> bool:
    """Global kill switch so the poller can be enabled/disabled from Coolify or .env."""
    raw = os.environ.get("CDC_ERP_ENABLED")
    if raw is None:
        base = root_dir or Path(__file__).resolve().parent.parent
        merged = {
            **load_dotenv_file(base / ".env"),
            **load_dotenv_file(base / ".enverp"),
        }
        raw = merged.get("CDC_ERP_ENABLED", "0")
    return (raw or "0").strip().lower() in {"1", "true", "yes", "on"}


class ErpSession:
    """Cookie-jar backed ERP session with 0600 disk persistence and liveness probing."""

    def __init__(
        self,
        credentials: ErpCredentials,
        *,
        timeout: float = 30.0,
        cookie_jar_path: Path | None = None,
    ) -> None:
        self._credentials = credentials
        self._timeout = timeout
        self._cookie_jar_path = cookie_jar_path
        self._jar: http.cookiejar.CookieJar = (
            http.cookiejar.LWPCookieJar(str(cookie_jar_path))
            if cookie_jar_path is not None
            else http.cookiejar.CookieJar()
        )
        self._opener = urllib.request.build_opener(
            urllib.request.HTTPCookieProcessor(self._jar)
        )
        self._authenticated = False
        self._requested_url = f"{ERP_HOST}/IIT_ERP3/"
        self._otp_provider: Any = None
        self._load_persisted_cookies()

    def _load_persisted_cookies(self) -> None:
        if not isinstance(self._jar, http.cookiejar.LWPCookieJar) or self._cookie_jar_path is None:
            return
        if not self._cookie_jar_path.is_file():
            return
        try:
            self._jar.load(ignore_discard=True, ignore_expires=True)
            if len(self._jar) > 0:
                self._authenticated = True
                logger.info("Loaded warm ERP session cookies from disk for %s", self._mask_rollno())
        except Exception as exc:  # noqa: BLE001
            logger.warning("Ignoring unreadable ERP cookie jar: %s", exc)
            self._jar.clear()
            self._authenticated = False

    def _save_persisted_cookies(self) -> None:
        if not isinstance(self._jar, http.cookiejar.LWPCookieJar) or self._cookie_jar_path is None:
            return
        try:
            self._cookie_jar_path.parent.mkdir(parents=True, exist_ok=True)
            fd, tmp_name = tempfile.mkstemp(
                prefix=".erp_cookies_",
                dir=str(self._cookie_jar_path.parent),
            )
            os.close(fd)
            os.chmod(tmp_name, 0o600)
            self._jar.save(tmp_name, ignore_discard=True, ignore_expires=True)
            os.replace(tmp_name, str(self._cookie_jar_path))
            os.chmod(str(self._cookie_jar_path), 0o600)
        except OSError as exc:
            logger.warning("Could not persist ERP session cookies: %s", exc)

    # ---------------------------------------------------------------- transport

    def _request(
        self,
        path: str,
        *,
        form: dict[str, str] | None = None,
        ajax: bool = False,
        method: str | None = None,
    ) -> tuple[int, str, str]:
        url = path if path.startswith("http") else f"{ERP_HOST}{path}"
        data = urllib.parse.urlencode(form).encode() if form is not None else None
        req = urllib.request.Request(
            url,
            data=data,
            method=method or ("POST" if data is not None else "GET"),
        )
        req.add_header("User-Agent", USER_AGENT)
        req.add_header("Accept", "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8")
        if data is not None:
            req.add_header("Content-Type", "application/x-www-form-urlencoded; charset=UTF-8")
        if ajax:
            req.add_header("X-Requested-With", "XMLHttpRequest")

        try:
            with self._opener.open(req, timeout=self._timeout) as resp:
                body = resp.read().decode("utf-8", errors="replace")
                return resp.status, resp.geturl(), body
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace") if exc.fp else ""
            return exc.code, url, body
        except urllib.error.URLError as exc:
            raise ErpAuthError(f"ERP is unreachable: {exc.reason}") from exc

    # -------------------------------------------------------------- session ops

    def is_alive(self) -> bool:
        """Cheap probe used every poll tick before deciding whether to re-login."""
        if not self._authenticated:
            return False
        try:
            status, final_url, body = self._request("/IIT_ERP3/home.htm")
        except ErpAuthError:
            return False
        if status >= 400 or _is_auth_landing_url(final_url):
            return False
        return "SSOAdministration/login" not in body

    def keepalive(self) -> bool:
        """Extend the ERP session without consuming an email OTP."""
        if not self._authenticated:
            return False
        rand_id = random.randint(10**14, 10**15 - 1)
        try:
            status, final_url, _ = self._request(f"/IIT_ERP3/holdSession.htm?rand_id={rand_id}")
        except ErpAuthError:
            return False
        if status >= 400 or _is_auth_landing_url(final_url):
            return False
        self._save_persisted_cookies()
        return True

    def ensure(self, otp_provider: Any = None) -> None:
        """Keepalive when possible; full login only when ERP actually dropped us."""
        if self.is_alive():
            if self.keepalive():
                return
            logger.info("holdSession failed; performing a full ERP re-login")
        self.login(otp_provider)

    # ------------------------------------------------------------------- login

    def _prime_sso_session(self) -> None:
        """Step 1.1 — reset stale cookies and start at /IIT_ERP3/ so SSO sets fresh JSID cookies."""
        status, final_url, _ = self._request("/IIT_ERP3/")
        if status >= 400:
            raise ErpAuthError(f"ERP SSO bootstrap failed with HTTP {status}")
        try:
            parsed = urllib.parse.urlparse(final_url)
            req_urls = urllib.parse.parse_qs(parsed.query).get("requestedUrl")
            if req_urls and req_urls[0].strip():
                self._requested_url = req_urls[0].strip()
            else:
                self._requested_url = f"{ERP_HOST}/IIT_ERP3/"
        except Exception:  # noqa: BLE001
            self._requested_url = f"{ERP_HOST}/IIT_ERP3/"

    def _fetch_matching_question(self, max_attempts: int = 16) -> str:
        """Step 1.2 — ERP rotates the question per call, so retry until we know one."""
        seen: list[str] = []
        for _ in range(max_attempts):
            _, _, body = self._request(
                "/SSOAdministration/getSecurityQues.htm",
                form={"user_id": self._credentials.rollno},
                ajax=True,
            )
            question = re.sub(r"<[^>]+>", " ", body or "").strip().strip('"').strip()
            if not question or question.upper() == "FALSE":
                raise ErpAuthError("ERP rejected the roll number for the security question")
            if question not in seen:
                seen.append(question)
            if self._credentials.lookup_answer(question) is not None:
                return question
        raise ErpAuthError(
            "None of the configured security answers match an ERP question "
            f"(saw {len(seen)} distinct question(s)); update CDC_ERP_SECURITY_ANSWERS"
        )

    def login(self, otp_provider: Any = None) -> None:
        """Full login: prime SSO, answer security question, poll fresh OTP, submit."""
        provider = otp_provider or self._otp_provider
        if provider is None:
            raise ErpAuthError(
                "An OTP provider is required for a full ERP login (set GMAIL_OAUTH_* credentials)"
            )

        # Best-effort server-side logout before wiping cookies so ERP never reuses a half-dead session
        try:
            self._request("/IIT_ERP3/logout.htm")
        except Exception:  # noqa: BLE001
            pass
        self._clear_cookies()
        self._prime_sso_session()
        question = self._fetch_matching_question()
        answer = self._credentials.lookup_answer(question) or ""

        pre_existing_ids: set[str] | None = None
        if hasattr(provider, "snapshot_existing_ids"):
            try:
                pre_existing_ids = provider.snapshot_existing_ids()
            except Exception:  # noqa: BLE001
                pre_existing_ids = None

        otp_sent_at = time.time()
        _, otp_url, body = self._request(
            "/SSOAdministration/getEmilOTP.htm",
            form={
                "user_id": self._credentials.rollno,
                "password": self._credentials.password,
                "answer": answer,
                "typeee": "SI",
                "sessionToken": "",
                "requestedUrl": self._requested_url,
                "email_otp": "",
            },
            ajax=True,
        )
        if _is_auth_landing_url(otp_url):
            raise ErpAuthError("ERP reset the login session while requesting OTP")

        try:
            message = str(json.loads(body).get("msg", ""))
        except json.JSONDecodeError:
            raise ErpAuthError("ERP getEmilOTP.htm did not return JSON") from None
        low = message.lower()
        if "unable" in low or "incorrect" in low or "invalid" in low or "error" in low:
            # Deliberately never includes password or security answer.
            raise ErpAuthError(f"ERP rejected the login attempt: {message[:200]}")

        otp = _call_fetch_otp(provider, not_before=otp_sent_at, exclude_ids=pre_existing_ids)
        if not otp:
            raise ErpAuthError("No fresh ERP OTP arrived in Gmail within the polling window")

        _, final_url, auth_body = self._request(
            "/SSOAdministration/auth.htm",
            form={
                "user_id": self._credentials.rollno,
                "password": self._credentials.password,
                "answer": answer,
                "typeee": "SI",
                "sessionToken": "",
                "requestedUrl": self._requested_url,
                "email_otp": otp.strip(),
            },
        )
        if _is_auth_landing_url(final_url):
            raise ErpAuthError("ERP stayed on the login page — the OTP was rejected or expired")
        if "SSOAdministration/login" in auth_body and "logout" not in auth_body.lower():
            raise ErpAuthError("ERP returned the login form after OTP submission")

        self._authenticated = True
        self._save_persisted_cookies()
        logger.info("ERP login succeeded for roll %s", self._mask_rollno())

    def open_placement(self) -> None:
        """Step 2 — hop the CDC module menu to mint a TrainingPlacementSSO session."""
        status, menu_url, menu = self._request(
            "/IIT_ERP3/showmenu.htm",
            form={
                "module_id": "26",
                "menu_id": "11",
                "link": f"{ERP_HOST}/TrainingPlacementSSO/TPStudent.jsp",
                "module_name": "CDC",
                "parent_display_name": "Student",
                "display_name": "Application of Placement/Internship",
            },
        )
        if status >= 400 or _is_auth_landing_url(menu_url):
            self._authenticated = False
            raise ErpSessionExpired("ERP session ended before opening CDC Placement")

        token_match = re.search(
            r'name=["\']ssoToken["\'][^>]*value=["\']([^"\']+)["\']', menu, re.IGNORECASE
        )
        if not token_match:
            self._authenticated = False
            raise ErpSessionExpired("CDC menu did not return an ssoToken")

        _, tp_url, _ = self._request(
            "/TrainingPlacementSSO/TPStudent.jsp",
            form={"ssoToken": token_match.group(1), "module_id": "26", "menu_id": "11"},
        )
        if _is_auth_landing_url(tp_url):
            self._authenticated = False
            raise ErpSessionExpired("ERP session ended while opening TPStudent.jsp")
        self._save_persisted_cookies()

    # ---------------------------------------------------------------- utilities

    def fetch_notice_grid(self, rows: int = 100) -> str:
        """Step 3 — the jqGrid XML feed of notices (same endpoint the browser uses)."""
        _, notice_url, _ = self._request("/TrainingPlacementSSO/Notice.jsp")
        if _is_auth_landing_url(notice_url):
            self._authenticated = False
            raise ErpSessionExpired("ERP session expired on Notice.jsp")

        clamped = max(1, min(200, rows))
        _, grid_url, payload = self._request(
            "/TrainingPlacementSSO/ERPMonitoring.htm"
            f"?action=fetchData&jqqueryid=54&_search=false&nd={int(time.time() * 1000)}"
            f"&rows={clamped}&page=1&sidx=&sord=asc&totalrows={clamped}"
        )
        if _is_auth_landing_url(grid_url) or "<rows" not in (payload or ""):
            self._authenticated = False
            raise ErpSessionExpired("ERP notice feed did not return jqGrid XML")
        return payload

    def fetch_attachment(self, notice_id: str, year: str) -> bytes:
        """Download a notice attachment strictly validated by numeric ID and academic year."""
        clean_id = (notice_id or "").strip()
        if not SAFE_NOTICE_ID_RE.match(clean_id):
            raise ValueError("Invalid ERP notice id for attachment download")
        clean_year = (year or "").strip()
        if clean_year and not SAFE_YEAR_RE.match(clean_year):
            clean_year = "2026-2027"
        query = urllib.parse.urlencode(
            {"type": "NOTICE", "year": clean_year or "2026-2027", "id": clean_id}
        )
        url = f"{ERP_HOST}/TrainingPlacementSSO/AdmFilePDF.htm?{query}"
        req = urllib.request.Request(url, method="GET")
        req.add_header("User-Agent", USER_AGENT)
        req.add_header("Accept", "application/pdf,*/*")
        with self._opener.open(req, timeout=self._timeout) as resp:
            # Cap read at 25 MB so a malformed response can never OOM the worker
            return resp.read(25 * 1024 * 1024)

    def _clear_cookies(self) -> None:
        self._jar.clear()
        self._authenticated = False
        if self._cookie_jar_path is not None and self._cookie_jar_path.is_file():
            try:
                self._cookie_jar_path.unlink()
            except OSError:
                pass

    def _mask_rollno(self) -> str:
        return self._credentials.masked_rollno()

    def close(self) -> None:
        self._clear_cookies()


def _call_fetch_otp(
    provider: Any,
    *,
    not_before: float,
    exclude_ids: set[str] | None,
) -> str | None:
    """Invoke provider.fetch_otp with exclude_ids if supported, else fallback cleanly for mocks."""
    fn = getattr(provider, "fetch_otp")
    try:
        sig = inspect.signature(fn)
        if "exclude_ids" in sig.parameters or any(
            p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()
        ):
            return fn(not_before=not_before, exclude_ids=exclude_ids)
    except (TypeError, ValueError):
        pass
    return fn(not_before=not_before)
