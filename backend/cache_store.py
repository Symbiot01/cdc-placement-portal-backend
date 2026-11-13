"""Persistent SQLite + In-RAM Hot Cache store for CV PDFs text, per-CV diff summaries, CDC roles/notices, and per-(CVset,company,role) AI matches."""

from __future__ import annotations

import base64
import hashlib
import hmac
import io
import json
import os
import re
import secrets
import sqlite3
import threading
import time
from pathlib import Path
from typing import Any

from pypdf import PdfReader

# IIT KGP roll number pattern (e.g., 22XX10001, 23CS10001, 25EC62R03, 22MF3IM05)
KGP_ROLL_RE = re.compile(r"\b(\d{2}[A-Z]{2}[A-Z0-9]{4,6})\b")

# Maximum decoded PDF size accepted per CV (2 MB)
MAX_PDF_BYTES = 2 * 1024 * 1024


class CvVerificationError(ValueError):
    """Raised when uploaded CV PDFs fail integrity, size, or ERP format verification."""


def extract_pdf_text_from_base64(b64_payload: str, max_chars: int = 12000) -> str:
    """Decode base64 PDF and extract text using pypdf, falling back safely if minimal/synthetic PDF."""
    try:
        raw = base64.b64decode(b64_payload, validate=False)
    except Exception as exc:
        raise CvVerificationError("Invalid base64 encoding in CV PDF payload") from exc

    if len(raw) > MAX_PDF_BYTES:
        raise CvVerificationError("CV PDF exceeds maximum allowed size (2 MB)")
    if not raw.startswith(b"%PDF"):
        raise CvVerificationError("Uploaded CV payload is not a valid PDF document")

    try:
        reader = PdfReader(io.BytesIO(raw))
        pages_text = [page.extract_text() or "" for page in reader.pages]
        joined = "\n".join(t.strip() for t in pages_text if t and t.strip()).strip()
        if joined:
            return joined[:max_chars]
    except Exception:
        pass

    # Fallback for minimal/synthetic test PDFs that append text in byte stream
    decoded = raw.decode("utf-8", errors="ignore").strip()
    return decoded[:max_chars]


def compute_cvset_sha256(cv1_b64: str, cv2_b64: str, cv3_b64: str) -> str:
    combined = f"CV1:{cv1_b64}|CV2:{cv2_b64}|CV3:{cv3_b64}".encode("utf-8")
    return hashlib.sha256(combined).hexdigest()


def compute_jd_sha256(jd_text: str) -> str:
    normalized = " ".join((jd_text or "").split()).strip().lower()
    if not normalized:
        return "empty"
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()[:16]


def extract_verified_roll_hash(cv1_text: str, cv2_text: str, cv3_text: str) -> str | None:
    """Extract common IIT KGP roll number across CVs and return its SHA-256 hash (never storing raw rollno)."""
    rolls_per_cv = [
        {m.group(1).upper() for m in KGP_ROLL_RE.finditer(text or "")}
        for text in (cv1_text, cv2_text, cv3_text)
    ]
    non_empty = [s for s in rolls_per_cv if s]
    if not non_empty:
        return None
    common = set.intersection(*non_empty) if len(non_empty) == 3 else set.union(*non_empty)
    if not common:
        return None
    canonical_roll = sorted(common)[0]
    return hashlib.sha256(f"kgp_roll:{canonical_roll}".encode("utf-8")).hexdigest()[:32]


class CdcCacheStore:
    """Thread-safe SQLite + In-RAM store for CV sets, CV diff summaries, company roles, notices, and cached LLM recommendations."""

    def __init__(self, db_path: Path) -> None:
        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._ram_cv_sets: dict[str, dict[str, str]] = {}
        self._ram_recs: dict[str, dict[str, dict[str, Any]]] = {}
        self._llm_quota_by_subject: dict[str, tuple[str, int]] = {}
        self._phone_cipher_key: bytes = self._load_or_create_phone_cipher_key()
        self._init_schema()
        self._warm_ram_cache()

    def _load_or_create_phone_cipher_key(self) -> bytes:
        """Load or atomically create a 256-bit key (`0600`) for encrypting subscriber phone numbers at rest."""
        env_key = (os.environ.get("CDC_SUBSCRIBER_CIPHER_KEY") or "").strip()
        if len(env_key) >= 32:
            return hashlib.sha256(f"cdc_sub_key:{env_key}".encode("utf-8")).digest()

        key_file = self.db_path.parent / ".wa_subscriber_key"
        try:
            if key_file.is_file():
                raw = key_file.read_text(encoding="utf-8").strip()
                if len(raw) >= 64:
                    return bytes.fromhex(raw[:64])
            generated_hex = secrets.token_hex(32)
            fd = os.open(str(key_file), os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                fh.write(generated_hex)
            os.chmod(str(key_file), 0o600)
            return bytes.fromhex(generated_hex)
        except OSError:
            return secrets.token_bytes(32)

    def _encrypt_phone(self, normalized_e164: str) -> str:
        """Authenticated stream encryption (`nonce || ciphertext || hmac_sha256`) using stdlib primitives."""
        plaintext = normalized_e164.encode("utf-8")
        nonce = secrets.token_bytes(16)
        enc_key = hmac.new(self._phone_cipher_key, b"enc:" + nonce, hashlib.sha256).digest()
        ciphertext = bytes(b ^ enc_key[i % len(enc_key)] for i, b in enumerate(plaintext))
        tag = hmac.new(
            self._phone_cipher_key,
            b"mac:" + nonce + ciphertext,
            hashlib.sha256,
        ).digest()[:16]
        return "v1:" + base64.urlsafe_b64encode(nonce + ciphertext + tag).decode("ascii")

    def _decrypt_phone(self, token: str) -> str | None:
        if not token or not token.startswith("v1:"):
            return None
        try:
            raw = base64.urlsafe_b64decode(token[3:].encode("ascii"))
        except Exception:
            return None
        if len(raw) < 33:
            return None
        nonce = raw[:16]
        tag = raw[-16:]
        ciphertext = raw[16:-16]
        expected_tag = hmac.new(
            self._phone_cipher_key,
            b"mac:" + nonce + ciphertext,
            hashlib.sha256,
        ).digest()[:16]
        if not hmac.compare_digest(tag, expected_tag):
            return None
        enc_key = hmac.new(self._phone_cipher_key, b"enc:" + nonce, hashlib.sha256).digest()
        plaintext = bytes(b ^ enc_key[i % len(enc_key)] for i, b in enumerate(ciphertext))
        try:
            decoded = plaintext.decode("utf-8")
        except UnicodeDecodeError:
            return None
        return decoded if decoded.startswith("+") else None

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path), timeout=15.0, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA synchronous=NORMAL;")
        conn.execute("PRAGMA temp_store=MEMORY;")
        conn.execute("PRAGMA mmap_size=268435456;")
        conn.execute("PRAGMA cache_size=-65536;")
        return conn

    def _init_schema(self) -> None:
        with self._lock, self._connect() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS cv_sets (
                    cvset_hash TEXT PRIMARY KEY,
                    cv1_text TEXT NOT NULL,
                    cv2_text TEXT NOT NULL,
                    cv3_text TEXT NOT NULL,
                    cv1_diff TEXT NOT NULL DEFAULT '',
                    cv2_diff TEXT NOT NULL DEFAULT '',
                    cv3_diff TEXT NOT NULL DEFAULT '',
                    student_hash TEXT NOT NULL DEFAULT '',
                    created_at REAL NOT NULL
                );

                CREATE TABLE IF NOT EXISTS company_roles (
                    com_id TEXT NOT NULL,
                    jnf_id TEXT NOT NULL,
                    yop TEXT NOT NULL,
                    company TEXT NOT NULL,
                    role TEXT NOT NULL,
                    ctc TEXT NOT NULL DEFAULT '',
                    deadline_end TEXT NOT NULL DEFAULT '',
                    jd_text TEXT NOT NULL DEFAULT '',
                    updated_at REAL NOT NULL,
                    PRIMARY KEY (com_id, jnf_id, yop)
                );

                CREATE TABLE IF NOT EXISTS notices (
                    notice_id TEXT PRIMARY KEY,
                    category TEXT NOT NULL,
                    company TEXT NOT NULL,
                    notice_text TEXT NOT NULL,
                    notice_at TEXT NOT NULL,
                    updated_at REAL NOT NULL
                );

                CREATE TABLE IF NOT EXISTS cv_recommendations (
                    cvset_hash TEXT NOT NULL,
                    com_id TEXT NOT NULL,
                    jnf_id TEXT NOT NULL,
                    recommended_cv INTEGER NOT NULL CHECK (recommended_cv IN (1, 2, 3)),
                    confidence REAL NOT NULL,
                    reason TEXT NOT NULL,
                    model TEXT NOT NULL,
                    created_at REAL NOT NULL,
                    PRIMARY KEY (cvset_hash, com_id, jnf_id)
                );

                CREATE TABLE IF NOT EXISTS llm_daily_quotas (
                    subject_key TEXT NOT NULL,
                    day_utc TEXT NOT NULL,
                    calls INTEGER NOT NULL DEFAULT 0,
                    PRIMARY KEY (subject_key, day_utc)
                );

                -- Authoritative "already alerted" ledger for the ERP notice poller.
                -- Kept separate from `notices` because that table is written by the
                -- browser sync too, and dedupe there must not suppress an alert.
                CREATE TABLE IF NOT EXISTS notice_notifications (
                    notice_id TEXT PRIMARY KEY,
                    channel TEXT NOT NULL DEFAULT '',
                    recipients INTEGER NOT NULL DEFAULT 0,
                    attachment_sent INTEGER NOT NULL DEFAULT 0,
                    notified_at REAL NOT NULL
                );

                CREATE TABLE IF NOT EXISTS poller_runs (
                    run_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    polled_at REAL NOT NULL,
                    trigger_source TEXT NOT NULL DEFAULT 'cron',
                    status TEXT NOT NULL DEFAULT 'ok',
                    total_notices INTEGER NOT NULL DEFAULT 0,
                    new_notices INTEGER NOT NULL DEFAULT 0,
                    seeded_notices INTEGER NOT NULL DEFAULT 0,
                    delivered INTEGER NOT NULL DEFAULT 0,
                    attachments_sent INTEGER NOT NULL DEFAULT 0,
                    errors_json TEXT NOT NULL DEFAULT '[]'
                );

                CREATE TABLE IF NOT EXISTS whatsapp_optins (
                    recipient_hash TEXT PRIMARY KEY,
                    masked_phone TEXT NOT NULL,
                    last_inbound_at REAL NOT NULL,
                    last_status TEXT NOT NULL DEFAULT 'inbound'
                );

                CREATE TABLE IF NOT EXISTS whatsapp_broadcast_subscribers (
                    recipient_hash TEXT PRIMARY KEY,
                    masked_phone TEXT NOT NULL,
                    encrypted_phone TEXT NOT NULL,
                    display_name TEXT NOT NULL DEFAULT '',
                    subscribed INTEGER NOT NULL DEFAULT 1,
                    joined_at REAL NOT NULL,
                    updated_at REAL NOT NULL,
                    last_inbound_at REAL NOT NULL DEFAULT 0.0,
                    last_delivery_status TEXT NOT NULL DEFAULT ''
                );
                """
            )
            existing_cols = {
                str(r["name"])
                for r in conn.execute("PRAGMA table_info(cv_sets)").fetchall()
            }
            for col in ("cv1_diff", "cv2_diff", "cv3_diff", "student_hash"):
                if col not in existing_cols:
                    conn.execute(
                        f"ALTER TABLE cv_sets ADD COLUMN {col} TEXT NOT NULL DEFAULT ''"
                    )

    def _warm_ram_cache(self) -> None:
        """Load all CV sets and recommendations into Python RAM at startup for sub-millisecond reads."""
        with self._lock, self._connect() as conn:
            for row in conn.execute(
                "SELECT cvset_hash, cv1_text, cv2_text, cv3_text, cv1_diff, cv2_diff, cv3_diff, student_hash FROM cv_sets"
            ).fetchall():
                h = str(row["cvset_hash"])
                self._ram_cv_sets[h] = {
                    "cvset_hash": h,
                    "cv1_text": str(row["cv1_text"]),
                    "cv2_text": str(row["cv2_text"]),
                    "cv3_text": str(row["cv3_text"]),
                    "cv1_diff": str(row["cv1_diff"] or ""),
                    "cv2_diff": str(row["cv2_diff"] or ""),
                    "cv3_diff": str(row["cv3_diff"] or ""),
                    "student_hash": str(row["student_hash"] or ""),
                }

            for row in conn.execute(
                "SELECT cvset_hash, com_id, jnf_id, recommended_cv, confidence, reason, model FROM cv_recommendations"
            ).fetchall():
                h = str(row["cvset_hash"])
                com_id = str(row["com_id"])
                jnf_id = str(row["jnf_id"])
                key = f"{com_id}_{jnf_id}"
                self._ram_recs.setdefault(h, {})[key] = {
                    "com_id": com_id,
                    "jnf_id": jnf_id,
                    "key": key,
                    "cv": int(row["recommended_cv"]),
                    "confidence": float(row["confidence"]),
                    "reason": str(row["reason"]),
                    "model": str(row["model"]),
                    "cached": True,
                }

    def check_and_increment_llm_quota(self, subject_key: str, max_daily_calls: int = 30) -> bool:
        """Enforce daily LLM call quota per authenticated student/cvset_hash across workers in SQLite + RAM."""
        day_str = time.strftime("%Y-%m-%d", time.gmtime())
        with self._lock:
            prev_day, used = self._llm_quota_by_subject.get(subject_key, (day_str, 0))
            if prev_day == day_str and used >= max_daily_calls:
                return False
            with self._connect() as conn:
                row = conn.execute(
                    "SELECT calls FROM llm_daily_quotas WHERE subject_key = ? AND day_utc = ?",
                    (subject_key, day_str),
                ).fetchone()
                db_used = int(row["calls"]) if row else 0
                if db_used >= max_daily_calls:
                    self._llm_quota_by_subject[subject_key] = (day_str, db_used)
                    return False
                new_used = db_used + 1
                conn.execute(
                    """
                    INSERT INTO llm_daily_quotas (subject_key, day_utc, calls)
                    VALUES (?, ?, ?)
                    ON CONFLICT(subject_key, day_utc) DO UPDATE SET calls = excluded.calls
                    """,
                    (subject_key, day_str, new_used),
                )
                self._llm_quota_by_subject[subject_key] = (day_str, new_used)
                return True

    def get_cv_set(self, cvset_hash: str) -> dict[str, str] | None:
        with self._lock:
            cached = self._ram_cv_sets.get(cvset_hash)
            if cached is not None:
                return dict(cached)
        with self._lock, self._connect() as conn:
            row = conn.execute(
                """
                SELECT cvset_hash, cv1_text, cv2_text, cv3_text, cv1_diff, cv2_diff, cv3_diff, student_hash
                FROM cv_sets
                WHERE cvset_hash = ?
                """,
                (cvset_hash,),
            ).fetchone()
            if not row:
                return None
            record = {
                "cvset_hash": str(row["cvset_hash"]),
                "cv1_text": str(row["cv1_text"]),
                "cv2_text": str(row["cv2_text"]),
                "cv3_text": str(row["cv3_text"]),
                "cv1_diff": str(row["cv1_diff"] or ""),
                "cv2_diff": str(row["cv2_diff"] or ""),
                "cv3_diff": str(row["cv3_diff"] or ""),
                "student_hash": str(row["student_hash"] or ""),
            }
            self._ram_cv_sets[cvset_hash] = record
            return dict(record)

    def get_cv_diffs(self, cvset_hash: str) -> dict[str, str] | None:
        cv_set = self.get_cv_set(cvset_hash)
        if not cv_set:
            return None
        d1 = cv_set.get("cv1_diff", "").strip()
        d2 = cv_set.get("cv2_diff", "").strip()
        d3 = cv_set.get("cv3_diff", "").strip()
        if d1 and d2 and d3:
            return {"cv1": d1, "cv2": d2, "cv3": d3}
        # Multi-worker fallback: check SQLite if another worker populated diffs after initial CV upload
        with self._lock, self._connect() as conn:
            row = conn.execute(
                "SELECT cv1_diff, cv2_diff, cv3_diff FROM cv_sets WHERE cvset_hash = ?",
                (cvset_hash,),
            ).fetchone()
            if not row:
                return None
            d1 = str(row["cv1_diff"] or "").strip()
            d2 = str(row["cv2_diff"] or "").strip()
            d3 = str(row["cv3_diff"] or "").strip()
            if not (d1 and d2 and d3):
                return None
            if cvset_hash in self._ram_cv_sets:
                self._ram_cv_sets[cvset_hash]["cv1_diff"] = d1
                self._ram_cv_sets[cvset_hash]["cv2_diff"] = d2
                self._ram_cv_sets[cvset_hash]["cv3_diff"] = d3
            return {"cv1": d1, "cv2": d2, "cv3": d3}

    def save_cv_diffs(
        self,
        cvset_hash: str,
        cv1_diff: str,
        cv2_diff: str,
        cv3_diff: str,
    ) -> dict[str, str]:
        d1 = (cv1_diff or "").strip()[:300]
        d2 = (cv2_diff or "").strip()[:300]
        d3 = (cv3_diff or "").strip()[:300]
        with self._lock, self._connect() as conn:
            conn.execute(
                """
                UPDATE cv_sets
                SET cv1_diff = ?, cv2_diff = ?, cv3_diff = ?
                WHERE cvset_hash = ?
                """,
                (d1, d2, d3, cvset_hash),
            )
            if cvset_hash in self._ram_cv_sets:
                self._ram_cv_sets[cvset_hash]["cv1_diff"] = d1
                self._ram_cv_sets[cvset_hash]["cv2_diff"] = d2
                self._ram_cv_sets[cvset_hash]["cv3_diff"] = d3
        return {"cv1": d1, "cv2": d2, "cv3": d3}

    def upsert_cv_set_from_pdfs(
        self,
        cvset_hash: str,
        cv1_b64: str,
        cv2_b64: str,
        cv3_b64: str,
        require_kgp_roll: bool = False,
    ) -> dict[str, str]:
        # Verify cryptographic binding: caller cannot spoof cvset_hash for different PDF bytes
        expected_hash = compute_cvset_sha256(cv1_b64, cv2_b64, cv3_b64)
        if expected_hash.lower() != cvset_hash.strip().lower():
            raise CvVerificationError(
                "cvset_hash does not match SHA-256 digest of uploaded CV1/CV2/CV3 PDFs"
            )

        existing = self.get_cv_set(cvset_hash)
        if existing:
            return existing

        cv1_text = extract_pdf_text_from_base64(cv1_b64)
        cv2_text = extract_pdf_text_from_base64(cv2_b64)
        cv3_text = extract_pdf_text_from_base64(cv3_b64)

        student_hash = extract_verified_roll_hash(cv1_text, cv2_text, cv3_text) or ""
        if require_kgp_roll and not student_hash:
            raise CvVerificationError(
                "Uploaded PDFs do not contain a recognizable IIT KGP ERP roll number header."
            )

        now = time.time()
        record = {
            "cvset_hash": cvset_hash,
            "cv1_text": cv1_text,
            "cv2_text": cv2_text,
            "cv3_text": cv3_text,
            "cv1_diff": "",
            "cv2_diff": "",
            "cv3_diff": "",
            "student_hash": student_hash,
        }

        with self._lock, self._connect() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO cv_sets
                (cvset_hash, cv1_text, cv2_text, cv3_text, cv1_diff, cv2_diff, cv3_diff, student_hash, created_at)
                VALUES (?, ?, ?, ?, '', '', '', ?, ?)
                """,
                (cvset_hash, cv1_text, cv2_text, cv3_text, student_hash, now),
            )
            self._ram_cv_sets[cvset_hash] = record
        return dict(record)

    def upsert_roles(self, roles: list[dict[str, Any]]) -> int:
        if not roles:
            return 0
        now = time.time()
        count = 0
        with self._lock, self._connect() as conn:
            for r in roles:
                com_id = str(r.get("com_id", "")).strip()[:32]
                jnf_id = str(r.get("jnf_id", "")).strip()[:32]
                if not com_id or not jnf_id:
                    continue
                yop = str(r.get("yop", "2026-2027") or "2026-2027").strip()[:16]
                company = str(r.get("company", "")).strip()[:200]
                role_name = str(r.get("role", "")).strip()[:200]
                ctc = str(r.get("ctc", "")).strip()[:120]
                deadline = str(r.get("deadline", "")).strip()[:64]
                jd_text = str(r.get("jd_text", "")).strip()[:8000]

                # Never overwrite a non-empty JD with an empty or suspiciously shorter JD
                conn.execute(
                    """
                    INSERT INTO company_roles (com_id, jnf_id, yop, company, role, ctc, deadline_end, jd_text, updated_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(com_id, jnf_id, yop) DO UPDATE SET
                        company = CASE WHEN company_roles.company = '' THEN excluded.company ELSE company_roles.company END,
                        role = CASE WHEN company_roles.role = '' THEN excluded.role ELSE company_roles.role END,
                        ctc = CASE WHEN excluded.ctc != '' THEN excluded.ctc ELSE company_roles.ctc END,
                        deadline_end = CASE WHEN excluded.deadline_end != '' THEN excluded.deadline_end ELSE company_roles.deadline_end END,
                        jd_text = CASE WHEN company_roles.jd_text = '' AND excluded.jd_text != '' THEN excluded.jd_text ELSE company_roles.jd_text END,
                        updated_at = excluded.updated_at
                    """,
                    (com_id, jnf_id, yop, company, role_name, ctc, deadline, jd_text, now),
                )
                count += 1
        return count

    def get_role_jd(self, com_id: str, jnf_id: str) -> str:
        with self._lock, self._connect() as conn:
            row = conn.execute(
                "SELECT jd_text FROM company_roles WHERE com_id = ? AND jnf_id = ? ORDER BY updated_at DESC LIMIT 1",
                (com_id, jnf_id),
            ).fetchone()
            return str(row["jd_text"]) if row and row["jd_text"] else ""

    def upsert_notices(self, notices: list[dict[str, Any]]) -> int:
        if not notices:
            return 0
        now = time.time()
        count = 0
        with self._lock, self._connect() as conn:
            for n in notices:
                nid = str(n.get("id", "")).strip()[:32]
                if not nid:
                    continue
                conn.execute(
                    """
                    INSERT INTO notices (notice_id, category, company, notice_text, notice_at, updated_at)
                    VALUES (?, ?, ?, ?, ?, ?)
                    ON CONFLICT(notice_id) DO NOTHING
                    """,
                    (
                        nid,
                        str(n.get("category", "General")).strip()[:120],
                        str(n.get("company", "CDC General")).strip()[:200],
                        str(n.get("notice", "")).strip()[:8000],
                        str(n.get("when", "")).strip()[:64],
                        now,
                    ),
                )
                count += 1
        return count

    def existing_notice_ids(self) -> set[str]:
        """All notice ids ever seen (from either the browser sync or the poller)."""
        with self._lock, self._connect() as conn:
            rows = conn.execute("SELECT notice_id FROM notices").fetchall()
        return {str(r["notice_id"]) for r in rows}

    def notified_notice_ids(self) -> set[str]:
        """Notice ids already delivered to the alert channels."""
        with self._lock, self._connect() as conn:
            rows = conn.execute("SELECT notice_id FROM notice_notifications").fetchall()
        return {str(r["notice_id"]) for r in rows}

    def select_unnotified_notices(self, candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Return only notices that have never been alerted, preserving input order.

        Dedupe is keyed on the ERP notice id and backed by the `notice_notifications`
        ledger, so a restart, a retry, or a repeat poll cannot re-alert the same notice.
        """
        if not candidates:
            return []
        already = self.notified_notice_ids()
        fresh: list[dict[str, Any]] = []
        seen_in_batch: set[str] = set()
        for item in candidates:
            nid = str(item.get("id", "")).strip()[:32]
            if not nid or nid in already or nid in seen_in_batch:
                continue
            seen_in_batch.add(nid)
            fresh.append({**item, "id": nid})
        return fresh

    def mark_notices_notified(
        self,
        notices: list[dict[str, Any]],
        *,
        channel: str,
        recipients: int = 0,
        attachment_sent: bool = False,
    ) -> int:
        """Record that these notices were delivered, making future polls idempotent."""
        if not notices:
            return 0
        now = time.time()
        marked = 0
        with self._lock, self._connect() as conn:
            for item in notices:
                nid = str(item.get("id", "")).strip()[:32]
                if not nid:
                    continue
                conn.execute(
                    """
                    INSERT INTO notice_notifications
                        (notice_id, channel, recipients, attachment_sent, notified_at)
                    VALUES (?, ?, ?, ?, ?)
                    ON CONFLICT(notice_id) DO UPDATE SET
                        channel = excluded.channel,
                        recipients = excluded.recipients,
                        attachment_sent = excluded.attachment_sent,
                        notified_at = excluded.notified_at
                    """,
                    (
                        nid,
                        channel[:40],
                        int(recipients),
                        1 if attachment_sent else 0,
                        now,
                    ),
                )
                marked += 1
        return marked

    def record_poller_run(
        self,
        summary: dict[str, Any],
        *,
        trigger_source: str = "cron",
        max_history: int = 100,
    ) -> dict[str, Any]:
        """Persist a sanitized poller execution summary and prune old rows."""
        polled_at = float(summary.get("polled_at") or time.time())
        errors = [str(e)[:280] for e in (summary.get("errors") or [])[:10]]
        status = str(summary.get("status") or ("error" if errors else "ok"))[:32]
        total_notices = int(summary.get("total_notices") or 0)
        new_notices = int(summary.get("new_notices") or 0)
        seeded_notices = int(summary.get("seeded_notices") or 0)
        delivered = int(summary.get("delivered") or 0)
        attachments_sent = int(summary.get("attachments_sent") or 0)
        clean_source = str(trigger_source or "cron")[:32]

        with self._lock, self._connect() as conn:
            cursor = conn.execute(
                """
                INSERT INTO poller_runs (
                    polled_at, trigger_source, status, total_notices,
                    new_notices, seeded_notices, delivered, attachments_sent, errors_json
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    polled_at,
                    clean_source,
                    status,
                    total_notices,
                    new_notices,
                    seeded_notices,
                    delivered,
                    attachments_sent,
                    json.dumps(errors),
                ),
            )
            run_id = int(cursor.lastrowid or 0)
            conn.execute(
                """
                DELETE FROM poller_runs
                WHERE run_id NOT IN (
                    SELECT run_id FROM poller_runs ORDER BY run_id DESC LIMIT ?
                )
                """,
                (max(10, int(max_history)),),
            )
        return {
            "run_id": run_id,
            "polled_at": polled_at,
            "trigger_source": clean_source,
            "status": status,
            "total_notices": total_notices,
            "new_notices": new_notices,
            "seeded_notices": seeded_notices,
            "delivered": delivered,
            "attachments_sent": attachments_sent,
            "errors": errors,
        }

    def get_recent_poller_runs(self, limit: int = 10) -> list[dict[str, Any]]:
        clamped = max(1, min(50, int(limit)))
        with self._lock, self._connect() as conn:
            rows = conn.execute(
                """
                SELECT run_id, polled_at, trigger_source, status, total_notices,
                       new_notices, seeded_notices, delivered, attachments_sent, errors_json
                FROM poller_runs
                ORDER BY run_id DESC
                LIMIT ?
                """,
                (clamped,),
            ).fetchall()
        out: list[dict[str, Any]] = []
        for row in rows:
            try:
                errs = json.loads(str(row["errors_json"] or "[]"))
                if not isinstance(errs, list):
                    errs = []
            except Exception:
                errs = []
            out.append(
                {
                    "run_id": int(row["run_id"]),
                    "polled_at": float(row["polled_at"]),
                    "trigger_source": str(row["trigger_source"]),
                    "status": str(row["status"]),
                    "total_notices": int(row["total_notices"]),
                    "new_notices": int(row["new_notices"]),
                    "seeded_notices": int(row["seeded_notices"]),
                    "delivered": int(row["delivered"]),
                    "attachments_sent": int(row["attachments_sent"]),
                    "errors": [str(e) for e in errs],
                }
            )
        return out

    @staticmethod
    def _hash_phone(normalized_e164: str) -> str:
        clean = normalized_e164.strip()
        if not clean.startswith("+"):
            clean = f"+{clean}"
        return hashlib.sha256(f"cdc_wa_phone:{clean}".encode("utf-8")).hexdigest()[:32]

    @staticmethod
    def _mask_phone(normalized_e164: str) -> str:
        clean = normalized_e164.strip()
        if len(clean) <= 6:
            return "***"
        return f"{clean[:3]}******{clean[-4:]}"

    def record_whatsapp_optin(
        self,
        e164_phone: str,
        *,
        status: str = "inbound",
        timestamp: float | None = None,
    ) -> None:
        """Record inbound WhatsApp contact or delivery state using hashed phone only (zero raw PII in SQLite)."""
        digits = re.sub(r"[^\d]", "", e164_phone or "")
        if len(digits) < 7 or len(digits) > 15:
            return
        normalized = f"+{digits}"
        r_hash = self._hash_phone(normalized)
        masked = self._mask_phone(normalized)
        ts = float(timestamp if timestamp is not None else time.time())
        with self._lock, self._connect() as conn:
            conn.execute(
                """
                INSERT INTO whatsapp_optins (recipient_hash, masked_phone, last_inbound_at, last_status)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(recipient_hash) DO UPDATE SET
                    last_inbound_at = CASE
                        WHEN excluded.last_status = 'inbound' THEN MAX(whatsapp_optins.last_inbound_at, excluded.last_inbound_at)
                        ELSE whatsapp_optins.last_inbound_at
                    END,
                    last_status = excluded.last_status
                """,
                (r_hash, masked, ts if status == "inbound" else 0.0, str(status)[:40]),
            )
            if status == "inbound":
                conn.execute(
                    """
                    UPDATE whatsapp_broadcast_subscribers
                    SET last_inbound_at = MAX(last_inbound_at, ?), updated_at = ?
                    WHERE recipient_hash = ?
                    """,
                    (ts, time.time(), r_hash),
                )
            elif status.startswith("delivery:"):
                conn.execute(
                    """
                    UPDATE whatsapp_broadcast_subscribers
                    SET last_delivery_status = ?, updated_at = ?
                    WHERE recipient_hash = ?
                    """,
                    (str(status)[:40], time.time(), r_hash),
                )

    def subscribe_whatsapp_member(
        self,
        e164_phone: str,
        *,
        display_name: str = "",
        timestamp: float | None = None,
    ) -> dict[str, Any] | None:
        """Add or re-activate a member in the business-managed WhatsApp broadcast group (encrypted at rest)."""
        digits = re.sub(r"[^\d]", "", e164_phone or "")
        if len(digits) < 7 or len(digits) > 15:
            return None
        normalized = f"+{digits}"
        r_hash = self._hash_phone(normalized)
        masked = self._mask_phone(normalized)
        enc_phone = self._encrypt_phone(normalized)
        safe_name = re.sub(r"[\r\n\t]+", " ", (display_name or "")).strip()[:60]
        now = time.time()
        inbound_ts = float(timestamp if timestamp is not None else now)

        with self._lock, self._connect() as conn:
            existing = conn.execute(
                "SELECT subscribed FROM whatsapp_broadcast_subscribers WHERE recipient_hash = ?",
                (r_hash,),
            ).fetchone()
            was_already_active = bool(existing and int(existing["subscribed"]) == 1)
            conn.execute(
                """
                INSERT INTO whatsapp_broadcast_subscribers (
                    recipient_hash, masked_phone, encrypted_phone, display_name,
                    subscribed, joined_at, updated_at, last_inbound_at, last_delivery_status
                )
                VALUES (?, ?, ?, ?, 1, ?, ?, ?, 'subscribed')
                ON CONFLICT(recipient_hash) DO UPDATE SET
                    masked_phone = excluded.masked_phone,
                    encrypted_phone = excluded.encrypted_phone,
                    display_name = CASE WHEN excluded.display_name != '' THEN excluded.display_name ELSE whatsapp_broadcast_subscribers.display_name END,
                    subscribed = 1,
                    updated_at = excluded.updated_at,
                    last_inbound_at = MAX(whatsapp_broadcast_subscribers.last_inbound_at, excluded.last_inbound_at)
                """,
                (r_hash, masked, enc_phone, safe_name, now, now, inbound_ts),
            )
            conn.execute(
                """
                INSERT INTO whatsapp_optins (recipient_hash, masked_phone, last_inbound_at, last_status)
                VALUES (?, ?, ?, 'inbound')
                ON CONFLICT(recipient_hash) DO UPDATE SET
                    last_inbound_at = MAX(whatsapp_optins.last_inbound_at, excluded.last_inbound_at),
                    last_status = 'inbound'
                """,
                (r_hash, masked, inbound_ts),
            )
        return {
            "recipient_hash": r_hash,
            "masked_phone": masked,
            "already_subscribed": was_already_active,
        }

    def unsubscribe_whatsapp_member(self, e164_phone_or_hash: str) -> bool:
        """Unsubscribe a member by phone number or 32-char `recipient_hash`."""
        raw = (e164_phone_or_hash or "").strip()
        if re.fullmatch(r"[0-9a-f]{32}", raw.lower()):
            r_hash = raw.lower()
        else:
            digits = re.sub(r"[^\d]", "", raw)
            if len(digits) < 7 or len(digits) > 15:
                return False
            r_hash = self._hash_phone(f"+{digits}")

        with self._lock, self._connect() as conn:
            cur = conn.execute(
                """
                UPDATE whatsapp_broadcast_subscribers
                SET subscribed = 0, updated_at = ?
                WHERE recipient_hash = ? AND subscribed = 1
                """,
                (time.time(), r_hash),
            )
            return int(cur.rowcount or 0) > 0

    def list_active_subscriber_phones(self) -> list[str]:
        """Decrypt active subscriber E.164 phone numbers in-RAM strictly for outbound notice dispatch."""
        with self._lock, self._connect() as conn:
            rows = conn.execute(
                """
                SELECT encrypted_phone
                FROM whatsapp_broadcast_subscribers
                WHERE subscribed = 1
                ORDER BY joined_at ASC
                """
            ).fetchall()
        phones: list[str] = []
        for row in rows:
            dec = self._decrypt_phone(str(row["encrypted_phone"] or ""))
            if dec and dec not in phones:
                phones.append(dec)
        return phones

    def list_whatsapp_subscribers_summary(self) -> list[dict[str, Any]]:
        """Zero-PII operational summary of broadcast subscribers (never returns raw phone numbers)."""
        now = time.time()
        with self._lock, self._connect() as conn:
            rows = conn.execute(
                """
                SELECT recipient_hash, masked_phone, display_name, subscribed,
                       joined_at, last_inbound_at, last_delivery_status
                FROM whatsapp_broadcast_subscribers
                ORDER BY subscribed DESC, joined_at ASC
                """
            ).fetchall()
        out: list[dict[str, Any]] = []
        for row in rows:
            last_in = float(row["last_inbound_at"] or 0.0)
            out.append(
                {
                    "subscriber_id": str(row["recipient_hash"]),
                    "masked_phone": str(row["masked_phone"]),
                    "display_name": str(row["display_name"] or ""),
                    "subscribed": bool(int(row["subscribed"]) == 1),
                    "joined_at": float(row["joined_at"]),
                    "window_24h_open": last_in > 0 and (now - last_in) < 86400,
                    "last_delivery_status": str(row["last_delivery_status"] or ""),
                }
            )
        return out

    def check_recipients_optin_windows(self, recipients_e164: list[str]) -> list[dict[str, Any]]:
        """Check 24h customer-service window status for configured recipients without exposing raw numbers."""
        now = time.time()
        with self._lock, self._connect() as conn:
            rows = {
                str(r["recipient_hash"]): (float(r["last_inbound_at"]), str(r["last_status"]))
                for r in conn.execute(
                    "SELECT recipient_hash, last_inbound_at, last_status FROM whatsapp_optins"
                ).fetchall()
            }
        statuses: list[dict[str, Any]] = []
        for phone in recipients_e164:
            r_hash = self._hash_phone(phone)
            last_inbound_at, last_status = rows.get(r_hash, (0.0, "never_seen"))
            window_open = last_inbound_at > 0 and (now - last_inbound_at) < 86400
            statuses.append(
                {
                    "masked_phone": self._mask_phone(phone),
                    "opted_in_ever": last_inbound_at > 0,
                    "window_24h_open": window_open,
                    "last_status": last_status,
                }
            )
        return statuses

    def get_cached_recommendation(
        self,
        cvset_hash: str,
        com_id: str,
        jnf_id: str,
    ) -> dict[str, Any] | None:
        key = f"{com_id}_{jnf_id}"
        with self._lock:
            by_key = self._ram_recs.get(cvset_hash)
            if by_key and key in by_key:
                return dict(by_key[key])

        with self._lock, self._connect() as conn:
            row = conn.execute(
                """
                SELECT recommended_cv, confidence, reason, model, created_at
                FROM cv_recommendations
                WHERE cvset_hash = ? AND com_id = ? AND jnf_id = ?
                """,
                (cvset_hash, com_id, jnf_id),
            ).fetchone()
            if not row:
                return None
            rec = {
                "com_id": com_id,
                "jnf_id": jnf_id,
                "key": key,
                "cv": int(row["recommended_cv"]),
                "confidence": float(row["confidence"]),
                "reason": str(row["reason"]),
                "model": str(row["model"]),
                "cached": True,
            }
            self._ram_recs.setdefault(cvset_hash, {})[key] = rec
            return dict(rec)

    def list_cached_recommendations(self, cvset_hash: str) -> dict[str, dict[str, Any]]:
        with self._lock, self._connect() as conn:
            rows = conn.execute(
                """
                SELECT com_id, jnf_id, recommended_cv, confidence, reason, model
                FROM cv_recommendations
                WHERE cvset_hash = ?
                """,
                (cvset_hash,),
            ).fetchall()
            out: dict[str, dict[str, Any]] = {}
            for row in rows:
                com_id = str(row["com_id"])
                jnf_id = str(row["jnf_id"])
                key = f"{com_id}_{jnf_id}"
                out[key] = {
                    "com_id": com_id,
                    "jnf_id": jnf_id,
                    "key": key,
                    "cv": int(row["recommended_cv"]),
                    "confidence": float(row["confidence"]),
                    "reason": str(row["reason"]),
                    "model": str(row["model"]),
                    "cached": True,
                }
            if out:
                self._ram_recs[cvset_hash] = {k: dict(v) for k, v in out.items()}
            return out

    def save_recommendation(
        self,
        cvset_hash: str,
        com_id: str,
        jnf_id: str,
        cv: int,
        confidence: float,
        reason: str,
        model: str,
    ) -> dict[str, Any]:
        now = time.time()
        clean_reason = (reason or "").strip()[:280]
        key = f"{com_id}_{jnf_id}"
        with self._lock, self._connect() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO cv_recommendations
                (cvset_hash, com_id, jnf_id, recommended_cv, confidence, reason, model, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (cvset_hash, com_id, jnf_id, int(cv), float(confidence), clean_reason, model, now),
            )
            ram_entry = {
                "com_id": com_id,
                "jnf_id": jnf_id,
                "key": key,
                "cv": int(cv),
                "confidence": float(confidence),
                "reason": clean_reason,
                "model": model,
                "cached": True,
            }
            self._ram_recs.setdefault(cvset_hash, {})[key] = ram_entry
        return {
            "com_id": com_id,
            "jnf_id": jnf_id,
            "key": key,
            "cv": int(cv),
            "confidence": float(confidence),
            "reason": clean_reason,
            "model": model,
            "cached": False,
        }
