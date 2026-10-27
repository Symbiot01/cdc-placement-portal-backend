"""Persistent SQLite + In-RAM Hot Cache store for CV PDFs text, per-CV diff summaries, CDC roles/notices, and per-(CVset,company,role) AI matches."""

from __future__ import annotations

import base64
import hashlib
import io
import re
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
        self._init_schema()
        self._warm_ram_cache()

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
