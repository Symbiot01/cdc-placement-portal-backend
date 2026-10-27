"""Auditable public interface for the AI CV Matcher.

Open-source callers can inspect `backend/app.py`, `backend/cache_store.py`, and this file
to verify that ONLY non-credential CV text and role/JD fields are passed to the matcher.
The proprietary prompt & scoring implementation lives in `backend/private/proprietary_matcher.py`
(gitignored via `**/private/`) or an external module specified by `CDC_PROPRIETARY_MATCHER_MODULE`.
"""

from __future__ import annotations

import importlib
import os
from pathlib import Path
from typing import Any


class LlmUnavailableError(RuntimeError):
    """Raised when server LLM keys/proprietary module are missing or the upstream LLM API fails."""


def load_dotenv_file(env_path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    if not env_path.is_file():
        return values
    for raw_line in env_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        key = k.strip()
        val = v.strip().strip("'").strip('"')
        if key:
            values[key] = val
    return values


def get_llm_config(root_dir: Path | None = None) -> dict[str, str]:
    base = root_dir or Path(__file__).resolve().parent.parent
    file_env = load_dotenv_file(base / ".env")

    gemini_key = os.environ.get("GEMINI_API_KEY") or file_env.get("GEMINI_API_KEY", "")
    gemini_model = (
        os.environ.get("GEMINI_MODEL")
        or file_env.get("GEMINI_MODEL", "")
        or "gemini-2.5-flash"
    )
    openrouter_key = os.environ.get("OPENROUTER_API_KEY") or file_env.get("OPENROUTER_API_KEY", "")
    openrouter_model = (
        os.environ.get("OPENROUTER_MODEL")
        or file_env.get("OPENROUTER_MODEL", "")
        or "google/gemini-2.5-flash"
    )

    return {
        "gemini_api_key": gemini_key.strip(),
        "gemini_model": gemini_model.strip(),
        "openrouter_api_key": openrouter_key.strip(),
        "openrouter_model": openrouter_model.strip(),
    }


def _load_proprietary_runner():
    module_name = os.environ.get(
        "CDC_PROPRIETARY_MATCHER_MODULE",
        "backend.private.proprietary_matcher",
    )
    try:
        mod = importlib.import_module(module_name)
        return getattr(mod, "run_proprietary_cv_analysis")
    except Exception as exc:
        raise LlmUnavailableError(
            "Proprietary CV matcher plugin is not installed on this server instance."
        ) from exc


def analyze_cvs_and_roles_with_llm(
    cv1_text: str,
    cv2_text: str,
    cv3_text: str,
    roles: list[dict[str, Any]],
    include_cv_diffs: bool = True,
    root_dir: Path | None = None,
) -> tuple[dict[str, dict[str, Any]], dict[str, str] | None, str]:
    """Run a single batched LLM call returning (recommendations_by_key, cv_diffs, model_name)."""
    runner = _load_proprietary_runner()
    return runner(
        cv1_text=cv1_text,
        cv2_text=cv2_text,
        cv3_text=cv3_text,
        roles=roles,
        include_cv_diffs=include_cv_diffs,
        root_dir=root_dir,
    )


def match_roles_with_llm(
    cv1_text: str,
    cv2_text: str,
    cv3_text: str,
    roles: list[dict[str, Any]],
    root_dir: Path | None = None,
) -> tuple[dict[str, dict[str, Any]], str]:
    """Backwards-compatible wrapper returning (recommendations_by_key, model_name)."""
    by_key, _cv_diffs, used_model = analyze_cvs_and_roles_with_llm(
        cv1_text=cv1_text,
        cv2_text=cv2_text,
        cv3_text=cv3_text,
        roles=roles,
        include_cv_diffs=False,
        root_dir=root_dir,
    )
    return by_key, used_model
