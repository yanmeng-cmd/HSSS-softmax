"""Shared cache paths and process-local environment setup for Longformer experiments."""

from __future__ import annotations

import os
import socket
import sys
import urllib.parse
from pathlib import Path


LONGFORMER_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = LONGFORMER_DIR.parent
HF_CACHE_DIR = LONGFORMER_DIR / ".hf-cache"
HF_HUB_CACHE_DIR = HF_CACHE_DIR / "hub"
HF_DATASETS_CACHE_DIR = HF_CACHE_DIR / "datasets"


def _drop_invalid_proxy_env() -> None:
    """Remove stale proxy schemes unsupported by httpx in this process only."""
    for key in ("ALL_PROXY", "all_proxy", "HTTPS_PROXY", "https_proxy", "HTTP_PROXY", "http_proxy"):
        value = os.environ.get(key, "")
        if _should_remove_proxy(value):
            os.environ.pop(key, None)


def _should_remove_proxy(value: str) -> bool:
    if not value:
        return False
    parsed = urllib.parse.urlparse(value)
    scheme = parsed.scheme.lower()
    if scheme.startswith("socks"):
        return True
    host = (parsed.hostname or "").lower()
    if host not in {"localhost", "::1"} and not host.startswith("127."):
        return False
    if parsed.port is None:
        return True
    try:
        with socket.create_connection((host, parsed.port), timeout=0.2):
            return False
    except OSError:
        return True


def configure_environment() -> None:
    """Pin Hugging Face caches under longformer and expose sibling packages."""
    _drop_invalid_proxy_env()
    os.environ.setdefault("HF_HOME", str(HF_CACHE_DIR))
    os.environ.setdefault("HF_HUB_CACHE", str(HF_HUB_CACHE_DIR))
    os.environ.setdefault("HUGGINGFACE_HUB_CACHE", str(HF_HUB_CACHE_DIR))
    os.environ.setdefault("HF_DATASETS_CACHE", str(HF_DATASETS_CACHE_DIR))
    os.environ.setdefault("EVALUATE_HOME", str(HF_CACHE_DIR / "evaluate"))
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    project_root = str(PROJECT_ROOT)
    if project_root not in sys.path:
        sys.path.insert(0, project_root)


def cached_longformer_model_path(model_repo: str = "allenai/longformer-base-4096") -> Path:
    """Return the latest downloaded model snapshot if it exists."""
    repo_cache_name = "models--" + model_repo.replace("/", "--")
    snapshots_dir = HF_HUB_CACHE_DIR / repo_cache_name / "snapshots"
    if not snapshots_dir.exists():
        return snapshots_dir
    snapshots = sorted(
        (path for path in snapshots_dir.iterdir() if path.is_dir()),
        key=lambda path: path.stat().st_mtime,
    )
    return snapshots[-1] if snapshots else snapshots_dir
