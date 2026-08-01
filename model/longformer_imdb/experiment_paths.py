"""Cache paths and process-local environment setup for Longformer IMDB experiments."""

from __future__ import annotations

import os
import socket
import sys
import urllib.parse
from pathlib import Path


LONGFORMER2_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = LONGFORMER2_DIR.parent
HF_CACHE_DIR = LONGFORMER2_DIR / ".hf-cache"
HF_HUB_CACHE_DIR = HF_CACHE_DIR / "hub"
HF_DATASETS_CACHE_DIR = HF_CACHE_DIR / "datasets"
RESULTS_DIR = LONGFORMER2_DIR / "results"


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
    """Pin Hugging Face caches under longformer2 and expose sibling packages."""
    _drop_invalid_proxy_env()
    os.environ.setdefault("HF_HOME", str(HF_CACHE_DIR))
    os.environ.setdefault("HF_HUB_CACHE", str(HF_HUB_CACHE_DIR))
    os.environ.setdefault("HUGGINGFACE_HUB_CACHE", str(HF_HUB_CACHE_DIR))
    os.environ.setdefault("HF_DATASETS_CACHE", str(HF_DATASETS_CACHE_DIR))
    os.environ.setdefault("EVALUATE_HOME", str(HF_CACHE_DIR / "evaluate"))
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    for path in (str(PROJECT_ROOT), str(LONGFORMER2_DIR)):
        if path not in sys.path:
            sys.path.insert(0, path)


def cached_model_path(model_repo: str) -> Path:
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


def direct_dataset_dir(dataset: str, config: str) -> Path:
    """Return the direct-download dataset directory."""
    return HF_DATASETS_CACHE_DIR / "direct" / dataset.replace("/", "--") / config
