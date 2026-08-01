#!/usr/bin/env python3
"""Download Longformer-IMDB model and IMDB dataset assets."""

from __future__ import annotations

import argparse
import os
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path


LONGFORMER2_DIR = Path(__file__).resolve().parent
if str(LONGFORMER2_DIR) not in sys.path:
    sys.path.insert(0, str(LONGFORMER2_DIR))

from experiment_paths import HF_DATASETS_CACHE_DIR, HF_HUB_CACHE_DIR, configure_environment, direct_dataset_dir


configure_environment()


DEFAULT_MODEL = "ahmed792002/Finetuning_Longformer_IMDb_movie_reviews_Classification"
DEFAULT_DATASET = "stanfordnlp/imdb"
DEFAULT_DATASET_CONFIG = "plain_text"

MODEL_ALLOW_PATTERNS = [
    "config.json",
    "vocab.json",
    "merges.txt",
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "model.safetensors",
    "pytorch_model.bin",
]

IMDB_PARQUET_FILES = {
    "plain_text": {
        "train": "plain_text/train-00000-of-00001.parquet",
        "test": "plain_text/test-00000-of-00001.parquet",
        "unsupervised": "plain_text/unsupervised-00000-of-00001.parquet",
    }
}


def endpoint_candidates(extra_endpoints: str) -> list[str]:
    candidates: list[str] = []
    for value in (extra_endpoints, os.environ.get("HF_ENDPOINT", ""), os.environ.get("HF_HUB_ENDPOINT", "")):
        for endpoint in value.split(","):
            endpoint = endpoint.strip().rstrip("/")
            if endpoint and endpoint not in candidates:
                candidates.append(endpoint)
    for endpoint in ("https://huggingface.co",):
        if endpoint not in candidates:
            candidates.append(endpoint)
    return candidates


def read_parquet_count(path: Path) -> tuple[int, dict[int, int]] | None:
    try:
        import pyarrow.parquet as pq
    except ModuleNotFoundError:
        return None

    table = pq.read_table(path, columns=["label"])
    labels = table["label"].to_pylist()
    counts: dict[int, int] = {}
    for label in labels:
        if label is None or int(label) < 0:
            continue
        counts[int(label)] = counts.get(int(label), 0) + 1
    return len(labels), counts


def download_file(url: str, output_path: Path, timeout: int) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = output_path.with_suffix(output_path.suffix + ".tmp")
    opener = urllib.request.build_opener()
    current_url = url
    for _ in range(8):
        request = urllib.request.Request(current_url, headers={"User-Agent": "softmax-longformer-imdb-experiment"})
        try:
            response_ctx = opener.open(request, timeout=timeout)
        except urllib.error.HTTPError as exc:
            if exc.code in (301, 302, 303, 307, 308) and exc.headers.get("Location"):
                current_url = urllib.parse.urljoin(current_url, exc.headers["Location"])
                continue
            raise
        with response_ctx as response:
            total = int(response.headers.get("Content-Length", "0") or 0)
            downloaded = 0
            with tmp_path.open("wb") as file_obj:
                while True:
                    chunk = response.read(1024 * 1024)
                    if not chunk:
                        break
                    file_obj.write(chunk)
                    downloaded += len(chunk)
                    if total:
                        print(f"\r{output_path.name}: {downloaded}/{total} bytes", end="", flush=True)
            if total:
                print()
        tmp_path.replace(output_path)
        return
    raise RuntimeError(f"too many redirects while downloading {url}")


def download_dataset_direct(
    dataset: str,
    config: str,
    splits: list[str],
    endpoints: str,
    timeout: int,
    retries: int,
) -> None:
    if dataset != DEFAULT_DATASET or config not in IMDB_PARQUET_FILES:
        raise ValueError(f"direct dataset mode only supports {DEFAULT_DATASET} {sorted(IMDB_PARQUET_FILES)}")

    dataset_dir = direct_dataset_dir(dataset, config)
    print(f"dataset_cache     : {HF_DATASETS_CACHE_DIR}")
    print(f"direct_dataset_dir: {dataset_dir}")
    print(f"direct_endpoints  : {', '.join(endpoint_candidates(endpoints))}")

    for split in splits:
        if split not in IMDB_PARQUET_FILES[config]:
            raise ValueError(f"unknown split for {config}: {split}")
        remote_path = IMDB_PARQUET_FILES[config][split]
        output_path = dataset_dir / Path(remote_path).name
        if output_path.exists() and output_path.stat().st_size > 0:
            print(f"{split}: exists {output_path} ({output_path.stat().st_size} bytes)")
        else:
            last_error: Exception | None = None
            for endpoint in endpoint_candidates(endpoints):
                url = f"{endpoint}/datasets/{dataset}/resolve/main/{remote_path}"
                for attempt in range(1, retries + 1):
                    try:
                        print(f"{split}: downloading {url} [attempt {attempt}/{retries}]")
                        download_file(url, output_path, timeout=timeout)
                        last_error = None
                        break
                    except (TimeoutError, urllib.error.URLError, urllib.error.HTTPError, OSError) as exc:
                        last_error = exc
                        print(f"{split}: failed from {endpoint}: {exc}")
                        time.sleep(min(attempt, 5))
                if output_path.exists() and output_path.stat().st_size > 0:
                    break
            if last_error is not None and not output_path.exists():
                raise RuntimeError(f"failed to download {split} parquet") from last_error

        counts = read_parquet_count(output_path)
        if counts is None:
            print(f"{dataset}/{config}/{split}: file={output_path} size={output_path.stat().st_size}")
        else:
            rows, label_counts = counts
            print(f"{dataset}/{config}/{split}: rows={rows} label_counts={label_counts} file={output_path}")


def download_dataset_hf(dataset: str, config: str | None, splits: list[str]) -> None:
    from datasets import load_dataset

    print(f"dataset_cache: {HF_DATASETS_CACHE_DIR}")
    for split in splits:
        if config:
            ds = load_dataset(dataset, config, split=split, cache_dir=str(HF_DATASETS_CACHE_DIR))
        else:
            ds = load_dataset(dataset, split=split, cache_dir=str(HF_DATASETS_CACHE_DIR))
        labels = [int(label) for label in ds["label"] if int(label) >= 0]
        label_counts = {label: labels.count(label) for label in sorted(set(labels))}
        print(f"{dataset}/{config or 'default'}/{split}: rows={len(ds)} label_counts={label_counts}")


def download_dataset(
    dataset: str,
    config: str,
    splits: list[str],
    mode: str,
    endpoints: str,
    timeout: int,
    retries: int,
) -> None:
    if mode in ("auto", "direct"):
        try:
            download_dataset_direct(dataset, config, splits, endpoints=endpoints, timeout=timeout, retries=retries)
            return
        except Exception as exc:
            if mode == "direct":
                raise
            print(f"direct dataset download failed, falling back to datasets API: {exc}")
    download_dataset_hf(dataset, config, splits)


def download_model(model: str, max_workers: int, endpoints: str, retries: int) -> str:
    from huggingface_hub import snapshot_download

    print(f"model_repo    : {model}")
    print(f"model_cache   : {HF_HUB_CACHE_DIR}")
    print(f"allow_patterns: {MODEL_ALLOW_PATTERNS}")
    last_error: Exception | None = None
    path = ""
    for endpoint in endpoint_candidates(endpoints):
        for attempt in range(1, retries + 1):
            try:
                print(f"model_endpoint: {endpoint} [attempt {attempt}/{retries}]")
                path = snapshot_download(
                    repo_id=model,
                    cache_dir=str(HF_HUB_CACHE_DIR),
                    allow_patterns=MODEL_ALLOW_PATTERNS,
                    max_workers=max_workers,
                    endpoint=endpoint,
                )
                last_error = None
                break
            except Exception as exc:
                last_error = exc
                print(f"model_download_failed: {type(exc).__name__}: {exc}")
                time.sleep(min(attempt, 5))
        if path:
            break
    if not path:
        raise RuntimeError(
            "failed to download model assets; check network access to Hugging Face "
            "or set HF_ENDPOINT/HTTPS_PROXY"
        ) from last_error
    print(f"snapshot_path : {path}")
    for file_path in sorted(Path(path).iterdir()):
        if file_path.is_file():
            print(f"{file_path.name}\t{file_path.stat().st_size}")
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description="Download cached assets for the Longformer IMDB experiment")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL)
    parser.add_argument("--dataset", type=str, default=DEFAULT_DATASET)
    parser.add_argument("--dataset-config", type=str, default=DEFAULT_DATASET_CONFIG)
    parser.add_argument("--splits", type=str, default="train,test")
    parser.add_argument(
        "--dataset-mode",
        choices=("auto", "direct", "hf"),
        default="auto",
        help="auto/direct avoids Hugging Face dataset metadata requests by downloading parquet files directly",
    )
    parser.add_argument(
        "--dataset-endpoints",
        type=str,
        default="",
        help="comma-separated dataset endpoints, for example https://hf-mirror.com,https://huggingface.co",
    )
    parser.add_argument(
        "--model-endpoints",
        type=str,
        default="",
        help="comma-separated model endpoints, for example https://hf-mirror.com,https://huggingface.co",
    )
    parser.add_argument("--timeout", type=int, default=60)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--skip-model", action="store_true")
    parser.add_argument("--skip-dataset", action="store_true")
    parser.add_argument("--max-workers", type=int, default=4)
    args = parser.parse_args()

    if not args.skip_dataset:
        splits = [split.strip() for split in args.splits.split(",") if split.strip()]
        download_dataset(
            args.dataset,
            args.dataset_config,
            splits,
            mode=args.dataset_mode,
            endpoints=args.dataset_endpoints,
            timeout=args.timeout,
            retries=args.retries,
        )
    if not args.skip_model:
        download_model(args.model, max_workers=args.max_workers, endpoints=args.model_endpoints, retries=args.retries)


if __name__ == "__main__":
    main()
