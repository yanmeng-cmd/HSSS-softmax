#!/usr/bin/env python3
"""Download Longformer and WikiText-2 assets for the fixed-mask MLM experiment."""

from __future__ import annotations

import argparse
import os
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path


LONGFORMER_DIR = Path(__file__).resolve().parent
if str(LONGFORMER_DIR) not in sys.path:
    sys.path.insert(0, str(LONGFORMER_DIR))

from experiment_paths import HF_DATASETS_CACHE_DIR, HF_HUB_CACHE_DIR, configure_environment


configure_environment()


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

WIKITEXT_PARQUET_FILES = {
    "wikitext-2-raw-v1": {
        "train": "train-00000-of-00001.parquet",
        "validation": "validation-00000-of-00001.parquet",
        "test": "test-00000-of-00001.parquet",
    }
}


def local_dataset_dir(dataset: str, config: str) -> Path:
    return HF_DATASETS_CACHE_DIR / "direct" / dataset.replace("/", "--") / config


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


def read_parquet_text_count(path: Path, text_column: str = "text") -> tuple[int, int] | None:
    try:
        import pyarrow.parquet as pq
    except ModuleNotFoundError:
        return None

    table = pq.read_table(path, columns=[text_column])
    texts = table[text_column].to_pylist()
    nonempty = sum(1 for text in texts if isinstance(text, str) and text.strip())
    return len(texts), nonempty


def download_file(url: str, output_path: Path, timeout: int) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = output_path.with_suffix(output_path.suffix + ".tmp")
    opener = urllib.request.build_opener()
    current_url = url
    for _ in range(8):
        request = urllib.request.Request(current_url, headers={"User-Agent": "softmax-longformer-experiment"})
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
    if dataset != "Salesforce/wikitext" or config not in WIKITEXT_PARQUET_FILES:
        raise ValueError(f"direct dataset mode only supports Salesforce/wikitext {sorted(WIKITEXT_PARQUET_FILES)}")

    dataset_dir = local_dataset_dir(dataset, config)
    print(f"dataset_cache: {HF_DATASETS_CACHE_DIR}")
    print(f"direct_dataset_dir: {dataset_dir}")
    print(f"direct_endpoints: {', '.join(endpoint_candidates(endpoints))}")

    for split in splits:
        if split not in WIKITEXT_PARQUET_FILES[config]:
            raise ValueError(f"unknown split for {config}: {split}")
        filename = WIKITEXT_PARQUET_FILES[config][split]
        output_path = dataset_dir / filename
        if output_path.exists() and output_path.stat().st_size > 0:
            print(f"{split}: exists {output_path} ({output_path.stat().st_size} bytes)")
        else:
            last_error: Exception | None = None
            for endpoint in endpoint_candidates(endpoints):
                url = f"{endpoint}/datasets/{dataset}/resolve/main/{config}/{filename}"
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

        counts = read_parquet_text_count(output_path)
        if counts is None:
            print(f"{dataset}/{config}/{split}: file={output_path} size={output_path.stat().st_size}")
        else:
            rows, nonempty = counts
            print(f"{dataset}/{config}/{split}: rows={rows} nonempty={nonempty} file={output_path}")


def download_dataset_hf(dataset: str, config: str, splits: list[str]) -> None:
    from datasets import load_dataset

    print(f"dataset_cache: {HF_DATASETS_CACHE_DIR}")
    for split in splits:
        ds = load_dataset(dataset, config, split=split, cache_dir=str(HF_DATASETS_CACHE_DIR))
        nonempty = sum(1 for text in ds["text"] if isinstance(text, str) and text.strip())
        print(f"{dataset}/{config}/{split}: rows={len(ds)} nonempty={nonempty} columns={ds.column_names}")


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
    parser = argparse.ArgumentParser(description="Download cached assets for the Longformer MLM experiment")
    parser.add_argument("--model", type=str, default="allenai/longformer-base-4096")
    parser.add_argument("--dataset", type=str, default="Salesforce/wikitext")
    parser.add_argument("--dataset-config", type=str, default="wikitext-2-raw-v1")
    parser.add_argument("--splits", type=str, default="train,validation,test")
    parser.add_argument(
        "--dataset-mode",
        choices=("auto", "direct", "hf"),
        default="auto",
        help="auto/direct avoids the Hugging Face dataset metadata HEAD request by downloading parquet files directly",
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
    parser.add_argument("--timeout", type=int, default=30)
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
