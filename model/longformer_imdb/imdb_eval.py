"""Evaluation helpers for Longformer IMDB sequence classification."""

from __future__ import annotations

import json
import math
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, List

import torch
import torch.nn.functional as F
from tqdm.auto import tqdm

try:
    from .experiment_paths import (
        HF_DATASETS_CACHE_DIR,
        RESULTS_DIR,
        cached_model_path,
        configure_environment,
        direct_dataset_dir,
    )
    from .hardware_softmax import DEFAULT_SOFTMAX_PROFILE, patched_softmax
except ImportError:  # pragma: no cover - supports direct script execution
    from experiment_paths import HF_DATASETS_CACHE_DIR, RESULTS_DIR, cached_model_path, configure_environment, direct_dataset_dir
    from hardware_softmax import DEFAULT_SOFTMAX_PROFILE, patched_softmax


configure_environment()


DEFAULT_MODEL = "ahmed792002/Finetuning_Longformer_IMDb_movie_reviews_Classification"
DEFAULT_DATASET = "stanfordnlp/imdb"
DEFAULT_DATASET_CONFIG = "plain_text"


@dataclass
class ClassificationResult:
    profile: str
    loss: float
    accuracy: float
    macro_f1: float
    correct: int
    examples: int
    label_counts: dict[int, int]
    prediction_counts: dict[int, int]
    predictions: List[int]
    labels: List[int]
    elapsed_sec: float
    softmax_stats: dict | None


def direct_dataset_file(dataset: str, config: str, split: str) -> Path:
    return direct_dataset_dir(dataset, config) / f"{split}-00000-of-00001.parquet"


def load_imdb_split(
    dataset: str,
    config: str,
    split: str,
    max_samples: int,
    local_files_only: bool,
    sample_mode: str,
    sample_seed: int,
) -> list[dict]:
    parquet_path = direct_dataset_file(dataset, config, split)
    if parquet_path.exists() and parquet_path.stat().st_size > 0:
        try:
            import pyarrow.parquet as pq
        except ModuleNotFoundError as exc:
            raise RuntimeError("pyarrow is required to read direct IMDB parquet files") from exc

        table = pq.read_table(parquet_path, columns=["text", "label"])
        rows = [
            {"text": str(text), "label": int(label)}
            for text, label in zip(table["text"].to_pylist(), table["label"].to_pylist())
            if isinstance(text, str) and text and int(label) >= 0
        ]
        print(f"Loaded local parquet corpus: {parquet_path}")
        return select_samples(rows, max_samples=max_samples, sample_mode=sample_mode, sample_seed=sample_seed)

    if local_files_only:
        raise FileNotFoundError(
            f"local IMDB parquet not found: {parquet_path}; run download_assets.py first or disable --local-files-only"
        )

    from datasets import load_dataset

    ds = load_dataset(dataset, config, split=split, cache_dir=str(HF_DATASETS_CACHE_DIR))
    if max_samples > 0:
        ds = ds.select(range(len(ds)))
    rows = [
        {"text": str(row["text"]), "label": int(row["label"])}
        for row in ds
        if isinstance(row["text"], str) and row["text"] and int(row["label"]) >= 0
    ]
    return select_samples(rows, max_samples=max_samples, sample_mode=sample_mode, sample_seed=sample_seed)


def select_samples(rows: list[dict], max_samples: int, sample_mode: str, sample_seed: int) -> list[dict]:
    if max_samples <= 0 or max_samples >= len(rows):
        return rows
    if sample_mode == "first":
        return rows[:max_samples]

    rng = random.Random(sample_seed)
    if sample_mode == "shuffle":
        indices = list(range(len(rows)))
        rng.shuffle(indices)
        selected = sorted(indices[:max_samples])
        return [rows[idx] for idx in selected]

    if sample_mode != "balanced":
        raise ValueError(f"unsupported sample_mode: {sample_mode}")

    by_label: dict[int, list[int]] = {}
    for idx, row in enumerate(rows):
        by_label.setdefault(int(row["label"]), []).append(idx)
    labels = sorted(by_label)
    if not labels:
        return []

    base = max_samples // len(labels)
    remainder = max_samples % len(labels)
    selected_indices: list[int] = []
    for pos, label in enumerate(labels):
        indices = by_label[label][:]
        rng.shuffle(indices)
        take = base + (1 if pos < remainder else 0)
        selected_indices.extend(indices[:take])
    return [rows[idx] for idx in sorted(selected_indices)]


def resolve_torch_dtype(dtype_name: str):
    if dtype_name in ("auto", "", "none"):
        return None
    mapping = {
        "float32": torch.float32,
        "fp32": torch.float32,
        "float16": torch.float16,
        "fp16": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
    }
    if dtype_name not in mapping:
        raise ValueError(f"unsupported model dtype: {dtype_name}")
    return mapping[dtype_name]


def resolve_device(device_name: str) -> torch.device:
    if device_name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device_name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    return device


def latest_or_repo_model(model_repo: str, local_files_only: bool) -> str:
    cached = cached_model_path(model_repo)
    if cached.exists() and cached.is_dir():
        return str(cached)
    if local_files_only:
        raise FileNotFoundError(f"local model snapshot not found for {model_repo}: {cached}")
    return model_repo


def load_model_and_tokenizer(model_repo: str, local_files_only: bool, model_dtype: str):
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    source = latest_or_repo_model(model_repo, local_files_only=local_files_only)
    dtype = resolve_torch_dtype(model_dtype)
    print(f"Loading {source}...")
    tokenizer = AutoTokenizer.from_pretrained(source, local_files_only=local_files_only, use_fast=True)
    model_kwargs = {"local_files_only": local_files_only}
    if dtype is not None:
        model_kwargs["torch_dtype"] = dtype
    model = AutoModelForSequenceClassification.from_pretrained(source, **model_kwargs)
    if getattr(model.config, "num_labels", None) != 2:
        print(f"warning: model num_labels={getattr(model.config, 'num_labels', None)}; IMDB expects 2")
    return model, tokenizer


def effective_max_length(model, requested_max_length: int) -> int:
    """Cap requested length to the model's safe RoBERTa/Longformer position range."""
    max_position_embeddings = getattr(getattr(model, "config", None), "max_position_embeddings", None)
    pad_token_id = getattr(getattr(model, "config", None), "pad_token_id", None)
    if not isinstance(max_position_embeddings, int) or max_position_embeddings <= 0:
        return requested_max_length

    # Longformer inherits RoBERTa-style position ids, where non-pad tokens start
    # after the pad id. In practice a 514-position checkpoint supports 512 tokens.
    reserved_positions = 2 if pad_token_id is not None else 0
    safe_length = max(1, max_position_embeddings - reserved_positions)
    if requested_max_length > safe_length:
        print(
            f"warning: requested max_length={requested_max_length} exceeds safe model length={safe_length} "
            f"(max_position_embeddings={max_position_embeddings}); using {safe_length}",
            flush=True,
        )
        return safe_length
    return requested_max_length


def set_torch_threads(num_threads: int, inter_op_threads: int) -> None:
    if num_threads > 0:
        torch.set_num_threads(num_threads)
    if inter_op_threads > 0:
        try:
            torch.set_num_interop_threads(inter_op_threads)
        except RuntimeError:
            pass
    print(f"torch_threads : intra_op={torch.get_num_threads()} inter_op={torch.get_num_interop_threads()}")


def batched(rows: list[dict], batch_size: int) -> Iterable[list[dict]]:
    for start in range(0, len(rows), batch_size):
        yield rows[start : start + batch_size]


def build_global_attention_mask(mode: str, input_ids: torch.Tensor) -> torch.Tensor | None:
    if mode == "auto":
        return None
    if mode == "none":
        return torch.zeros_like(input_ids, dtype=torch.long)
    if mode != "cls":
        raise ValueError(f"unsupported global attention mode: {mode}")
    mask = torch.zeros_like(input_ids, dtype=torch.long)
    if input_ids.numel() > 0:
        mask[:, 0] = 1
    return mask


def macro_f1(labels: list[int], predictions: list[int], num_labels: int) -> float:
    scores: list[float] = []
    for label_id in range(num_labels):
        tp = sum(1 for label, pred in zip(labels, predictions) if label == label_id and pred == label_id)
        fp = sum(1 for label, pred in zip(labels, predictions) if label != label_id and pred == label_id)
        fn = sum(1 for label, pred in zip(labels, predictions) if label == label_id and pred != label_id)
        if tp == 0 and fp == 0 and fn == 0:
            scores.append(0.0)
            continue
        precision = tp / (tp + fp) if tp + fp > 0 else 0.0
        recall = tp / (tp + fn) if tp + fn > 0 else 0.0
        scores.append(2 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0)
    return sum(scores) / len(scores) if scores else 0.0


def count_values(values: list[int]) -> dict[int, int]:
    counts: dict[int, int] = {}
    for value in values:
        counts[int(value)] = counts.get(int(value), 0) + 1
    return dict(sorted(counts.items()))


def order_rows_for_batching(
    rows: list[dict],
    tokenizer,
    max_length: int,
    batch_order: str,
) -> list[dict]:
    """Optionally group similar-length samples before batching."""
    if batch_order == "original":
        return rows
    if batch_order not in ("length_asc", "length_desc"):
        raise ValueError(f"unsupported batch_order: {batch_order}")

    print(f"ordering_samples: {batch_order}", flush=True)

    def effective_token_length(row: dict) -> int:
        encoded = tokenizer(
            row["text"],
            add_special_tokens=True,
            truncation=True,
            max_length=max_length,
            return_attention_mask=False,
        )
        return len(encoded["input_ids"])

    reverse = batch_order == "length_desc"
    return sorted(rows, key=effective_token_length, reverse=reverse)


def summarize_softmax_stats(stats: dict | None) -> str:
    if not stats:
        return "n/a"
    return (
        f"calls={stats['calls']} rows={stats['total_rows']} "
        f"pruned={stats['total_pruned_elements']}/{stats['total_elements']} ({100.0 * stats['total_pruned_rate']:.2f}%) "
        f"block_pruned={stats['block_pruned_blocks']}/{stats['total_blocks']} "
        f"({100.0 * stats['block_pruned_block_rate']:.2f}%) "
        f"padded={stats['padded_elements']}"
    )


def evaluate_profile(
    model,
    tokenizer,
    rows: list[dict],
    profile: str,
    device: torch.device,
    batch_size: int,
    max_length: int,
    global_attention: str,
    input_format: str,
) -> ClassificationResult:
    model.eval()
    model.to(device)
    num_labels = int(getattr(model.config, "num_labels", 2) or 2)

    total_loss = 0.0
    total_examples = 0
    correct = 0
    all_predictions: list[int] = []
    all_labels: list[int] = []
    start_time = time.time()

    with torch.no_grad(), patched_softmax(None if profile == "exact" else profile, input_format=input_format) as runtime:
        for batch_rows in tqdm(list(batched(rows, batch_size)), desc=f"imdb:{profile}"):
            texts = [row["text"] for row in batch_rows]
            labels = torch.tensor([int(row["label"]) for row in batch_rows], dtype=torch.long, device=device)
            encoded = tokenizer(
                texts,
                truncation=True,
                max_length=max_length,
                padding=True,
                return_tensors="pt",
            )
            encoded = {key: value.to(device) for key, value in encoded.items()}
            global_attention_mask = build_global_attention_mask(global_attention, encoded["input_ids"])
            if global_attention_mask is not None:
                encoded["global_attention_mask"] = global_attention_mask.to(device)

            outputs = model(**encoded)
            logits = outputs.logits.float()
            loss = F.cross_entropy(logits, labels, reduction="sum")
            predictions = logits.argmax(dim=-1)

            total_loss += float(loss.item())
            total_examples += int(labels.numel())
            correct += int((predictions == labels).sum().item())
            all_predictions.extend(int(value) for value in predictions.detach().cpu().tolist())
            all_labels.extend(int(value) for value in labels.detach().cpu().tolist())

        softmax_stats = runtime.summarize_stats() if runtime is not None else None

    elapsed_sec = time.time() - start_time
    mean_loss = total_loss / total_examples if total_examples > 0 else math.nan
    return ClassificationResult(
        profile=profile,
        loss=mean_loss,
        accuracy=correct / total_examples if total_examples > 0 else 0.0,
        macro_f1=macro_f1(all_labels, all_predictions, num_labels=num_labels),
        correct=correct,
        examples=total_examples,
        label_counts=count_values(all_labels),
        prediction_counts=count_values(all_predictions),
        predictions=all_predictions,
        labels=all_labels,
        elapsed_sec=elapsed_sec,
        softmax_stats=softmax_stats,
    )


def prediction_agreement(exact_result: ClassificationResult, result: ClassificationResult) -> float:
    if exact_result.examples != result.examples:
        raise ValueError("cannot compare predictions with different example counts")
    if exact_result.examples == 0:
        return 0.0
    return sum(
        1 for exact_pred, pred in zip(exact_result.predictions, result.predictions) if exact_pred == pred
    ) / exact_result.examples


def print_result(result: ClassificationResult, exact_result: ClassificationResult | None = None) -> None:
    print(f"\n[{result.profile}]")
    print(f"loss          : {result.loss:.8f}")
    print(f"accuracy      : {result.accuracy:.6f}")
    print(f"macro_f1      : {result.macro_f1:.6f}")
    print(f"correct       : {result.correct}/{result.examples}")
    print(f"label_counts  : {result.label_counts}")
    print(f"pred_counts   : {result.prediction_counts}")
    print(f"elapsed_sec   : {result.elapsed_sec:.2f}")
    if exact_result is not None and result.profile != "exact":
        print(f"agreement_vs_exact : {prediction_agreement(exact_result, result):.6f}")
    print(f"softmax_stats : {summarize_softmax_stats(result.softmax_stats)}")


def result_for_json(result: ClassificationResult, exact_result: ClassificationResult | None = None) -> dict:
    payload = asdict(result)
    if exact_result is not None and result.profile != "exact":
        payload["agreement_vs_exact"] = prediction_agreement(exact_result, result)
    return payload


def save_results(output_path: Path, args: dict, results: list[ClassificationResult]) -> None:
    exact_result = next((result for result in results if result.profile == "exact"), None)
    payload = {
        "args": args,
        "results": [result_for_json(result, exact_result=exact_result) for result in results],
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as file_obj:
        json.dump(payload, file_obj, indent=2, sort_keys=True)
    print(f"wrote_results : {output_path}")


def default_output_path(prefix: str = "imdb_eval") -> Path:
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    return RESULTS_DIR / f"{prefix}_{timestamp}.json"


__all__ = [
    "DEFAULT_DATASET",
    "DEFAULT_DATASET_CONFIG",
    "DEFAULT_MODEL",
    "DEFAULT_SOFTMAX_PROFILE",
    "ClassificationResult",
    "default_output_path",
    "evaluate_profile",
    "effective_max_length",
    "load_imdb_split",
    "load_model_and_tokenizer",
    "order_rows_for_batching",
    "prediction_agreement",
    "print_result",
    "resolve_device",
    "save_results",
    "set_torch_threads",
]
