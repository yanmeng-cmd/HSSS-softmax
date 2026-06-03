#!/usr/bin/env python3

from __future__ import annotations

import argparse
import concurrent.futures
import contextlib
import csv
import functools
import itertools
import json
import math
import multiprocessing as mp
from numbers import Integral
import os
import shutil
import sys
import time
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[1]
HF_CACHE_DIR = PROJECT_ROOT / ".hf-cache"
os.environ.setdefault("HF_HOME", str(HF_CACHE_DIR))
os.environ.setdefault("HF_HUB_CACHE", str(HF_CACHE_DIR / "hub"))
os.environ.setdefault("HUGGINGFACE_HUB_CACHE", str(HF_CACHE_DIR / "hub"))
os.environ.setdefault("HF_DATASETS_CACHE", str(HF_CACHE_DIR / "datasets"))
os.environ.setdefault("EVALUATE_HOME", str(HF_CACHE_DIR / "evaluate"))
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

DEPENDENCY_IMPORT_ERROR: ModuleNotFoundError | None = None
try:
    import evaluate
    import numpy as np
    import torch
    import torch.nn.functional as F
    from datasets import Dataset, DatasetDict, load_dataset
    from transformers import AutoModelForQuestionAnswering, AutoModelForSequenceClassification, AutoTokenizer
except ModuleNotFoundError as exc:
    DEPENDENCY_IMPORT_ERROR = exc
    evaluate = None
    np = None
    torch = None
    F = None
    Dataset = None
    DatasetDict = None
    load_dataset = None
    AutoModelForQuestionAnswering = None
    AutoModelForSequenceClassification = None
    AutoTokenizer = None

try:
    from python.softmax_edge_model import (
        SoftmaxEdgeModel,
        SoftmaxModelConfig,
        expand_block_flags_to_elements,
        get_precision_profile_overrides,
    )
except ModuleNotFoundError:
    from softmax_edge_model import (
        SoftmaxEdgeModel,
        SoftmaxModelConfig,
        expand_block_flags_to_elements,
        get_precision_profile_overrides,
    )


GLUE_TASKS: Dict[str, dict] = {
    "cola": {
        "model_name": "textattack/bert-base-uncased-CoLA",
        "sentence_keys": ("sentence",),
        "metric_name": "glue",
        "metric_config": "cola",
        "split": "validation",
        "label_mode": "classification",
    },
    "mrpc": {
        "model_name": "textattack/bert-base-uncased-MRPC",
        "sentence_keys": ("sentence1", "sentence2"),
        "metric_name": "glue",
        "metric_config": "mrpc",
        "split": "validation",
        "label_mode": "classification",
    },
    "qnli": {
        "model_name": "textattack/bert-base-uncased-QNLI",
        "sentence_keys": ("question", "sentence"),
        "metric_name": "glue",
        "metric_config": "qnli",
        "split": "validation",
        "label_mode": "classification",
    },
    "qqp": {
        "model_name": "textattack/bert-base-uncased-QQP",
        "sentence_keys": ("question1", "question2"),
        "metric_name": "glue",
        "metric_config": "qqp",
        "split": "validation",
        "label_mode": "classification",
    },
    "rte": {
        "model_name": "textattack/bert-base-uncased-RTE",
        "sentence_keys": ("sentence1", "sentence2"),
        "metric_name": "glue",
        "metric_config": "rte",
        "split": "validation",
        "label_mode": "classification",
    },
    "sst2": {
        "model_name": "textattack/bert-base-uncased-SST-2",
        "sentence_keys": ("sentence",),
        "metric_name": "glue",
        "metric_config": "sst2",
        "split": "validation",
        "label_mode": "classification",
    },
    "stsb": {
        "model_name": "textattack/bert-base-uncased-STS-B",
        "sentence_keys": ("sentence1", "sentence2"),
        "metric_name": "glue",
        "metric_config": "stsb",
        "split": "validation",
        "label_mode": "regression",
    },
    "mnli": {
        "model_name": "textattack/bert-base-uncased-MNLI",
        "sentence_keys": ("premise", "hypothesis"),
        "metric_name": "glue",
        "metric_config": "mnli",
        "split": "validation_matched",
        "label_mode": "classification",
    },
    "wnli": {
        "model_name": "textattack/bert-base-uncased-WNLI",
        "sentence_keys": ("sentence1", "sentence2"),
        "metric_name": "glue",
        "metric_config": "wnli",
        "split": "validation",
        "label_mode": "classification",
    },
}
QA_TASKS: Dict[str, dict] = {
    "squad": {
        "model_name": "csarron/bert-base-uncased-squad-v1",
        "metric_name": "squad",
        "metric_config": None,
        "split": "validation",
        "version": "1.1",
    },
}

LOCAL_METRIC_SCRIPTS: Dict[str, Path] = {
    "glue": PROJECT_ROOT / "third_party" / "metrics" / "glue" / "glue.py",
    "squad": PROJECT_ROOT / "third_party" / "metrics" / "squad_v1" / "squad.py",
}
LOCAL_GLUE_FALLBACKS: Dict[str, Dict[str, Path]] = {
    "cola": {
        "validation": PROJECT_ROOT / ".data" / "glue" / "CoLA" / "dev.tsv",
    },
    "mrpc": {
        "validation": PROJECT_ROOT / ".data" / "glue" / "MRPC" / "dev.tsv",
    },
    "qnli": {
        "validation": PROJECT_ROOT / ".data" / "glue" / "QNLI" / "dev.tsv",
    },
    "qqp": {
        "validation": PROJECT_ROOT / ".data" / "glue" / "QQP" / "dev.tsv",
    },
    "rte": {
        "validation": PROJECT_ROOT / ".data" / "glue" / "RTE" / "dev.tsv",
    },
    "sst2": {
        "validation": PROJECT_ROOT / ".data" / "glue" / "SST-2" / "dev.tsv",
    },
    "stsb": {
        "validation": PROJECT_ROOT / ".data" / "glue" / "STS-B" / "dev.tsv",
    },
    "mnli": {
        "validation_matched": PROJECT_ROOT / ".data" / "glue" / "MNLI" / "dev_matched.tsv",
    },
    "wnli": {
        "validation": PROJECT_ROOT / ".data" / "glue" / "WNLI" / "dev.tsv",
    },
}
LOCAL_GLUE_LABEL_TO_ID: Dict[str, Dict[str, int]] = {
    "mnli": {
        "contradiction": 0,
        "entailment": 1,
        "neutral": 2,
    },
    "rte": {
        "entailment": 0,
        "not_entailment": 1,
    },
    "qnli": {
        "entailment": 0,
        "not_entailment": 1,
    },
}
LOCAL_SQUAD_FALLBACKS: Dict[str, Dict[str, Path]] = {
    "squad": {
        "validation": PROJECT_ROOT / ".data" / "squad" / "v1.1" / "dev-v1.1.json",
    },
}


@dataclass
class TaskEvalResult:
    task: str
    profile: str
    num_samples: int
    metrics: dict
    softmax_metrics: dict = None


@dataclass
class TaskDiffScanResult:
    task: str
    profile: str
    num_samples: int
    changed_prediction_count: int
    changed_prediction_rate: float
    max_logit_abs_diff: float


@dataclass
class LoadedTaskResources:
    task_name: str
    task_cfg: dict
    encoded: object
    model: object
    raw_examples: object | None = None
    qa_examples: list[tuple[str, str]] | None = None
    qa_references: list[dict] | None = None
    qa_offset_mappings: list | None = None
    qa_features_per_example: Dict[str, List[int]] | None = None


TASK_RESOURCE_CACHE: Dict[tuple, LoadedTaskResources] = {}
_THREADS_CONFIGURED = False


def model_cache_repo_dir(model_name: str) -> Path:
    return HF_CACHE_DIR / "hub" / f"models--{model_name.replace('/', '--')}"


def _snapshot_has_any(snapshot_dir: Path, file_names: Sequence[str]) -> bool:
    return any((snapshot_dir / file_name).exists() for file_name in file_names)


def _pick_snapshot_with_files(repo_dir: Path, file_names: Sequence[str]) -> Path | None:
    snapshots_dir = repo_dir / "snapshots"
    if not snapshots_dir.exists():
        return None

    ref_path = repo_dir / "refs" / "main"
    ref_name = ref_path.read_text().strip() if ref_path.exists() else ""
    candidates = sorted((path for path in snapshots_dir.iterdir() if path.is_dir()), key=lambda path: path.name)

    if ref_name:
        ref_snapshot = snapshots_dir / ref_name
        if ref_snapshot.exists() and _snapshot_has_any(ref_snapshot, file_names):
            return ref_snapshot

    with_files = [path for path in candidates if _snapshot_has_any(path, file_names)]
    if not with_files:
        return None
    return with_files[-1]


def resolve_local_model_dir(model_name: str) -> Path | None:
    repo_dir = model_cache_repo_dir(model_name)
    if not repo_dir.exists():
        return None

    tokenizer_snapshot = _pick_snapshot_with_files(
        repo_dir,
        (
            "tokenizer.json",
            "tokenizer_config.json",
            "vocab.txt",
            "sentencepiece.bpe.model",
            "spiece.model",
        ),
    )
    config_snapshot = _pick_snapshot_with_files(repo_dir, ("config.json",))
    weight_snapshot = _pick_snapshot_with_files(
        repo_dir,
        (
            "model.safetensors",
            "model.safetensors.index.json",
            "pytorch_model.bin",
            "pytorch_model.bin.index.json",
        ),
    )

    if config_snapshot is None or weight_snapshot is None:
        return None

    offline_root = HF_CACHE_DIR / "offline-models"
    offline_root.mkdir(parents=True, exist_ok=True)
    offline_dir = offline_root / model_name.replace("/", "--")
    offline_dir.mkdir(parents=True, exist_ok=True)

    source_dirs = []
    for snapshot in (tokenizer_snapshot, config_snapshot, weight_snapshot):
        if snapshot is not None and snapshot not in source_dirs:
            source_dirs.append(snapshot)

    for source_dir in source_dirs:
        for item in source_dir.iterdir():
            target = offline_dir / item.name
            if target.exists():
                continue
            if item.is_symlink():
                target.symlink_to(item.resolve())
            elif item.is_file():
                shutil.copy2(item, target)

    if not (offline_dir / "config.json").exists():
        return None
    if not _snapshot_has_any(
        offline_dir,
        ("model.safetensors", "model.safetensors.index.json", "pytorch_model.bin", "pytorch_model.bin.index.json"),
    ):
        return None
    return offline_dir


def ensure_runtime_dependencies() -> None:
    if DEPENDENCY_IMPORT_ERROR is None:
        return
    missing_name = getattr(DEPENDENCY_IMPORT_ERROR, "name", "unknown")
    venv_python = PROJECT_ROOT / ".venv-softmax" / "bin" / "python"
    hint = ""
    if venv_python.exists():
        hint = f"; try running with {venv_python}"
    raise ModuleNotFoundError(
        f"missing Python dependency '{missing_name}'; install the evaluation stack before running this script{hint}"
    ) from DEPENDENCY_IMPORT_ERROR


def _resolve_per_process_budget(parallel_jobs: int, reserve_cpus: int) -> int:
    cpu_total = os.cpu_count() or 1
    reserve = min(max(0, reserve_cpus), max(0, cpu_total - 1))
    available = max(1, cpu_total - reserve)
    return max(1, available // max(1, parallel_jobs))


def resolve_torch_threads(requested: int, parallel_jobs: int, reserve_cpus: int) -> int:
    if requested > 0:
        return requested
    return _resolve_per_process_budget(parallel_jobs, reserve_cpus)


def resolve_aux_workers(requested: int, parallel_jobs: int, reserve_cpus: int, cap: int) -> int:
    if requested > 0:
        return requested
    per_process_budget = _resolve_per_process_budget(parallel_jobs, reserve_cpus)
    if per_process_budget < 4:
        return 1
    return min(cap, per_process_budget)


def resolve_qa_inference_workers(
    requested: int,
    tasks: Sequence[str],
    profiles: Sequence[str],
    task_workers: int,
    run_workers: int,
    reserve_cpus: int,
) -> int:
    if requested > 0:
        return requested
    if task_workers > 1 or run_workers > 1:
        return 1
    if len(tasks) != 1 or tasks[0] not in QA_TASKS:
        return 1
    if len(profiles) != 1 or profiles[0] in (None, "", "exact"):
        return 1

    cpu_total = os.cpu_count() or 1
    reserve = min(max(0, reserve_cpus), max(0, cpu_total - 1))
    worker_budget = max(1, cpu_total - reserve)
    return max(1, min(worker_budget, 6))


def configure_runtime_threads(torch_threads: int, torch_interop_threads: int) -> None:
    global _THREADS_CONFIGURED
    if _THREADS_CONFIGURED or DEPENDENCY_IMPORT_ERROR is not None or torch is None:
        return

    if torch_threads > 0:
        torch.set_num_threads(torch_threads)
    if torch_interop_threads > 0:
        try:
            torch.set_num_interop_threads(torch_interop_threads)
        except RuntimeError:
            pass
    _THREADS_CONFIGURED = True


@dataclass(frozen=True)
class TaskRunRequest:
    task_name: str
    profiles: Tuple[str, ...]
    max_samples: int
    batch_size: int
    max_length: int
    qa_max_length: int
    qa_doc_stride: int
    qa_n_best_size: int
    qa_max_answer_length: int
    device: str
    input_format: str
    tokenize_workers: int
    qa_postprocess_workers: int
    torch_threads: int
    torch_interop_threads: int
    qa_inference_workers: int
    collect_softmax_stats: bool
    scan_diff_only: bool
    progress_interval_sec: float


@dataclass(frozen=True)
class TaskProfileRunRequest:
    bundle_index: int
    profile_index: int
    task_name: str
    profile: str
    max_samples: int
    batch_size: int
    max_length: int
    qa_max_length: int
    qa_doc_stride: int
    qa_n_best_size: int
    qa_max_answer_length: int
    device: str
    input_format: str
    tokenize_workers: int
    qa_postprocess_workers: int
    torch_threads: int
    torch_interop_threads: int
    qa_inference_workers: int
    collect_softmax_stats: bool
    scan_diff_only: bool
    progress_interval_sec: float


@dataclass(frozen=True)
class QaInferenceShardRequest:
    task_name: str
    profile: str
    max_samples: int
    batch_size: int
    max_length: int
    qa_max_length: int
    qa_doc_stride: int
    device: str
    input_format: str
    tokenize_workers: int
    torch_threads: int
    torch_interop_threads: int
    collect_softmax_stats: bool
    shard_index: int
    feature_start: int
    feature_end: int


@dataclass
class QaInferenceShardResult:
    shard_index: int
    feature_count: int
    batch_count: int
    start_logits: np.ndarray
    end_logits: np.ndarray
    softmax_stats: dict | None = None


@dataclass
class TaskRunBundle:
    task_name: str
    results: List[TaskEvalResult] = field(default_factory=list)
    diff_results: List[TaskDiffScanResult] = field(default_factory=list)


@dataclass
class TaskProfileRunBundleItem:
    bundle_index: int
    profile_index: int
    task_name: str
    result: TaskEvalResult | None = None
    diff_result: TaskDiffScanResult | None = None


class ProgressPrinter:

    def __init__(self, task_name: str, profile: str, interval_sec: float):
        self.task_name = task_name
        self.profile = profile
        self.interval_sec = max(1.0, float(interval_sec))
        self.phase = "init"
        self.unit = "items"
        self.total = 0
        self.phase_started_at = time.monotonic()
        self.last_emit_at = 0.0

    def _prefix(self) -> str:
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        return f"[{timestamp}] [task={self.task_name} profile={self.profile}]"

    @staticmethod
    def _format_eta(seconds: float | None) -> str:
        if seconds is None:
            return "unknown"
        seconds_int = max(0, int(round(seconds)))
        hours, rem = divmod(seconds_int, 3600)
        minutes, secs = divmod(rem, 60)
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"

    def start_phase(self, phase: str, total: int, unit: str, extra: str = "") -> None:
        self.phase = phase
        self.total = max(0, int(total))
        self.unit = unit
        self.phase_started_at = time.monotonic()
        self.last_emit_at = 0.0
        self.emit(0, force=True, extra=extra or "started")

    def emit(self, current: int, force: bool = False, extra: str = "") -> None:
        now = time.monotonic()
        current = min(max(0, int(current)), self.total) if self.total > 0 else max(0, int(current))
        if not force and self.last_emit_at and (now - self.last_emit_at) < self.interval_sec:
            return

        elapsed = now - self.phase_started_at
        if self.total > 0:
            percent = 100.0 * current / self.total
            if current > 0 and elapsed > 0.0:
                eta = self._format_eta((self.total - current) * (elapsed / current))
            else:
                eta = "unknown"
            progress_text = f"{percent:6.2f}% {current}/{self.total} {self.unit}"
        else:
            eta = "unknown"
            progress_text = f"{current} {self.unit}"

        message = (
            f"{self._prefix()} phase={self.phase:<11} progress={progress_text} "
            f"elapsed={self._format_eta(elapsed)} eta={eta}"
        )
        if extra:
            message = f"{message} {extra}"
        print(message, flush=True)
        self.last_emit_at = now

    def finish_phase(self, extra: str = "") -> None:
        current = self.total if self.total > 0 else 0
        self.emit(current, force=True, extra=extra or "done")


def print_progress_event(task_name: str, profile: str, phase: str, message: str) -> None:
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] [task={task_name} profile={profile}] phase={phase:<11} {message}", flush=True)


class ApproxSoftmaxRuntime:

    def __init__(
        self,
        profile: str,
        input_format: str = "fp16",
        fallback_softmax=None,
        collect_stats: bool = False,
    ):
        self.profile = profile
        self.input_format = input_format
        self.fallback_softmax = fallback_softmax or F.softmax
        self.collect_stats = collect_stats
        self.model_cache: Dict[int, SoftmaxEdgeModel] = {}
        self.stats = self.empty_stats()

    @staticmethod
    def empty_stats() -> dict:
        return {
            "sum_kl": 0.0,
            "count": 0,
            "topk_matches": {1: 0, 3: 0, 5: 0},
            "total_rows": 0,
            "total_elements": 0,
            "total_pruned_elements": 0,
            "block_pruned_blocks": 0,
            "total_blocks": 0,
            "block_pruned_elements": 0,
            "element_pruned_only": 0,
            "aggressive_blocks": 0,
            "conservative_blocks": 0,
        }

    @classmethod
    def merge_stats(cls, stats_items: Sequence[dict | None]) -> dict:
        merged = cls.empty_stats()
        for item in stats_items:
            if not item:
                continue
            merged["sum_kl"] += item.get("sum_kl", 0.0)
            merged["count"] += int(item.get("count", 0))
            merged["total_rows"] += int(item.get("total_rows", 0))
            merged["total_elements"] += int(item.get("total_elements", 0))
            merged["total_pruned_elements"] += int(item.get("total_pruned_elements", 0))
            merged["block_pruned_blocks"] += int(item.get("block_pruned_blocks", 0))
            merged["total_blocks"] += int(item.get("total_blocks", 0))
            merged["block_pruned_elements"] += int(item.get("block_pruned_elements", 0))
            merged["element_pruned_only"] += int(item.get("element_pruned_only", 0))
            merged["aggressive_blocks"] += int(item.get("aggressive_blocks", 0))
            merged["conservative_blocks"] += int(item.get("conservative_blocks", 0))
            topk_matches = item.get("topk_matches", {})
            for k in merged["topk_matches"]:
                merged["topk_matches"][k] += int(topk_matches.get(k, 0))
        return merged

    @classmethod
    def summarize_stats(cls, stats: dict) -> dict:
        if stats["total_rows"] == 0:
            return {}
        summary = {
            "total_rows": stats["total_rows"],
            "total_elements": stats["total_elements"],
            "total_pruned_elements": stats["total_pruned_elements"],
            "total_pruned_rate": (
                stats["total_pruned_elements"] / stats["total_elements"]
                if stats["total_elements"] > 0
                else 0.0
            ),
            "block_pruned_blocks": stats["block_pruned_blocks"],
            "total_blocks": stats["total_blocks"],
            "block_pruned_block_rate": (
                stats["block_pruned_blocks"] / stats["total_blocks"]
                if stats["total_blocks"] > 0
                else 0.0
            ),
            "block_pruned_elements": stats["block_pruned_elements"],
            "block_pruned_element_rate": (
                stats["block_pruned_elements"] / stats["total_elements"]
                if stats["total_elements"] > 0
                else 0.0
            ),
            "element_pruned_only": stats["element_pruned_only"],
            "element_pruned_only_rate": (
                stats["element_pruned_only"] / stats["total_elements"]
                if stats["total_elements"] > 0
                else 0.0
            ),
            "aggressive_blocks": stats["aggressive_blocks"],
            "aggressive_block_rate": (
                stats["aggressive_blocks"] / stats["total_blocks"]
                if stats["total_blocks"] > 0
                else 0.0
            ),
            "conservative_blocks": stats["conservative_blocks"],
            "conservative_block_rate": (
                stats["conservative_blocks"] / stats["total_blocks"]
                if stats["total_blocks"] > 0
                else 0.0
            ),
        }
        if stats["count"] > 0:
            summary.update(
                {
                    "avg_kl": stats["sum_kl"] / stats["total_rows"],
                    "topk_rates": {k: v / stats["total_rows"] for k, v in stats["topk_matches"].items()},
                    "ppl_proxy": math.exp(stats["sum_kl"] / stats["total_rows"]),
                }
            )
        return summary

    def _get_model(self, row_depth: int) -> SoftmaxEdgeModel:
        model = self.model_cache.get(row_depth)
        if model is not None:
            return model

        cfg_kwargs = {
            "input_format": self.input_format,
            "row_depth": row_depth,
        }
        cfg_kwargs.update(get_precision_profile_overrides(self.profile))
        model = SoftmaxEdgeModel(SoftmaxModelConfig(**cfg_kwargs))
        self.model_cache[row_depth] = model
        return model

    def __call__(self, input_tensor: torch.Tensor, dim: int = -1, _stacklevel: int = 3, dtype=None) -> torch.Tensor:
        if dim != -1:
            return self.fallback_softmax(input_tensor, dim=dim, _stacklevel=_stacklevel, dtype=dtype)

        target_dtype = dtype or input_tensor.dtype
        row_depth = int(input_tensor.shape[-1])
        model = self._get_model(row_depth)
        cpu_float32_fast_path = (
            input_tensor.device.type == "cpu"
            and input_tensor.dtype == torch.float32
            and target_dtype == torch.float32
        )

        if cpu_float32_fast_path:
            flat = input_tensor.detach().reshape(-1, row_depth)
        else:
            flat = input_tensor.detach().to(torch.float32).cpu().reshape(-1, row_depth)
        if self.collect_stats:
            approx_rows: List[List[float]] = []
            for row in flat:
                row_list = row.tolist()
                result = model.simulate_row(row_list)
                approx_probs = result.approx_probs
                approx_rows.append(approx_probs)

                self.stats["sum_kl"] += result.kl_divergence
                self.stats["count"] += 1
                self.stats["total_rows"] += 1
                self.stats["total_elements"] += len(result.prune_flags)
                self.stats["total_pruned_elements"] += sum(1 for flag in result.prune_flags if flag)
                self.stats["block_pruned_blocks"] += sum(1 for flag in result.block_prune_flags if flag)
                self.stats["total_blocks"] += len(result.block_prune_flags)
                block_level_flags = expand_block_flags_to_elements(
                    result.block_prune_flags,
                    len(result.prune_flags),
                    model.cfg.block_size_effective,
                )
                self.stats["block_pruned_elements"] += sum(1 for flag in block_level_flags if flag)
                self.stats["element_pruned_only"] += sum(
                    1
                    for elem_pruned, final_pruned, block_pruned in zip(
                        result.element_prune_flags,
                        result.prune_flags,
                        block_level_flags,
                    )
                    if elem_pruned and final_pruned and not block_pruned
                )
                self.stats["aggressive_blocks"] += sum(1 for meta in result.block_metas if meta.tau_elem_value == -2)
                self.stats["conservative_blocks"] += sum(1 for meta in result.block_metas if meta.tau_elem_value == -4)

                ref_probs = result.reference_probs

                def get_topk_set(probs, k):
                    return set(sorted(range(len(probs)), key=lambda i: probs[i], reverse=True)[:k])

                for k in self.stats["topk_matches"]:
                    ref_k = get_topk_set(ref_probs, k)
                    approx_k = get_topk_set(approx_probs, k)
                    if ref_k == approx_k:
                        self.stats["topk_matches"][k] += 1
            approx_rows_tensor = torch.tensor(approx_rows, device=input_tensor.device, dtype=target_dtype)
            return approx_rows_tensor.reshape_as(input_tensor)

        approx_rows, prune_stats = model.simulate_rows_fast(flat.numpy())
        self.stats["total_rows"] += int(flat.shape[0])
        self.stats["total_elements"] += prune_stats["total_elements"]
        self.stats["total_pruned_elements"] += prune_stats["total_pruned_elements"]
        self.stats["block_pruned_blocks"] += prune_stats["block_pruned_blocks"]
        self.stats["total_blocks"] += prune_stats["total_blocks"]
        self.stats["block_pruned_elements"] += prune_stats["block_pruned_elements"]
        self.stats["element_pruned_only"] += prune_stats["element_pruned_only"]
        self.stats["aggressive_blocks"] += prune_stats["aggressive_blocks"]
        self.stats["conservative_blocks"] += prune_stats["conservative_blocks"]

        if cpu_float32_fast_path and not isinstance(approx_rows, torch.Tensor):
            return torch.from_numpy(approx_rows).reshape_as(input_tensor)
        if isinstance(approx_rows, torch.Tensor):
            approx_rows_tensor = approx_rows.to(device=input_tensor.device, dtype=target_dtype)
        else:
            approx_rows_tensor = torch.as_tensor(approx_rows, device=input_tensor.device, dtype=target_dtype)
        return approx_rows_tensor.reshape_as(input_tensor)

    def get_summary(self) -> dict:
        return self.summarize_stats(self.stats)

    def get_raw_stats(self) -> dict:
        stats = dict(self.stats)
        stats["topk_matches"] = dict(self.stats["topk_matches"])
        return stats


@contextlib.contextmanager
def patched_softmax(profile: str | None, input_format: str, collect_stats: bool = False):
    if profile in (None, "", "exact"):
        yield
        return

    original_softmax = F.softmax
    runtime = ApproxSoftmaxRuntime(
        profile=profile,
        input_format=input_format,
        fallback_softmax=original_softmax,
        collect_stats=collect_stats,
    )

    def replacement(input_tensor: torch.Tensor, dim: int = -1, _stacklevel: int = 3, dtype=None):
        return runtime(input_tensor, dim=dim, _stacklevel=_stacklevel, dtype=dtype)

    F.softmax = replacement
    try:
        yield runtime
    finally:
        F.softmax = original_softmax


def profile_requires_fp16_input(profile: str | None) -> bool:
    if profile in (None, "", "exact"):
        return False
    return bool(get_precision_profile_overrides(profile).get("rtl_exact"))


def resolve_profile_input_format(profile: str | None, requested_input_format: str) -> str:
    if profile_requires_fp16_input(profile):
        return "fp16"
    return requested_input_format


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate BERT tasks with approximate softmax")
    parser.add_argument(
        "--tasks",
        type=str,
        default="sst2,rte,mrpc",
        help="逗号分隔任务列表，例如 sst2,rte,mrpc,squad",
    )
    parser.add_argument(
        "--profiles",
        type=str,
        default="exact,doc_adaptive_desc9_q7_special4",
        help=(
            "逗号分隔 softmax 配置列表，例如 "
            "exact,doc_adaptive_desc9_q7_special4 或 "
            "exact,doc_adaptive_desc9_q7_special4_block4"
        ),
    )
    parser.add_argument("--max-samples", type=int, default=128, help="每个任务最多评估多少条样本")
    parser.add_argument("--batch-size", type=int, default=8, help="推理批大小")
    parser.add_argument(
        "--task-workers",
        type=int,
        default=1,
        help="按 task 并行评估的进程数；同一 task 内 profile 仍串行以复用模型和数据",
    )
    parser.add_argument(
        "--run-workers",
        type=int,
        default=1,
        help="按 task x profile 独立并行的进程数；并行度更高，但每个作业都会单独加载模型和数据",
    )
    parser.add_argument(
        "--auto-workers",
        action="store_true",
        help="自动选择保守并行度，默认给系统保留一部分 CPU 余量",
    )
    parser.add_argument(
        "--reserve-cpus",
        type=int,
        default=2,
        help="自动并行时至少预留多少个 CPU 给系统和其它任务",
    )
    parser.add_argument(
        "--max-auto-workers",
        type=int,
        default=3,
        help="自动并行时的最大 worker 数；设为 0 表示只受 CPU/任务数限制",
    )
    parser.add_argument("--max-length", type=int, default=128, help="tokenizer 最大长度")
    parser.add_argument("--qa-max-length", type=int, default=384, help="SQuAD 特征最大长度")
    parser.add_argument("--qa-doc-stride", type=int, default=128, help="SQuAD context 滑窗步长")
    parser.add_argument("--qa-n-best-size", type=int, default=20, help="SQuAD 解码保留的候选跨度数")
    parser.add_argument("--qa-max-answer-length", type=int, default=30, help="SQuAD 最大答案跨度长度")
    parser.add_argument(
        "--tokenize-workers",
        type=int,
        default=0,
        help="tokenize/map 进程数；0 表示按当前并行度自动选择保守值",
    )
    parser.add_argument(
        "--qa-postprocess-workers",
        type=int,
        default=0,
        help="SQuAD 后处理线程数；0 表示按当前并行度自动选择保守值",
    )
    parser.add_argument(
        "--qa-inference-workers",
        type=int,
        default=0,
        help="SQuAD 单个 profile 的 feature 分片并行进程数；0 表示仅在单任务单近似 profile 时自动启用",
    )
    parser.add_argument("--device", choices=("cpu",), default="cpu")
    parser.add_argument(
        "--input-format",
        choices=("fp16", "bf16"),
        default="bf16",
        help="任务级 attention score 动态范围较大，默认用 bf16 以避免 fp16 溢出",
    )
    parser.add_argument(
        "--torch-threads",
        type=int,
        default=0,
        help="每个评测进程内部的 PyTorch CPU 线程数；0 表示自动按并行度分配",
    )
    parser.add_argument(
        "--torch-interop-threads",
        type=int,
        default=1,
        help="每个评测进程的 PyTorch inter-op 线程数",
    )
    parser.add_argument("--output-json", type=str, help="把汇总结果写到 JSON 文件")
    parser.add_argument(
        "--progress-interval-sec",
        type=float,
        default=30.0,
        help="进度打印时间间隔（秒）；后台跑时可配合 tail -f 观察",
    )
    parser.add_argument(
        "--collect-softmax-stats",
        action="store_true",
        help="额外统计 attention softmax 行级 KL/topk；会显著拖慢任务级评测",
    )
    parser.add_argument(
        "--scan-diff-only",
        action="store_true",
        help="不算任务指标，只扫描 exact 与近似 softmax 下预测是否变化",
    )
    return parser.parse_args()


def select_examples(dataset_split, max_samples: int):
    if max_samples <= 0 or max_samples >= len(dataset_split):
        return dataset_split
    return dataset_split.select(range(max_samples))


def tokenize_batch(examples, tokenizer, sentence_keys: Sequence[str], max_length: int):
    texts = [examples[sentence_keys[0]]]
    if len(sentence_keys) == 2:
        texts.append(examples[sentence_keys[1]])
    return tokenizer(*texts, truncation=True, padding="max_length", max_length=max_length)


def tokenize_qa_validation_features(examples, tokenizer, max_length: int, doc_stride: int):
    tokenized = tokenizer(
        [question.lstrip() for question in examples["question"]],
        examples["context"],
        truncation="only_second",
        max_length=max_length,
        stride=doc_stride,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding="max_length",
    )

    sample_mapping = tokenized.pop("overflow_to_sample_mapping")
    example_ids = []
    for feature_index, sample_index in enumerate(sample_mapping):
        sequence_ids = tokenized.sequence_ids(feature_index)
        example_ids.append(examples["id"][sample_index])
        tokenized["offset_mapping"][feature_index] = [
            offset if sequence_ids[token_index] == 1 else None
            for token_index, offset in enumerate(tokenized["offset_mapping"][feature_index])
        ]

    tokenized["example_id"] = example_ids
    return tokenized


def build_batch_inputs(batch, device: str) -> dict:
    inputs = {
        "input_ids": torch.tensor(batch["input_ids"], device=device),
        "attention_mask": torch.tensor(batch["attention_mask"], device=device),
    }
    if "token_type_ids" in batch:
        inputs["token_type_ids"] = torch.tensor(batch["token_type_ids"], device=device)
    return inputs


def load_glue_dataset_with_fallback(task_name: str):
    local_dataset = load_local_glue_dataset(task_name)
    if local_dataset is not None:
        return local_dataset
    return load_dataset("glue", task_name, cache_dir=str(HF_CACHE_DIR / "datasets"))


def load_squad_dataset_with_fallback(task_name: str):
    local_dataset = load_local_squad_dataset(task_name)
    if local_dataset is not None:
        return local_dataset
    if task_name == "squad":
        return load_dataset("squad", cache_dir=str(HF_CACHE_DIR / "datasets"))
    raise ValueError(f"unsupported SQuAD task: {task_name}")


def load_metric_with_local_script(metric_name: str, metric_config: str | None = None):
    local_script = LOCAL_METRIC_SCRIPTS.get(metric_name)
    if local_script is not None:
        if not local_script.exists():
            raise FileNotFoundError(f"local metric script not found: {local_script}")
        return evaluate.load(
            str(local_script),
            config_name=metric_config,
            cache_dir=str(HF_CACHE_DIR / "evaluate"),
        )

    if metric_config is None:
        return evaluate.load(metric_name, cache_dir=str(HF_CACHE_DIR / "evaluate"))
    return evaluate.load(
        metric_name,
        config_name=metric_config,
        cache_dir=str(HF_CACHE_DIR / "evaluate"),
    )


def _read_dict_rows(path: Path, encoding: str = "utf-8") -> list[dict]:
    with path.open("r", encoding=encoding) as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        return list(reader)


def _flatten_squad_examples(path: Path):
    payload = json.loads(path.read_text(encoding="utf-8"))
    records = []
    for article in payload.get("data", []):
        title = article.get("title", "")
        for paragraph in article.get("paragraphs", []):
            context = paragraph.get("context", "")
            for qa in paragraph.get("qas", []):
                answers = qa.get("answers", [])
                records.append(
                    {
                        "id": qa["id"],
                        "title": title,
                        "context": context,
                        "question": qa["question"],
                        "answers": {
                            "text": [answer["text"] for answer in answers],
                            "answer_start": [int(answer["answer_start"]) for answer in answers],
                        },
                    }
                )
    return Dataset.from_list(records)


def _load_local_glue_split(task_name: str, split_name: str):
    fallback_files = LOCAL_GLUE_FALLBACKS.get(task_name)
    if not fallback_files or split_name not in fallback_files:
        return None

    path = fallback_files[split_name]
    if not path.exists():
        return None

    records: list[dict] = []
    if task_name == "cola":
        with path.open("r", encoding="utf-8") as handle:
            reader = csv.reader(handle, delimiter="\t")
            for row in reader:
                if len(row) < 4:
                    continue
                records.append(
                    {
                        "sentence": row[3],
                        "label": int(row[1]),
                    }
                )
    elif task_name == "sst2":
        for row in _read_dict_rows(path):
            records.append({"sentence": row["sentence"], "label": int(row["label"])})
    elif task_name == "mrpc":
        for row in _read_dict_rows(path, encoding="utf-8-sig"):
            label_key = "Quality" if "Quality" in row else "\ufeffQuality"
            sent1 = row.get("#1 String", "")
            sent2 = row.get("#2 String", "")
            if not sent1 or not sent2:
                continue
            records.append(
                {
                    "sentence1": sent1,
                    "sentence2": sent2,
                    "label": int(row[label_key]),
                }
            )
    elif task_name == "qqp":
        for row in _read_dict_rows(path):
            sent1 = row.get("question1", "")
            sent2 = row.get("question2", "")
            label = row.get("is_duplicate", "")
            if not sent1 or not sent2 or label == "":
                continue
            records.append(
                {
                    "question1": sent1,
                    "question2": sent2,
                    "label": int(label),
                }
            )
    elif task_name == "stsb":
        with path.open("r", encoding="utf-8") as handle:
            _ = handle.readline()
            for line in handle:
                parts = line.rstrip("\n").split("\t")
                if len(parts) < 10:
                    continue
                records.append(
                    {
                        "sentence1": parts[7],
                        "sentence2": parts[8],
                        "label": float(parts[9]),
                    }
                )
    elif task_name == "mnli":
        label_to_id = LOCAL_GLUE_LABEL_TO_ID["mnli"]
        for row in _read_dict_rows(path):
            label = row["gold_label"].strip()
            if label not in label_to_id:
                continue
            records.append(
                {
                    "premise": row["sentence1"],
                    "hypothesis": row["sentence2"],
                    "label": label_to_id[label],
                }
            )
    elif task_name == "qnli":
        label_to_id = LOCAL_GLUE_LABEL_TO_ID["qnli"]
        with path.open("r", encoding="utf-8") as handle:
            _ = handle.readline()
            for line in handle:
                parts = line.rstrip("\n").split("\t")
                if len(parts) < 4:
                    continue
                label = parts[-1].strip()
                if label not in label_to_id:
                    continue
                records.append(
                    {
                        "question": parts[1],
                        "sentence": "\t".join(parts[2:-1]),
                        "label": label_to_id[label],
                    }
                )
    elif task_name == "rte":
        label_to_id = LOCAL_GLUE_LABEL_TO_ID["rte"]
        with path.open("r", encoding="utf-8") as handle:
            _ = handle.readline()
            for line in handle:
                parts = line.rstrip("\n").split("\t")
                if len(parts) < 4:
                    continue
                label = parts[-1].strip()
                if label not in label_to_id:
                    continue
                records.append(
                    {
                        "sentence1": parts[1],
                        "sentence2": "\t".join(parts[2:-1]),
                        "label": label_to_id[label],
                    }
                )
    elif task_name == "wnli":
        for row in _read_dict_rows(path):
            records.append(
                {
                    "sentence1": row["sentence1"],
                    "sentence2": row["sentence2"],
                    "label": int(row["label"]),
                }
            )

    return Dataset.from_list(records)


def load_local_glue_dataset(task_name: str):
    fallback_files = LOCAL_GLUE_FALLBACKS.get(task_name)
    if not fallback_files:
        return None

    splits = {}
    for split_name in fallback_files:
        dataset = _load_local_glue_split(task_name, split_name)
        if dataset is None:
            return None
        splits[split_name] = dataset
    return DatasetDict(splits)


def load_local_squad_dataset(task_name: str):
    fallback_files = LOCAL_SQUAD_FALLBACKS.get(task_name)
    if not fallback_files:
        return None

    splits = {}
    for split_name, path in fallback_files.items():
        if not path.exists():
            return None
        splits[split_name] = _flatten_squad_examples(path)
    return DatasetDict(splits)


def load_model_with_safetensor_fallback(model_cls, model_name: str, **kwargs):
    try:
        return model_cls.from_pretrained(model_name, use_safetensors=True, **kwargs)
    except Exception as exc:
        message = str(exc).lower()
        if "safetensor" not in message and "safetensors" not in message:
            print(
                f"warning: safetensors load failed for {model_name}, fallback to default loader: {exc}",
                file=sys.stderr,
        )
        return model_cls.from_pretrained(model_name, **kwargs)


def load_tokenizer_with_local_cache_fallback(model_name: str):
    local_model_dir = resolve_local_model_dir(model_name)
    if local_model_dir is not None:
        return AutoTokenizer.from_pretrained(str(local_model_dir), use_fast=True, local_files_only=True)
    return AutoTokenizer.from_pretrained(
        model_name,
        cache_dir=str(HF_CACHE_DIR / "hub"),
        use_fast=True,
    )


def load_transformer_model_with_local_cache_fallback(model_cls, model_name: str, **kwargs):
    local_model_dir = resolve_local_model_dir(model_name)
    if local_model_dir is not None:
        return load_model_with_safetensor_fallback(
            model_cls,
            str(local_model_dir),
            local_files_only=True,
            **kwargs,
        )
    return load_model_with_safetensor_fallback(
        model_cls,
        model_name,
        cache_dir=str(HF_CACHE_DIR / "hub"),
        **kwargs,
    )


def get_task_cfg(task_name: str) -> dict:
    if task_name in GLUE_TASKS:
        return GLUE_TASKS[task_name]
    if task_name in QA_TASKS:
        return QA_TASKS[task_name]
    raise KeyError(f"unsupported task: {task_name}")


def normalize_local_glue_labels(task_name: str, dataset_split):
    label_to_id = LOCAL_GLUE_LABEL_TO_ID.get(task_name)
    if label_to_id is None or "label" not in dataset_split.column_names:
        return dataset_split
    if len(dataset_split) == 0:
        return dataset_split

    first_label = dataset_split[0]["label"]
    if isinstance(first_label, Integral):
        label_feature = getattr(dataset_split, "features", {}).get("label")
        feature_names = getattr(label_feature, "names", None)
        if not feature_names:
            return dataset_split

        try:
            remap = {index: label_to_id[name] for index, name in enumerate(feature_names)}
        except KeyError:
            return dataset_split

        if all(remap[index] == index for index in remap):
            return dataset_split

        return dataset_split.map(lambda example: {"label": remap[int(example["label"])]})

    return dataset_split.map(lambda example: {"label": label_to_id[str(example["label"]).strip()]})


def load_task_resources(
    task_name: str,
    max_samples: int,
    max_length: int,
    qa_max_length: int,
    qa_doc_stride: int,
    device: str,
    tokenize_workers: int,
) -> LoadedTaskResources:
    cache_key = (task_name, max_samples, max_length, qa_max_length, qa_doc_stride, device)
    cached = TASK_RESOURCE_CACHE.get(cache_key)
    if cached is not None:
        print_progress_event(task_name, "shared", "load", "reuse cached resources")
        return cached

    task_cfg = get_task_cfg(task_name)
    print_progress_event(task_name, "shared", "load", f"loading tokenizer/model for {task_cfg['model_name']}")
    tokenizer = load_tokenizer_with_local_cache_fallback(task_cfg["model_name"])

    if task_name in GLUE_TASKS:
        print_progress_event(task_name, "shared", "tokenize", "loading GLUE dataset split")
        dataset = load_glue_dataset_with_fallback(task_name)
        split = normalize_local_glue_labels(task_name, dataset[task_cfg["split"]])
        split = select_examples(split, max_samples)
        map_num_proc = min(tokenize_workers, len(split)) if tokenize_workers > 1 else None
        model = load_transformer_model_with_local_cache_fallback(
            AutoModelForSequenceClassification,
            task_cfg["model_name"],
            attn_implementation="eager",
        )
        tokenize_fn = functools.partial(
            tokenize_batch,
            tokenizer=tokenizer,
            sentence_keys=task_cfg["sentence_keys"],
            max_length=max_length,
        )
        encoded = split.map(
            tokenize_fn,
            batched=True,
            num_proc=map_num_proc,
        )
        raw_examples = split
        qa_examples = None
        qa_references = None
        qa_offset_mappings = None
        qa_features_per_example = None
    else:
        print_progress_event(task_name, "shared", "tokenize", "loading SQuAD dataset split")
        dataset = load_squad_dataset_with_fallback(task_name)
        split = select_examples(dataset[task_cfg["split"]], max_samples)
        map_num_proc = min(tokenize_workers, len(split)) if tokenize_workers > 1 else None
        model = load_transformer_model_with_local_cache_fallback(
            AutoModelForQuestionAnswering,
            task_cfg["model_name"],
            attn_implementation="eager",
        )
        tokenize_fn = functools.partial(
            tokenize_qa_validation_features,
            tokenizer=tokenizer,
            max_length=qa_max_length,
            doc_stride=qa_doc_stride,
        )
        encoded = split.map(
            tokenize_fn,
            batched=True,
            remove_columns=split.column_names,
            num_proc=map_num_proc,
        )
        raw_examples = split
        print_progress_event(task_name, "shared", "tokenize", "building SQuAD postprocess cache")
        example_ids = raw_examples["id"]
        contexts = raw_examples["context"]
        answers = raw_examples["answers"]
        qa_examples = list(zip(example_ids, contexts))
        qa_references = [{"id": example_id, "answers": answer} for example_id, answer in zip(example_ids, answers)]
        qa_offset_mappings = list(encoded["offset_mapping"])
        qa_features_per_example = defaultdict(list)
        for feature_index, example_id in enumerate(encoded["example_id"]):
            qa_features_per_example[example_id].append(feature_index)

    model.eval()
    model.to(device)
    if task_name in GLUE_TASKS:
        summary = f"samples={len(raw_examples)} encoded={len(encoded)} device={device}"
    else:
        summary = f"examples={len(raw_examples)} features={len(encoded)} device={device}"
    print_progress_event(task_name, "shared", "load", f"ready {summary}")

    resources = LoadedTaskResources(
        task_name=task_name,
        task_cfg=task_cfg,
        encoded=encoded,
        model=model,
        raw_examples=raw_examples,
        qa_examples=qa_examples,
        qa_references=qa_references,
        qa_offset_mappings=qa_offset_mappings,
        qa_features_per_example=qa_features_per_example,
    )
    TASK_RESOURCE_CACHE[cache_key] = resources
    return resources


def _topk_descending_indexes(values: np.ndarray, k: int) -> list[int]:
    if k <= 0 or values.size == 0:
        return []
    if k >= values.size:
        return np.argsort(values)[::-1].tolist()
    topk = np.argpartition(values, -k)[-k:]
    return topk[np.argsort(values[topk])[::-1]].tolist()


def _build_qa_prediction(
    example_id: str,
    context: str,
    offset_mappings,
    features_per_example: Dict[str, List[int]],
    start_logits: np.ndarray,
    end_logits: np.ndarray,
    n_best_size: int,
    max_answer_length: int,
    allow_no_answer: bool,
) -> dict:
    best_answer_text = ""
    best_answer_score = float("-inf")
    best_null_score = None

    for feature_index in features_per_example.get(example_id, []):
        offsets = offset_mappings[feature_index]
        feature_start_logits = start_logits[feature_index]
        feature_end_logits = end_logits[feature_index]
        cls_score = float(feature_start_logits[0] + feature_end_logits[0])
        if best_null_score is None or cls_score > best_null_score:
            best_null_score = cls_score

        start_indexes = _topk_descending_indexes(feature_start_logits, n_best_size)
        end_indexes = _topk_descending_indexes(feature_end_logits, n_best_size)
        for start_index in start_indexes:
            start_offset = offsets[start_index]
            if start_offset is None:
                continue
            start_char, _ = start_offset
            max_end_index = start_index + max_answer_length - 1
            for end_index in end_indexes:
                if end_index < start_index or end_index > max_end_index:
                    continue
                end_offset = offsets[end_index]
                if end_offset is None:
                    continue
                _, end_char = end_offset
                score = float(feature_start_logits[start_index] + feature_end_logits[end_index])
                if score > best_answer_score:
                    best_answer_score = score
                    best_answer_text = context[start_char:end_char]

    if allow_no_answer:
        null_score = 0.0 if best_null_score is None else best_null_score
        score_gap = max(-60.0, min(60.0, best_answer_score - null_score))
        no_answer_probability = 1.0 / (1.0 + math.exp(score_gap))
        prediction_text = "" if null_score >= best_answer_score else best_answer_text
        return {
            "id": example_id,
            "prediction_text": prediction_text,
            "no_answer_probability": float(no_answer_probability),
        }
    return {"id": example_id, "prediction_text": best_answer_text}


def _build_qa_prediction_chunk(
    examples_chunk: Sequence[tuple[str, str]],
    offset_mappings,
    features_per_example: Dict[str, List[int]],
    start_logits: np.ndarray,
    end_logits: np.ndarray,
    n_best_size: int,
    max_answer_length: int,
    allow_no_answer: bool,
) -> list[dict]:
    return [
        _build_qa_prediction(
            example_id=example_id,
            context=context,
            offset_mappings=offset_mappings,
            features_per_example=features_per_example,
            start_logits=start_logits,
            end_logits=end_logits,
            n_best_size=n_best_size,
            max_answer_length=max_answer_length,
            allow_no_answer=allow_no_answer,
        )
        for example_id, context in examples_chunk
    ]


def build_qa_predictions(
    examples: Sequence[tuple[str, str]],
    offset_mappings,
    features_per_example: Dict[str, List[int]],
    start_logits: np.ndarray,
    end_logits: np.ndarray,
    n_best_size: int,
    max_answer_length: int,
    version: str = "1.1",
    progress: ProgressPrinter | None = None,
    workers: int = 1,
):
    predictions: list[dict] = []
    allow_no_answer = version == "2.0"
    total_examples = len(examples)
    if total_examples == 0:
        return predictions

    worker_count = min(max(1, workers), total_examples)
    if worker_count <= 1 or total_examples < 512:
        for example_index, (example_id, context) in enumerate(examples, start=1):
            predictions.append(
                _build_qa_prediction(
                    example_id=example_id,
                    context=context,
                    offset_mappings=offset_mappings,
                    features_per_example=features_per_example,
                    start_logits=start_logits,
                    end_logits=end_logits,
                    n_best_size=n_best_size,
                    max_answer_length=max_answer_length,
                    allow_no_answer=allow_no_answer,
                )
            )
            if progress is not None:
                progress.emit(example_index, extra=f"examples={example_index}/{total_examples}")
        return predictions

    chunk_size = math.ceil(total_examples / worker_count)
    chunks = [examples[start : start + chunk_size] for start in range(0, total_examples, chunk_size)]
    with concurrent.futures.ThreadPoolExecutor(max_workers=worker_count) as executor:
        for chunk_predictions in executor.map(
            _build_qa_prediction_chunk,
            chunks,
            itertools.repeat(offset_mappings),
            itertools.repeat(features_per_example),
            itertools.repeat(start_logits),
            itertools.repeat(end_logits),
            itertools.repeat(n_best_size),
            itertools.repeat(max_answer_length),
            itertools.repeat(allow_no_answer),
        ):
            predictions.extend(chunk_predictions)
            if progress is not None:
                progress.emit(len(predictions), extra=f"examples={len(predictions)}/{total_examples}")

    return predictions


def _build_qa_feature_shards(total_features: int, batch_size: int, worker_count: int) -> List[Tuple[int, int]]:
    if total_features <= 0:
        return []
    total_batches = max(1, math.ceil(total_features / batch_size))
    target_shards = max(worker_count, worker_count * 4)
    shard_batch_span = max(1, math.ceil(total_batches / target_shards))

    shards: List[Tuple[int, int]] = []
    batch_start = 0
    while batch_start < total_batches:
        batch_end = min(total_batches, batch_start + shard_batch_span)
        feature_start = batch_start * batch_size
        feature_end = min(total_features, batch_end * batch_size)
        shards.append((feature_start, feature_end))
        batch_start = batch_end
    return shards


def _run_qa_inference_shard(request: QaInferenceShardRequest) -> QaInferenceShardResult:
    configure_runtime_threads(request.torch_threads, request.torch_interop_threads)
    cache_key = (
        request.task_name,
        request.max_samples,
        request.max_length,
        request.qa_max_length,
        request.qa_doc_stride,
        request.device,
    )
    resources = TASK_RESOURCE_CACHE.get(cache_key)
    if resources is None:
        resources = load_task_resources(
            task_name=request.task_name,
            max_samples=request.max_samples,
            max_length=request.max_length,
            qa_max_length=request.qa_max_length,
            qa_doc_stride=request.qa_doc_stride,
            device=request.device,
            tokenize_workers=request.tokenize_workers,
        )

    encoded = resources.encoded
    model = resources.model
    runtime_input_format = resolve_profile_input_format(request.profile, request.input_format)
    all_start_logits = []
    all_end_logits = []

    with torch.inference_mode():
        with patched_softmax(request.profile, runtime_input_format, request.collect_softmax_stats) as runtime:
            for start in range(request.feature_start, request.feature_end, request.batch_size):
                batch = encoded[start : min(start + request.batch_size, request.feature_end)]
                inputs = build_batch_inputs(batch, request.device)
                outputs = model(**inputs)
                all_start_logits.append(outputs.start_logits.detach().cpu().numpy())
                all_end_logits.append(outputs.end_logits.detach().cpu().numpy())

    if all_start_logits:
        start_logits = np.concatenate(all_start_logits, axis=0)
        end_logits = np.concatenate(all_end_logits, axis=0)
    else:
        start_logits = np.zeros((0, request.qa_max_length), dtype=np.float32)
        end_logits = np.zeros((0, request.qa_max_length), dtype=np.float32)

    return QaInferenceShardResult(
        shard_index=request.shard_index,
        feature_count=request.feature_end - request.feature_start,
        batch_count=max(0, math.ceil((request.feature_end - request.feature_start) / request.batch_size)),
        start_logits=start_logits,
        end_logits=end_logits,
        softmax_stats=runtime.get_raw_stats() if request.profile not in (None, "", "exact") else None,
    )


def _run_qa_inference_parallel(
    task_name: str,
    profile: str,
    max_samples: int,
    batch_size: int,
    max_length: int,
    qa_max_length: int,
    qa_doc_stride: int,
    device: str,
    input_format: str,
    tokenize_workers: int,
    torch_threads: int,
    torch_interop_threads: int,
    collect_softmax_stats: bool,
    progress: ProgressPrinter,
    total_features: int,
    total_examples: int,
    worker_count: int,
) -> Tuple[np.ndarray, np.ndarray, dict | None]:
    shards = _build_qa_feature_shards(total_features, batch_size, worker_count)
    shard_requests = [
        QaInferenceShardRequest(
            task_name=task_name,
            profile=profile,
            max_samples=max_samples,
            batch_size=batch_size,
            max_length=max_length,
            qa_max_length=qa_max_length,
            qa_doc_stride=qa_doc_stride,
            device=device,
            input_format=input_format,
            tokenize_workers=tokenize_workers,
            torch_threads=torch_threads,
            torch_interop_threads=torch_interop_threads,
            collect_softmax_stats=collect_softmax_stats,
            shard_index=index,
            feature_start=feature_start,
            feature_end=feature_end,
        )
        for index, (feature_start, feature_end) in enumerate(shards)
    ]

    progress.start_phase(
        "inference",
        max(1, math.ceil(total_features / batch_size)),
        "batches",
        extra=(
            f"features={total_features} examples={total_examples} "
            f"batch_size={batch_size} qa_infer_workers={worker_count}"
        ),
    )
    processed_features = 0
    processed_batches = 0
    results_by_index: Dict[int, QaInferenceShardResult] = {}

    try:
        with concurrent.futures.ProcessPoolExecutor(
            max_workers=worker_count,
            **_process_pool_executor_kwargs(),
        ) as executor:
            futures = [executor.submit(_run_qa_inference_shard, request) for request in shard_requests]
            for future in concurrent.futures.as_completed(futures):
                result = future.result()
                results_by_index[result.shard_index] = result
                processed_features += result.feature_count
                processed_batches += result.batch_count
                progress.emit(
                    processed_batches,
                    extra=f"features={processed_features}/{total_features}",
                )
    except (PermissionError, OSError) as exc:
        print(
            f"qa inference parallelism disabled, falling back to serial execution: {exc}",
            file=sys.stderr,
        )
        raise

    ordered_results = [results_by_index[index] for index in range(len(shard_requests))]
    progress.finish_phase(extra=f"features={total_features}/{total_features}")

    start_logits = np.concatenate([item.start_logits for item in ordered_results], axis=0)
    end_logits = np.concatenate([item.end_logits for item in ordered_results], axis=0)
    softmax_stats = None
    if profile not in (None, "", "exact"):
        softmax_stats = ApproxSoftmaxRuntime.merge_stats([item.softmax_stats for item in ordered_results])
    return start_logits, end_logits, softmax_stats


def evaluate_task(
    task_name: str,
    profile: str,
    max_samples: int,
    batch_size: int,
    max_length: int,
    qa_max_length: int,
    qa_doc_stride: int,
    qa_n_best_size: int,
    qa_max_answer_length: int,
    device: str,
    input_format: str,
    tokenize_workers: int,
    qa_postprocess_workers: int,
    qa_inference_workers: int,
    torch_threads: int,
    torch_interop_threads: int,
    collect_softmax_stats: bool = False,
    resources: LoadedTaskResources | None = None,
    progress_interval_sec: float = 30.0,
) -> TaskEvalResult:
    resources = resources or load_task_resources(
        task_name,
        max_samples,
        max_length,
        qa_max_length,
        qa_doc_stride,
        device,
        tokenize_workers,
    )
    task_cfg = resources.task_cfg
    encoded = resources.encoded
    model = resources.model
    progress = ProgressPrinter(task_name=task_name, profile=profile, interval_sec=progress_interval_sec)

    metric = load_metric_with_local_script(task_cfg["metric_name"], task_cfg["metric_config"])
    runtime_input_format = resolve_profile_input_format(profile, input_format)
    total_batches = max(1, math.ceil(len(encoded) / batch_size))

    softmax_stats = None
    runtime = None
    with torch.inference_mode():
        if task_name in QA_TASKS and qa_inference_workers > 1:
            total_examples = len(resources.raw_examples)
            try:
                start_logits, end_logits, softmax_stats = _run_qa_inference_parallel(
                    task_name=task_name,
                    profile=profile,
                    max_samples=max_samples,
                    batch_size=batch_size,
                    max_length=max_length,
                    qa_max_length=qa_max_length,
                    qa_doc_stride=qa_doc_stride,
                    device=device,
                    input_format=input_format,
                    tokenize_workers=tokenize_workers,
                    torch_threads=torch_threads,
                    torch_interop_threads=torch_interop_threads,
                    collect_softmax_stats=collect_softmax_stats,
                    progress=progress,
                    total_features=len(encoded),
                    total_examples=total_examples,
                    worker_count=qa_inference_workers,
                )
            except (PermissionError, OSError):
                qa_inference_workers = 1
        if task_name in GLUE_TASKS or qa_inference_workers <= 1:
            with patched_softmax(profile, runtime_input_format, collect_softmax_stats) as runtime:
                if task_name in GLUE_TASKS:
                    progress.start_phase(
                        "inference",
                        total_batches,
                        "batches",
                        extra=f"samples={len(encoded)} batch_size={batch_size}",
                    )
                    processed_samples = 0
                    for start in range(0, len(encoded), batch_size):
                        batch = encoded[start : start + batch_size]
                        inputs = build_batch_inputs(batch, device)
                        outputs = model(**inputs)
                        logits = outputs.logits.detach().cpu()
                        labels = batch["label"]

                        if task_cfg["label_mode"] == "classification":
                            predictions = logits.argmax(dim=-1).tolist()
                        else:
                            predictions = logits.squeeze(-1).tolist()

                        metric.add_batch(predictions=predictions, references=labels)
                        processed_samples += len(labels)
                        batch_index = min(total_batches, start // batch_size + 1)
                        progress.emit(
                            batch_index,
                            extra=f"samples={processed_samples}/{len(encoded)}",
                        )
                    progress.finish_phase(extra=f"samples={len(encoded)}/{len(encoded)}")
                else:
                    total_examples = len(resources.raw_examples)
                    progress.start_phase(
                        "inference",
                        total_batches,
                        "batches",
                        extra=f"features={len(encoded)} examples={total_examples} batch_size={batch_size}",
                    )
                    all_start_logits = []
                    all_end_logits = []
                    processed_features = 0
                    for start in range(0, len(encoded), batch_size):
                        batch = encoded[start : start + batch_size]
                        inputs = build_batch_inputs(batch, device)
                        outputs = model(**inputs)
                        all_start_logits.append(outputs.start_logits.detach().cpu().numpy())
                        all_end_logits.append(outputs.end_logits.detach().cpu().numpy())
                        processed_features += len(batch["input_ids"])
                        batch_index = min(total_batches, start // batch_size + 1)
                        progress.emit(
                            batch_index,
                            extra=f"features={processed_features}/{len(encoded)}",
                        )
                    progress.finish_phase(extra=f"features={len(encoded)}/{len(encoded)}")

                    start_logits = np.concatenate(all_start_logits, axis=0)
                    end_logits = np.concatenate(all_end_logits, axis=0)

        if task_name in QA_TASKS:
            total_examples = len(resources.raw_examples)
            progress.start_phase(
                "postprocess",
                total_examples,
                "examples",
                extra=f"features={len(encoded)}",
            )
            predictions = build_qa_predictions(
                examples=resources.qa_examples or [],
                offset_mappings=resources.qa_offset_mappings or [],
                features_per_example=resources.qa_features_per_example or {},
                start_logits=start_logits,
                end_logits=end_logits,
                n_best_size=qa_n_best_size,
                max_answer_length=qa_max_answer_length,
                version=task_cfg.get("version", "1.1"),
                progress=progress,
                workers=qa_postprocess_workers,
            )
            progress.finish_phase(extra=f"examples={total_examples}/{total_examples}")
            metric.add_batch(predictions=predictions, references=resources.qa_references or [])

    metrics = metric.compute()
    if task_name in QA_TASKS and qa_inference_workers > 1 and profile not in (None, "", "exact"):
        softmax_metrics = ApproxSoftmaxRuntime.summarize_stats(softmax_stats)
    else:
        softmax_metrics = runtime.get_summary() if profile not in (None, "", "exact") else None
    num_samples = len(resources.raw_examples) if task_name in QA_TASKS else len(encoded)

    return TaskEvalResult(
        task=task_name,
        profile=profile,
        num_samples=num_samples,
        metrics=metrics,
        softmax_metrics=softmax_metrics,
    )


def scan_task_prediction_diff(
    task_name: str,
    profile: str,
    max_samples: int,
    batch_size: int,
    max_length: int,
    qa_max_length: int,
    qa_doc_stride: int,
    device: str,
    input_format: str,
    tokenize_workers: int,
    resources: LoadedTaskResources | None = None,
    progress_interval_sec: float = 30.0,
) -> TaskDiffScanResult:
    resources = resources or load_task_resources(
        task_name,
        max_samples,
        max_length,
        qa_max_length,
        qa_doc_stride,
        device,
        tokenize_workers,
    )
    task_cfg = resources.task_cfg
    if task_name not in GLUE_TASKS or task_cfg["label_mode"] != "classification":
        raise ValueError("scan-diff-only currently supports classification tasks only")
    encoded = resources.encoded
    model = resources.model
    runtime_input_format = resolve_profile_input_format(profile, input_format)
    progress = ProgressPrinter(task_name=task_name, profile=profile, interval_sec=progress_interval_sec)
    total_batches = max(1, math.ceil(len(encoded) / batch_size))

    changed_prediction_count = 0
    max_logit_abs_diff = 0.0

    with torch.inference_mode():
        progress.start_phase(
            "diff_scan",
            total_batches,
            "batches",
            extra=f"samples={len(encoded)} batch_size={batch_size}",
        )
        processed_samples = 0
        for start in range(0, len(encoded), batch_size):
            batch = encoded[start : start + batch_size]
            inputs = build_batch_inputs(batch, device)
            logits_exact = model(**inputs).logits.detach().cpu()
            with patched_softmax(profile, runtime_input_format):
                logits_approx = model(**inputs).logits.detach().cpu()

            pred_exact = logits_exact.argmax(dim=-1)
            pred_approx = logits_approx.argmax(dim=-1)
            changed_prediction_count += int((pred_exact != pred_approx).sum().item())
            max_logit_abs_diff = max(max_logit_abs_diff, float((logits_exact - logits_approx).abs().max().item()))
            processed_samples += len(batch["input_ids"])
            batch_index = min(total_batches, start // batch_size + 1)
            progress.emit(batch_index, extra=f"samples={processed_samples}/{len(encoded)}")
        progress.finish_phase(extra=f"samples={len(encoded)}/{len(encoded)}")

    num_samples = len(encoded)
    return TaskDiffScanResult(
        task=task_name,
        profile=profile,
        num_samples=num_samples,
        changed_prediction_count=changed_prediction_count,
        changed_prediction_rate=changed_prediction_count / num_samples,
        max_logit_abs_diff=max_logit_abs_diff,
    )


def print_results(results: Sequence[TaskEvalResult]) -> None:
    grouped: Dict[str, List[TaskEvalResult]] = {}
    for result in results:
        grouped.setdefault(result.task, []).append(result)

    print("BERT task evaluation summary")
    for task, task_results in grouped.items():
        print()
        print(f"task              : {task}")
        for result in task_results:
            metrics_text = ", ".join(f"{key}={value:.6f}" for key, value in result.metrics.items())
            print(
                f"profile={result.profile:<8} "
                f"samples={result.num_samples:<5} "
                f"{metrics_text}"
            )
            if result.softmax_metrics:
                sm = result.softmax_metrics
                prune_text = (
                    f"prune={sm['total_pruned_rate']:.4f}, "
                    f"blk={sm['block_pruned_element_rate']:.4f}, "
                    f"elem_only={sm['element_pruned_only_rate']:.4f}, "
                    f"aggr_blk={sm['aggressive_block_rate']:.4f}"
                )
                if "topk_rates" in sm:
                    topk_text = ", ".join(f"top{k}={rate:.4f}" for k, rate in sm["topk_rates"].items())
                    print(
                        f"      [Softmax] KL={sm['avg_kl']:.6e}, PPL_Proxy={sm['ppl_proxy']:.4f}, "
                        f"{topk_text}, {prune_text}"
                    )
                else:
                    print(f"      [Softmax] {prune_text}")


def print_diff_scan_results(results: Sequence[TaskDiffScanResult]) -> None:
    print("BERT task prediction-diff scan")
    for result in results:
        print()
        print(f"task                    : {result.task}")
        print(f"profile                 : {result.profile}")
        print(f"samples                 : {result.num_samples}")
        print(f"changed_prediction_count: {result.changed_prediction_count}")
        print(f"changed_prediction_rate : {result.changed_prediction_rate:.6f}")
        print(f"max_logit_abs_diff      : {result.max_logit_abs_diff:.8e}")


def _execute_task_profile(
    request: TaskRunRequest | TaskProfileRunRequest,
    profile: str,
    resources: LoadedTaskResources | None = None,
) -> Tuple[TaskEvalResult | None, TaskDiffScanResult | None]:
    if request.scan_diff_only:
        if profile == "exact":
            return None, None
        return (
            None,
            scan_task_prediction_diff(
                task_name=request.task_name,
                profile=profile,
                max_samples=request.max_samples,
                batch_size=request.batch_size,
                max_length=request.max_length,
                qa_max_length=request.qa_max_length,
                qa_doc_stride=request.qa_doc_stride,
                device=request.device,
                input_format=request.input_format,
                tokenize_workers=request.tokenize_workers,
                resources=resources,
                progress_interval_sec=request.progress_interval_sec,
            ),
        )

    return (
        evaluate_task(
            task_name=request.task_name,
            profile=profile,
            max_samples=request.max_samples,
            batch_size=request.batch_size,
            max_length=request.max_length,
            qa_max_length=request.qa_max_length,
            qa_doc_stride=request.qa_doc_stride,
            qa_n_best_size=request.qa_n_best_size,
            qa_max_answer_length=request.qa_max_answer_length,
            device=request.device,
            input_format=request.input_format,
            tokenize_workers=request.tokenize_workers,
            qa_postprocess_workers=request.qa_postprocess_workers,
            qa_inference_workers=request.qa_inference_workers,
            torch_threads=request.torch_threads,
            torch_interop_threads=request.torch_interop_threads,
            collect_softmax_stats=request.collect_softmax_stats,
            resources=resources,
            progress_interval_sec=request.progress_interval_sec,
        ),
        None,
    )


def run_task_bundle(request: TaskRunRequest) -> TaskRunBundle:
    configure_runtime_threads(request.torch_threads, request.torch_interop_threads)
    resources = load_task_resources(
        task_name=request.task_name,
        max_samples=request.max_samples,
        max_length=request.max_length,
        qa_max_length=request.qa_max_length,
        qa_doc_stride=request.qa_doc_stride,
        device=request.device,
        tokenize_workers=request.tokenize_workers,
    )
    bundle = TaskRunBundle(task_name=request.task_name)
    for profile in request.profiles:
        print_progress_event(request.task_name, profile, "task", "profile run started")
        result, diff_result = _execute_task_profile(request, profile, resources=resources)
        if result is not None:
            bundle.results.append(result)
        if diff_result is not None:
            bundle.diff_results.append(diff_result)
        print_progress_event(request.task_name, profile, "task", "profile run finished")
    return bundle


def _expand_task_profile_requests(requests: Sequence[TaskRunRequest]) -> List[TaskProfileRunRequest]:
    expanded: List[TaskProfileRunRequest] = []
    for bundle_index, request in enumerate(requests):
        for profile_index, profile in enumerate(request.profiles):
            if request.scan_diff_only and profile == "exact":
                continue
            expanded.append(
                TaskProfileRunRequest(
                    bundle_index=bundle_index,
                    profile_index=profile_index,
                    task_name=request.task_name,
                    profile=profile,
                    max_samples=request.max_samples,
                    batch_size=request.batch_size,
                    max_length=request.max_length,
                    qa_max_length=request.qa_max_length,
                    qa_doc_stride=request.qa_doc_stride,
                    qa_n_best_size=request.qa_n_best_size,
                    qa_max_answer_length=request.qa_max_answer_length,
                    device=request.device,
                    input_format=request.input_format,
                    tokenize_workers=request.tokenize_workers,
                    qa_postprocess_workers=request.qa_postprocess_workers,
                    torch_threads=request.torch_threads,
                    torch_interop_threads=request.torch_interop_threads,
                    qa_inference_workers=request.qa_inference_workers,
                    collect_softmax_stats=request.collect_softmax_stats,
                    scan_diff_only=request.scan_diff_only,
                    progress_interval_sec=request.progress_interval_sec,
                )
            )
    return expanded


def run_task_profile_request(request: TaskProfileRunRequest) -> TaskProfileRunBundleItem:
    configure_runtime_threads(request.torch_threads, request.torch_interop_threads)
    print_progress_event(request.task_name, request.profile, "task", "profile run started")
    result, diff_result = _execute_task_profile(request, request.profile, resources=None)
    print_progress_event(request.task_name, request.profile, "task", "profile run finished")
    return TaskProfileRunBundleItem(
        bundle_index=request.bundle_index,
        profile_index=request.profile_index,
        task_name=request.task_name,
        result=result,
        diff_result=diff_result,
    )


def _process_pool_executor_kwargs() -> dict:
    if sys.platform == "win32":
        return {}
    try:
        return {"mp_context": mp.get_context("fork")}
    except ValueError:
        return {}


def _prefork_profile_resources(requests: Sequence[TaskRunRequest], run_workers: int) -> None:
    if run_workers <= 1 or len(requests) != 1:
        return
    if sys.platform == "win32":
        return
    request = requests[0]
    configure_runtime_threads(request.torch_threads, request.torch_interop_threads)
    print_progress_event(request.task_name, "shared", "load", "prefork loading shared resources for profile parallelism")
    load_task_resources(
        task_name=request.task_name,
        max_samples=request.max_samples,
        max_length=request.max_length,
        qa_max_length=request.qa_max_length,
        qa_doc_stride=request.qa_doc_stride,
        device=request.device,
        tokenize_workers=request.tokenize_workers,
    )


def run_task_profile_bundles(requests: Sequence[TaskRunRequest], run_workers: int) -> List[TaskRunBundle]:
    if not requests:
        return []

    profile_requests = _expand_task_profile_requests(requests)
    bundles = [TaskRunBundle(task_name=request.task_name) for request in requests]
    if not profile_requests:
        return bundles

    if run_workers <= 1 or len(profile_requests) == 1:
        items = [run_task_profile_request(request) for request in profile_requests]
    else:
        max_workers = min(run_workers, len(profile_requests))
        items_by_order: Dict[Tuple[int, int], TaskProfileRunBundleItem] = {}
        _prefork_profile_resources(requests, max_workers)
        try:
            with concurrent.futures.ProcessPoolExecutor(
                max_workers=max_workers,
                **_process_pool_executor_kwargs(),
            ) as executor:
                future_to_order = {
                    executor.submit(run_task_profile_request, request): (request.bundle_index, request.profile_index)
                    for request in profile_requests
                }
                for future in concurrent.futures.as_completed(future_to_order):
                    item = future.result()
                    items_by_order[future_to_order[future]] = item
        except (PermissionError, OSError) as exc:
            print(
                f"run parallelism disabled, falling back to serial execution: {exc}",
                file=sys.stderr,
            )
            items = [run_task_profile_request(request) for request in profile_requests]
        else:
            items = [items_by_order[(request.bundle_index, request.profile_index)] for request in profile_requests]

    for item in items:
        bundle = bundles[item.bundle_index]
        if item.result is not None:
            bundle.results.append(item.result)
        if item.diff_result is not None:
            bundle.diff_results.append(item.diff_result)
    return bundles


def run_task_bundles(requests: Sequence[TaskRunRequest], task_workers: int) -> List[TaskRunBundle]:
    if not requests:
        return []
    if task_workers <= 1 or len(requests) == 1:
        return [run_task_bundle(request) for request in requests]

    max_workers = min(task_workers, len(requests))
    bundles_by_index: Dict[int, TaskRunBundle] = {}
    try:
        with concurrent.futures.ProcessPoolExecutor(
            max_workers=max_workers,
            **_process_pool_executor_kwargs(),
        ) as executor:
            future_to_index = {
                executor.submit(run_task_bundle, request): index for index, request in enumerate(requests)
            }
            for future in concurrent.futures.as_completed(future_to_index):
                bundle = future.result()
                bundles_by_index[future_to_index[future]] = bundle
    except (PermissionError, OSError) as exc:
        print(
            f"task parallelism disabled, falling back to serial execution: {exc}",
            file=sys.stderr,
        )
        return [run_task_bundle(request) for request in requests]

    return [bundles_by_index[index] for index in range(len(requests))]


def resolve_worker_plan(
    tasks: Sequence[str],
    profiles: Sequence[str],
    task_workers: int,
    run_workers: int,
    auto_workers: bool,
    reserve_cpus: int,
    max_auto_workers: int,
) -> Tuple[int, int, str]:
    if task_workers > 1 and run_workers > 1:
        raise ValueError("use either --task-workers or --run-workers, not both at the same time")
    if task_workers > 1:
        return task_workers, 1, "manual-task"
    if run_workers > 1:
        return 1, run_workers, "manual-run"
    if not auto_workers:
        return 1, 1, "serial"

    cpu_total = os.cpu_count() or 1
    reserve = min(max(0, reserve_cpus), max(0, cpu_total - 1))
    worker_budget = max(1, cpu_total - reserve)
    if max_auto_workers > 0:
        worker_budget = min(worker_budget, max_auto_workers)

    if len(tasks) > 1:
        return min(worker_budget, len(tasks)), 1, "auto-task"
    if len(profiles) > 1:
        return 1, min(worker_budget, len(profiles)), "auto-run"
    return 1, 1, "auto-serial"


def print_worker_plan(mode: str, task_workers: int, run_workers: int, reserve_cpus: int) -> None:
    cpu_total = os.cpu_count() or 1
    print("parallel_plan     :", mode)
    print(f"cpu_total         : {cpu_total}")
    print(f"reserve_cpus      : {max(0, reserve_cpus)}")
    print(f"task_workers      : {task_workers}")
    print(f"run_workers       : {run_workers}")


def print_runtime_tuning(
    torch_threads: int,
    torch_interop_threads: int,
    tokenize_workers: int,
    qa_postprocess_workers: int,
    qa_inference_workers: int,
) -> None:
    print(f"torch_threads     : {torch_threads}")
    print(f"torch_interop_thr : {torch_interop_threads}")
    print(f"tokenize_workers  : {tokenize_workers}")
    print(f"qa_post_workers   : {qa_postprocess_workers}")
    print(f"qa_infer_workers  : {qa_inference_workers}")


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(line_buffering=True)
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(line_buffering=True)

    args = parse_args()
    ensure_runtime_dependencies()
    tasks = [task.strip() for task in args.tasks.split(",") if task.strip()]
    profiles = [profile.strip() for profile in args.profiles.split(",") if profile.strip()]
    fp16_profiles = [profile for profile in profiles if profile_requires_fp16_input(profile)]
    if fp16_profiles and args.input_format != "fp16":
        print(
            f"note: forcing input_format=fp16 for rtl_exact profile(s): {','.join(fp16_profiles)}",
            file=sys.stderr,
        )
    task_workers, run_workers, worker_mode = resolve_worker_plan(
        tasks=tasks,
        profiles=profiles,
        task_workers=args.task_workers,
        run_workers=args.run_workers,
        auto_workers=args.auto_workers,
        reserve_cpus=args.reserve_cpus,
        max_auto_workers=args.max_auto_workers,
    )
    if args.qa_inference_workers > 1 and (task_workers > 1 or run_workers > 1):
        raise ValueError("use either --task-workers/--run-workers or --qa-inference-workers, not both at the same time")
    qa_inference_workers = resolve_qa_inference_workers(
        requested=args.qa_inference_workers,
        tasks=tasks,
        profiles=profiles,
        task_workers=task_workers,
        run_workers=run_workers,
        reserve_cpus=args.reserve_cpus,
    )
    parallel_jobs = max(1, task_workers, run_workers, qa_inference_workers)
    torch_threads = resolve_torch_threads(args.torch_threads, parallel_jobs, args.reserve_cpus)
    tokenize_workers = resolve_aux_workers(args.tokenize_workers, parallel_jobs, args.reserve_cpus, cap=4)
    qa_postprocess_workers = resolve_aux_workers(
        args.qa_postprocess_workers,
        parallel_jobs,
        args.reserve_cpus,
        cap=4,
    )
    print_worker_plan(worker_mode, task_workers, run_workers, args.reserve_cpus)
    print_runtime_tuning(
        torch_threads=torch_threads,
        torch_interop_threads=args.torch_interop_threads,
        tokenize_workers=tokenize_workers,
        qa_postprocess_workers=qa_postprocess_workers,
        qa_inference_workers=qa_inference_workers,
    )

    unknown_tasks = [task for task in tasks if task not in GLUE_TASKS and task not in QA_TASKS]
    if unknown_tasks:
        raise ValueError(f"unsupported tasks: {','.join(unknown_tasks)}")

    requests = [
        TaskRunRequest(
            task_name=task,
            profiles=tuple(profiles),
            max_samples=args.max_samples,
            batch_size=args.batch_size,
            max_length=args.max_length,
            qa_max_length=args.qa_max_length,
            qa_doc_stride=args.qa_doc_stride,
            qa_n_best_size=args.qa_n_best_size,
            qa_max_answer_length=args.qa_max_answer_length,
            device=args.device,
            input_format=args.input_format,
            tokenize_workers=tokenize_workers,
            qa_postprocess_workers=qa_postprocess_workers,
            torch_threads=torch_threads,
            torch_interop_threads=args.torch_interop_threads,
            qa_inference_workers=qa_inference_workers,
            collect_softmax_stats=args.collect_softmax_stats,
            scan_diff_only=args.scan_diff_only,
            progress_interval_sec=args.progress_interval_sec,
        )
        for task in tasks
    ]
    if run_workers > 1:
        bundles = run_task_profile_bundles(requests, run_workers)
    else:
        bundles = run_task_bundles(requests, task_workers)

    results: List[TaskEvalResult] = []
    diff_results: List[TaskDiffScanResult] = []
    for bundle in bundles:
        results.extend(bundle.results)
        diff_results.extend(bundle.diff_results)

    if args.scan_diff_only:
        print_diff_scan_results(diff_results)
    else:
        print_results(results)

    if args.output_json:
        path = Path(args.output_json)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = [asdict(result) for result in (diff_results if args.scan_diff_only else results)]
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
