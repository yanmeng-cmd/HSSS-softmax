"""Model, corpus, and fixed-mask MLM evaluation helpers for Longformer."""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import torch
from tqdm import tqdm
from transformers import AutoModelForMaskedLM, AutoTokenizer

try:
    from .experiment_paths import HF_DATASETS_CACHE_DIR, configure_environment
except ImportError:  # pragma: no cover - supports direct script execution
    from experiment_paths import HF_DATASETS_CACHE_DIR, configure_environment


configure_environment()


WIKITEXT_PARQUET_FILES = {
    "wikitext-2-raw-v1": {
        "train": "train-00000-of-00001.parquet",
        "validation": "validation-00000-of-00001.parquet",
        "test": "test-00000-of-00001.parquet",
    }
}


@dataclass
class MlmWindow:
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    labels: torch.Tensor
    global_attention_mask: torch.Tensor | None
    content_tokens: int
    masked_tokens: int


@dataclass
class MlmResult:
    profile: str
    mlm_loss: float
    masked_pseudo_perplexity: float
    total_loss: float
    masked_tokens: int
    top1_correct: int
    top5_correct: int
    top1_predictions: List[int]
    top5_predictions: List[List[int]]
    input_tokens: int
    num_windows: int
    elapsed_sec: float
    softmax_stats: Dict[str, int] | None = None

    @property
    def top1_accuracy(self) -> float:
        return self.top1_correct / self.masked_tokens if self.masked_tokens > 0 else 0.0

    @property
    def top5_accuracy(self) -> float:
        return self.top5_correct / self.masked_tokens if self.masked_tokens > 0 else 0.0


def resolve_device(device_text: str) -> torch.device:
    if device_text == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_text)


def resolve_dtype(dtype_text: str, device: torch.device) -> torch.dtype:
    if dtype_text == "auto":
        return torch.float16 if device.type == "cuda" else torch.float32
    mapping = {
        "float16": torch.float16,
        "fp16": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float32": torch.float32,
        "fp32": torch.float32,
    }
    if dtype_text not in mapping:
        raise ValueError(f"unsupported model dtype: {dtype_text}")
    dtype = mapping[dtype_text]
    if device.type == "cpu" and dtype == torch.float16:
        print("warning: CPU float16 is slow/unsupported for many ops; falling back to float32", flush=True)
        return torch.float32
    return dtype


def configure_torch_threads(num_threads: int, inter_op_threads: int) -> None:
    """Configure PyTorch CPU parallelism for reproducible CPU-only runs."""
    if num_threads > 0:
        torch.set_num_threads(num_threads)
    if inter_op_threads > 0:
        try:
            torch.set_num_interop_threads(inter_op_threads)
        except RuntimeError as exc:
            print(f"warning: could not set inter-op threads after parallel work started: {exc}", flush=True)
    print(
        f"torch_threads : intra_op={torch.get_num_threads()} "
        f"inter_op={torch.get_num_interop_threads()}",
        flush=True,
    )


def resolve_model_input_device(model, fallback_device: torch.device) -> torch.device:
    if hasattr(model, "device"):
        try:
            return model.device
        except Exception:
            pass
    try:
        return next(model.parameters()).device
    except StopIteration:
        return fallback_device


def load_text_corpus(args: argparse.Namespace) -> str:
    if args.text_file:
        return Path(args.text_file).read_text(encoding=args.text_encoding)

    local_text = load_local_wikitext_parquet(args)
    if local_text is not None:
        return local_text

    try:
        from datasets import load_dataset
    except ModuleNotFoundError as exc:  # pragma: no cover - optional fallback path
        raise RuntimeError("datasets is required for the default WikiText-2 path") from exc

    dataset = load_dataset(
        args.dataset,
        args.dataset_config,
        split=args.dataset_split,
        cache_dir=str(HF_DATASETS_CACHE_DIR),
    )
    if args.text_column not in dataset.column_names:
        raise ValueError(
            f"text column {args.text_column!r} not found in dataset columns {dataset.column_names!r}"
        )
    texts = [text for text in dataset[args.text_column] if isinstance(text, str) and text.strip()]
    if not texts:
        raise ValueError("loaded dataset does not contain any non-empty text rows")
    return "\n\n".join(texts)


def load_local_wikitext_parquet(args: argparse.Namespace) -> str | None:
    if args.dataset != "Salesforce/wikitext":
        return None
    config_files = WIKITEXT_PARQUET_FILES.get(args.dataset_config)
    if not config_files:
        return None
    filename = config_files.get(args.dataset_split)
    if not filename:
        return None

    parquet_path = (
        HF_DATASETS_CACHE_DIR
        / "direct"
        / args.dataset.replace("/", "--")
        / args.dataset_config
        / filename
    )
    if not parquet_path.exists():
        return None

    try:
        import pyarrow.parquet as pq
    except ModuleNotFoundError as exc:
        raise RuntimeError("pyarrow is required to read the locally downloaded WikiText parquet file") from exc

    table = pq.read_table(parquet_path, columns=[args.text_column])
    texts = [text for text in table[args.text_column].to_pylist() if isinstance(text, str) and text.strip()]
    if not texts:
        raise ValueError(f"local parquet file does not contain non-empty text rows: {parquet_path}")
    print(f"Loaded local parquet corpus: {parquet_path}", flush=True)
    return "\n\n".join(texts)


def load_model_and_tokenizer(args: argparse.Namespace):
    device = resolve_device(args.device)
    dtype = resolve_dtype(args.model_dtype, device)

    load_kwargs = {
        "torch_dtype": dtype,
        "local_files_only": args.local_files_only,
        "trust_remote_code": args.trust_remote_code,
    }
    if args.device_map:
        load_kwargs["device_map"] = args.device_map

    try:
        load_kwargs["attn_implementation"] = "eager"
        model = AutoModelForMaskedLM.from_pretrained(args.model, **load_kwargs)
    except TypeError as exc:
        if "attn_implementation" not in str(exc):
            raise
        load_kwargs.pop("attn_implementation", None)
        model = AutoModelForMaskedLM.from_pretrained(args.model, **load_kwargs)

    if hasattr(model, "set_attn_implementation"):
        try:
            model.set_attn_implementation("eager")
        except Exception:
            pass
    elif hasattr(model, "config") and hasattr(model.config, "_attn_implementation"):
        try:
            model.config._attn_implementation = "eager"
        except Exception:
            pass

    tokenizer = AutoTokenizer.from_pretrained(
        args.model,
        local_files_only=args.local_files_only,
        trust_remote_code=args.trust_remote_code,
    )
    if tokenizer.mask_token_id is None:
        raise ValueError("tokenizer must define a mask token for MLM evaluation")
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token or tokenizer.mask_token

    model.eval()
    if not args.device_map:
        model.to(device)

    return model, tokenizer, device, dtype


def build_token_ids(tokenizer, text: str, max_tokens: int) -> torch.Tensor:
    encoded = tokenizer(text, add_special_tokens=False, return_attention_mask=False)
    input_ids = torch.tensor(encoded["input_ids"], dtype=torch.long)
    if max_tokens > 0 and input_ids.numel() > max_tokens:
        input_ids = input_ids[:max_tokens]
    if input_ids.numel() == 0:
        raise ValueError("tokenized text is empty")
    return input_ids


def cap_max_length(model, requested_max_length: int) -> int:
    model_max = getattr(getattr(model, "config", None), "max_position_embeddings", None)
    if not isinstance(model_max, int) or model_max <= 0:
        return requested_max_length
    if requested_max_length <= 0:
        return model_max
    if requested_max_length > model_max:
        print(
            f"warning: max_length={requested_max_length} exceeds model max_position_embeddings={model_max}; "
            f"using {model_max}",
            flush=True,
        )
        return model_max
    return requested_max_length


def _build_global_attention_mask(mode: str, seq_len: int) -> torch.Tensor | None:
    if mode == "none":
        return None
    mask = torch.zeros((1, seq_len), dtype=torch.long)
    if mode == "cls" and seq_len > 0:
        mask[0, 0] = 1
    return mask


def _select_mask_positions(
    eligible_positions: List[int],
    mlm_probability: float,
    generator: torch.Generator,
) -> torch.Tensor:
    if not eligible_positions:
        raise ValueError("no maskable positions in window")
    probs = torch.rand(len(eligible_positions), generator=generator)
    selected = probs < mlm_probability
    if not bool(selected.any()):
        fallback_idx = int(torch.randint(len(eligible_positions), (1,), generator=generator).item())
        selected[fallback_idx] = True
    return torch.tensor([eligible_positions[idx] for idx, flag in enumerate(selected.tolist()) if flag], dtype=torch.long)


def _build_input_with_specials(tokenizer, token_ids: List[int]) -> tuple[List[int], List[int]]:
    build_inputs = getattr(tokenizer, "build_inputs_with_special_tokens", None)
    if callable(build_inputs):
        ids = build_inputs(token_ids)
    else:
        prefix_token = tokenizer.cls_token_id if tokenizer.cls_token_id is not None else tokenizer.bos_token_id
        suffix_token = tokenizer.sep_token_id if tokenizer.sep_token_id is not None else tokenizer.eos_token_id
        prefix = [] if prefix_token is None else [prefix_token]
        suffix = [] if suffix_token is None else [suffix_token]
        ids = prefix + token_ids + suffix

    get_special_mask = getattr(tokenizer, "get_special_tokens_mask", None)
    if callable(get_special_mask):
        special_mask = get_special_mask(ids, already_has_special_tokens=True)
    else:
        all_special_ids = set(getattr(tokenizer, "all_special_ids", []))
        special_mask = [1 if token_id in all_special_ids else 0 for token_id in ids]
    return ids, special_mask


def build_mlm_windows(
    model,
    tokenizer,
    text: str,
    max_length: int,
    stride: int,
    max_eval_tokens: int,
    mlm_probability: float,
    mask_seed: int,
    global_attention: str,
) -> List[MlmWindow]:
    if not (0.0 < mlm_probability <= 1.0):
        raise ValueError("mlm_probability must be in (0, 1]")

    max_length = cap_max_length(model, max_length)
    special_tokens = tokenizer.num_special_tokens_to_add(pair=False)
    content_window = max_length - special_tokens
    if content_window <= 0:
        raise ValueError(f"max_length={max_length} is too small for tokenizer special tokens")
    if stride <= 0:
        stride = content_window
    if stride > content_window:
        raise ValueError(f"stride={stride} must be <= content window size {content_window}")

    token_ids = build_token_ids(tokenizer, text, max_eval_tokens)
    generator = torch.Generator()
    generator.manual_seed(mask_seed)
    windows: List[MlmWindow] = []

    for start in range(0, int(token_ids.numel()), stride):
        content = token_ids[start : start + content_window]
        if content.numel() == 0:
            continue

        ids, special_mask = _build_input_with_specials(tokenizer, content.tolist())
        input_ids = torch.tensor([ids], dtype=torch.long)
        labels = torch.full_like(input_ids, -100)
        attention_mask = torch.ones_like(input_ids)

        eligible = [
            idx
            for idx, is_special in enumerate(special_mask)
            if not is_special and input_ids[0, idx].item() != tokenizer.pad_token_id
        ]
        mask_positions = _select_mask_positions(eligible, mlm_probability, generator)
        labels[0, mask_positions] = input_ids[0, mask_positions]
        input_ids[0, mask_positions] = tokenizer.mask_token_id

        global_attention_mask = _build_global_attention_mask(global_attention, input_ids.size(1))
        windows.append(
            MlmWindow(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
                global_attention_mask=global_attention_mask,
                content_tokens=int(content.numel()),
                masked_tokens=int(mask_positions.numel()),
            )
        )

        if start + content_window >= token_ids.numel():
            break

    if not windows:
        raise ValueError("no MLM windows were built")
    return windows


def evaluate_mlm_loss(
    model,
    windows: List[MlmWindow],
    device: torch.device,
) -> tuple[float, float, float, int, int, int, List[int], List[List[int]], int]:
    total_loss = 0.0
    total_masked_tokens = 0
    top1_correct = 0
    top5_correct = 0
    top1_predictions: List[int] = []
    top5_predictions: List[List[int]] = []
    total_input_tokens = 0

    for window in tqdm(windows, desc="mlm", unit="window"):
        valid_tokens = window.masked_tokens
        if valid_tokens <= 0:
            continue

        input_ids = window.input_ids.to(device)
        attention_mask = window.attention_mask.to(device)
        labels = window.labels.to(device)
        forward_kwargs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }
        if window.global_attention_mask is not None:
            forward_kwargs["global_attention_mask"] = window.global_attention_mask.to(device)

        with torch.inference_mode():
            outputs = model(**forward_kwargs)
        loss_value = float(outputs.loss.detach().cpu())
        total_loss += loss_value * valid_tokens
        total_masked_tokens += valid_tokens
        total_input_tokens += window.content_tokens

        masked_positions = labels != -100
        masked_logits = outputs.logits[masked_positions]
        target_ids = labels[masked_positions]
        top1 = masked_logits.argmax(dim=-1)
        top1_correct += int((top1 == target_ids).sum().item())
        topk = min(5, int(masked_logits.shape[-1]))
        top5 = masked_logits.topk(k=topk, dim=-1).indices
        top5_correct += int((top5 == target_ids.unsqueeze(-1)).any(dim=-1).sum().item())
        top1_predictions.extend(int(value) for value in top1.detach().cpu().tolist())
        top5_predictions.extend([int(value) for value in row] for row in top5.detach().cpu().tolist())

    if total_masked_tokens == 0:
        raise ValueError("no valid masked tokens were accumulated")

    mean_loss = total_loss / total_masked_tokens
    return (
        mean_loss,
        math.exp(mean_loss),
        total_loss,
        total_masked_tokens,
        top1_correct,
        top5_correct,
        top1_predictions,
        top5_predictions,
        total_input_tokens,
    )
