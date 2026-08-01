"""Command-line runner for Longformer fixed-mask MLM experiments."""

from __future__ import annotations

import argparse
import math
import time
from typing import List

import torch

try:
    from .experiment_paths import cached_longformer_model_path, configure_environment
    from .hardware_softmax import DEFAULT_SOFTMAX_PROFILE, patched_softmax
    from .mlm_eval import (
        MlmResult,
        build_mlm_windows,
        configure_torch_threads,
        evaluate_mlm_loss,
        load_model_and_tokenizer,
        load_text_corpus,
        resolve_model_input_device,
    )
except ImportError:  # pragma: no cover - supports direct script execution
    from experiment_paths import cached_longformer_model_path, configure_environment
    from hardware_softmax import DEFAULT_SOFTMAX_PROFILE, patched_softmax
    from mlm_eval import (
        MlmResult,
        build_mlm_windows,
        configure_torch_threads,
        evaluate_mlm_loss,
        load_model_and_tokenizer,
        load_text_corpus,
        resolve_model_input_device,
    )


configure_environment()


def default_model_name() -> str:
    cached_path = cached_longformer_model_path()
    if cached_path.exists() and (cached_path / "config.json").exists():
        return str(cached_path)
    return "allenai/longformer-base-4096"


def parse_profiles(args: argparse.Namespace) -> List[str]:
    if args.profiles.strip():
        profiles = [profile.strip() for profile in args.profiles.split(",") if profile.strip()]
    elif args.compare:
        profiles = ["exact", args.profile]
    elif args.exact:
        profiles = ["exact"]
    else:
        profiles = [args.profile]

    if not profiles:
        raise ValueError("at least one profile is required")
    return profiles


def format_stats(stats: dict | None) -> str:
    if not stats:
        return "n/a"
    return (
        f"calls={stats['calls']} "
        f"rows={stats['total_rows']} "
        f"pruned={stats['total_pruned_elements']}/{stats['total_elements']} "
        f"({stats['total_pruned_rate']:.2%}) "
        f"block_pruned={stats['block_pruned_blocks']}/{stats['total_blocks']} "
        f"({stats['block_pruned_block_rate']:.2%}) "
        f"padded={stats['padded_elements']}"
    )


def build_agreement_stats(exact_result: MlmResult, result: MlmResult) -> dict:
    if exact_result.masked_tokens != result.masked_tokens:
        raise ValueError("cannot compare prediction agreement with different masked token counts")
    masked_tokens = result.masked_tokens
    if masked_tokens <= 0:
        return {
            "top1_agreement": 0.0,
            "exact_top1_in_hw_top5": 0.0,
            "avg_top5_overlap": 0.0,
        }

    top1_agree = 0
    exact_top1_in_hw_top5 = 0
    top5_overlap_sum = 0.0
    for exact_top1, result_top1, exact_top5, result_top5 in zip(
        exact_result.top1_predictions,
        result.top1_predictions,
        exact_result.top5_predictions,
        result.top5_predictions,
    ):
        top1_agree += int(exact_top1 == result_top1)
        result_top5_set = set(result_top5)
        exact_top1_in_hw_top5 += int(exact_top1 in result_top5_set)
        top5_overlap_sum += len(set(exact_top5) & result_top5_set) / max(1, len(exact_top5))

    return {
        "top1_agreement": top1_agree / masked_tokens,
        "exact_top1_in_hw_top5": exact_top1_in_hw_top5 / masked_tokens,
        "avg_top5_overlap": top5_overlap_sum / masked_tokens,
    }


def print_result(
    result: MlmResult,
    exact_baseline: MlmResult | None = None,
    exact_loss_value: float | None = None,
) -> None:
    print(f"\n[{result.profile}]")
    print(f"mlm_loss      : {result.mlm_loss:.8f}")
    print(f"masked_ppl    : {result.masked_pseudo_perplexity:.6f}")
    print(f"total_loss    : {result.total_loss:.6f}")
    print(f"masked_tokens : {result.masked_tokens}")
    print(f"top1_correct  : {result.top1_correct}/{result.masked_tokens}")
    print(f"top1_accuracy : {result.top1_accuracy:.6f}")
    print(f"top5_correct  : {result.top5_correct}/{result.masked_tokens}")
    print(f"top5_accuracy : {result.top5_accuracy:.6f}")
    print(f"input_tokens  : {result.input_tokens}")
    print(f"windows       : {result.num_windows}")
    print(f"elapsed_sec   : {result.elapsed_sec:.2f}")
    baseline_loss = exact_baseline.mlm_loss if exact_baseline is not None else exact_loss_value
    if baseline_loss is not None and result.profile != "exact":
        exact_ppl = math.exp(baseline_loss)
        print(f"delta_loss    : {result.mlm_loss - baseline_loss:.8f}")
        print(f"ratio_loss    : {result.mlm_loss / baseline_loss if baseline_loss > 0 else float('nan'):.8f}")
        print(f"delta_ppl     : {result.masked_pseudo_perplexity - exact_ppl:.6f}")
        print(f"ratio_ppl     : {result.masked_pseudo_perplexity / exact_ppl if exact_ppl > 0 else float('nan'):.8f}")
    if exact_baseline is not None and result.profile != "exact":
        print(f"delta_top1_acc: {result.top1_accuracy - exact_baseline.top1_accuracy:.6f}")
        print(f"delta_top5_acc: {result.top5_accuracy - exact_baseline.top5_accuracy:.6f}")
        agreement = build_agreement_stats(exact_baseline, result)
        print(f"top1_agreement_vs_exact : {agreement['top1_agreement']:.6f}")
        print(f"exact_top1_in_hw_top5   : {agreement['exact_top1_in_hw_top5']:.6f}")
        print(f"avg_top5_overlap_vs_exact: {agreement['avg_top5_overlap']:.6f}")
    print(f"softmax_stats : {format_stats(result.softmax_stats)}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate Longformer MLM loss with hardware softmax")
    parser.add_argument("--model", type=str, default=default_model_name())
    parser.add_argument(
        "--profile",
        type=str,
        default=DEFAULT_SOFTMAX_PROFILE,
        help="hardware softmax profile; default is RTL-aligned HSSS-Softmax-block8",
    )
    parser.add_argument(
        "--profiles",
        type=str,
        default="",
        help="comma-separated profile list, for example exact,HSSS-Softmax-block8",
    )
    parser.add_argument(
        "--compare",
        action="store_true",
        help="run exact softmax and the selected hardware profile",
    )
    parser.add_argument("--exact", action="store_true", help="run exact softmax only")
    parser.add_argument(
        "--exact-loss",
        type=float,
        default=0.0,
        help="previous exact MLM loss for reporting delta when running hardware-only",
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=4096,
        help="Longformer input length cap; allenai/longformer-base-4096 supports 4096",
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=0,
        help="content-token stride; 0 means non-overlapping windows",
    )
    parser.add_argument(
        "--max-eval-tokens",
        type=int,
        default=8192,
        help="maximum corpus tokens before adding special tokens; 0 evaluates the whole split",
    )
    parser.add_argument("--mlm-probability", type=float, default=0.15, help="fixed mask ratio")
    parser.add_argument("--mask-seed", type=int, default=2026, help="deterministic mask seed")
    parser.add_argument(
        "--global-attention",
        choices=("cls", "none"),
        default="cls",
        help="cls enables one 4K-scale global-attention softmax row per head/layer",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="model device; default cpu, pass cuda or auto explicitly if needed",
    )
    parser.add_argument(
        "--model-dtype",
        type=str,
        default="auto",
        help="model dtype: auto/float16/bfloat16/float32",
    )
    parser.add_argument(
        "--device-map",
        type=str,
        default="",
        help="optional transformers device_map, for example auto",
    )
    parser.add_argument(
        "--torch-num-threads",
        type=int,
        default=0,
        help="PyTorch CPU intra-op threads; 0 keeps PyTorch default",
    )
    parser.add_argument(
        "--torch-inter-op-threads",
        type=int,
        default=0,
        help="PyTorch CPU inter-op threads; 0 keeps PyTorch default",
    )
    parser.add_argument("--local-files-only", action="store_true", help="load model/tokenizer only from local cache")
    parser.add_argument("--trust-remote-code", action="store_true", help="trust Hugging Face remote code")
    parser.add_argument("--dataset", type=str, default="Salesforce/wikitext", help="default text dataset")
    parser.add_argument("--dataset-config", type=str, default="wikitext-2-raw-v1", help="default dataset config")
    parser.add_argument("--dataset-split", type=str, default="test", help="default dataset split")
    parser.add_argument("--text-column", type=str, default="text", help="dataset text column")
    parser.add_argument("--text-file", type=str, default="", help="read a local long-text file instead of a dataset")
    parser.add_argument("--text-encoding", type=str, default="utf-8", help="encoding for --text-file")
    return parser


def run_profile(args: argparse.Namespace, model, windows, device, profile: str) -> MlmResult:
    start = time.perf_counter()
    input_device = resolve_model_input_device(model, device)
    if profile in (None, "", "exact"):
        with torch.inference_mode():
            (
                loss,
                pseudo_ppl,
                total_loss,
                masked_tokens,
                top1_correct,
                top5_correct,
                top1_predictions,
                top5_predictions,
                input_tokens,
            ) = evaluate_mlm_loss(
                model=model,
                windows=windows,
                device=input_device,
            )
        stats = None
    else:
        with patched_softmax(profile, input_format="fp16") as runtime:
            (
                loss,
                pseudo_ppl,
                total_loss,
                masked_tokens,
                top1_correct,
                top5_correct,
                top1_predictions,
                top5_predictions,
                input_tokens,
            ) = evaluate_mlm_loss(
                model=model,
                windows=windows,
                device=input_device,
            )
            stats = runtime.summarize_stats() if runtime is not None else None

    return MlmResult(
        profile=profile,
        mlm_loss=loss,
        masked_pseudo_perplexity=pseudo_ppl,
        total_loss=total_loss,
        masked_tokens=masked_tokens,
        top1_correct=top1_correct,
        top5_correct=top5_correct,
        top1_predictions=top1_predictions,
        top5_predictions=top5_predictions,
        input_tokens=input_tokens,
        num_windows=len(windows),
        elapsed_sec=time.perf_counter() - start,
        softmax_stats=stats,
    )


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    profiles = parse_profiles(args)
    configure_torch_threads(args.torch_num_threads, args.torch_inter_op_threads)

    print(f"Loading {args.model}...")
    model, tokenizer, device, dtype = load_model_and_tokenizer(args)
    print(f"model_device    : {device}")
    print(f"model_dtype     : {dtype}")
    print(f"profiles        : {', '.join(profiles)}")
    print(f"global_attention: {args.global_attention}")

    print("Loading text corpus...")
    text = load_text_corpus(args)
    print("Building fixed-mask MLM windows...")
    windows = build_mlm_windows(
        model=model,
        tokenizer=tokenizer,
        text=text,
        max_length=args.max_length,
        stride=args.stride,
        max_eval_tokens=args.max_eval_tokens,
        mlm_probability=args.mlm_probability,
        mask_seed=args.mask_seed,
        global_attention=args.global_attention,
    )
    print(
        f"windows         : {len(windows)} "
        f"input_tokens={sum(window.content_tokens for window in windows)} "
        f"masked_tokens={sum(window.masked_tokens for window in windows)}",
        flush=True,
    )

    results: List[MlmResult] = []
    exact_loss = args.exact_loss if args.exact_loss > 0 else None
    exact_result = None
    for profile in profiles:
        result = run_profile(args, model, windows, device, profile)
        results.append(result)
        if profile == "exact":
            exact_loss = result.mlm_loss
            exact_result = result
        print_result(result, exact_baseline=exact_result, exact_loss_value=exact_loss)

    if len(results) > 1 and exact_loss is not None:
        print("\nsummary:")
        for result in results:
            if result.profile == "exact":
                continue
            print(
                f"{result.profile:<32} loss={result.mlm_loss:.8f} "
                f"delta_loss={result.mlm_loss - exact_loss:.8f} "
                f"top1={result.top1_accuracy:.6f} "
                f"top5={result.top5_accuracy:.6f} "
                f"masked_ppl={result.masked_pseudo_perplexity:.6f}"
            )


if __name__ == "__main__":
    main()
