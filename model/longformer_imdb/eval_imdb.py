#!/usr/bin/env python3
"""Run Longformer IMDB classification with exact and hardware softmax."""

from __future__ import annotations

import argparse
from pathlib import Path

try:
    from .imdb_eval import (
        DEFAULT_DATASET,
        DEFAULT_DATASET_CONFIG,
        DEFAULT_MODEL,
        DEFAULT_SOFTMAX_PROFILE,
        default_output_path,
        effective_max_length,
        evaluate_profile,
        load_imdb_split,
        load_model_and_tokenizer,
        order_rows_for_batching,
        print_result,
        resolve_device,
        save_results,
        set_torch_threads,
    )
except ImportError:  # pragma: no cover - supports direct script execution
    from imdb_eval import (
        DEFAULT_DATASET,
        DEFAULT_DATASET_CONFIG,
        DEFAULT_MODEL,
        DEFAULT_SOFTMAX_PROFILE,
        default_output_path,
        effective_max_length,
        evaluate_profile,
        load_imdb_split,
        load_model_and_tokenizer,
        order_rows_for_batching,
        print_result,
        resolve_device,
        save_results,
        set_torch_threads,
    )


def parse_profiles(compare: bool, profiles: str, hardware_profile: str) -> list[str]:
    if compare:
        return ["exact", hardware_profile]
    parsed = [profile.strip() for profile in profiles.split(",") if profile.strip()]
    return parsed or ["exact"]


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate Longformer IMDB classification")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL)
    parser.add_argument("--dataset", type=str, default=DEFAULT_DATASET)
    parser.add_argument("--dataset-config", type=str, default=DEFAULT_DATASET_CONFIG)
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--compare", action="store_true", help="run exact and hardware-softmax profiles")
    parser.add_argument("--profiles", type=str, default="exact")
    parser.add_argument("--hardware-profile", type=str, default=DEFAULT_SOFTMAX_PROFILE)
    parser.add_argument("--input-format", type=str, default="fp16")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--model-dtype", type=str, default="auto")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument(
        "--batch-order",
        choices=("original", "length_asc", "length_desc"),
        default="original",
        help="length ordering can accelerate batch_size > 1 by reducing padding",
    )
    parser.add_argument("--max-length", type=int, default=4096)
    parser.add_argument("--max-samples", type=int, default=0, help="0 means full split")
    parser.add_argument(
        "--sample-mode",
        choices=("balanced", "shuffle", "first"),
        default="balanced",
        help="sampling strategy when --max-samples is set",
    )
    parser.add_argument("--sample-seed", type=int, default=2026)
    parser.add_argument(
        "--global-attention",
        choices=("auto", "cls", "none"),
        default="cls",
        help="cls is the standard sequence-classification setting for Longformer",
    )
    parser.add_argument("--torch-num-threads", type=int, default=12)
    parser.add_argument("--torch-inter-op-threads", type=int, default=2)
    parser.add_argument("--output", type=str, default="")
    args = parser.parse_args()

    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive")
    if args.max_length <= 0:
        raise ValueError("--max-length must be positive")

    set_torch_threads(args.torch_num_threads, args.torch_inter_op_threads)
    device = resolve_device(args.device)
    model, tokenizer = load_model_and_tokenizer(
        args.model,
        local_files_only=args.local_files_only,
        model_dtype=args.model_dtype,
    )
    print(f"model_device    : {device}")
    print(f"model_dtype     : {next(model.parameters()).dtype}")
    print(f"profiles        : {', '.join(parse_profiles(args.compare, args.profiles, args.hardware_profile))}")
    print(f"global_attention: {args.global_attention}")
    max_length = effective_max_length(model, args.max_length)
    print(f"requested_length: {args.max_length}")
    print(f"effective_length: {max_length}")
    print("Loading IMDB corpus...")
    rows = load_imdb_split(
        args.dataset,
        args.dataset_config,
        args.split,
        max_samples=args.max_samples,
        local_files_only=args.local_files_only,
        sample_mode=args.sample_mode,
        sample_seed=args.sample_seed,
    )
    rows = order_rows_for_batching(rows, tokenizer, max_length=max_length, batch_order=args.batch_order)
    print(f"examples        : {len(rows)} split={args.split} max_length={max_length} batch_size={args.batch_size}")

    results = []
    exact_result = None
    for profile in parse_profiles(args.compare, args.profiles, args.hardware_profile):
        result = evaluate_profile(
            model,
            tokenizer,
            rows,
            profile=profile,
            device=device,
            batch_size=args.batch_size,
            max_length=max_length,
            global_attention=args.global_attention,
            input_format=args.input_format,
        )
        print_result(result, exact_result=exact_result)
        if profile == "exact":
            exact_result = result
        results.append(result)

    print("\nsummary:")
    for result in results:
        line = f"{result.profile:<32} acc={result.accuracy:.6f} macro_f1={result.macro_f1:.6f}"
        if exact_result is not None and result.profile != "exact":
            agreement = sum(
                1 for exact_pred, pred in zip(exact_result.predictions, result.predictions) if exact_pred == pred
            ) / exact_result.examples
            line += f" agreement={agreement:.6f}"
        print(line)

    output_path = Path(args.output) if args.output else default_output_path(prefix=f"imdb_{args.split}")
    save_results(output_path, vars(args), results)


if __name__ == "__main__":
    main()
