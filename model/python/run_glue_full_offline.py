#!/usr/bin/env python3
"""Run GLUE evaluation for exact vs approximate softmax profiles."""

from __future__ import annotations

import argparse
import os
import shlex
import subprocess
import sys
from datetime import datetime
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]
EVAL_SCRIPT = SCRIPT_DIR / "eval_bert_softmax_tasks.py"
RESULTS_DIR = SCRIPT_DIR / "results"
DEFAULT_GLUE_TASKS = "cola,mrpc,qnli,qqp,rte,sst2,stsb,mnli,wnli"
DEFAULT_APPROX_PROFILE_BY_BLOCK_SIZE = {
    8: "doc_adaptive_desc9_q7_special4",
    4: "doc_adaptive_desc9_q7_special4_block4",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the full GLUE evaluation with conservative parallelism")
    parser.add_argument("--tasks", type=str, default=DEFAULT_GLUE_TASKS, help="GLUE task subset; default is the full GLUE set")
    parser.add_argument("--profiles", type=str, default="", help="Softmax profiles; overrides --block-size when set")
    parser.add_argument("--block-size", type=int, choices=(4, 8), default=8, help="Select the default approximate profile by block size")
    parser.add_argument("--max-samples", type=int, default=0, help="0 means full validation split")
    parser.add_argument("--batch-size", type=int, default=8, help="Inference batch size")
    parser.add_argument("--max-length", type=int, default=128, help="Tokenizer max length")
    parser.add_argument("--input-format", choices=("fp16", "bf16"), default="fp16")
    parser.add_argument("--device", choices=("cpu",), default="cpu")
    parser.add_argument("--task-workers", type=int, default=1, help="Manual task-level parallelism")
    parser.add_argument("--run-workers", type=int, default=1, help="Manual task x profile parallelism")
    parser.add_argument("--disable-auto-workers", action="store_true", help="Disable conservative auto parallelism")
    parser.add_argument("--reserve-cpus", type=int, default=2, help="CPUs reserved for the system under auto parallelism")
    parser.add_argument("--max-auto-workers", type=int, default=3, help="Upper bound for auto-selected workers; 0 means uncapped")
    parser.add_argument("--collect-softmax-stats", action="store_true", help="Collect per-softmax stats; much slower")
    parser.add_argument("--scan-diff-only", action="store_true", help="Only scan prediction changes, skip task metrics")
    parser.add_argument("--progress-interval-sec", type=float, default=30.0, help="Progress print interval in seconds")
    parser.add_argument("--output-json", type=str, help="Optional output JSON path")
    parser.add_argument("--log-file", type=str, help="Optional log file path")
    parser.add_argument("--background", action="store_true", help="Launch in background and return immediately")
    return parser.parse_args()


def build_default_paths(scan_diff_only: bool) -> tuple[Path, Path]:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    stem = "glue_full_diff_scan" if scan_diff_only else "glue_full_eval"
    return (
        RESULTS_DIR / f"{stem}_{timestamp}.json",
        RESULTS_DIR / f"{stem}_{timestamp}.log",
    )


def resolve_profiles(args: argparse.Namespace) -> str:
    if args.profiles.strip():
        return args.profiles
    return f"exact,{DEFAULT_APPROX_PROFILE_BY_BLOCK_SIZE[args.block_size]}"


def build_command(args: argparse.Namespace, output_json: Path) -> list[str]:
    profiles = resolve_profiles(args)
    command = [
        sys.executable,
        str(EVAL_SCRIPT),
        "--tasks",
        args.tasks,
        "--profiles",
        profiles,
        "--max-samples",
        str(args.max_samples),
        "--batch-size",
        str(args.batch_size),
        "--max-length",
        str(args.max_length),
        "--device",
        args.device,
        "--input-format",
        args.input_format,
        "--output-json",
        str(output_json),
        "--progress-interval-sec",
        str(args.progress_interval_sec),
    ]
    if args.task_workers > 1:
        command.extend(["--task-workers", str(args.task_workers)])
    if args.run_workers > 1:
        command.extend(["--run-workers", str(args.run_workers)])
    if not args.disable_auto_workers and args.task_workers <= 1 and args.run_workers <= 1:
        command.append("--auto-workers")
        command.extend(["--reserve-cpus", str(args.reserve_cpus)])
        command.extend(["--max-auto-workers", str(args.max_auto_workers)])
    if args.collect_softmax_stats:
        command.append("--collect-softmax-stats")
    if args.scan_diff_only:
        command.append("--scan-diff-only")
    return command


def main() -> int:
    args = parse_args()
    output_json_default, log_default = build_default_paths(args.scan_diff_only)
    output_json = Path(args.output_json) if args.output_json else output_json_default
    log_file = Path(args.log_file) if args.log_file else log_default
    output_json.parent.mkdir(parents=True, exist_ok=True)
    log_file.parent.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    env.setdefault("PYTHONUNBUFFERED", "1")
    profiles = resolve_profiles(args)
    command = build_command(args, output_json)
    command_text = shlex.join(command)

    print(f"tasks             : {args.tasks}")
    print(f"profiles          : {profiles}")
    print(f"max_samples       : {args.max_samples} (0 means full dataset)")
    print(f"output_json       : {output_json}")
    print(f"log_file          : {log_file}")
    print(f"progress_interval : {args.progress_interval_sec}s")
    print(f"command           : {command_text}")

    if args.background:
        with log_file.open("w", encoding="utf-8") as handle:
            process = subprocess.Popen(
                command,
                cwd=str(PROJECT_ROOT),
                env=env,
                stdout=handle,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
        print(f"background_pid    : {process.pid}")
        print(f"monitor           : tail -f {log_file}")
        print("status            : launched")
        return 0

    with log_file.open("w", encoding="utf-8") as handle:
        process = subprocess.run(
            command,
            cwd=str(PROJECT_ROOT),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
        handle.write(process.stdout)
        print(process.stdout, end="")
    return process.returncode


if __name__ == "__main__":
    raise SystemExit(main())
