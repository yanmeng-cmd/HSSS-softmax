#!/usr/bin/env python3
"""Compact Makefile runner for BERT and Longformer HSSS softmax evaluations."""

from __future__ import annotations

import json
import os
import re
import selectors
import shutil
import socket
import subprocess
import sys
import tarfile
import time
import urllib.parse
from datetime import datetime
from pathlib import Path
from typing import Iterable


try:
    sys.stdout.reconfigure(line_buffering=True)
    sys.stderr.reconfigure(line_buffering=True)
except AttributeError:
    pass

ROOT = Path(__file__).resolve().parent
RESULTS_DIR = ROOT / "results"
LOG_DIR = RESULTS_DIR / "logs"
VENV_DIR = ROOT / ".venv"
VENV_PYTHON = VENV_DIR / "bin" / "python"
REQUIREMENTS = ROOT / "requirements.txt"
PROFILE = "HSSS-Softmax-block8"
BLOCK4_PROFILE = "HSSS-Softmax-block4"
PROFILE_ALIASES = {
    PROFILE.lower(): PROFILE,
    "hsss-softmax": PROFILE,
    "hsss": PROFILE,
    BLOCK4_PROFILE.lower(): BLOCK4_PROFILE,
}
GLUE_TASKS = "cola,mrpc,qnli,qqp,rte,sst2,stsb,mnli,wnli"
CONTEXTS = {
    "1k": 1024,
    "1024": 1024,
    "2k": 2048,
    "2048": 2048,
    "4k": 4096,
    "4096": 4096,
}
REQUIRED_IMPORTS = {
    "torch": "torch",
    "transformers": "transformers",
    "datasets": "datasets",
    "evaluate": "evaluate",
    "huggingface_hub": "huggingface_hub",
    "safetensors": "safetensors",
    "numpy": "numpy",
    "pyarrow": "pyarrow",
    "tqdm": "tqdm",
}
OFFLINE_ENV_KEYS = ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "HF_DATASETS_OFFLINE")
PROXY_ENV_KEYS = ("ALL_PROXY", "all_proxy", "HTTPS_PROXY", "https_proxy", "HTTP_PROXY", "http_proxy")


def env_flag(name: str, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.lower() not in {"0", "false", "no", "off", ""}


def env_value(name: str, default: str) -> str:
    value = os.environ.get(name)
    return default if value is None or value == "" else value


def timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def ensure_dirs() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)


def command_env(force_online: bool = False, offline_default: bool = True) -> dict[str, str]:
    env = os.environ.copy()
    sanitized_proxy_vars = sanitize_proxy_env(env)
    if sanitized_proxy_vars:
        env["HSSS_SANITIZED_PROXY_VARS"] = ",".join(sanitized_proxy_vars)
    env["PYTHONUNBUFFERED"] = "1"
    env.setdefault("TOKENIZERS_PARALLELISM", "false")
    pythonpath = str(ROOT)
    if env.get("PYTHONPATH"):
        pythonpath = pythonpath + os.pathsep + env["PYTHONPATH"]
    env["PYTHONPATH"] = pythonpath
    if force_online:
        for key in OFFLINE_ENV_KEYS:
            env.pop(key, None)
    elif env_flag("OFFLINE", offline_default):
        env.setdefault("HF_HUB_OFFLINE", "1")
        env.setdefault("TRANSFORMERS_OFFLINE", "1")
        env.setdefault("HF_DATASETS_OFFLINE", "1")
    return env


def sanitize_proxy_env(env: dict[str, str]) -> list[str]:
    sanitized: list[str] = []
    for key in PROXY_ENV_KEYS:
        value = env.get(key, "").strip()
        if should_remove_proxy(value):
            env.pop(key, None)
            sanitized.append(key)
    return sanitized


def should_remove_proxy(value: str) -> bool:
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


def print_proxy_notice(env: dict[str, str]) -> None:
    sanitized = env.get("HSSS_SANITIZED_PROXY_VARS")
    if sanitized and not env_flag("QUIET_PROXY_NOTICE", False):
        print(f"proxy   : ignored stale/unsupported proxy variables for child process: {sanitized}")


def eval_python() -> str:
    override = os.environ.get("EVAL_PYTHON")
    if override:
        return override
    if VENV_PYTHON.exists():
        return str(VENV_PYTHON)
    return sys.executable


def bootstrap_python() -> str:
    return env_value("BOOTSTRAP_PYTHON", sys.executable)


def shlex_join(command: Iterable[str]) -> str:
    import shlex

    return shlex.join(list(command))


def resolve_profile(profile: str) -> str:
    clean = profile.strip()
    return PROFILE_ALIASES.get(clean.lower(), clean)


def compact_profile(profile: str) -> str:
    return resolve_profile(profile)


def public_output_text(text: str) -> str:
    return (
        text.replace("doc_adaptive_desc9_q7_special4_block4", BLOCK4_PROFILE)
        .replace("doc_adaptive_desc9_q7_special4", PROFILE)
    )


def fmt_percent(value: float | None) -> str:
    if value is None:
        return "-"
    return f"{100.0 * value:.2f}%"


def print_table(headers: list[str], rows: list[list[str]]) -> None:
    widths = [len(header) for header in headers]
    for row in rows:
        for index, cell in enumerate(row):
            widths[index] = max(widths[index], len(cell))
    fmt = "  ".join("{:<" + str(width) + "}" for width in widths)
    print(fmt.format(*headers))
    print(fmt.format(*["-" * width for width in widths]))
    for row in rows:
        print(fmt.format(*row))


def missing_imports(python: str) -> list[str]:
    command = [
        python,
        "-c",
        (
            "import importlib.util, sys\n"
            "missing = [name for name in sys.argv[1:] if importlib.util.find_spec(name) is None]\n"
            "print('\\n'.join(missing))\n"
            "raise SystemExit(1 if missing else 0)\n"
        ),
        *REQUIRED_IMPORTS.values(),
    ]
    completed = subprocess.run(command, text=True, capture_output=True, check=False)
    if completed.returncode == 0:
        return []
    missing = [line.strip() for line in completed.stdout.splitlines() if line.strip()]
    return missing or ["dependency check failed"]


def env_command(target: str) -> str:
    return {
        "bert": "make bert env",
        "longformer wiki": "make longformer wiki env",
        "longformer imdb": "make longformer imdb env",
        "longformer": "make longformer wiki env  # or: make longformer imdb env",
    }.get(target, "make env")


def next_hint(target: str) -> str | None:
    return {
        "bert": "make bert glue  # or: make bert squad",
        "longformer wiki": "make longformer wiki 2k",
        "longformer imdb": "make longformer imdb 2k",
        "longformer": "make longformer wiki 2k  # or: make longformer imdb 2k",
    }.get(target)


def asset_commands(target: str, python: str | None = None) -> list[list[str]]:
    runner = python or eval_python()
    if target == "longformer wiki":
        return [[runner, str(ROOT / "longformer_wiki" / "download_assets.py")]]
    if target == "longformer imdb":
        return [[runner, str(ROOT / "longformer_imdb" / "download_assets.py")]]
    return []


def print_env_not_ready(target: str, reason: str, missing: list[str] | None = None) -> None:
    print(f"environment: not ready for {target}")
    print(f"reason     : {reason}")
    if missing:
        print(f"missing    : {', '.join(missing)}")
    print(f"run        : {env_command(target)}")
    if asset_commands(target):
        print("assets     : included by the env target by default; use DOWNLOAD_ASSETS=0 to skip")
    followup = next_hint(target)
    if followup:
        print(f"then       : {followup}")


def ensure_environment_ready(target: str) -> None:
    if env_flag("DRY_RUN", False) or env_flag("SKIP_ENV_CHECK", False):
        return
    if not VENV_PYTHON.exists():
        print_env_not_ready(target, f"missing virtualenv at {VENV_DIR}")
        raise SystemExit(2)
    missing = missing_imports(str(VENV_PYTHON))
    if missing:
        print_env_not_ready(target, f"incomplete virtualenv at {VENV_DIR}", missing)
        raise SystemExit(2)


def run_setup_step(command: list[str], force_online: bool = False) -> None:
    env = command_env(force_online=force_online)
    print(f"command : {shlex_join(command)}")
    print_proxy_notice(env)
    if env_flag("DRY_RUN", False):
        print("status  : dry-run")
        return
    completed = subprocess.run(
        command,
        cwd=str(ROOT),
        env=env,
        text=True,
        check=False,
    )
    if completed.returncode == 0:
        return
    print(f"status  : failed ({completed.returncode})")
    if force_online:
        print_hf_network_hint()
    raise SystemExit(completed.returncode)


def print_hf_network_hint() -> None:
    print("hint    : this usually means the server cannot reach Hugging Face, not that login is required")
    print("proxy   : set HTTPS_PROXY/HTTP_PROXY if this server must use a proxy")
    print("mirror  : if the official Hub is unreachable in your network, retry with HF_ENDPOINT=https://hf-mirror.com")
    print("cache   : if assets were downloaded elsewhere, copy the corresponding .hf-cache directory before rerunning")


def setup_environment(target: str) -> None:
    ensure_dirs()
    print(f"target  : {target} env")
    print(f"venv    : {VENV_DIR}")
    print(f"python  : {VENV_PYTHON}")
    if not VENV_PYTHON.exists():
        run_setup_step([bootstrap_python(), "-m", "venv", str(VENV_DIR)])
    run_setup_step([str(VENV_PYTHON), "-m", "pip", "install", "--upgrade", "pip"])
    run_setup_step([str(VENV_PYTHON), "-m", "pip", "install", "-r", str(REQUIREMENTS)])
    if not env_flag("DRY_RUN", False):
        missing = missing_imports(str(VENV_PYTHON))
        if missing:
            print_env_not_ready(target, f"dependency install did not complete in {VENV_DIR}", missing)
            raise SystemExit(2)
    commands = asset_commands(target, python=str(VENV_PYTHON))
    if commands and env_flag("DOWNLOAD_ASSETS", True):
        print("assets  : downloading task assets; set DOWNLOAD_ASSETS=0 to skip")
        for command in commands:
            run_setup_step(command, force_online=True)
    if env_flag("DRY_RUN", False):
        return
    print("status  : ready")
    followup = next_hint(target)
    if followup:
        print(f"next    : {followup}")


def run_logged(command: list[str], log_path: Path, offline_default: bool = True) -> int:
    ensure_dirs()
    log_path.parent.mkdir(parents=True, exist_ok=True)
    env = command_env(offline_default=offline_default)

    print(f"command : {shlex_join(command)}")
    print(f"log     : {log_path}")
    print_proxy_notice(env)
    if env_flag("DRY_RUN", False):
        print("status  : dry-run")
        return 0

    stream_logs = env_flag("STREAM_LOGS", True) or env_flag("VERBOSE", False)
    heartbeat_sec = float(env_value("HEARTBEAT_SEC", "30"))
    if stream_logs:
        print("status  : running; output is shown below and saved to the log")
        with log_path.open("w", encoding="utf-8") as handle:
            process = subprocess.Popen(
                command,
                cwd=str(ROOT),
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )
            assert process.stdout is not None
            selector = selectors.DefaultSelector()
            selector.register(process.stdout, selectors.EVENT_READ)
            start = time.monotonic()
            next_heartbeat = start + heartbeat_sec
            while True:
                for key, _ in selector.select(timeout=1.0):
                    line = key.fileobj.readline()
                    if line:
                        line = public_output_text(line)
                        handle.write(line)
                        handle.flush()
                        print(line, end="")
                    else:
                        selector.unregister(key.fileobj)
                returncode = process.poll()
                if returncode is not None and not selector.get_map():
                    return returncode
                if heartbeat_sec > 0:
                    now = time.monotonic()
                    if now >= next_heartbeat:
                        elapsed = int(now - start)
                        print(f"status  : still running ({elapsed}s); log={log_path}")
                        next_heartbeat = now + heartbeat_sec

    print("status  : running; detailed output is being written to the log")
    with log_path.open("w", encoding="utf-8") as handle:
        process = subprocess.Popen(
            command,
            cwd=str(ROOT),
            env=env,
            stdout=handle,
            stderr=subprocess.STDOUT,
            text=True,
        )
        start = time.monotonic()
        next_heartbeat = start + heartbeat_sec
        while True:
            returncode = process.poll()
            if returncode is not None:
                return returncode
            if heartbeat_sec > 0:
                now = time.monotonic()
                if now >= next_heartbeat:
                    elapsed = int(now - start)
                    print(f"status  : still running ({elapsed}s); log={log_path}")
                    next_heartbeat = now + heartbeat_sec
            time.sleep(1.0)


def print_log_tail(log_path: Path, lines: int = 80) -> None:
    if not log_path.exists():
        return
    content = log_path.read_text(encoding="utf-8", errors="replace").splitlines()
    print()
    print(f"last {min(lines, len(content))} log lines:")
    for line in content[-lines:]:
        print(line)


def fail_if_needed(returncode: int, log_path: Path) -> None:
    if returncode == 0:
        return
    print(f"status  : failed ({returncode})")
    print_log_tail(log_path)
    raise SystemExit(returncode)


def parse_context(args: list[str]) -> tuple[str, int]:
    token = args[0].lower() if args else env_value("CONTEXT", "4k").lower()
    if token not in CONTEXTS:
        raise SystemExit(f"unknown context {token!r}; use 1k, 2k, or 4k")
    return token, CONTEXTS[token]


def parse_pruning_percent(stats_line: str) -> float | None:
    match = re.search(r"\((\d+(?:\.\d+)?)%\)", stats_line)
    if not match:
        return None
    return float(match.group(1)) / 100.0


def parse_wiki_log(log_path: Path, approx_profile: str) -> dict[str, dict]:
    blocks: dict[str, dict] = {}
    current: str | None = None
    for line in log_path.read_text(encoding="utf-8", errors="replace").splitlines():
        stripped = line.strip()
        if stripped.startswith("[") and stripped.endswith("]"):
            current = stripped.strip("[]")
            blocks.setdefault(current, {})
            continue
        if current is None or ":" not in stripped:
            continue
        key, value = [part.strip() for part in stripped.split(":", 1)]
        if key in {"mlm_loss", "masked_ppl", "top1_accuracy", "top5_accuracy"}:
            try:
                blocks[current][key] = float(value)
            except ValueError:
                pass
        elif key == "softmax_stats":
            blocks[current]["pruning"] = parse_pruning_percent(value)
    return {"exact": blocks.get("exact", {}), "approx": blocks.get(approx_profile, {})}


def print_wiki_summary(context_label: str, parsed: dict[str, dict], max_eval_tokens: str) -> None:
    exact = parsed["exact"]
    approx = parsed["approx"]
    print()
    print("Longformer WikiText-2 MLM summary")
    print_table(
        ["Context", "Tokens", "Metric", "Exact", "Approx.", "Pruning"],
        [
            [
                context_label.upper(),
                max_eval_tokens,
                "Top-1",
                fmt_percent(exact.get("top1_accuracy")),
                fmt_percent(approx.get("top1_accuracy")),
                fmt_percent(approx.get("pruning")),
            ],
            [
                context_label.upper(),
                max_eval_tokens,
                "Top-5",
                fmt_percent(exact.get("top5_accuracy")),
                fmt_percent(approx.get("top5_accuracy")),
                fmt_percent(approx.get("pruning")),
            ],
        ],
    )


def run_longformer_wiki(args: list[str]) -> None:
    if args and args[0].lower() == "env":
        setup_environment("longformer wiki")
        return
    ensure_environment_ready("longformer wiki")
    context_label, context = parse_context(args)
    max_eval_tokens = env_value("MAX_EVAL_TOKENS", "0")
    profile = resolve_profile(env_value("PROFILE", PROFILE))
    stamp = timestamp()
    log_path = LOG_DIR / f"longformer_wiki_{context_label}_{stamp}.log"
    command = [
        eval_python(),
        str(ROOT / "longformer_wiki" / "eval_mlm.py"),
        "--compare",
        "--device",
        env_value("DEVICE", "cpu"),
        "--model-dtype",
        env_value("MODEL_DTYPE", "auto"),
        "--torch-num-threads",
        env_value("TORCH_THREADS", "0"),
        "--torch-inter-op-threads",
        env_value("TORCH_INTEROP_THREADS", "0"),
        "--max-length",
        str(context),
        "--stride",
        env_value("STRIDE", "0"),
        "--max-eval-tokens",
        max_eval_tokens,
        "--global-attention",
        env_value("GLOBAL_ATTENTION", "none"),
        "--profile",
        profile,
    ]
    if env_flag("LOCAL_FILES_ONLY", True):
        command.append("--local-files-only")

    print(f"target  : longformer wiki {context_label}")
    print(f"profile : {compact_profile(profile)}")
    print(f"tokens  : {max_eval_tokens} (0 means full split)")
    returncode = run_logged(command, log_path)
    fail_if_needed(returncode, log_path)
    if env_flag("DRY_RUN", False):
        return
    parsed = parse_wiki_log(log_path, profile)
    print_wiki_summary(context_label, parsed, max_eval_tokens)


def print_imdb_summary(context_label: str, result_path: Path) -> None:
    payload = json.loads(result_path.read_text(encoding="utf-8"))
    results = {item["profile"]: item for item in payload["results"]}
    exact = results.get("exact", {})
    approx_profiles = [profile for profile in results if profile != "exact"]
    rows: list[list[str]] = []
    for profile in sorted(approx_profiles):
        approx = results[profile]
        agreement = approx.get("agreement_vs_exact")
        softmax = approx.get("softmax_stats") or {}
        rows.append(
            [
                context_label.upper(),
                str(approx.get("examples", exact.get("examples", "-"))),
                fmt_percent(exact.get("accuracy")),
                fmt_percent(approx.get("accuracy")),
                fmt_percent(exact.get("macro_f1")),
                fmt_percent(approx.get("macro_f1")),
                fmt_percent(agreement),
                fmt_percent(softmax.get("total_pruned_rate")),
                compact_profile(profile),
            ]
        )

    print()
    print("Longformer IMDB summary")
    print_table(
        ["Context", "Samples", "Exact Acc.", "Approx. Acc.", "Exact F1", "Approx. F1", "Agreement", "Pruning", "Profile"],
        rows,
    )


def default_imdb_samples(context: int) -> str:
    if context in {1024, 2048}:
        return "5000"
    return "0"


def run_longformer_imdb(args: list[str]) -> None:
    if args and args[0].lower() == "env":
        setup_environment("longformer imdb")
        return
    ensure_environment_ready("longformer imdb")
    context_label, context = parse_context(args)
    profile = resolve_profile(env_value("PROFILE", PROFILE))
    max_samples = env_value("MAX_SAMPLES", default_imdb_samples(context))
    stamp = timestamp()
    result_path = RESULTS_DIR / f"longformer_imdb_{context_label}_{stamp}.json"
    log_path = LOG_DIR / f"longformer_imdb_{context_label}_{stamp}.log"
    command = [
        eval_python(),
        str(ROOT / "longformer_imdb" / "eval_imdb.py"),
        "--compare",
        "--device",
        env_value("DEVICE", "cpu"),
        "--model-dtype",
        env_value("MODEL_DTYPE", "auto"),
        "--torch-num-threads",
        env_value("TORCH_THREADS", "12"),
        "--torch-inter-op-threads",
        env_value("TORCH_INTEROP_THREADS", "2"),
        "--max-length",
        str(context),
        "--max-samples",
        max_samples,
        "--sample-mode",
        env_value("SAMPLE_MODE", "balanced"),
        "--sample-seed",
        env_value("SAMPLE_SEED", "2026"),
        "--batch-size",
        env_value("BATCH_SIZE", "1"),
        "--batch-order",
        env_value("BATCH_ORDER", "original"),
        "--global-attention",
        env_value("GLOBAL_ATTENTION", "cls"),
        "--hardware-profile",
        profile,
        "--input-format",
        env_value("INPUT_FORMAT", "fp16"),
        "--output",
        str(result_path),
    ]
    if env_flag("LOCAL_FILES_ONLY", True):
        command.append("--local-files-only")

    print(f"target  : longformer imdb {context_label}")
    print(f"profile : {compact_profile(profile)}")
    print(f"samples : {max_samples} (0 means full split)")
    returncode = run_logged(command, log_path)
    fail_if_needed(returncode, log_path)
    if env_flag("DRY_RUN", False):
        return
    print(f"results : {result_path}")
    print_imdb_summary(context_label, result_path)


def run_longformer(args: list[str]) -> None:
    if not args:
        raise SystemExit("usage: make longformer wiki|imdb 1k|2k|4k")
    task = args[0].lower()
    rest = args[1:]
    if task == "env":
        setup_environment("longformer")
        return
    if task in {"wiki", "wikitext", "mlm"}:
        run_longformer_wiki(rest)
        return
    if task in {"imdb", "classification", "cls"}:
        run_longformer_imdb(rest)
        return
    raise SystemExit(f"unknown longformer task {task!r}; use wiki or imdb")


def bert_block_size() -> str:
    explicit = os.environ.get("BLOCK_SIZE")
    if explicit:
        if explicit not in {"4", "8"}:
            raise SystemExit("BLOCK_SIZE must be 4 or 8")
        return explicit

    profile = resolve_profile(env_value("PROFILE", PROFILE))
    if profile == BLOCK4_PROFILE:
        return "4"
    if profile == PROFILE:
        return "8"
    raise SystemExit("BERT wrapper supports PROFILE=HSSS-Softmax-block8 or HSSS-Softmax-block4")


def primary_bert_metric(task: str, metrics: dict) -> tuple[str, float | None]:
    if task == "cola":
        return "MCC", metrics.get("matthews_correlation")
    if task == "stsb":
        return "Pearson", metrics.get("pearson")
    if task in {"mrpc", "qqp"} and "f1" in metrics:
        return "F1", metrics.get("f1")
    if task in {"squad", "squad_v2"}:
        return "F1", metrics.get("f1")
    if "accuracy" in metrics:
        return "Acc.", metrics.get("accuracy")
    if metrics:
        key = sorted(metrics)[0]
        return key, metrics.get(key)
    return "-", None


def fmt_bert_metric(task: str, metric_name: str, value: float | None) -> str:
    if value is None:
        return "-"
    if task in {"squad", "squad_v2"}:
        return f"{value:.2f}"
    if metric_name in {"MCC", "Pearson", "Spearman"}:
        return f"{100.0 * value:.2f}"
    return f"{100.0 * value:.2f}%"


def print_bert_summary(result_path: Path) -> None:
    payload = json.loads(result_path.read_text(encoding="utf-8"))
    grouped: dict[str, dict[str, dict]] = {}
    for item in payload:
        grouped.setdefault(item["task"], {})[item["profile"]] = item

    rows: list[list[str]] = []
    for task in sorted(grouped):
        profiles = grouped[task]
        exact = profiles.get("exact")
        approx_profiles = [profile for profile in profiles if profile != "exact"]
        for profile in sorted(approx_profiles):
            approx = profiles[profile]
            metric_name, exact_value = primary_bert_metric(task, exact["metrics"] if exact else {})
            _, approx_value = primary_bert_metric(task, approx["metrics"])
            softmax = approx.get("softmax_metrics") or {}
            rows.append(
                [
                    task,
                    metric_name,
                    str(approx.get("num_samples", exact.get("num_samples") if exact else "-")),
                    fmt_bert_metric(task, metric_name, exact_value),
                    fmt_bert_metric(task, metric_name, approx_value),
                    fmt_percent(softmax.get("total_pruned_rate")),
                ]
            )

    print()
    print("BERT summary")
    print_table(["Task", "Metric", "Samples", "Exact", "Approx.", "Pruning"], rows)


def append_common_bert_args(command: list[str]) -> None:
    command.extend(
        [
            "--block-size",
            bert_block_size(),
            "--max-samples",
            env_value("MAX_SAMPLES", "0"),
            "--batch-size",
            env_value("BATCH_SIZE", "8"),
            "--max-length",
            env_value("MAX_LENGTH", "128"),
            "--device",
            env_value("DEVICE", "cpu"),
            "--input-format",
            env_value("INPUT_FORMAT", "fp16"),
            "--task-workers",
            env_value("TASK_WORKERS", "1"),
            "--run-workers",
            env_value("RUN_WORKERS", "1"),
            "--reserve-cpus",
            env_value("RESERVE_CPUS", "2"),
            "--max-auto-workers",
            env_value("MAX_AUTO_WORKERS", "3"),
            "--progress-interval-sec",
            env_value("PROGRESS_INTERVAL", "30"),
        ]
    )
    if not env_flag("AUTO_WORKERS", True):
        command.append("--disable-auto-workers")
    if env_flag("COLLECT_SOFTMAX_STATS", False):
        command.append("--collect-softmax-stats")


def run_bert(args: list[str]) -> None:
    mode = args[0].lower() if args else "glue"
    if mode == "env":
        setup_environment("bert")
        return
    ensure_environment_ready("bert")

    stamp = timestamp()
    result_path = RESULTS_DIR / f"bert_{mode}_{stamp}.json"
    log_path = LOG_DIR / f"bert_{mode}_{stamp}.log"
    script_log_path = LOG_DIR / f"bert_{mode}_{stamp}_eval.log"

    if mode == "glue":
        tasks = env_value("TASKS", GLUE_TASKS)
        command = [
            eval_python(),
            str(ROOT / "python" / "run_glue_full_offline.py"),
            "--tasks",
            tasks,
            "--output-json",
            str(result_path),
            "--log-file",
            str(script_log_path),
        ]
    elif mode == "squad":
        tasks = env_value("TASKS", "squad")
        command = [
            eval_python(),
            str(ROOT / "python" / "run_squad_offline.py"),
            "--tasks",
            tasks,
            "--qa-max-length",
            env_value("QA_MAX_LENGTH", "384"),
            "--qa-doc-stride",
            env_value("QA_DOC_STRIDE", "128"),
            "--qa-n-best-size",
            env_value("QA_N_BEST_SIZE", "20"),
            "--qa-max-answer-length",
            env_value("QA_MAX_ANSWER_LENGTH", "30"),
            "--torch-threads",
            env_value("TORCH_THREADS", "0"),
            "--torch-interop-threads",
            env_value("TORCH_INTEROP_THREADS", "1"),
            "--output-json",
            str(result_path),
            "--log-file",
            str(script_log_path),
        ]
    else:
        raise SystemExit("usage: make bert env|glue|squad")

    append_common_bert_args(command)
    print(f"target  : bert {mode}")
    print(f"tasks   : {tasks}")
    print(f"profile : {BLOCK4_PROFILE if bert_block_size() == '4' else PROFILE}")
    print(f"samples : {env_value('MAX_SAMPLES', '0')} (0 means full split)")
    returncode = run_logged(command, log_path, offline_default=False)
    fail_if_needed(returncode, log_path)
    if env_flag("DRY_RUN", False):
        return
    print(f"results : {result_path}")
    print(f"log     : {script_log_path}")
    print_bert_summary(result_path)


def run_download(args: list[str]) -> None:
    if not args:
        raise SystemExit("usage: make download wiki|imdb")
    target = args[0].lower()
    if target in {"wiki", "wikitext"}:
        ensure_environment_ready("longformer wiki")
        commands = asset_commands("longformer wiki")
    elif target == "imdb":
        ensure_environment_ready("longformer imdb")
        commands = asset_commands("longformer imdb")
    else:
        raise SystemExit(f"unknown download target {target!r}")

    for index, command in enumerate(commands, start=1):
        log_path = LOG_DIR / f"download_{target}_{index}_{timestamp()}.log"
        print(f"download: {target} step {index}/{len(commands)}")
        returncode = run_logged(command, log_path)
        fail_if_needed(returncode, log_path)
        if env_flag("DRY_RUN", False):
            continue


def package_model() -> None:
    ensure_dirs()
    package_name = ROOT.name
    output = ROOT.parent / f"{package_name}_package_{timestamp()}.tar.gz"
    exclude_parts = {
        ".hf-cache",
        ".venv",
        "__pycache__",
        ".pytest_cache",
    }

    def should_skip(path: Path) -> bool:
        rel = path.relative_to(ROOT)
        if any(part in exclude_parts for part in rel.parts):
            return True
        if rel.parts and rel.parts[0] == "results" and rel.name != ".gitkeep":
            return True
        if path.suffix in {".log", ".pyc", ".tar", ".gz"}:
            return True
        return False

    with tarfile.open(output, "w:gz") as archive:
        for path in sorted(ROOT.rglob("*")):
            if should_skip(path):
                continue
            archive.add(path, arcname=Path(package_name) / path.relative_to(ROOT), recursive=False)
    print(f"package : {output}")


def print_help() -> None:
    print(
        """Usage:
  make bert env
  make bert glue [MAX_SAMPLES=0]
  make bert squad [MAX_SAMPLES=0]
  make longformer wiki env
  make longformer wiki 1k|2k|4k [MAX_EVAL_TOKENS=0]
  make longformer imdb env
  make longformer imdb 1k|2k|4k [MAX_SAMPLES=5000|0]
  make env
  make download wiki|imdb
  make package

Common knobs:
  PROFILE=HSSS-Softmax-block8   # default profile
  PROFILE=HSSS-Softmax-block4   # BERT block-4 comparison
  OFFLINE=1                 # default for Longformer runs; BERT runs allow downloads unless OFFLINE=1 is set
  LOCAL_FILES_ONLY=1         # default for Longformer
  DOWNLOAD_ASSETS=1          # default for task-specific env targets
  STREAM_LOGS=1              # default; stream child output while saving logs
  STREAM_LOGS=0              # quieter mode with heartbeat and log path only
  HEARTBEAT_SEC=30           # status interval while logs are written to file
  TORCH_THREADS=<n>
"""
    )


def main(argv: list[str]) -> None:
    if not argv or argv[0] in {"help", "-h", "--help"}:
        print_help()
        return
    command, rest = argv[0].lower(), argv[1:]
    if command == "bert":
        run_bert(rest)
    elif command == "longformer":
        run_longformer(rest)
    elif command == "download":
        run_download(rest)
    elif command == "env":
        setup_environment("bert/longformer")
    elif command == "package":
        package_model()
    elif command == "clean":
        if RESULTS_DIR.exists():
            shutil.rmtree(RESULTS_DIR)
        ensure_dirs()
        (RESULTS_DIR / ".gitkeep").touch()
        (LOG_DIR / ".gitkeep").touch()
        print(f"cleaned {ROOT.name}/results")
    else:
        raise SystemExit(f"unknown command {command!r}; run make help")


if __name__ == "__main__":
    main(sys.argv[1:])
