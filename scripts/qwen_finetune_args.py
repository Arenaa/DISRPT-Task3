"""
Shared argparse and TSV path resolution for Qwen training scripts (instruction SFT, etc.).

Keeps ``--qwen3`` / ``--qwen3-4b`` presets aligned with ``finetune_xlm_roberta`` data flags.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from finetune_xlm_roberta import build_arg_parser as xlm_build_arg_parser

QWEN25_DEFAULT_MODEL = "Qwen/Qwen2.5-0.5B"
QWEN25_DEFAULT_OUTPUT = "results/qwen_finetune_results"
QWEN3_DEFAULT_MODEL = "Qwen/Qwen3-0.6B-Base"
QWEN3_DEFAULT_OUTPUT = "results/qwen3_finetune_results"
QWEN3_4B_DEFAULT_MODEL = "Qwen/Qwen3-4B"
QWEN3_4B_DEFAULT_OUTPUT = "results/qwen3_4b_finetune_results"
QWEN_TSV_FEATURES_DEFAULT_OUTPUT = "results/qwen_finetune_tsv_features_results"
QWEN3_TSV_FEATURES_DEFAULT_OUTPUT = "results/qwen3_finetune_tsv_features_results"
QWEN3_4B_TSV_FEATURES_DEFAULT_OUTPUT = "results/qwen3_4b_finetune_tsv_features_results"


def resolve_split_data_dir(
    data_dir: str | Path,
    train_file: str,
    dev_file: str,
    test_file: str,
) -> Path:
    _ = dev_file, test_file
    train_path = Path(train_file)
    if train_path.is_absolute():
        return train_path.parent.resolve()
    return Path(data_dir).expanduser().resolve()


def resolve_by_source_data_dir(by_source_dir: str | Path) -> Path:
    return Path(by_source_dir).expanduser().resolve()


def build_qwen_arg_parser() -> argparse.ArgumentParser:
    p = xlm_build_arg_parser()
    p.description = "Qwen fine-tuning CLI (shared data/model arguments)."
    p.set_defaults(
        model_name=QWEN25_DEFAULT_MODEL,
        max_length=512,
        output_dir=QWEN25_DEFAULT_OUTPUT,
        train_batch_size=8,
        eval_batch_size=16,
        lr=1e-5,
    )
    p.add_argument(
        "--qwen3",
        action="store_true",
        help=(
            f"Use Qwen3 defaults: --model-name {QWEN3_DEFAULT_MODEL} and --output-dir {QWEN3_DEFAULT_OUTPUT} "
            f"(only when those still match the Qwen2.5 defaults)."
        ),
    )
    p.add_argument(
        "--qwen3-4b",
        action="store_true",
        dest="qwen3_4b",
        help=(
            f"Use Qwen3-4B defaults: --model-name {QWEN3_4B_DEFAULT_MODEL} and --output-dir {QWEN3_4B_DEFAULT_OUTPUT} "
            f"(mutually exclusive with --qwen3)."
        ),
    )
    p.add_argument(
        "--trust-remote-code",
        action="store_true",
        default=False,
        help="Pass trust_remote_code=True to from_pretrained.",
    )
    p.add_argument(
        "--bf16",
        action="store_true",
        help="Use torch autocast bfloat16 on CUDA during training/eval.",
    )
    p.add_argument(
        "--fp16",
        action="store_true",
        help="Use torch autocast float16 on CUDA (not together with --bf16).",
    )
    p.add_argument(
        "--gradient-checkpointing",
        action="store_true",
        help="Enable gradient checkpointing to save activation memory.",
    )
    return p


def _apply_qwen3_cli_defaults(args: argparse.Namespace) -> None:
    if getattr(args, "qwen3", False) and getattr(args, "qwen3_4b", False):
        raise ValueError("Use at most one of --qwen3 and --qwen3-4b.")
    if not getattr(args, "qwen3", False):
        return
    if args.model_name == QWEN25_DEFAULT_MODEL:
        args.model_name = QWEN3_DEFAULT_MODEL
    if args.output_dir == QWEN25_DEFAULT_OUTPUT:
        args.output_dir = QWEN3_DEFAULT_OUTPUT
    elif args.output_dir == QWEN_TSV_FEATURES_DEFAULT_OUTPUT:
        args.output_dir = QWEN3_TSV_FEATURES_DEFAULT_OUTPUT


def _apply_qwen3_4b_cli_defaults(args: argparse.Namespace) -> None:
    if not getattr(args, "qwen3_4b", False):
        return
    if args.model_name == QWEN25_DEFAULT_MODEL:
        args.model_name = QWEN3_4B_DEFAULT_MODEL
    if args.output_dir == QWEN25_DEFAULT_OUTPUT:
        args.output_dir = QWEN3_4B_DEFAULT_OUTPUT
    elif args.output_dir == QWEN_TSV_FEATURES_DEFAULT_OUTPUT:
        args.output_dir = QWEN3_4B_TSV_FEATURES_DEFAULT_OUTPUT


def _resolved_tsv_paths(args: argparse.Namespace) -> tuple[Path, Path, Path, Path, Path]:
    data_dir = resolve_split_data_dir(
        args.data_dir, args.train_file, args.dev_file, args.test_file
    )
    t_user = Path(args.train_file)
    d_user = Path(args.dev_file)
    s_user = Path(args.test_file)
    train_path = t_user.resolve() if t_user.is_absolute() else (data_dir / args.train_file).resolve()
    dev_path = d_user.resolve() if d_user.is_absolute() else (data_dir / args.dev_file).resolve()
    test_path = s_user.resolve() if s_user.is_absolute() else (data_dir / args.test_file).resolve()
    by_source_dir = resolve_by_source_data_dir(args.by_source_dir).resolve()
    return data_dir, train_path, dev_path, test_path, by_source_dir
