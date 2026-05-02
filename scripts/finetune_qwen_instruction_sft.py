"""
Supervised instruction fine-tuning (SFT) of Qwen causal LMs on DISRPT Task 3.

Uses ``AutoModelForCausalLM`` with next-token loss only on the assistant reply (the gold
unified ``label``), using the tokenizer's ``chat_template`` when available (Qwen2 / Qwen3).

Data layout: pooled ``--data-dir``/{train,dev,test}.tsv plus
optional per-corpus test eval under ``--by-source-dir``. Rows should include ``dir`` and
``rel_type`` when present in processed TSVs; prompts reuse ``qwen_prompt_baseline.build_prompt``
with a synthetic corpus id for pooled splits (language/framework/corpus marked as mixed).

Optional LoRA (``--lora``) requires ``peft``.

Example::

    python scripts/finetune_qwen_instruction_sft.py --qwen3-4b --max-length 1024 \\
        --train-batch-size 1 --gradient-accumulation-steps 16 --epochs 1

    python scripts/finetune_qwen_instruction_sft.py --model-name Qwen/Qwen3-4B --lora \\
        --dtype bfloat16 --gradient-checkpointing
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from typing import Any

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel, PreTrainedTokenizerBase
from transformers import get_linear_schedule_with_warmup

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from qwen_finetune_args import (  # noqa: E402
    _apply_qwen3_4b_cli_defaults,
    _apply_qwen3_cli_defaults,
    _resolved_tsv_paths,
    build_qwen_arg_parser,
)
from finetune_xlm_roberta import read_tsv, set_seed
from qwen_prompt_baseline import (  # noqa: E402
    ChatInstructionSFTDataset,
    DEFAULT_INSTRUCTION_SYSTEM,
    Example,
    LABEL_SET,
    aggregate_predictions_by_corpus,
    corpus_framework,
    corpus_language,
    corpus_name,
    evaluate_predictions,
    generate_predictions_chat,
    move_batch_to_device,
    pad_batch,
    prompt_framework,
    read_examples_from_processed_tsv,
)

SYNTHETIC_POOL_CORPUS_ID = "unk.mixed.pooled"


def pooled_rows_to_examples(rows: list[dict[str, str]], *, split: str) -> list[Example]:
    cid = SYNTHETIC_POOL_CORPUS_ID
    lang = corpus_language(cid)
    corp = "pooled"
    fw_prompt = prompt_framework(cid)
    fw_group = corpus_framework(cid)
    examples: list[Example] = []
    for row in rows:
        if any(not (row.get(k) or "").strip() for k in ("unit1_txt", "unit2_txt", "label")):
            continue
        examples.append(
            Example(
                split=split,
                corpus_id=cid,
                language=lang,
                corpus=corp,
                framework_prompt=fw_prompt,
                framework_group=fw_group,
                unit1_txt=(row.get("unit1_txt") or "").strip(),
                unit2_txt=(row.get("unit2_txt") or "").strip(),
                direction=(row.get("dir") or "").strip(),
                rel_type=(row.get("rel_type") or "").strip(),
                label=(row.get("label") or "").strip(),
            )
        )
    return examples


def examples_from_corpus_tsv(tsv_path: Path, corpus_id: str, *, split: str) -> list[Example]:
    rows = read_tsv(tsv_path)
    lang = corpus_language(corpus_id)
    corp = corpus_name(corpus_id)
    fw_prompt = prompt_framework(corpus_id)
    fw_group = corpus_framework(corpus_id)
    out: list[Example] = []
    for row in rows:
        if any(not (row.get(c) or "").strip() for c in ("unit1_txt", "unit2_txt", "label")):
            continue
        out.append(
            Example(
                split=split,
                corpus_id=corpus_id,
                language=lang,
                corpus=corp,
                framework_prompt=fw_prompt,
                framework_group=fw_group,
                unit1_txt=(row.get("unit1_txt") or "").strip(),
                unit2_txt=(row.get("unit2_txt") or "").strip(),
                direction=(row.get("dir") or "").strip(),
                rel_type=(row.get("rel_type") or "").strip(),
                label=(row.get("label") or "").strip(),
            )
        )
    return out


def per_corpus_test_string_preds(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    by_source_dir: Path,
    device: torch.device,
    *,
    max_new_tokens: int,
    system: str | None,
) -> dict[str, dict[str, list[str]]]:
    """Run generation on each ``<corpus>/<corpus>_test.tsv`` for per-dataset metrics."""
    bundle: dict[str, dict[str, list[str]]] = {}
    for corpus_dir in sorted(p for p in by_source_dir.iterdir() if p.is_dir()):
        corpus = corpus_dir.name
        tsv_path = corpus_dir / f"{corpus}_test.tsv"
        if not tsv_path.exists():
            continue
        examples = examples_from_corpus_tsv(tsv_path, corpus, split="test")
        if not examples:
            continue
        unknown = {ex.label for ex in examples if ex.label not in LABEL_SET}
        if unknown:
            raise ValueError(f"{corpus}: labels outside LABEL_SET: {unknown}")
        _, gold, pred = generate_predictions_chat(
            model,
            tokenizer,
            examples,
            device,
            max_new_tokens=max_new_tokens,
            system=system,
            log_every=0,
        )
        bundle[corpus] = {"gold_label": gold, "pred_label": pred}
    return bundle


def load_pooled_examples(
    train_path: Path,
    dev_path: Path,
    test_path: Path,
    *,
    max_train: int | None,
    max_dev: int | None,
    max_test: int | None,
) -> tuple[list[Example], list[Example], list[Example]]:
    train_ex = pooled_rows_to_examples(read_tsv(train_path), split="train")
    dev_ex = pooled_rows_to_examples(read_tsv(dev_path), split="dev")
    test_ex = pooled_rows_to_examples(read_tsv(test_path), split="test")
    if max_train is not None:
        train_ex = train_ex[:max_train]
    if max_dev is not None:
        dev_ex = dev_ex[:max_dev]
    if max_test is not None:
        test_ex = test_ex[:max_test]
    return train_ex, dev_ex, test_ex


def resolve_torch_dtype(name: str) -> torch.dtype:
    m = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    if name not in m:
        raise ValueError(f"Unknown dtype {name!r}")
    return m[name]


def load_causal_model(
    model_name: str,
    *,
    torch_dtype: torch.dtype,
    trust_remote_code: bool,
    gradient_checkpointing: bool,
    lora: bool,
    lora_r: int,
    lora_alpha: int,
) -> PreTrainedModel:
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch_dtype,
        trust_remote_code=trust_remote_code,
    )
    if gradient_checkpointing:
        model.gradient_checkpointing_enable()
        model.config.use_cache = False
    if lora:
        try:
            from peft import LoraConfig, TaskType, get_peft_model
        except ImportError as e:
            raise ImportError("Install peft for --lora, e.g. pip install peft") from e
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
        config = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            target_modules=target_modules,
            lora_dropout=0.05,
            bias="none",
            task_type=TaskType.CAUSAL_LM,
        )
        model = get_peft_model(model, config)
    return model


def train_one_epoch(
    model: PreTrainedModel,
    loader: DataLoader,
    optimizer: AdamW,
    scheduler: Any,
    device: torch.device,
    *,
    grad_accum: int,
    use_autocast: bool,
    amp_dtype: torch.dtype,
) -> float:
    model.train()
    total_loss = 0.0
    optimizer.zero_grad(set_to_none=True)
    use_fp16_scaler = device.type == "cuda" and amp_dtype == torch.float16
    try:
        scaler = torch.amp.GradScaler("cuda", enabled=use_fp16_scaler)
    except (AttributeError, TypeError):
        scaler = torch.cuda.amp.GradScaler(enabled=use_fp16_scaler)
    for step, batch in enumerate(loader, start=1):
        batch = move_batch_to_device(batch, device)
        autocast_ctx = (
            torch.amp.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_autocast)
            if hasattr(torch, "amp") and hasattr(torch.amp, "autocast")
            else torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_autocast)
        )
        with autocast_ctx:
            outputs = model(**batch)
            loss = outputs.loss / grad_accum
        if scaler.is_enabled():
            scaler.scale(loss).backward()
        else:
            loss.backward()
        total_loss += float(loss.item()) * grad_accum
        if step % grad_accum == 0 or step == len(loader):
            if scaler.is_enabled():
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
    return total_loss / max(1, len(loader))


def extend_parser() -> Any:
    p = build_qwen_arg_parser()
    p.description = __doc__
    p.set_defaults(
        max_length=1024,
        train_batch_size=1,
        eval_batch_size=1,
        output_dir="results/qwen_instruction_sft_results",
    )
    p.add_argument(
        "--gradient-accumulation-steps",
        type=int,
        default=16,
        help="Effective batch = train_batch_size * this value.",
    )
    p.add_argument("--eval-max-new-tokens", type=int, default=24)
    p.add_argument(
        "--dtype",
        default="bfloat16",
        choices=["float32", "float16", "bfloat16"],
        help="Model weights / autocast dtype (default bfloat16).",
    )
    p.add_argument(
        "--system-prompt",
        default=DEFAULT_INSTRUCTION_SYSTEM,
        help="System message for chat template; empty string disables.",
    )
    p.add_argument("--lora", action="store_true", help="Train LoRA adapters (requires peft).")
    p.add_argument("--lora-r", type=int, default=16)
    p.add_argument("--lora-alpha", type=int, default=32)
    p.add_argument(
        "--data-from-by-source-only",
        action="store_true",
        help="Train/dev/test from merged per-corpus TSVs under --by-source-dir (rich LCF metadata).",
    )
    return p


def parse_extended_args():
    return extend_parser().parse_args()


def _prepare_instruction_sft_args(args: argparse.Namespace) -> None:
    _apply_qwen3_cli_defaults(args)
    _apply_qwen3_4b_cli_defaults(args)
    if getattr(args, "bf16", False) and getattr(args, "fp16", False):
        raise ValueError("Use at most one of --bf16 and --fp16.")
    if getattr(args, "bf16", False):
        args.dtype = "bfloat16"
    elif getattr(args, "fp16", False):
        args.dtype = "float16"


def run_instruction_sft(args: argparse.Namespace) -> Path | None:
    """
    Run instruction SFT on ``args`` (same namespace as :func:`parse_extended_args`).

    Returns the resolved directory of the saved causal LM checkpoint (``.../best_model``)
    when ``save_best`` and a best state exist; otherwise ``None``.

    When ``--lora`` was used, weights are merged (``merge_and_unload``) before save so the
    checkpoint is a full model usable with ``AutoModelForSequenceClassification``.
    """
    import argparse as _argparse

    if not isinstance(args, _argparse.Namespace):
        raise TypeError("run_instruction_sft expects an argparse.Namespace")

    _prepare_instruction_sft_args(args)
    set_seed(args.seed)
    device = torch.device(args.device)

    _, train_path, dev_path, test_path, by_source_dir = _resolved_tsv_paths(args)

    if getattr(args, "data_from_by_source_only", False):
        examples_by_split = read_examples_from_processed_tsv(
            by_source_dir,
            max_examples_by_split={
                "train": args.max_train_examples,
                "dev": args.max_dev_examples,
                "test": args.max_test_examples,
            },
        )
        train_ex = examples_by_split["train"]
        dev_ex = examples_by_split["dev"]
        test_ex = examples_by_split["test"]
        data_note = f"by_source_dir={by_source_dir}"
    else:
        if not train_path.exists():
            raise FileNotFoundError(f"Missing pooled train TSV: {train_path}")
        train_ex, dev_ex, test_ex = load_pooled_examples(
            train_path,
            dev_path,
            test_path,
            max_train=args.max_train_examples,
            max_dev=args.max_dev_examples,
            max_test=args.max_test_examples,
        )
        data_note = f"pooled train={train_path}"

    if not train_ex:
        raise ValueError("No training examples loaded.")

    unknown_labels = {ex.label for ex in train_ex + dev_ex + test_ex if ex.label not in LABEL_SET}
    if unknown_labels:
        raise ValueError(f"Examples contain labels outside LABEL_SET: {sorted(unknown_labels)}")

    tok_kw: dict[str, Any] = {"use_fast": True}
    if args.trust_remote_code:
        tok_kw["trust_remote_code"] = True
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, **tok_kw)
    if tokenizer.pad_token is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token

    dtype = resolve_torch_dtype(args.dtype)
    use_autocast = device.type == "cuda" and dtype in (torch.float16, torch.bfloat16)

    model = load_causal_model(
        args.model_name,
        torch_dtype=dtype,
        trust_remote_code=bool(args.trust_remote_code),
        gradient_checkpointing=bool(args.gradient_checkpointing),
        lora=bool(args.lora),
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
    )
    model.to(device)

    system_prompt = (args.system_prompt or "").strip() or None

    train_ds = ChatInstructionSFTDataset(train_ex, tokenizer, args.max_length, system_prompt=system_prompt)
    dev_ds = ChatInstructionSFTDataset(dev_ex, tokenizer, args.max_length, system_prompt=system_prompt)
    test_ds = ChatInstructionSFTDataset(test_ex, tokenizer, args.max_length, system_prompt=system_prompt)

    pad_id = int(tokenizer.pad_token_id or 0)
    collate = lambda b: pad_batch(b, pad_token_id=pad_id)
    train_loader = DataLoader(train_ds, batch_size=args.train_batch_size, shuffle=True, collate_fn=collate)
    # Dev loss optional; generation metrics used like baseline

    optimizer = AdamW((p for p in model.parameters() if p.requires_grad), lr=args.lr, weight_decay=args.weight_decay)
    accum = max(1, args.gradient_accumulation_steps)
    updates_per_epoch = max(1, math.ceil(len(train_loader) / accum))
    total_steps = max(1, updates_per_epoch * args.epochs)
    warmup = int(args.warmup_ratio * total_steps)
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup, total_steps)

    best_f1 = -1.0
    best_state: dict[str, torch.Tensor] | None = None
    history: list[dict[str, Any]] = []

    print(
        f"Instruction SFT: {len(train_ex)} train, {len(dev_ex)} dev, {len(test_ex)} test | "
        f"{data_note} | model={args.model_name}",
        flush=True,
    )

    for epoch in range(1, args.epochs + 1):
        avg_loss = train_one_epoch(
            model,
            train_loader,
            optimizer,
            scheduler,
            device,
            grad_accum=accum,
            use_autocast=use_autocast,
            amp_dtype=dtype,
        )
        _, _, dev_pred = generate_predictions_chat(
            model,
            tokenizer,
            dev_ex,
            device,
            max_new_tokens=args.eval_max_new_tokens,
            system=system_prompt,
            log_every=0,
        )
        dev_metrics = evaluate_predictions([e.label for e in dev_ex], dev_pred)
        history.append(
            {
                "epoch": epoch,
                "train_loss": avg_loss,
                "dev_accuracy": dev_metrics["accuracy"],
                "dev_macro_f1": dev_metrics["macro_f1"],
                "dev_invalid_predictions": dev_metrics["invalid_predictions"],
            }
        )
        print(
            f"epoch {epoch}/{args.epochs}  loss={avg_loss:.4f}  "
            f"dev_acc={dev_metrics['accuracy']:.4f}  dev_macro_f1={dev_metrics['macro_f1']:.4f}  "
            f"invalid={dev_metrics['invalid_predictions']}",
            flush=True,
        )
        if dev_metrics["macro_f1"] > best_f1:
            best_f1 = dev_metrics["macro_f1"]
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict({k: v.to(device) for k, v in best_state.items()})

    _, _, dev_pred = generate_predictions_chat(
        model,
        tokenizer,
        dev_ex,
        device,
        max_new_tokens=args.eval_max_new_tokens,
        system=system_prompt,
        log_every=0,
    )
    dev_metrics = evaluate_predictions([e.label for e in dev_ex], dev_pred)

    test_rows, _, test_pred = generate_predictions_chat(
        model,
        tokenizer,
        test_ex,
        device,
        max_new_tokens=args.eval_max_new_tokens,
        system=system_prompt,
        log_every=0,
    )
    test_metrics = evaluate_predictions([e.label for e in test_ex], test_pred)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    setup_payload = {
        "model_name": args.model_name,
        "training": "instruction_sft_causal_lm",
        "chat_template": bool(getattr(tokenizer, "chat_template", None)),
        "max_length": args.max_length,
        "dtype": args.dtype,
        "lora": bool(args.lora),
        "gradient_accumulation_steps": accum,
        "epochs": args.epochs,
        "lr": args.lr,
        "data_note": data_note,
        "system_prompt": system_prompt,
        "label_set": LABEL_SET,
        "defaults_note": "Uses pooled TSV paths unless --data-from-by-source-only.",
    }
    (out_dir / "metrics.json").write_text(
        json.dumps(
            {"setup": setup_payload, "dev_metrics": dev_metrics, "test_metrics": test_metrics, "history": history},
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    checkpoint_dir: Path | None = None
    if args.save_best and best_state is not None:
        mp = out_dir / "best_model"
        mp.mkdir(parents=True, exist_ok=True)
        to_save = model
        if bool(args.lora):
            if not hasattr(model, "merge_and_unload"):
                raise RuntimeError(
                    "LoRA is enabled but the model has no merge_and_unload(); "
                    "install/use peft and ensure load_causal_model wraps with get_peft_model."
                )
            to_save = model.merge_and_unload()
            model = to_save
        to_save.save_pretrained(mp)
        tokenizer.save_pretrained(mp)
        checkpoint_dir = mp.resolve()

    if not args.skip_per_dataset and by_source_dir.exists():
        gold_pred = per_corpus_test_string_preds(
            model,
            tokenizer,
            by_source_dir,
            device,
            max_new_tokens=args.eval_max_new_tokens,
            system=system_prompt,
        )
        if gold_pred:
            agg = aggregate_predictions_by_corpus(gold_pred)
            (out_dir / "per_dataset_test_metrics.json").write_text(
                json.dumps(
                    {
                        "model": f"Instruction SFT Qwen ({args.model_name})",
                        "model_name": args.model_name,
                        "max_length": args.max_length,
                        "label_set": LABEL_SET,
                        **agg,
                    },
                    indent=2,
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
    elif not args.skip_per_dataset:
        print(f"[warn] skip per-dataset eval: by-source dir missing {by_source_dir}", flush=True)

    if not args.no_save_test_predictions:
        from qwen_prompt_baseline import write_predictions

        write_predictions(out_dir / "test_predictions.tsv", test_rows)

    print(json.dumps({"test_accuracy": test_metrics["accuracy"], "test_macro_f1": test_metrics["macro_f1"]}, indent=2))
    return checkpoint_dir


def run() -> None:
    args = parse_extended_args()
    run_instruction_sft(args)


if __name__ == "__main__":
    run()
