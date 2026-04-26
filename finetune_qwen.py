"""
Fine-tune a Qwen (Qwen2.5 or Qwen3) model for DISRPT pair classification.

Same TSV format and training loop as finetune_xlm_roberta.py, with adjustments for
causal-encoder classifiers: padding token, left padding, and optional bf16/gradient checkpointing.

Qwen3 uses the same code path; pass ``--qwen3`` to load Qwen3 default checkpoint and output paths,
or set ``--model-name`` to any Hub id your ``transformers`` build maps to
``*ForSequenceClassification`` (e.g. ``Qwen/Qwen3-0.6B-Base``, ``Qwen/Qwen3-1.7B-Base``).
"""
from __future__ import annotations

import sys
from pathlib import Path

# Resolve `finetune_xlm_roberta` when the script lives next to it (any cwd, or copied as a single file pair).
_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

import argparse
import csv
import json
import math
from typing import Any, Dict, List

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)

try:
    from tqdm.auto import tqdm as _tqdm
except ImportError:  # pragma: no cover
    _tqdm = None  # type: ignore[misc, assignment]


def resolve_split_data_dir(
    data_dir: str | Path,
    train_file: str,
    dev_file: str,
    test_file: str,
) -> Path:
    """Same semantics as finetune_xlm_roberta; kept here so clusters with an older xlm script still run Qwen."""
    _ = dev_file, test_file
    train_path = Path(train_file)
    if train_path.is_absolute():
        return train_path.parent.resolve()
    return Path(data_dir).expanduser().resolve()


def resolve_by_source_data_dir(by_source_dir: str | Path) -> Path:
    return Path(by_source_dir).expanduser().resolve()


def error_missing_split_tsv(
    train_path: Path,
    train_file: str,
    dev_file: str,
    test_file: str,
) -> str:
    return (
        f"Missing train split TSV: {train_path}\n"
        f"  Check --data-dir and filenames ({train_file!r}, {dev_file!r}, {test_file!r}). "
        f"Generate TSVs with: python data_pipeline.py"
    )


from finetune_xlm_roberta import (
    PairDataset,
    aggregate_predictions,
    load_split,
    move_batch_to_device,
    print_aggregate_summary,
    set_seed,
)
from finetune_xlm_roberta import build_arg_parser as xlm_build_arg_parser

try:
    from finetune_xlm_roberta import write_test_prediction_artifacts
except ImportError:  # pragma: no cover — older finetune_xlm_roberta.py on shared clusters

    def write_test_prediction_artifacts(
        out_dir: Path,
        split: str,
        preds_by_corpus: Dict[str, Dict[str, Any]],
        id2label: Dict[int, str],
    ) -> None:
        """Same as finetune_xlm_roberta.write_test_prediction_artifacts; kept here for self-contained runs."""
        pred_dir = out_dir / "test_predictions"
        pred_dir.mkdir(parents=True, exist_ok=True)
        corpora_payload: Dict[str, Any] = {}
        tsv_path = pred_dir / f"{split}_gold_vs_pred.tsv"
        with tsv_path.open("w", encoding="utf-8", newline="") as f:
            w = csv.writer(f, delimiter="\t", lineterminator="\n")
            w.writerow(["corpus", "row_in_corpus", "gold_label", "pred_label"])
            for corpus, arrs in sorted(preds_by_corpus.items()):
                yt, yp = arrs["y_true"], arrs["y_pred"]
                gold_str = arrs.get("gold_label")
                pred_str = arrs.get("pred_label")
                if gold_str is None:
                    gold_str = [id2label[int(t)] for t in yt]
                if pred_str is None:
                    pred_str = [id2label[int(p)] for p in yp]
                corpora_payload[corpus] = {
                    "gold_label": gold_str,
                    "pred_label": pred_str,
                    "y_true": [int(t) for t in yt.tolist()],
                    "y_pred": [int(p) for p in yp.tolist()],
                }
                for i, (g, p) in enumerate(zip(gold_str, pred_str)):
                    w.writerow([corpus, str(i), g, p])
        summary = {
            "split": split,
            "id2label": {str(k): v for k, v in id2label.items()},
            "corpora": corpora_payload,
        }
        json_path = pred_dir / f"{split}_per_corpus_labels.json"
        json_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"Saved per-example labels → {json_path} and {tsv_path}")

# Presets: pair with --qwen3 to switch defaults, or set --model-name / --output-dir yourself.
QWEN25_DEFAULT_MODEL = "Qwen/Qwen2.5-0.5B"
QWEN25_DEFAULT_OUTPUT = "results/qwen_finetune_results"
QWEN3_DEFAULT_MODEL = "Qwen/Qwen3-0.6B-Base"
QWEN3_DEFAULT_OUTPUT = "results/qwen3_finetune_results"


def _no_weight_decay_param(name: str) -> bool:
    n = name.lower()
    if n.endswith("bias"):
        return True
    if "norm" in n or "ln" in n or "layernorm" in n:
        return True
    return False


def _prepare_batch_for_qwen(batch: Dict[str, Any], device: torch.device) -> Dict[str, Any]:
    """Move to device and drop token_type_ids (BERT-only); Qwen2/Qwen3 have no segment embeddings."""
    batch = move_batch_to_device(batch, device)
    batch.pop("token_type_ids", None)
    return batch


def _apply_pad_token_to_tokenizer(tokenizer: Any) -> int:
    """
    Causal LMs often omit a dedicated pad. GenericForSequenceClassification also requires
    a non-None pad token id in practice for batch size > 1.
    """
    if getattr(tokenizer, "pad_token", None) is None:
        if getattr(tokenizer, "eos_token", None) is not None:
            tokenizer.pad_token = tokenizer.eos_token
        elif getattr(tokenizer, "unk_token", None) is not None:
            tokenizer.pad_token = tokenizer.unk_token
    pad_id = getattr(tokenizer, "pad_token_id", None)
    if pad_id is None and getattr(tokenizer, "eos_token_id", None) is not None:
        tokenizer.pad_token = tokenizer.eos_token
        pad_id = int(tokenizer.eos_token_id)  # type: ignore[attr-defined]
    if pad_id is None:
        raise ValueError(
            "Tokenizer has no pad_token_id (and no eos/unk to reuse). "
            "Cannot run batched sequence classification. Try another checkpoint or set tokenizer.pad_token."
        )
    return int(pad_id)


def _set_model_config_pad_token_id(model: nn.Module, pad_id: int) -> None:
    """HF modeling_layers checks self.config.pad_token_id for multi-example batches."""
    if hasattr(model, "config") and model.config is not None and hasattr(model.config, "pad_token_id"):
        model.config.pad_token_id = pad_id


def build_qwen_arg_parser() -> argparse.ArgumentParser:
    p = xlm_build_arg_parser()
    p.description = "Fine-tune a Qwen model on DISRPT pair classification."
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
            f"(only applied when those args still match the Qwen2.5 defaults, so you can override either)."
        ),
    )
    p.add_argument(
        "--trust-remote-code",
        action="store_true",
        default=False,
        help="Pass trust_remote_code=True to from_pretrained (older custom checkpoints).",
    )
    p.add_argument(
        "--bf16",
        action="store_true",
        help="Use torch autocast bfloat16 on CUDA during training/eval (recommended on Ampere+).",
    )
    p.add_argument(
        "--fp16",
        action="store_true",
        help="Use torch autocast float16 on CUDA (pick either bf16 or fp16, not both).",
    )
    p.add_argument(
        "--gradient-checkpointing",
        action="store_true",
        help="Enable gradient checkpointing to save activation memory.",
    )
    return p


def parse_qwen_args() -> argparse.Namespace:
    return build_qwen_arg_parser().parse_args()


def _apply_qwen3_cli_defaults(args: argparse.Namespace) -> None:
    """If --qwen3, swap in Qwen3 model/output defaults when the user did not change them from Qwen2.5."""
    if not getattr(args, "qwen3", False):
        return
    if args.model_name == QWEN25_DEFAULT_MODEL:
        args.model_name = QWEN3_DEFAULT_MODEL
    if args.output_dir == QWEN25_DEFAULT_OUTPUT:
        args.output_dir = QWEN3_DEFAULT_OUTPUT


def _resolved_tsv_paths(args: argparse.Namespace) -> tuple[Path, Path, Path, Path, Path]:
    """
    Canonical absolute paths for pooled splits and per-corpus TSV root.
    Uses the same --data-dir / --by-source-dir fallbacks as finetune_xlm_roberta.
    Pooled file args may be relative to the resolved data dir or absolute files.
    """
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


def run_qwen_finetune(args: argparse.Namespace) -> None:
    _apply_qwen3_cli_defaults(args)
    if args.bf16 and args.fp16:
        raise ValueError("Use at most one of --bf16 and --fp16.")

    set_seed(args.seed)

    data_dir, train_path, dev_path, test_path, by_source_dir = _resolved_tsv_paths(args)

    if not train_path.exists():
        raise FileNotFoundError(
            error_missing_split_tsv(
                train_path, args.train_file, args.dev_file, args.test_file
            )
        )

    device = torch.device(args.device)

    train_data = load_split(train_path, args.max_train_examples)
    dev_data = load_split(dev_path, args.max_dev_examples)
    test_data = load_split(test_path, args.max_test_examples)

    label_set = sorted(set(train_data.labels) | set(dev_data.labels) | set(test_data.labels))
    if not label_set:
        raise ValueError("No labels found across splits.")

    label2id = {lab: i for i, lab in enumerate(label_set)}
    id2label = {i: lab for lab, i in label2id.items()}
    num_labels = len(label_set)

    tok_kwargs: Dict[str, Any] = {"use_fast": True}
    if args.trust_remote_code:
        tok_kwargs["trust_remote_code"] = True
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, **tok_kwargs)
    pad_id = _apply_pad_token_to_tokenizer(tokenizer)
    tokenizer.padding_side = "left"

    model_kwargs: Dict[str, Any] = {
        "num_labels": num_labels,
        "id2label": id2label,
        "label2id": label2id,
        # Base checkpoints often ship with a tiny default head (e.g. num_labels=2). Without this,
        # weights load into the old head and logits are [batch, 2] while labels are 0..16 — CUDA
        # nll_loss then fails: t >= 0 && t < n_classes.
        "ignore_mismatched_sizes": True,
    }
    if args.trust_remote_code:
        model_kwargs["trust_remote_code"] = True
    model = AutoModelForSequenceClassification.from_pretrained(args.model_name, **model_kwargs)
    if getattr(model.config, "num_labels", None) != num_labels:
        raise ValueError(
            f"Model num_labels={getattr(model.config, 'num_labels', None)} != dataset {num_labels=}"
        )
    _set_model_config_pad_token_id(model, pad_id)
    if args.gradient_checkpointing and getattr(model.config, "use_cache", None) is not None:
        model.config.use_cache = False  # required with gradient checkpointing; avoids a HF warning
    model.to(device)
    if args.gradient_checkpointing and hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()

    train_ds = PairDataset(train_data, label2id=label2id, tokenizer=tokenizer, max_length=args.max_length)
    dev_ds = PairDataset(dev_data, label2id=label2id, tokenizer=tokenizer, max_length=args.max_length)
    test_ds = PairDataset(test_data, label2id=label2id, tokenizer=tokenizer, max_length=args.max_length)

    train_loader = DataLoader(train_ds, batch_size=args.train_batch_size, shuffle=True)
    dev_loader = DataLoader(dev_ds, batch_size=args.eval_batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=args.eval_batch_size, shuffle=False)

    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if p.requires_grad and not _no_weight_decay_param(n)],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if p.requires_grad and _no_weight_decay_param(n)],
            "weight_decay": 0.0,
        },
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.lr)

    num_update_steps_per_epoch = math.ceil(len(train_loader))
    t_total = num_update_steps_per_epoch * args.epochs
    warmup_steps = int(args.warmup_ratio * t_total)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=t_total,
    )

    use_cuda_amp = (args.bf16 or args.fp16) and device.type == "cuda"
    amp_dtype = torch.bfloat16 if args.bf16 else (torch.float16 if args.fp16 else torch.float32)

    def forward_train(batch: Dict[str, Any]) -> torch.Tensor:
        batch = _prepare_batch_for_qwen(batch, device)
        if use_cuda_amp:
            with torch.autocast(device_type="cuda", dtype=amp_dtype):
                outputs = model(**batch)
            return outputs.loss
        outputs = model(**batch)
        return outputs.loss

    def evaluate_wrapped(loader: DataLoader) -> Dict[str, Any]:
        return _evaluate_qwen(
            model, loader, device, num_labels, use_autocast=use_cuda_amp, amp_dtype=amp_dtype
        )

    best_dev_macro_f1 = -1.0
    best_state_dict: Dict[str, torch.Tensor] | None = None
    history: List[Dict[str, Any]] = []

    n_steps = len(train_loader)
    log_every = max(1, n_steps // 20)
    print(
        f"Starting training: {len(train_ds)} train examples, {n_steps} steps/epoch, "
        f"{args.epochs} epochs, device={device}.\n"
        f"(HF load may show MISSING `classifier`/`score` — new head; fine-tune updates it.)\n"
        f"(After load, no output until step 1 finishes — first step can take a while on GPU.)\n"
        f"({'tqdm shows loss in postfix;' if _tqdm is not None else 'Install tqdm for a progress bar;'} "
        f"otherwise ~{log_every} printed steps/epoch with batch_loss.)\n",
        flush=True,
    )

    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss = 0.0
        train_iter: Any = train_loader
        if _tqdm is not None:
            train_iter = _tqdm(
                train_loader,
                desc=f"train epoch {epoch}/{args.epochs}",
                total=n_steps,
                unit="step",
            )
        for step, batch in enumerate(train_iter, start=1):
            loss = forward_train(batch)
            loss_f = float(loss.item())
            loss.backward()
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            epoch_loss += loss_f
            if _tqdm is not None:
                train_iter.set_postfix(loss=f"{loss_f:.4f}", refresh=(step == 1 or step == n_steps or step % log_every == 0))
            elif step == 1 or step % log_every == 0 or step == n_steps:
                print(
                    f"  epoch {epoch}/{args.epochs}  step {step}/{n_steps}  batch_loss={loss_f:.4f}",
                    flush=True,
                )
        if _tqdm is not None:
            train_iter.close()

        print(
            f"epoch {epoch}/{args.epochs} done — running dev eval …",
            flush=True,
        )
        avg_loss = epoch_loss / max(1, len(train_loader))
        dev_metrics = evaluate_wrapped(dev_loader)
        print(
            f"epoch {epoch}/{args.epochs}  train_loss={avg_loss:.4f}  dev_acc={dev_metrics['accuracy']:.4f}  "
            f"dev_macro_f1={dev_metrics['macro_f1']:.4f}",
            flush=True,
        )
        history.append(
            {
                "epoch": epoch,
                "train_loss": avg_loss,
                "dev_accuracy": dev_metrics["accuracy"],
                "dev_macro_f1": dev_metrics["macro_f1"],
            }
        )

        if dev_metrics["macro_f1"] > best_dev_macro_f1:
            best_dev_macro_f1 = dev_metrics["macro_f1"]
            best_state_dict = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    if best_state_dict is not None:
        model.load_state_dict({k: v.to(device) for k, v in best_state_dict.items()})

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    dev_full = evaluate_wrapped(dev_loader)
    test_full = evaluate_wrapped(test_loader)

    setup_payload: Dict[str, Any] = {
        "model_name": args.model_name,
        "max_length": args.max_length,
        "num_labels": num_labels,
        "label_set": label_set,
        "tsv_data_dir": str(data_dir),
        "tsv_train_path": str(train_path),
        "tsv_dev_path": str(dev_path),
        "tsv_test_path": str(test_path),
        "tsv_by_source_dir": str(by_source_dir),
        "train_batch_size": args.train_batch_size,
        "eval_batch_size": args.eval_batch_size,
        "epochs": args.epochs,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "warmup_ratio": args.warmup_ratio,
        "seed": args.seed,
        "bf16": bool(args.bf16),
        "fp16": bool(args.fp16),
        "gradient_checkpointing": bool(args.gradient_checkpointing),
        "qwen3_cli": bool(getattr(args, "qwen3", False)),
    }

    payload = {
        "setup": setup_payload,
        "dev_metrics": dev_full,
        "test_metrics": test_full,
        "training_history": history,
    }
    (out_dir / "metrics.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    if args.save_best and best_state_dict is not None:
        model.save_pretrained(out_dir / "best_model")
        tokenizer.save_pretrained(out_dir / "best_model")
        print(f"Saved best model to {out_dir / 'best_model'}")

    print(f"Saved fine-tuning metrics to {out_dir / 'metrics.json'}")

    _fam = "Qwen3" if "Qwen3" in args.model_name else "Qwen2.5" if "Qwen2" in args.model_name else "Qwen"
    per_label = f"Fine-tuned {_fam} ({args.model_name})"
    if not args.skip_per_dataset:
        if not by_source_dir.exists():
            print(f"[warn] by-source dir not found: {by_source_dir} — skipping per-dataset eval.")
        else:
            print(f"\nRunning per-dataset test evaluation from {by_source_dir}...")
            preds_by_corpus = _predict_per_corpus_qwen(
                by_source_dir=by_source_dir,
                model=model,
                tokenizer=tokenizer,
                label2id=label2id,
                device=device,
                max_length=args.max_length,
                eval_batch_size=args.eval_batch_size,
                use_cuda_amp=use_cuda_amp,
                amp_dtype=amp_dtype,
            )
            agg = aggregate_predictions(preds_by_corpus, id2label)
            print_aggregate_summary(agg)
            if not args.no_save_test_predictions:
                write_test_prediction_artifacts(out_dir, "test", preds_by_corpus, id2label)
            per_dataset_path = out_dir / "per_dataset_test_metrics.json"
            per_dataset_payload = {
                "model": per_label,
                "model_name": args.model_name,
                "max_length": args.max_length,
                "label_set": label_set,
                **agg,
            }
            per_dataset_path.write_text(json.dumps(per_dataset_payload, indent=2, ensure_ascii=False), encoding="utf-8")
            print(f"\nSaved per-dataset metrics to {per_dataset_path}")


def _evaluate_qwen(
    model: nn.Module,
    data_loader: DataLoader,
    device: torch.device,
    num_labels: int,
    *,
    use_autocast: bool,
    amp_dtype: torch.dtype,
) -> Dict[str, Any]:
    from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
    import numpy as np

    model.eval()
    all_logits: list = []
    all_labels: list = []
    with torch.no_grad():
        for batch in data_loader:
            batch = _prepare_batch_for_qwen(batch, device)
            labels = batch.pop("labels")
            if use_autocast and device.type == "cuda":
                with torch.autocast(device_type="cuda", dtype=amp_dtype):
                    outputs = model(**batch)
            else:
                outputs = model(**batch)
            all_logits.append(outputs.logits.detach().float().cpu().numpy())
            all_labels.append(labels.detach().cpu().numpy())
    y_true = np.concatenate(all_labels, axis=0)
    y_pred = np.concatenate(all_logits, axis=0).argmax(axis=1)
    acc = float(accuracy_score(y_true, y_pred))
    report = classification_report(
        y_true, y_pred, labels=list(range(num_labels)), output_dict=True, zero_division=0
    )
    macro_f1 = float(report["macro avg"]["f1-score"])
    cm = confusion_matrix(y_true, y_pred, labels=list(range(num_labels))).tolist()
    return {
        "accuracy": acc,
        "macro_f1": macro_f1,
        "classification_report": report,
        "confusion_matrix": cm,
    }


def _predict_per_corpus_qwen(
    *,
    by_source_dir: Path,
    model: nn.Module,
    tokenizer: Any,
    label2id: Dict[str, int],
    device: torch.device,
    max_length: int,
    eval_batch_size: int,
    split: str = "test",
    use_cuda_amp: bool = False,
    amp_dtype: torch.dtype = torch.bfloat16,
) -> Dict[str, Dict[str, Any]]:
    from finetune_xlm_roberta import load_split, SplitData, PairDataset
    from torch.utils.data import DataLoader
    import numpy as np

    id2label_local = {i: lab for lab, i in label2id.items()}
    preds: Dict[str, Dict[str, Any]] = {}
    for corpus_dir in sorted(p for p in by_source_dir.iterdir() if p.is_dir()):
        corpus = corpus_dir.name
        tsv_path = (corpus_dir / f"{corpus}_{split}.tsv").resolve()
        if not tsv_path.exists():
            continue
        split_data = load_split(tsv_path, None)
        if not split_data.labels:
            continue
        unknown = sorted({lab for lab in split_data.labels if lab not in label2id})
        if unknown:
            print(f"[warn] {corpus}: {len(unknown)} label(s) outside training vocab — rows dropped: {unknown}")
            keep = [i for i, lab in enumerate(split_data.labels) if lab in label2id]
            split_data = SplitData(
                unit1=[split_data.unit1[i] for i in keep],
                unit2=[split_data.unit2[i] for i in keep],
                labels=[split_data.labels[i] for i in keep],
            )
            if not split_data.labels:
                continue
        ds = PairDataset(split_data, label2id=label2id, tokenizer=tokenizer, max_length=max_length)
        loader = DataLoader(ds, batch_size=eval_batch_size, shuffle=False)
        all_logits: list = []
        all_labels: list = []
        model.eval()
        with torch.no_grad():
            for batch in loader:
                batch = _prepare_batch_for_qwen(batch, device)
                labels = batch.pop("labels")
                if use_cuda_amp:
                    with torch.autocast(device_type="cuda", dtype=amp_dtype):
                        outputs = model(**batch)
                else:
                    outputs = model(**batch)
                all_logits.append(outputs.logits.detach().float().cpu().numpy())
                all_labels.append(labels.detach().cpu().numpy())
        yt = np.concatenate(all_labels, axis=0).astype(np.int64)
        yp = np.concatenate(all_logits, axis=0).argmax(axis=1).astype(np.int64)
        gold_labs = list(split_data.labels)
        pred_labs = [id2label_local[int(p)] for p in yp]
        preds[corpus] = {
            "y_true": yt,
            "y_pred": yp,
            "gold_label": gold_labs,
            "pred_label": pred_labs,
        }
    return preds


def main() -> None:
    run_qwen_finetune(parse_qwen_args())


if __name__ == "__main__":
    main()
