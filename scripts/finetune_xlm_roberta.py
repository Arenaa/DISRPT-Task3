from __future__ import annotations

import argparse
import csv
import json
import math
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def read_tsv(path: Path) -> List[Dict[str, str]]:
    # Match loader robustness from the frozen baseline script.
    try:
        csv.field_size_limit(sys.maxsize)
    except OverflowError:
        csv.field_size_limit(10**9)

    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t", restval="")
        rows = list(reader)
    return rows


@dataclass(frozen=True)
class SplitData:
    unit1: List[str]
    unit2: List[str]
    labels: List[str]


def limit_examples(rows: List[Dict[str, str]], n: int | None) -> List[Dict[str, str]]:
    if n is None:
        return rows
    return rows[:n]


def load_split(split_path: Path, max_examples: int | None) -> SplitData:
    rows = read_tsv(split_path)
    rows = limit_examples(rows, max_examples)

    unit1: List[str] = []
    unit2: List[str] = []
    labels: List[str] = []

    for r in rows:
        lab = r.get("label", "")
        u1 = r.get("unit1_txt", "")
        u2 = r.get("unit2_txt", "")
        if not lab or not u1 or not u2:
            continue
        unit1.append(u1)
        unit2.append(u2)
        labels.append(lab)

    return SplitData(unit1=unit1, unit2=unit2, labels=labels)


class PairDataset(Dataset):
    def __init__(self, data: SplitData, label2id: Dict[str, int], tokenizer: AutoTokenizer, max_length: int):
        self.data = data
        self.label2id = label2id
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.data.labels)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        u1 = self.data.unit1[idx]
        u2 = self.data.unit2[idx]
        label_idx = self.label2id[self.data.labels[idx]]

        encoded = self.tokenizer(
            u1,
            u2,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt",
        )
        item = {k: v.squeeze(0) for k, v in encoded.items()}
        item["labels"] = torch.tensor(label_idx, dtype=torch.long)
        return item


def move_batch_to_device(batch: Dict[str, torch.Tensor], device: torch.device) -> Dict[str, torch.Tensor]:
    return {k: v.to(device) for k, v in batch.items()}


def _predict(
    model: nn.Module,
    data_loader: DataLoader,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    all_logits: List[np.ndarray] = []
    all_labels: List[np.ndarray] = []
    with torch.no_grad():
        for batch in data_loader:
            batch = move_batch_to_device(batch, device)
            labels = batch.pop("labels")
            outputs = model(**batch)
            all_logits.append(outputs.logits.detach().cpu().numpy())
            all_labels.append(labels.detach().cpu().numpy())
    y_true = np.concatenate(all_labels, axis=0)
    y_pred = np.concatenate(all_logits, axis=0).argmax(axis=1)
    return y_true, y_pred


def evaluate(
    model: nn.Module,
    data_loader: DataLoader,
    device: torch.device,
    num_labels: int,
) -> Dict[str, Any]:
    y_true, y_pred = _predict(model, data_loader, device=device)
    acc = float(accuracy_score(y_true, y_pred))
    report = classification_report(
        y_true,
        y_pred,
        labels=list(range(num_labels)),
        output_dict=True,
        zero_division=0,
    )
    macro_f1 = float(report["macro avg"]["f1-score"])
    cm = confusion_matrix(y_true, y_pred, labels=list(range(num_labels))).tolist()

    return {
        "accuracy": acc,
        "macro_f1": macro_f1,
        "classification_report": report,
        "confusion_matrix": cm,
    }


CORPUS_FRAMEWORK_OVERRIDES = {"eng.erst.gum": "rst"}


def corpus_framework(corpus: str) -> str:
    if corpus in CORPUS_FRAMEWORK_OVERRIDES:
        return CORPUS_FRAMEWORK_OVERRIDES[corpus]
    parts = corpus.split(".")
    return parts[1] if len(parts) >= 2 else "unknown"


def corpus_language(corpus: str) -> str:
    parts = corpus.split(".")
    return parts[0] if parts else "unknown"


def predict_per_corpus(
    *,
    by_source_dir: Path,
    model: nn.Module,
    tokenizer: AutoTokenizer,
    label2id: Dict[str, int],
    device: torch.device,
    max_length: int,
    eval_batch_size: int,
    split: str = "test",
    load_split_fn: Callable[[Path, int | None], SplitData] | None = None,
) -> Dict[str, Dict[str, Any]]:
    _load = load_split_fn or load_split
    id2label_local = {i: lab for lab, i in label2id.items()}
    preds: Dict[str, Dict[str, Any]] = {}
    for corpus_dir in sorted(p for p in by_source_dir.iterdir() if p.is_dir()):
        corpus = corpus_dir.name
        tsv_path = corpus_dir / f"{corpus}_{split}.tsv"
        if not tsv_path.exists():
            continue
        split_data = _load(tsv_path, None)
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
        yt, yp = _predict(model, loader, device=device)
        yti = yt.astype(np.int64)
        ypi = yp.astype(np.int64)
        gold_labs = list(split_data.labels)
        pred_labs = [id2label_local[int(p)] for p in ypi]
        preds[corpus] = {
            "y_true": yti,
            "y_pred": ypi,
            "gold_label": gold_labs,
            "pred_label": pred_labs,
        }
    return preds


def _slice_report(y_true: np.ndarray, y_pred: np.ndarray, id2label: Dict[int, str]) -> Dict[str, Any]:
    if len(y_true) == 0:
        return {"support": 0, "accuracy": 0.0, "macro_f1": 0.0, "weighted_f1": 0.0, "per_label": {}}
    present_ids = sorted(set(int(v) for v in y_true.tolist()) | set(int(v) for v in y_pred.tolist()))
    present_names = [id2label[i] for i in present_ids]
    acc = float(accuracy_score(y_true, y_pred))
    report = classification_report(
        y_true, y_pred,
        labels=present_ids, target_names=present_names,
        output_dict=True, zero_division=0,
    )
    per_label = {
        name: {
            "precision": float(report.get(name, {}).get("precision", 0.0)),
            "recall": float(report.get(name, {}).get("recall", 0.0)),
            "f1": float(report.get(name, {}).get("f1-score", 0.0)),
            "support": int(report.get(name, {}).get("support", 0)),
        }
        for name in present_names
    }
    return {
        "support": int(len(y_true)),
        "num_gold_labels": int(len(set(int(v) for v in y_true.tolist()))),
        "accuracy": acc,
        "macro_f1": float(report["macro avg"]["f1-score"]),
        "weighted_f1": float(report["weighted avg"]["f1-score"]),
        "per_label": per_label,
    }


def write_test_prediction_artifacts(
    out_dir: Path,
    split: str,
    preds_by_corpus: Dict[str, Dict[str, Any]],
    id2label: Dict[int, str],
) -> None:
    """Save each example's gold and predicted class name (and ids) for the given split."""
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


def aggregate_predictions(
    preds_by_corpus: Dict[str, Dict[str, Any]],
    id2label: Dict[int, str],
) -> Dict[str, Any]:
    per_corpus: Dict[str, Dict[str, Any]] = {}
    by_framework: Dict[str, List[tuple]] = {}
    by_language: Dict[str, List[tuple]] = {}
    all_true: List[np.ndarray] = []
    all_pred: List[np.ndarray] = []
    for corpus, arrs in preds_by_corpus.items():
        yt, yp = arrs["y_true"], arrs["y_pred"]
        per_corpus[corpus] = {
            "framework": corpus_framework(corpus),
            "language": corpus_language(corpus),
            **_slice_report(yt, yp, id2label),
        }
        by_framework.setdefault(corpus_framework(corpus), []).append((yt, yp))
        by_language.setdefault(corpus_language(corpus), []).append((yt, yp))
        all_true.append(yt); all_pred.append(yp)
    per_framework = {}
    for fw, items in sorted(by_framework.items()):
        yt = np.concatenate([a for a, _ in items]); yp = np.concatenate([b for _, b in items])
        per_framework[fw] = {
            "corpora": sorted(c for c in preds_by_corpus if corpus_framework(c) == fw),
            **_slice_report(yt, yp, id2label),
        }
    per_language = {}
    for lg, items in sorted(by_language.items()):
        yt = np.concatenate([a for a, _ in items]); yp = np.concatenate([b for _, b in items])
        per_language[lg] = {
            "corpora": sorted(c for c in preds_by_corpus if corpus_language(c) == lg),
            **_slice_report(yt, yp, id2label),
        }
    gyt = np.concatenate(all_true) if all_true else np.array([], dtype=np.int64)
    gyp = np.concatenate(all_pred) if all_pred else np.array([], dtype=np.int64)
    pooled = _slice_report(gyt, gyp, id2label)
    return {
        "pooled": pooled,
        "per_corpus": per_corpus,
        "per_framework": per_framework,
        "per_language": per_language,
        "per_label_global": pooled["per_label"],
    }


def print_aggregate_summary(agg: Dict[str, Any]) -> None:
    def _row(name, r):
        return f"  {name:24s} n={r['support']:>6d}  acc={r['accuracy']:.4f}  macroF1={r['macro_f1']:.4f}  weightedF1={r['weighted_f1']:.4f}"
    print("\n-- Per corpus --")
    for c, r in sorted(agg["per_corpus"].items()):
        print(_row(c, r))
    print("\n-- Per framework --")
    for f, r in sorted(agg["per_framework"].items()):
        print(_row(f, r))
    print("\n-- Per language --")
    for l, r in sorted(agg["per_language"].items()):
        print(_row(l, r))
    print("\n-- Pooled global --")
    print(_row("ALL", agg["pooled"]))


def resolve_split_data_dir(
    data_dir: str | Path,
    train_file: str,
    dev_file: str,
    test_file: str,
) -> Path:
    """Base directory for pooled train/dev/test TSVs when using relative file names."""
    _ = dev_file, test_file  # same root as train
    train_path = Path(train_file)
    if train_path.is_absolute():
        return train_path.parent.resolve()
    return Path(data_dir).expanduser().resolve()


def resolve_by_source_data_dir(by_source_dir: str | Path) -> Path:
    """Root directory with per-corpus `<corpus>/<corpus>_{train,dev,test}.tsv`."""
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
        f"Generate TSVs with: python scripts/data_pipeline.py"
    )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Fine-tune xlm-roberta-base on DISRPT pair classification.")

    parser.add_argument("--data-dir", default="results/processed_tsv/by_split")
    parser.add_argument("--train-file", default="train.tsv")
    parser.add_argument("--dev-file", default="dev.tsv")
    parser.add_argument("--test-file", default="test.tsv")

    parser.add_argument("--model-name", default="FacebookAI/xlm-roberta-base")
    parser.add_argument("--max-length", type=int, default=256)

    parser.add_argument("--train-batch-size", type=int, default=16)
    parser.add_argument("--eval-batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-ratio", type=float, default=0.1)

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")

    parser.add_argument("--max-train-examples", type=int, default=None)
    parser.add_argument("--max-dev-examples", type=int, default=None)
    parser.add_argument("--max-test-examples", type=int, default=None)

    parser.add_argument("--output-dir", default="results/xlmr_finetune_results")
    parser.add_argument(
        "--save-best",
        dest="save_best",
        action="store_true",
        default=True,
        help="Save best dev checkpoint (default: True).",
    )
    parser.add_argument(
        "--no-save-best",
        dest="save_best",
        action="store_false",
        help="Do not save the best dev checkpoint.",
    )
    parser.add_argument(
        "--by-source-dir",
        default="results/processed_tsv/by_source_file",
        help="Root dir with per-corpus `<corpus>/<corpus>_{train,dev,test}.tsv` for per-dataset eval.",
    )
    parser.add_argument(
        "--skip-per-dataset",
        action="store_true",
        help="Skip per-corpus test evaluation (pooled metrics only).",
    )
    parser.add_argument(
        "--no-save-test-predictions",
        action="store_true",
        help="If set, do not write test_predictions/ with gold vs pred label per example.",
    )

    return parser


def parse_args() -> argparse.Namespace:
    return build_arg_parser().parse_args()


def run_finetune(
    args: argparse.Namespace,
    *,
    load_split_fn: Callable[[Path, int | None], SplitData] = load_split,
    setup_extras: Dict[str, Any] | None = None,
    per_dataset_model_label: str = "Fine-tuned XLM-RoBERTa (xlm-roberta-base)",
) -> None:
    set_seed(args.seed)

    data_dir = Path(args.data_dir)
    train_path = data_dir / args.train_file
    dev_path = data_dir / args.dev_file
    test_path = data_dir / args.test_file

    if not train_path.exists():
        raise FileNotFoundError(f"Missing {train_path}")

    device = torch.device(args.device)

    train_data = load_split_fn(train_path, args.max_train_examples)
    dev_data = load_split_fn(dev_path, args.max_dev_examples)
    test_data = load_split_fn(test_path, args.max_test_examples)

    # Use label union so that dev/test labels are always covered.
    label_set = sorted(set(train_data.labels) | set(dev_data.labels) | set(test_data.labels))
    if not label_set:
        raise ValueError("No labels found across splits.")

    label2id = {lab: i for i, lab in enumerate(label_set)}
    id2label = {i: lab for lab, i in label2id.items()}
    num_labels = len(label_set)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)
    # Use AutoModelForSequenceClassification so pooling/classification is handled by the model.
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name,
        num_labels=num_labels,
        id2label=id2label,
        label2id=label2id,
    )
    model.to(device)

    train_ds = PairDataset(train_data, label2id=label2id, tokenizer=tokenizer, max_length=args.max_length)
    dev_ds = PairDataset(dev_data, label2id=label2id, tokenizer=tokenizer, max_length=args.max_length)
    test_ds = PairDataset(test_data, label2id=label2id, tokenizer=tokenizer, max_length=args.max_length)

    train_loader = DataLoader(train_ds, batch_size=args.train_batch_size, shuffle=True)
    dev_loader = DataLoader(dev_ds, batch_size=args.eval_batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=args.eval_batch_size, shuffle=False)

    # Standard AdamW + linear warmup/decay.
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
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

    best_dev_macro_f1 = -1.0
    best_state_dict: Dict[str, torch.Tensor] | None = None
    history: List[Dict[str, Any]] = []

    steps_per_epoch = len(train_loader)
    log_every = max(1, steps_per_epoch // 20)
    print(
        f"Starting training: {len(train_ds)} train examples, {steps_per_epoch} steps/epoch, "
        f"{args.epochs} epochs, device={device}.\n"
        f"(After the HF load report there is no output until the first log line below — "
        f"epoch 1 can take several minutes on GPU.)\n",
        flush=True,
    )

    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss = 0.0
        for step, batch in enumerate(train_loader, start=1):
            batch = move_batch_to_device(batch, device)
            outputs = model(**batch)
            loss = outputs.loss
            loss.backward()

            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)

            loss_f = float(loss.item())
            epoch_loss += loss_f
            if step == 1 or step % log_every == 0 or step == steps_per_epoch:
                print(
                    f"  epoch {epoch}/{args.epochs}  step {step}/{steps_per_epoch}  batch_loss={loss_f:.4f}",
                    flush=True,
                )

        avg_loss = epoch_loss / max(1, len(train_loader))
        print(f"epoch {epoch}/{args.epochs} done — running dev eval …", flush=True)
        dev_metrics = evaluate(model, dev_loader, device=device, num_labels=num_labels)
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

    dev_full = evaluate(model, dev_loader, device=device, num_labels=num_labels)
    test_full = evaluate(model, test_loader, device=device, num_labels=num_labels)

    setup_payload: Dict[str, Any] = {
        "model_name": args.model_name,
        "max_length": args.max_length,
        "num_labels": num_labels,
        "label_set": label_set,
        "train_batch_size": args.train_batch_size,
        "eval_batch_size": args.eval_batch_size,
        "epochs": args.epochs,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "warmup_ratio": args.warmup_ratio,
        "seed": args.seed,
    }
    if setup_extras:
        setup_payload.update(setup_extras)

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

    if not args.skip_per_dataset:
        by_source_dir = Path(args.by_source_dir)
        if not by_source_dir.exists():
            print(f"[warn] by-source dir not found: {by_source_dir} — skipping per-dataset eval.")
        else:
            print(f"\nRunning per-dataset test evaluation from {by_source_dir}...")
            preds_by_corpus = predict_per_corpus(
                by_source_dir=by_source_dir,
                model=model,
                tokenizer=tokenizer,
                label2id=label2id,
                device=device,
                max_length=args.max_length,
                eval_batch_size=args.eval_batch_size,
                split="test",
                load_split_fn=load_split_fn,
            )
            agg = aggregate_predictions(preds_by_corpus, id2label)
            print_aggregate_summary(agg)
            if not getattr(args, "no_save_test_predictions", False):
                write_test_prediction_artifacts(out_dir, "test", preds_by_corpus, id2label)

            per_dataset_path = out_dir / "per_dataset_test_metrics.json"
            per_dataset_payload = {
                "model": per_dataset_model_label,
                "model_name": args.model_name,
                "max_length": args.max_length,
                "label_set": label_set,
                **agg,
            }
            per_dataset_path.write_text(json.dumps(per_dataset_payload, indent=2, ensure_ascii=False), encoding="utf-8")
            print(f"\nSaved per-dataset metrics to {per_dataset_path}")


def main() -> None:
    run_finetune(parse_args())


if __name__ == "__main__":
    main()

