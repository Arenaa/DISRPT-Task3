from __future__ import annotations

import argparse
import csv
import json
import math
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

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


def evaluate(
    model: nn.Module,
    data_loader: DataLoader,
    device: torch.device,
    num_labels: int,
) -> Dict[str, Any]:
    model.eval()
    all_logits: List[np.ndarray] = []
    all_labels: List[np.ndarray] = []

    with torch.no_grad():
        for batch in data_loader:
            batch = move_batch_to_device(batch, device)
            labels = batch.pop("labels")
            outputs = model(**batch)
            logits = outputs.logits
            all_logits.append(logits.detach().cpu().numpy())
            all_labels.append(labels.detach().cpu().numpy())

    y_true = np.concatenate(all_labels, axis=0)
    y_pred = np.concatenate(all_logits, axis=0).argmax(axis=1)

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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tune xlm-roberta-base on DISRPT pair classification.")

    parser.add_argument("--data-dir", default="processed_tsv/by_split")
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

    parser.add_argument("--output-dir", default="xlmr_finetune_results")
    parser.add_argument("--save-best", action="store_true", help="Save best dev checkpoint.")

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    data_dir = Path(args.data_dir)
    train_path = data_dir / args.train_file
    dev_path = data_dir / args.dev_file
    test_path = data_dir / args.test_file

    if not train_path.exists():
        raise FileNotFoundError(f"Missing {train_path}")

    device = torch.device(args.device)

    train_data = load_split(train_path, args.max_train_examples)
    dev_data = load_split(dev_path, args.max_dev_examples)
    test_data = load_split(test_path, args.max_test_examples)

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

            epoch_loss += float(loss.item())

        avg_loss = epoch_loss / max(1, len(train_loader))
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

    payload = {
        "setup": {
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
        },
        "dev_metrics": dev_full,
        "test_metrics": test_full,
        "training_history": history,
    }

    (out_dir / "metrics.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    if args.save_best and best_state_dict is not None:
        model.save_pretrained(out_dir / "best_model")
        tokenizer.save_pretrained(out_dir / "best_model")

    print(f"Saved fine-tuning metrics to {out_dir / 'metrics.json'}")


if __name__ == "__main__":
    main()

