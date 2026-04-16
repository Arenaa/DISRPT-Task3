from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from torch.utils.data import DataLoader, Dataset, TensorDataset
from transformers import AutoModel, AutoTokenizer


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def read_tsv(path: Path) -> list[dict[str, str]]:
    # Some DISRPT fields (unit texts) can exceed Python's default csv field size limit.
    # Raising it prevents _csv.Error: field larger than field limit.
    try:
        csv.field_size_limit(sys.maxsize)
    except OverflowError:
        csv.field_size_limit(10**9)

    with path.open("r", encoding="utf-8", newline="") as f:
        # Some TSV lines are malformed (missing trailing columns). DictReader uses
        # `restval` for missing fields; set it to "" so downstream code doesn't
        # see `None` values.
        reader = csv.DictReader(f, delimiter="\t", restval="")
        rows = list(reader)
    return rows


@dataclass(frozen=True)
class SplitData:
    unit1: list[str]
    unit2: list[str]
    labels: list[str]


def limit_examples(rows: list[dict[str, str]], n: int | None) -> list[dict[str, str]]:
    if n is None:
        return rows
    return rows[:n]


def load_split(split_path: Path, max_examples: int | None) -> SplitData:
    rows = read_tsv(split_path)
    rows = limit_examples(rows, max_examples)
    unit1: list[str] = []
    unit2: list[str] = []
    labels: list[str] = []

    for r in rows:
        lab = r.get("label", "")
        u1 = r.get("unit1_txt", "")
        u2 = r.get("unit2_txt", "")

        # Skip malformed rows (label/unit missing). This can happen for some
        # lines in the exported TSVs.
        if not lab or not u1 or not u2:
            continue

        unit1.append(u1)
        unit2.append(u2)
        labels.append(lab)

    return SplitData(unit1=unit1, unit2=unit2, labels=labels)


def sanitize_filename(s: str) -> str:
    # Keep it simple: avoid slashes and spaces, which break filenames.
    return "".join(ch if ch.isalnum() or ch in ("-", "_", ".") else "_" for ch in s)


class PairTextDataset(Dataset):
    def __init__(self, data: SplitData, label2id: dict[str, int]):
        self.data = data
        self.label2id = label2id

    def __len__(self) -> int:
        return len(self.data.labels)

    def __getitem__(self, idx: int) -> tuple[str, str, int]:
        return self.data.unit1[idx], self.data.unit2[idx], self.label2id[self.data.labels[idx]]


def compute_cls_pooled_embeddings(
    *,
    encoder: AutoModel,
    tokenizer: AutoTokenizer,
    dataset: PairTextDataset,
    device: torch.device,
    batch_size: int,
    max_length: int,
    num_workers: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Baseline pooling: pooled embedding = last_hidden_state[:, 0, :] (first token, i.e., <s> in RoBERTa).

    Input formatting (conceptually): [CLS] unit1 [SEP] unit2 [SEP].
    In Hugging Face tokenizers for RoBERTa, passing (text, text_pair) yields:
      <s> text </s></s> text_pair </s>
    """

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=_collate_text_pairs,
    )

    encoder.eval()

    all_pooled: list[torch.Tensor] = []
    all_labels: list[torch.Tensor] = []
    with torch.no_grad():
        for unit1_batch, unit2_batch, y_batch in loader:
            encoded = tokenizer(
                unit1_batch,
                unit2_batch,
                return_tensors="pt",
                padding="max_length",
                truncation=True,
                max_length=max_length,
            )
            encoded = {k: v.to(device) for k, v in encoded.items()}

            outputs = encoder(**encoded)
            pooled = outputs.last_hidden_state[:, 0, :]  # (batch, hidden)
            all_pooled.append(pooled.cpu())
            all_labels.append(torch.tensor(y_batch, dtype=torch.long))

    X = torch.cat(all_pooled, dim=0)
    y = torch.cat(all_labels, dim=0)
    return X, y


def _collate_text_pairs(batch: list[tuple[str, str, int]]) -> tuple[list[str], list[str], list[int]]:
    unit1 = [b[0] for b in batch]
    unit2 = [b[1] for b in batch]
    labels = [b[2] for b in batch]
    return unit1, unit2, labels


def train_linear_head(
    *,
    X_train: torch.Tensor,
    y_train: torch.Tensor,
    X_dev: torch.Tensor,
    y_dev: torch.Tensor,
    num_labels: int,
    lr: float,
    weight_decay: float,
    batch_size: int,
    epochs: int,
    device: torch.device,
) -> tuple[nn.Module, dict[str, object]]:
    hidden_size = X_train.shape[1]
    head = nn.Linear(hidden_size, num_labels).to(device)

    # Keep initialization deterministic-ish.
    nn.init.xavier_uniform_(head.weight)
    nn.init.zeros_(head.bias)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=weight_decay)

    train_ds = TensorDataset(X_train, y_train)
    dev_ds = TensorDataset(X_dev, y_dev)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

    best_state: dict[str, torch.Tensor] | None = None
    best_dev_macro_f1 = -1.0
    history: list[dict[str, object]] = []

    for epoch in range(1, epochs + 1):
        head.train()
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)

            logits = head(xb)
            loss = criterion(logits, yb)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        dev_metrics = evaluate_head(head, dev_ds, device=device)
        history.append({"epoch": epoch, **dev_metrics})

        if dev_metrics["macro_f1"] > best_dev_macro_f1:
            best_dev_macro_f1 = dev_metrics["macro_f1"]
            best_state = {k: v.detach().cpu().clone() for k, v in head.state_dict().items()}

    if best_state is not None:
        head.load_state_dict({k: v.to(device) for k, v in best_state.items()})

    return head, {"best_macro_f1": best_dev_macro_f1, "history": history}


def evaluate_head(head: nn.Module, ds: TensorDataset, *, device: torch.device) -> dict[str, object]:
    head.eval()
    X = ds.tensors[0].to(device)
    y_true = ds.tensors[1].cpu().numpy()

    with torch.no_grad():
        logits = head(X).cpu().numpy()

    y_pred = logits.argmax(axis=1)
    acc = float(accuracy_score(y_true, y_pred))

    report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
    macro_f1 = float(report["macro avg"]["f1-score"])

    return {"accuracy": acc, "macro_f1": macro_f1}


def full_report(
    head: nn.Module,
    *,
    X: torch.Tensor,
    y: torch.Tensor,
    id2label: dict[int, str],
    device: torch.device,
) -> dict[str, object]:
    head.eval()
    with torch.no_grad():
        logits = head(X.to(device)).cpu().numpy()
    y_true = y.cpu().numpy()
    y_pred = logits.argmax(axis=1)

    acc = float(accuracy_score(y_true, y_pred))
    labels = [id2label[i] for i in range(len(id2label))]

    report = classification_report(
        y_true,
        y_pred,
        labels=list(range(len(id2label))),
        target_names=labels,
        output_dict=True,
        zero_division=0,
    )
    macro_f1 = float(report["macro avg"]["f1-score"])

    cm = confusion_matrix(y_true, y_pred).tolist()

    return {
        "accuracy": acc,
        "macro_f1": macro_f1,
        "classification_report": report,
        "confusion_matrix": cm,
        "labels": labels,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark frozen XLM-RoBERTa (xlm-roberta-base) + linear head.")
    parser.add_argument("--data-dir", default="processed_tsv/by_split", help="Directory with train/dev/test TSVs.")
    parser.add_argument("--train-file", default="train.tsv")
    parser.add_argument("--dev-file", default="dev.tsv")
    parser.add_argument("--test-file", default="test.tsv")

    parser.add_argument("--model-name", default="xlm-roberta-base")
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--emb-batch-size", type=int, default=16)
    parser.add_argument("--head-batch-size", type=int, default=64)

    parser.add_argument("--emb-num-workers", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.01)

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")

    parser.add_argument("--cache-dir", default="xlmr_frozen_linear_cache")
    parser.add_argument("--no-cache", action="store_true")

    parser.add_argument("--max-train-examples", type=int, default=None)
    parser.add_argument("--max-dev-examples", type=int, default=None)
    parser.add_argument("--max-test-examples", type=int, default=None)

    parser.add_argument("--output-dir", default="xlmr_frozen_linear_results")
    return parser.parse_args()


def cache_path(cache_dir: Path, split: str, *, model_name: str, max_length: int, pooling: str) -> Path:
    tag = sanitize_filename(model_name)
    fname = f"{tag}__{split}__maxlen{max_length}__pool{pooling}.pt"
    return cache_dir / fname


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

    # If we limit examples (e.g., for a quick smoke test), dev/test might contain
    # labels that do not appear in the limited train subset. Use the union so
    # the benchmark can still run end-to-end.
    label_set = sorted(set(train_data.labels) | set(dev_data.labels) | set(test_data.labels))
    if not label_set:
        raise ValueError("No labels found in train split.")

    label2id = {lab: i for i, lab in enumerate(label_set)}
    id2label = {i: lab for lab, i in label2id.items()}

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)
    encoder = AutoModel.from_pretrained(args.model_name)
    for p in encoder.parameters():
        p.requires_grad = False

    encoder.to(device)

    cache_dir = Path(args.cache_dir)
    if not args.no_cache:
        cache_dir.mkdir(parents=True, exist_ok=True)

    pooling = "cls"

    def get_split_features(split_name: str, split_data: SplitData) -> tuple[torch.Tensor, torch.Tensor]:
        if not args.no_cache:
            p = cache_path(
                cache_dir,
                split_name,
                model_name=args.model_name,
                max_length=args.max_length,
                pooling=pooling,
            )
            if p.exists():
                payload = torch.load(p, map_location="cpu")
                return payload["X"], payload["y"]

        dataset = PairTextDataset(split_data, label2id=label2id)
        X, y = compute_cls_pooled_embeddings(
            encoder=encoder,
            tokenizer=tokenizer,
            dataset=dataset,
            device=device,
            batch_size=args.emb_batch_size,
            max_length=args.max_length,
            num_workers=args.emb_num_workers,
        )

        if not args.no_cache:
            torch.save({"X": X, "y": y}, cache_path(
                cache_dir,
                split_name,
                model_name=args.model_name,
                max_length=args.max_length,
                pooling=pooling,
            ))
        return X, y

    print("Computing frozen encoder embeddings (this can take a while)...")
    X_train, y_train = get_split_features("train", train_data)
    X_dev, y_dev = get_split_features("dev", dev_data)
    X_test, y_test = get_split_features("test", test_data)

    if X_train.shape[1] != X_dev.shape[1]:
        raise RuntimeError("Embedding dimension mismatch between splits.")

    print(f"Training linear head on {len(y_train)} examples; #labels={len(label_set)}")
    head, head_train_info = train_linear_head(
        X_train=X_train,
        y_train=y_train,
        X_dev=X_dev,
        y_dev=y_dev,
        num_labels=len(label_set),
        lr=args.lr,
        weight_decay=args.weight_decay,
        batch_size=args.head_batch_size,
        epochs=args.epochs,
        device=device,
    )

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = out_dir / "metrics.json"

    dev_full = full_report(
        head,
        X=X_dev,
        y=y_dev,
        id2label=id2label,
        device=device,
    )
    test_full = full_report(
        head,
        X=X_test,
        y=y_test,
        id2label=id2label,
        device=device,
    )

    payload = {
        "baseline": "Frozen XLM-RoBERTa (xlm-roberta-base) + linear head",
        "pooling": pooling,
        "input_format": "tokenizer(text=unit1, text_pair=unit2) pooled from first token",
        "model_name": args.model_name,
        "max_length": args.max_length,
        "num_labels": len(label_set),
        "label_set": label_set,
        "train_info": {
            "epochs": args.epochs,
            "lr": args.lr,
            "weight_decay": args.weight_decay,
            "emb_batch_size": args.emb_batch_size,
            "head_batch_size": args.head_batch_size,
            "seed": args.seed,
        },
        "dev_metrics": dev_full,
        "test_metrics": test_full,
        "head_training": head_train_info,
    }

    metrics_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Saved metrics to {metrics_path}")


if __name__ == "__main__":
    main()

