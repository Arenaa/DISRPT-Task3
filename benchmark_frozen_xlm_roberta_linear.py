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


CORPUS_FRAMEWORK_OVERRIDES = {
    # eRST is grouped with RST for reporting.
    "eng.erst.gum": "rst",
}


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
    encoder: AutoModel,
    tokenizer: AutoTokenizer,
    head: nn.Module,
    label2id: dict[str, int],
    device: torch.device,
    max_length: int,
    emb_batch_size: int,
    emb_num_workers: int,
    split: str = "test",
) -> dict[str, dict[str, np.ndarray]]:
    """Run the frozen-encoder + linear-head on each `<corpus>_{split}.tsv`
    and return raw prediction arrays keyed by corpus name."""
    head.eval()
    preds: dict[str, dict[str, np.ndarray]] = {}
    for corpus_dir in sorted(p for p in by_source_dir.iterdir() if p.is_dir()):
        corpus = corpus_dir.name
        tsv_path = corpus_dir / f"{corpus}_{split}.tsv"
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

        dataset = PairTextDataset(split_data, label2id=label2id)
        X, y = compute_cls_pooled_embeddings(
            encoder=encoder,
            tokenizer=tokenizer,
            dataset=dataset,
            device=device,
            batch_size=emb_batch_size,
            max_length=max_length,
            num_workers=emb_num_workers,
        )
        with torch.no_grad():
            logits = head(X.to(device)).cpu().numpy()
        preds[corpus] = {
            "y_true": y.cpu().numpy().astype(np.int64),
            "y_pred": logits.argmax(axis=1).astype(np.int64),
        }
    return preds


def _slice_report(y_true: np.ndarray, y_pred: np.ndarray, id2label: dict[int, str]) -> dict[str, object]:
    """Build an accuracy / macro-F1 / weighted-F1 / per-label report for a slice."""
    if len(y_true) == 0:
        return {"support": 0, "accuracy": 0.0, "macro_f1": 0.0, "weighted_f1": 0.0, "per_label": {}}
    present_ids = sorted(set(int(v) for v in y_true.tolist()) | set(int(v) for v in y_pred.tolist()))
    present_names = [id2label[i] for i in present_ids]
    acc = float(accuracy_score(y_true, y_pred))
    report = classification_report(
        y_true,
        y_pred,
        labels=present_ids,
        target_names=present_names,
        output_dict=True,
        zero_division=0,
    )
    per_label: dict[str, dict[str, float]] = {}
    for name in present_names:
        r = report.get(name, {})
        per_label[name] = {
            "precision": float(r.get("precision", 0.0)),
            "recall": float(r.get("recall", 0.0)),
            "f1": float(r.get("f1-score", 0.0)),
            "support": int(r.get("support", 0)),
        }
    return {
        "support": int(len(y_true)),
        "num_gold_labels": int(len(set(int(v) for v in y_true.tolist()))),
        "accuracy": acc,
        "macro_f1": float(report["macro avg"]["f1-score"]),
        "weighted_f1": float(report["weighted avg"]["f1-score"]),
        "per_label": per_label,
    }


def aggregate_predictions(
    preds_by_corpus: dict[str, dict[str, np.ndarray]],
    id2label: dict[int, str],
) -> dict[str, object]:
    """Aggregate raw per-corpus predictions into per-corpus, per-framework,
    per-language, and pooled-global metrics."""
    per_corpus: dict[str, dict[str, object]] = {}
    by_framework: dict[str, list[tuple[np.ndarray, np.ndarray]]] = {}
    by_language: dict[str, list[tuple[np.ndarray, np.ndarray]]] = {}
    all_true: list[np.ndarray] = []
    all_pred: list[np.ndarray] = []

    for corpus, arrs in preds_by_corpus.items():
        yt, yp = arrs["y_true"], arrs["y_pred"]
        per_corpus[corpus] = {
            "framework": corpus_framework(corpus),
            "language": corpus_language(corpus),
            **_slice_report(yt, yp, id2label),
        }
        by_framework.setdefault(corpus_framework(corpus), []).append((yt, yp))
        by_language.setdefault(corpus_language(corpus), []).append((yt, yp))
        all_true.append(yt)
        all_pred.append(yp)

    per_framework: dict[str, dict[str, object]] = {}
    for fw, items in sorted(by_framework.items()):
        yt = np.concatenate([a for a, _ in items])
        yp = np.concatenate([b for _, b in items])
        per_framework[fw] = {
            "corpora": sorted(c for c in preds_by_corpus if corpus_framework(c) == fw),
            **_slice_report(yt, yp, id2label),
        }

    per_language: dict[str, dict[str, object]] = {}
    for lg, items in sorted(by_language.items()):
        yt = np.concatenate([a for a, _ in items])
        yp = np.concatenate([b for _, b in items])
        per_language[lg] = {
            "corpora": sorted(c for c in preds_by_corpus if corpus_language(c) == lg),
            **_slice_report(yt, yp, id2label),
        }

    global_yt = np.concatenate(all_true) if all_true else np.array([], dtype=np.int64)
    global_yp = np.concatenate(all_pred) if all_pred else np.array([], dtype=np.int64)
    pooled = _slice_report(global_yt, global_yp, id2label)

    return {
        "pooled": pooled,
        "per_corpus": per_corpus,
        "per_framework": per_framework,
        "per_language": per_language,
        "per_label_global": pooled["per_label"],
    }


def print_aggregate_summary(agg: dict[str, object]) -> None:
    def _row(name: str, r: dict[str, object]) -> str:
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
    parser.add_argument(
        "--by-source-dir",
        default="processed_tsv/by_source_file",
        help="Root dir with per-corpus `<corpus>/<corpus>_{train,dev,test}.tsv` used for per-dataset evaluation.",
    )
    parser.add_argument(
        "--skip-per-dataset",
        action="store_true",
        help="Skip per-corpus test evaluation (pooled metrics only).",
    )
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

    head_state_path = out_dir / "linear_head.pt"
    torch.save(
        {
            "state_dict": {k: v.detach().cpu() for k, v in head.state_dict().items()},
            "label_set": label_set,
            "label2id": label2id,
            "model_name": args.model_name,
            "max_length": args.max_length,
            "pooling": pooling,
        },
        head_state_path,
    )
    print(f"Saved linear head to {head_state_path}")

    if not args.skip_per_dataset:
        by_source_dir = Path(args.by_source_dir)
        if not by_source_dir.exists():
            print(f"[warn] by-source dir not found: {by_source_dir} — skipping per-dataset eval.")
        else:
            print(f"\nRunning per-dataset test evaluation from {by_source_dir}...")
            preds_by_corpus = predict_per_corpus(
                by_source_dir=by_source_dir,
                encoder=encoder,
                tokenizer=tokenizer,
                head=head,
                label2id=label2id,
                device=device,
                max_length=args.max_length,
                emb_batch_size=args.emb_batch_size,
                emb_num_workers=args.emb_num_workers,
                split="test",
            )
            agg = aggregate_predictions(preds_by_corpus, id2label)
            print_aggregate_summary(agg)

            per_dataset_path = out_dir / "per_dataset_test_metrics.json"
            per_dataset_payload = {
                "model": payload["baseline"],
                "model_name": args.model_name,
                "max_length": args.max_length,
                "label_set": label_set,
                **agg,
            }
            per_dataset_path.write_text(json.dumps(per_dataset_payload, indent=2, ensure_ascii=False), encoding="utf-8")
            print(f"\nSaved per-dataset metrics to {per_dataset_path}")


if __name__ == "__main__":
    main()

