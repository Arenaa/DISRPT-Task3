"""finetune_xlm_roberta_framework_conditioned.py

Fine-tune XLM-RoBERTa-base on DISRPT Task 3 with framework conditioning.

Design is deliberately aligned with the plain fine-tune and frozen baselines
(`finetune_xlm_roberta.py`, `benchmark_frozen_xlm_roberta_linear.py`):

    * data is loaded **per corpus** from `processed_tsv/by_source_file/` so every
      row is tagged with (corpus, framework, language) without needing a
      `dataset` column in the pooled TSV (which doesn't exist there);
    * framework names are the canonical lowercase ones used elsewhere — `rst`,
      `pdtb`, `sdrt`, `dep` — with `eng.erst.gum` folded into `rst`;
    * at the end of a run the script produces the same 4-way test breakdown
      (per corpus / framework / language / label) and saves the best checkpoint
      under `<output_dir>/best_model/` so downstream scripts can re-use it;
    * the best checkpoint is compatible with
      `evaluate_with_official_script.py --mode framework_cond` which wraps the
      DISRPT 2025 official scorer around these predictions.

Two framework-specific additions are preserved from the original design:

    1. **Framework token** — `[RST]`, `[PDTB]`, `[SDRT]`, `[DEP]` is prepended to
       unit1 so the encoder gets an explicit framework signal.
    2. **Output masking** — at evaluation time, logits for labels that never
       co-occur with a row's framework in the training data are set to -inf.
       Disable with `--no-mask` for an ablation that keeps only the token.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, classification_report
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)

from benchmark_frozen_xlm_roberta_linear import (
    CORPUS_FRAMEWORK_OVERRIDES,
    _slice_report,
    aggregate_predictions,
    corpus_framework,
    corpus_language,
    print_aggregate_summary,
)

FRAMEWORK_TOKENS: List[str] = ["[RST]", "[PDTB]", "[SDRT]", "[DEP]"]


def framework_token(framework: str) -> str:
    """Map a lowercase framework name to its special token. Unknown → [RST]."""
    up = framework.upper()
    tok = f"[{up}]"
    if tok in FRAMEWORK_TOKENS:
        return tok
    # erst is folded into rst for reporting and for the token.
    if framework == "erst":
        return "[RST]"
    return "[RST]"


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _read_tsv(path: Path) -> List[Dict[str, str]]:
    try:
        csv.field_size_limit(sys.maxsize)
    except OverflowError:
        csv.field_size_limit(10 ** 9)
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f, delimiter="\t", restval=""))

@dataclass(frozen=True)
class SplitData:
    unit1: List[str]
    unit2: List[str]
    labels: List[str]
    corpora: List[str]
    frameworks: List[str]
    languages: List[str]


def load_split_by_corpus(by_source_dir: Path, split: str, max_examples: Optional[int] = None) -> SplitData:
    """Pool per-corpus `<corpus>_{split}.tsv` files into a single SplitData
    with per-row corpus / framework / language tags."""
    unit1: List[str] = []
    unit2: List[str] = []
    labels: List[str] = []
    corpora: List[str] = []
    frameworks: List[str] = []
    languages: List[str] = []

    for corpus_dir in sorted(p for p in by_source_dir.iterdir() if p.is_dir()):
        corpus = corpus_dir.name
        tsv_path = corpus_dir / f"{corpus}_{split}.tsv"
        if not tsv_path.exists():
            continue
        fw = corpus_framework(corpus)
        lg = corpus_language(corpus)
        for r in _read_tsv(tsv_path):
            u1 = (r.get("unit1_txt") or "").strip()
            u2 = (r.get("unit2_txt") or "").strip()
            lab = (r.get("label") or "").strip()
            if not u1 or not u2 or not lab:
                continue
            unit1.append(u1); unit2.append(u2); labels.append(lab)
            corpora.append(corpus); frameworks.append(fw); languages.append(lg)

    if max_examples is not None and max_examples < len(labels):
        idx = list(range(len(labels)))[:max_examples]
        unit1 = [unit1[i] for i in idx]
        unit2 = [unit2[i] for i in idx]
        labels = [labels[i] for i in idx]
        corpora = [corpora[i] for i in idx]
        frameworks = [frameworks[i] for i in idx]
        languages = [languages[i] for i in idx]

    return SplitData(
        unit1=unit1, unit2=unit2, labels=labels,
        corpora=corpora, frameworks=frameworks, languages=languages,
    )


def build_framework_label_map(
    splits: List[SplitData],
    label2id: Dict[str, int],
) -> Dict[str, Set[int]]:
    """Label indices observed for each framework across given splits."""
    fw_labels: Dict[str, Set[int]] = defaultdict(set)
    for data in splits:
        for fw, lab in zip(data.frameworks, data.labels):
            if lab in label2id:
                fw_labels[fw].add(label2id[lab])
    return dict(fw_labels)


def build_logit_masks(framework_label_map: Dict[str, Set[int]], num_labels: int) -> Dict[str, torch.Tensor]:
    """Additive logit masks per framework: 0.0 where allowed, -inf where blocked."""
    masks: Dict[str, torch.Tensor] = {}
    for fw, allowed_ids in framework_label_map.items():
        mask = torch.full((num_labels,), float("-inf"))
        for idx in allowed_ids:
            mask[idx] = 0.0
        masks[fw] = mask
    masks["unknown"] = torch.zeros(num_labels)
    return masks

class FrameworkPairDataset(Dataset):
    def __init__(
        self,
        data: SplitData,
        label2id: Dict[str, int],
        tokenizer: AutoTokenizer,
        max_length: int,
    ) -> None:
        self.data = data
        self.label2id = label2id
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.data.labels)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        fw = self.data.frameworks[idx]
        tok = framework_token(fw)
        u1_conditioned = f"{tok} {self.data.unit1[idx]}"
        u2 = self.data.unit2[idx]
        encoded = self.tokenizer(
            u1_conditioned, u2,
            truncation=True, max_length=self.max_length,
            padding="max_length", return_tensors="pt",
        )
        item = {k: v.squeeze(0) for k, v in encoded.items()}
        lab = self.data.labels[idx]
        item["labels"] = torch.tensor(self.label2id[lab], dtype=torch.long)
        item["framework"] = fw
        return item


def collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    frameworks = [b.pop("framework") for b in batch]
    collated = torch.utils.data.default_collate(batch)
    collated["framework"] = frameworks
    return collated


def move_batch_to_device(batch: Dict[str, Any], device: torch.device) -> Dict[str, Any]:
    return {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

def _predict(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    logit_masks: Dict[str, torch.Tensor],
    *,
    apply_mask: bool = True,
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    model.eval()
    all_logits: List[np.ndarray] = []
    all_labels: List[np.ndarray] = []
    all_fws: List[str] = []
    with torch.no_grad():
        for batch in loader:
            batch = move_batch_to_device(batch, device)
            fws = batch.pop("framework")
            labels = batch.pop("labels")
            logits = model(**batch).logits.cpu()
            if apply_mask:
                for i, fw in enumerate(fws):
                    mask = logit_masks.get(fw, logit_masks["unknown"])
                    logits[i] = logits[i] + mask
            all_logits.append(logits.numpy())
            all_labels.append(labels.cpu().numpy())
            all_fws.extend(fws)
    y_true = np.concatenate(all_labels, axis=0)
    y_pred = np.concatenate(all_logits, axis=0).argmax(axis=1)
    return y_true, y_pred, all_fws


def evaluate_pooled(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    num_labels: int,
    logit_masks: Dict[str, torch.Tensor],
    *,
    apply_mask: bool,
) -> Dict[str, Any]:
    y_true, y_pred, _ = _predict(model, loader, device, logit_masks, apply_mask=apply_mask)
    acc = float(accuracy_score(y_true, y_pred))
    report = classification_report(
        y_true, y_pred,
        labels=list(range(num_labels)),
        output_dict=True, zero_division=0,
    )
    return {
        "accuracy": acc,
        "macro_f1": float(report["macro avg"]["f1-score"]),
        "weighted_f1": float(report["weighted avg"]["f1-score"]),
        "classification_report": report,
    }


def predict_per_corpus(
    *,
    model: nn.Module,
    tokenizer: AutoTokenizer,
    label2id: Dict[str, int],
    device: torch.device,
    logit_masks: Dict[str, torch.Tensor],
    max_length: int,
    eval_batch_size: int,
    by_source_dir: Path,
    split: str = "test",
    apply_mask: bool = True,
) -> Dict[str, Dict[str, np.ndarray]]:
    """Run the fine-tuned framework-conditioned model on each corpus's
    `<corpus>_{split}.tsv` separately, returning raw prediction arrays.

    Output shape mirrors `predict_per_corpus` in the sibling scripts so
    `aggregate_predictions` from the frozen script can consume it directly.
    """
    preds: Dict[str, Dict[str, np.ndarray]] = {}
    for corpus_dir in sorted(p for p in by_source_dir.iterdir() if p.is_dir()):
        corpus = corpus_dir.name
        tsv_path = corpus_dir / f"{corpus}_{split}.tsv"
        if not tsv_path.exists():
            continue
        fw = corpus_framework(corpus)
        lg = corpus_language(corpus)

        rows = _read_tsv(tsv_path)
        unit1 = [(r.get("unit1_txt") or "").strip() for r in rows]
        unit2 = [(r.get("unit2_txt") or "").strip() for r in rows]
        labels = [(r.get("label") or "").strip() for r in rows]
        keep = [i for i in range(len(rows)) if unit1[i] and unit2[i] and labels[i]]
        if not keep:
            continue
        unit1 = [unit1[i] for i in keep]
        unit2 = [unit2[i] for i in keep]
        labels = [labels[i] for i in keep]

        unknown = sorted({lab for lab in labels if lab not in label2id})
        if unknown:
            print(f"[warn] {corpus}: {len(unknown)} label(s) outside training vocab — rows dropped: {unknown}")
            keep2 = [i for i, lab in enumerate(labels) if lab in label2id]
            unit1 = [unit1[i] for i in keep2]
            unit2 = [unit2[i] for i in keep2]
            labels = [labels[i] for i in keep2]
            if not labels:
                continue

        split_data = SplitData(
            unit1=unit1, unit2=unit2, labels=labels,
            corpora=[corpus] * len(labels),
            frameworks=[fw] * len(labels),
            languages=[lg] * len(labels),
        )
        ds = FrameworkPairDataset(split_data, label2id, tokenizer, max_length)
        loader = DataLoader(ds, batch_size=eval_batch_size, shuffle=False, collate_fn=collate_fn)
        yt, yp, _ = _predict(model, loader, device, logit_masks, apply_mask=apply_mask)
        preds[corpus] = {"y_true": yt.astype(np.int64), "y_pred": yp.astype(np.int64)}
    return preds


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)

    p.add_argument("--by-source-dir", default="processed_tsv/by_source_file",
                   help="Root dir with `<corpus>/<corpus>_{train,dev,test}.tsv`.")
    p.add_argument("--model-name", default="FacebookAI/xlm-roberta-base")
    p.add_argument("--max-length", type=int, default=256)

    p.add_argument("--train-batch-size", type=int, default=16)
    p.add_argument("--eval-batch-size", type=int, default=32)
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--warmup-ratio", type=float, default=0.1)

    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")

    p.add_argument("--max-train-examples", type=int, default=None)
    p.add_argument("--max-dev-examples", type=int, default=None)
    p.add_argument("--max-test-examples", type=int, default=None)

    p.add_argument("--output-dir", default="xlmr_framework_results")
    p.add_argument(
        "--save-best",
        dest="save_best",
        action="store_true",
        default=True,
        help="Save best dev checkpoint (default: True).",
    )
    p.add_argument(
        "--no-save-best",
        dest="save_best",
        action="store_false",
        help="Do not save the best dev checkpoint.",
    )
    p.add_argument("--no-mask", action="store_true",
                   help="Disable output masking (token conditioning only — ablation).")
    p.add_argument("--skip-per-dataset", action="store_true",
                   help="Skip per-corpus test evaluation.")
    return p.parse_args()

def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    by_source_dir = Path(args.by_source_dir)
    if not by_source_dir.exists():
        raise FileNotFoundError(f"Missing by-source-file dir: {by_source_dir}")

    device = torch.device(args.device)

    print("Loading splits from per-corpus TSVs …")
    train_data = load_split_by_corpus(by_source_dir, "train", args.max_train_examples)
    dev_data = load_split_by_corpus(by_source_dir, "dev", args.max_dev_examples)
    test_data = load_split_by_corpus(by_source_dir, "test", args.max_test_examples)
    if not train_data.labels:
        raise ValueError(f"No training rows found under {by_source_dir}.")

    label_set = sorted(set(train_data.labels) | set(dev_data.labels) | set(test_data.labels))
    label2id = {lab: i for i, lab in enumerate(label_set)}
    id2label = {i: lab for lab, i in label2id.items()}
    num_labels = len(label_set)
    print(f"  {num_labels} unique labels across all corpora.")

    framework_label_map = build_framework_label_map([train_data, dev_data, test_data], label2id)
    logit_masks = build_logit_masks(framework_label_map, num_labels)
    frameworks_found = sorted(framework_label_map.keys())
    print(f"  Frameworks detected: {frameworks_found}")
    for fw in frameworks_found:
        ids = sorted(framework_label_map[fw])
        print(f"    [{fw}]  {len(ids)} labels: {[id2label[i] for i in ids]}")

    print("\nLoading tokenizer …")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)
    num_added = tokenizer.add_special_tokens({"additional_special_tokens": FRAMEWORK_TOKENS})
    print(f"  Added {num_added} framework special tokens: {FRAMEWORK_TOKENS}")

    print("Loading model …")
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name,
        num_labels=num_labels,
        id2label=id2label,
        label2id=label2id,
    )
    model.resize_token_embeddings(len(tokenizer))
    model.to(device)

    mk_ds = lambda d: FrameworkPairDataset(d, label2id, tokenizer, args.max_length)
    train_loader = DataLoader(mk_ds(train_data), batch_size=args.train_batch_size, shuffle=True,  collate_fn=collate_fn)
    dev_loader = DataLoader(mk_ds(dev_data), batch_size=args.eval_batch_size, shuffle=False, collate_fn=collate_fn)
    test_loader = DataLoader(mk_ds(test_data), batch_size=args.eval_batch_size, shuffle=False, collate_fn=collate_fn)

    no_decay = ["bias", "LayerNorm.weight"]
    optimizer = AdamW(
        [
            {"params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
             "weight_decay": args.weight_decay},
            {"params": [p for n, p in model.named_parameters() if     any(nd in n for nd in no_decay)],
             "weight_decay": 0.0},
        ],
        lr=args.lr,
    )
    t_total = math.ceil(len(train_loader)) * args.epochs
    warmup_steps = int(args.warmup_ratio * t_total)
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, t_total)

    apply_mask = not args.no_mask
    best_dev_macro_f1 = -1.0
    best_state_dict: Optional[Dict[str, torch.Tensor]] = None
    history: List[Dict[str, Any]] = []

    print(f"\nTraining for {args.epochs} epoch(s). Output masking: {apply_mask}\n")
    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss = 0.0
        for batch in train_loader:
            batch = move_batch_to_device(batch, device)
            batch.pop("framework")
            outputs = model(**batch)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            epoch_loss += float(loss.item())

        avg_loss = epoch_loss / max(1, len(train_loader))
        dev_pooled = evaluate_pooled(model, dev_loader, device, num_labels, logit_masks, apply_mask=apply_mask)
        history.append({
            "epoch": epoch,
            "train_loss": avg_loss,
            "dev_accuracy": dev_pooled["accuracy"],
            "dev_macro_f1": dev_pooled["macro_f1"],
        })
        print(f"Epoch {epoch}/{args.epochs}  loss={avg_loss:.4f}  "
              f"dev_acc={dev_pooled['accuracy']:.4f}  dev_macro_f1={dev_pooled['macro_f1']:.4f}")

        if dev_pooled["macro_f1"] > best_dev_macro_f1:
            best_dev_macro_f1 = dev_pooled["macro_f1"]
            best_state_dict = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    if best_state_dict is not None:
        model.load_state_dict({k: v.to(device) for k, v in best_state_dict.items()})

    # ── Final pooled metrics ─────────────────────────────────────────────────
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    print("\nRunning final evaluation on dev and test (pooled) …")
    dev_full = evaluate_pooled(model, dev_loader, device, num_labels, logit_masks, apply_mask=apply_mask)
    test_full = evaluate_pooled(model, test_loader, device, num_labels, logit_masks, apply_mask=apply_mask)

    payload = {
        "setup": {
            "model_name": args.model_name,
            "framework_tokens": FRAMEWORK_TOKENS,
            "output_masking": apply_mask,
            "max_length": args.max_length,
            "num_labels": num_labels,
            "label_set": label_set,
            "frameworks_found": frameworks_found,
            "framework_label_map": {fw: sorted(ids) for fw, ids in framework_label_map.items()},
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
    print(f"Saved metrics → {out_dir / 'metrics.json'}")

    if args.save_best and best_state_dict is not None:
        best_dir = out_dir / "best_model"
        model.save_pretrained(best_dir)
        tokenizer.save_pretrained(best_dir)
        # Persist the masks + framework map alongside the HF checkpoint so
        # `evaluate_with_official_script.py` and other re-evaluators can reuse
        # them without recomputing from the pooled training data.
        aux = {
            "framework_tokens": FRAMEWORK_TOKENS,
            "framework_label_map": {fw: sorted(ids) for fw, ids in framework_label_map.items()},
            "label_set": label_set,
            "label2id": label2id,
            "output_masking_default": apply_mask,
        }
        (best_dir / "framework_conditioning.json").write_text(
            json.dumps(aux, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        print(f"Saved best model → {best_dir}")

    # ── Per-dataset test evaluation (4-way) ─────────────────────────────────
    if not args.skip_per_dataset:
        print(f"\nRunning per-dataset test evaluation from {by_source_dir} …")
        preds_by_corpus = predict_per_corpus(
            model=model,
            tokenizer=tokenizer,
            label2id=label2id,
            device=device,
            logit_masks=logit_masks,
            max_length=args.max_length,
            eval_batch_size=args.eval_batch_size,
            by_source_dir=by_source_dir,
            split="test",
            apply_mask=apply_mask,
        )
        agg = aggregate_predictions(preds_by_corpus, id2label)
        print_aggregate_summary(agg)

        per_dataset_path = out_dir / "per_dataset_test_metrics.json"
        per_dataset_payload = {
            "model": f"Fine-tuned XLM-RoBERTa (framework-conditioned, mask={apply_mask})",
            "model_name": args.model_name,
            "max_length": args.max_length,
            "label_set": label_set,
            "framework_tokens": FRAMEWORK_TOKENS,
            "output_masking": apply_mask,
            **agg,
        }
        per_dataset_path.write_text(
            json.dumps(per_dataset_payload, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        print(f"\nSaved per-dataset metrics → {per_dataset_path}")


if __name__ == "__main__":
    main()
