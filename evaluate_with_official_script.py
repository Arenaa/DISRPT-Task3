"""Run official DISRPT 2025 evaluation on our trained models.

Pipeline per corpus:

    data_subset/<corpus>/<corpus>_<split>.rels  (gold, 15 cols, ~353-label space)
              │
              ▼  (1) iterate row-by-row, tokenize unit1_txt + unit2_txt, predict with
              │      the saved model (use --finetune-use-tsv-features if the run was
              │      finetune_xlm_roberta_tsv_features.py)
              │  (2) map both gold.label and pred to the DISRPT-25 17-label space
              │      using disrpt_official/mapping_disrpt25.json
              ▼
    results/official_eval/<model>/<corpus>/gold_mapped.rels   (same 15 cols, label∈VALID17)
    results/official_eval/<model>/<corpus>/system.rels
              │
              ▼  (3) invoke disrpt_official/disrpt_eval_2024.py -g gold -p system -t R
              ▼
    results/official_eval/<model>/<corpus>/eval.json

Then the script aggregates the predictions itself into per-framework,
per-language, and pooled-global metrics (all in the 17-label space) and writes
`results/official_eval/<model>/summary.json` + `.csv` summaries so you get the full
4-way breakdown under the official label space too.

Usage
-----

    python evaluate_with_official_script.py --mode frozen \
        --frozen-dir results/xlmr_frozen_linear_results

    python evaluate_with_official_script.py --mode finetune \
        --finetune-dir results/xlmr_finetune_results/best_model

    python evaluate_with_official_script.py --mode both
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import AutoModel, AutoModelForSequenceClassification, AutoTokenizer

from benchmark_frozen_xlm_roberta_linear import (
    PairTextDataset,
    SplitData,
    _slice_report,
    aggregate_predictions,
    compute_cls_pooled_embeddings,
    corpus_framework,
    corpus_language,
    print_aggregate_summary,
)
from disrpt_tsv_features import disrpt_feature_prefix_from_rels_row
from finetune_xlm_roberta import (
    PairDataset,
    _predict as ft_predict,
)
from finetune_xlm_roberta_framework_conditioned import (
    FRAMEWORK_TOKENS,
    FrameworkPairDataset,
    SplitData as FwSplitData,
    _predict as fw_predict,
    build_logit_masks,
    collate_fn as fw_collate_fn,
)


VALID17 = {
    "alternation", "attribution", "causal", "comment", "concession", "condition",
    "conjunction", "contrast", "elaboration", "explanation", "frame", "label",
    "mode", "organization", "purpose", "query", "reformulation", "temporal",
}

DISRPT_RELS_HEADER = (
    "doc\tunit1_toks\tunit2_toks\tunit1_txt\tunit2_txt\tu1_raw\tu2_raw\t"
    "s1_toks\ts2_toks\tunit1_sent\tunit2_sent\tdir\trel_type\torig_label\tlabel"
)


def load_mapping(path: Path) -> dict[str, str]:
    return json.loads(path.read_text(encoding="utf-8"))


def to17(label: str, mapping: dict[str, str]) -> str:
    """Collapse any label to the DISRPT-25 17-label space.

    Order: (a) identity if already in VALID17, (b) mapping lookup,
    (c) fallback 'label' (the official "unclassified" bucket).
    """
    if label in VALID17:
        return label
    if label in mapping:
        return mapping[label]
    return "label"



def read_rels(path: Path) -> tuple[list[list[str]], str]:
    """Return (rows, header). Each row is a list of 15 str."""
    csv.field_size_limit(sys.maxsize)
    lines = path.read_text(encoding="utf-8").splitlines()
    if not lines:
        raise ValueError(f"{path}: empty file")
    header = lines[0]
    rows = [line.split("\t") for line in lines[1:] if line.strip()]
    if rows and any(len(r) != 15 for r in rows):
        raise ValueError(f"{path}: expected 15 cols, got e.g. {len(rows[0])}")
    return rows, header


def write_rels(path: Path, rows: list[list[str]], header: str = DISRPT_RELS_HEADER) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    body = "\n".join("\t".join(r) for r in rows)
    path.write_text(header + "\n" + body + "\n", encoding="utf-8")



def _predict_rows_frozen(
    rows: list[list[str]],
    *,
    encoder,
    tokenizer,
    head: torch.nn.Module,
    label2id: dict[str, int],
    id2label: dict[int, str],
    device: torch.device,
    max_length: int,
    batch_size: int,
) -> list[str]:
    unit1 = [r[3] for r in rows]
    unit2 = [r[4] for r in rows]
    # PairTextDataset indexes label2id[label]; substitute any unseen label with
    # an arbitrary known one (labels are ignored at inference).
    default_label = next(iter(label2id))
    labels = [r[14] if r[14] in label2id else default_label for r in rows]

    split = SplitData(unit1=unit1, unit2=unit2, labels=labels)
    ds = PairTextDataset(split, label2id=label2id)
    X, _ = compute_cls_pooled_embeddings(
        encoder=encoder, tokenizer=tokenizer, dataset=ds,
        device=device, batch_size=batch_size, max_length=max_length, num_workers=0,
    )
    head.eval()
    with torch.no_grad():
        logits = head(X.to(device)).cpu().numpy()
    preds = logits.argmax(axis=1)
    return [id2label[int(i)] for i in preds]


def _predict_rows_finetune(
    rows: list[list[str]],
    *,
    model,
    tokenizer,
    label2id: dict[str, int],
    id2label: dict[int, str],
    device: torch.device,
    max_length: int,
    batch_size: int,
    use_tsv_features: bool = False,
) -> list[str]:
    if use_tsv_features:
        unit1 = []
        for r in rows:
            fp = disrpt_feature_prefix_from_rels_row(r)
            u1 = f"{fp}{r[3]}" if fp else r[3]
            unit1.append(u1)
    else:
        unit1 = [r[3] for r in rows]
    unit2 = [r[4] for r in rows]
    default_label = next(iter(label2id))
    labels = [r[14] if r[14] in label2id else default_label for r in rows]

    split = SplitData(unit1=unit1, unit2=unit2, labels=labels)
    ds = PairDataset(split, label2id=label2id, tokenizer=tokenizer, max_length=max_length)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False)
    _, y_pred = ft_predict(model, loader, device=device)
    return [id2label[int(i)] for i in y_pred]


def _predict_rows_framework_cond(
    rows: list[list[str]],
    corpus: str,
    *,
    model,
    tokenizer,
    label2id: dict[str, int],
    id2label: dict[int, str],
    device: torch.device,
    max_length: int,
    batch_size: int,
    logit_masks: dict[str, torch.Tensor],
    apply_mask: bool,
) -> list[str]:
    unit1 = [r[3] for r in rows]
    unit2 = [r[4] for r in rows]
    default_label = next(iter(label2id))
    labels = [r[14] if r[14] in label2id else default_label for r in rows]
    fw = corpus_framework(corpus)
    lg = corpus_language(corpus)

    split = FwSplitData(
        unit1=unit1, unit2=unit2, labels=labels,
        corpora=[corpus] * len(rows),
        frameworks=[fw] * len(rows),
        languages=[lg] * len(rows),
    )
    ds = FrameworkPairDataset(split, label2id, tokenizer, max_length)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, collate_fn=fw_collate_fn)
    _, y_pred, _ = fw_predict(model, loader, device=device, logit_masks=logit_masks, apply_mask=apply_mask)
    return [id2label[int(i)] for i in y_pred]



def run_official_scorer(
    *,
    script: Path,
    gold: Path,
    pred: Path,
) -> dict[str, Any]:
    cmd = [sys.executable, str(script), "-g", str(gold), "-p", str(pred), "-t", "R"]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(
            f"Official scorer failed on {gold.name}\n"
            f"STDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
        )
    return json.loads(result.stdout)



def evaluate_model_officially(
    *,
    model_tag: str,
    predict_rows,
    mapping: dict[str, str],
    data_dir: Path,
    split: str,
    out_root: Path,
    official_script: Path,
) -> dict[str, Any]:
    out_root.mkdir(parents=True, exist_ok=True)
    corpora = sorted(p.name for p in data_dir.iterdir() if p.is_dir())

    per_corpus: dict[str, dict[str, Any]] = {}
    all_y_true: list[str] = []
    all_y_pred: list[str] = []
    by_framework: dict[str, list[tuple[list[str], list[str]]]] = {}
    by_language: dict[str, list[tuple[list[str], list[str]]]] = {}

    for corpus in corpora:
        gold_raw = data_dir / corpus / f"{corpus}_{split}.rels"
        if not gold_raw.exists():
            print(f"[skip] {corpus}: no {split} split at {gold_raw}")
            continue

        rows, header = read_rels(gold_raw)
        if header != DISRPT_RELS_HEADER:
            print(f"[warn] {corpus}: non-canonical header; overriding for output")

        gold_41 = [r[14] for r in rows]
        pred_41 = predict_rows(rows, corpus)

        if len(pred_41) != len(rows):
            raise RuntimeError(
                f"{corpus}: predictor returned {len(pred_41)} labels for {len(rows)} rows"
            )

        gold_17 = [to17(g, mapping) for g in gold_41]
        pred_17 = [to17(p, mapping) for p in pred_41]

        corpus_out = out_root / corpus
        corpus_out.mkdir(parents=True, exist_ok=True)

        gold_rows = [r.copy() for r in rows]
        for i, r in enumerate(gold_rows):
            r[14] = gold_17[i]
        gold_out = corpus_out / f"{corpus}_{split}.gold17.rels"
        write_rels(gold_out, gold_rows)

        sys_rows = [r.copy() for r in rows]
        for i, r in enumerate(sys_rows):
            r[14] = pred_17[i]
        sys_out = corpus_out / f"{corpus}_{split}.system.rels"
        write_rels(sys_out, sys_rows)

        official = run_official_scorer(script=official_script, gold=gold_out, pred=sys_out)
        (corpus_out / "eval.json").write_text(json.dumps(official, indent=2, ensure_ascii=False), encoding="utf-8")

        acc_key = "labels_accuracy"
        rep_key = "labels_classification_report"
        acc = float(official.get(acc_key, 0.0))
        report = official.get(rep_key, {})
        per_corpus[corpus] = {
            "framework": corpus_framework(corpus),
            "language": corpus_language(corpus),
            "support": len(gold_17),
            "accuracy_official": acc,
            "macro_f1_official": float(report.get("macro avg", {}).get("f1-score", 0.0)),
            "weighted_f1_official": float(report.get("weighted avg", {}).get("f1-score", 0.0)),
            "per_label_official": {
                name: {
                    "precision": float(v.get("precision", 0.0)),
                    "recall": float(v.get("recall", 0.0)),
                    "f1": float(v.get("f1-score", 0.0)),
                    "support": int(v.get("support", 0)),
                }
                for name, v in report.items()
                if name not in {"accuracy", "macro avg", "weighted avg"} and isinstance(v, dict)
            },
        }
        print(
            f"[{corpus:20s}] n={len(gold_17):5d}  "
            f"acc={acc:.4f}  macroF1={per_corpus[corpus]['macro_f1_official']:.4f}  "
            f"weightedF1={per_corpus[corpus]['weighted_f1_official']:.4f}"
        )

        all_y_true.extend(gold_17)
        all_y_pred.extend(pred_17)
        by_framework.setdefault(corpus_framework(corpus), []).append((gold_17, pred_17))
        by_language.setdefault(corpus_language(corpus), []).append((gold_17, pred_17))

    def _agg(groups, group_kind):
        out = {}
        for k, items in sorted(groups.items()):
            yt = [x for g, _ in items for x in g]
            yp = [x for _, p in items for x in p]
            out[k] = _slice_report_from_strings(yt, yp)
            out[k]["corpora"] = sorted(c for c, r in per_corpus.items() if r.get(group_kind) == k)
        return out

    per_framework = _agg(by_framework, "framework")
    per_language = _agg(by_language, "language")
    pooled = _slice_report_from_strings(all_y_true, all_y_pred)

    print_pretty_summary(per_corpus, per_framework, per_language, pooled)

    summary = {
        "model": model_tag,
        "label_space": "DISRPT-25 17-label",
        "valid_labels": sorted(VALID17),
        "per_corpus": per_corpus,
        "per_framework": per_framework,
        "per_language": per_language,
        "pooled": pooled,
    }
    (out_root / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    _write_summary_csvs(out_root, per_corpus, per_framework, per_language, pooled)

    print(f"\nSaved official summary to {out_root / 'summary.json'}")
    return summary


def _slice_report_from_strings(y_true: list[str], y_pred: list[str]) -> dict[str, Any]:
    """Reuse the numeric slice helper by encoding strings → ids."""
    vocab = sorted(set(y_true) | set(y_pred))
    l2i = {lab: i for i, lab in enumerate(vocab)}
    i2l = {i: lab for lab, i in l2i.items()}
    yt = np.array([l2i[l] for l in y_true], dtype=np.int64)
    yp = np.array([l2i[l] for l in y_pred], dtype=np.int64)
    return _slice_report(yt, yp, i2l)


def print_pretty_summary(per_corpus, per_framework, per_language, pooled) -> None:
    def _row(name, r, acc_key="accuracy", mf1_key="macro_f1", wf1_key="weighted_f1"):
        return (
            f"  {name:24s} n={r['support']:>6d}  "
            f"acc={r[acc_key]:.4f}  macroF1={r[mf1_key]:.4f}  weightedF1={r[wf1_key]:.4f}"
        )
    print("\n========  OFFICIAL DISRPT-25 evaluation (17-class space)  ========")
    print("\n-- Per corpus (official scorer) --")
    for c, r in sorted(per_corpus.items()):
        print(_row(c, r, "accuracy_official", "macro_f1_official", "weighted_f1_official"))
    print("\n-- Per framework --")
    for f, r in sorted(per_framework.items()):
        print(_row(f, r))
    print("\n-- Per language --")
    for l, r in sorted(per_language.items()):
        print(_row(l, r))
    print("\n-- Pooled global (17-class) --")
    print(_row("ALL", pooled))


def _write_summary_csvs(out_root, per_corpus, per_framework, per_language, pooled) -> None:
    with (out_root / "summary_per_corpus.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["corpus", "framework", "language", "support", "accuracy", "macro_f1", "weighted_f1"])
        for c, r in sorted(per_corpus.items()):
            w.writerow([c, r["framework"], r["language"], r["support"],
                        r["accuracy_official"], r["macro_f1_official"], r["weighted_f1_official"]])
    with (out_root / "summary_per_framework.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["framework", "support", "accuracy", "macro_f1", "weighted_f1", "corpora"])
        for k, r in sorted(per_framework.items()):
            w.writerow([k, r["support"], r["accuracy"], r["macro_f1"], r["weighted_f1"], ",".join(r["corpora"])])
    with (out_root / "summary_per_language.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["language", "support", "accuracy", "macro_f1", "weighted_f1", "corpora"])
        for k, r in sorted(per_language.items()):
            w.writerow([k, r["support"], r["accuracy"], r["macro_f1"], r["weighted_f1"], ",".join(r["corpora"])])
    with (out_root / "per_label_global.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["label", "support", "precision", "recall", "f1"])
        for lab, m in sorted(pooled["per_label"].items()):
            w.writerow([lab, m["support"], m["precision"], m["recall"], m["f1"]])



def load_frozen(frozen_dir: Path, device: torch.device):
    ckpt_path = frozen_dir / "linear_head.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(
            f"{ckpt_path} missing; re-run benchmark_frozen_xlm_roberta_linear.py first."
        )
    ckpt = torch.load(ckpt_path, map_location="cpu")
    label_set = ckpt["label_set"]
    label2id = ckpt["label2id"]
    id2label = {i: lab for lab, i in label2id.items()}
    model_name = ckpt["model_name"]
    max_length = int(ckpt.get("max_length", 256))

    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    encoder = AutoModel.from_pretrained(model_name)
    for p in encoder.parameters():
        p.requires_grad = False
    encoder.to(device)

    sd = ckpt["state_dict"]
    hidden = sd["weight"].shape[1]
    head = torch.nn.Linear(hidden, len(label_set)).to(device)
    head.load_state_dict(sd)
    head.eval()

    return encoder, tokenizer, head, label2id, id2label, max_length


def load_finetune(model_dir: Path, device: torch.device):
    if not model_dir.exists():
        raise FileNotFoundError(
            f"{model_dir} missing; re-run finetune_xlm_roberta.py first."
        )
    tokenizer = AutoTokenizer.from_pretrained(model_dir, use_fast=True)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    model.to(device)
    label2id = {str(k): int(v) for k, v in model.config.label2id.items()}
    id2label = {int(v): k for k, v in label2id.items()}
    return model, tokenizer, label2id, id2label


def load_framework_cond(model_dir: Path, device: torch.device):
    """Load HF checkpoint + `framework_conditioning.json` produced by
    finetune_xlm_roberta_framework_conditioned.py, and rebuild the logit masks."""
    if not model_dir.exists():
        raise FileNotFoundError(
            f"{model_dir} missing; re-run finetune_xlm_roberta_framework_conditioned.py first."
        )
    aux_path = model_dir / "framework_conditioning.json"
    if not aux_path.exists():
        raise FileNotFoundError(
            f"{aux_path} missing; this directory was not written by the aligned "
            f"framework-conditioned script. Retrain with that script to regenerate it."
        )
    aux = json.loads(aux_path.read_text(encoding="utf-8"))
    label_set: list[str] = aux["label_set"]
    label2id: dict[str, int] = {str(k): int(v) for k, v in aux["label2id"].items()}
    id2label: dict[int, str] = {v: k for k, v in label2id.items()}
    framework_label_map: dict[str, set[int]] = {
        fw: set(int(i) for i in ids) for fw, ids in aux["framework_label_map"].items()
    }
    num_labels = len(label_set)
    logit_masks = build_logit_masks(framework_label_map, num_labels)

    tokenizer = AutoTokenizer.from_pretrained(model_dir, use_fast=True)
    # Framework tokens must already be in the tokenizer vocab (added at train time).
    missing = [t for t in FRAMEWORK_TOKENS if t not in tokenizer.get_vocab()]
    if missing:
        print(f"[warn] framework tokens {missing} absent from tokenizer vocab; "
              f"the model may not have been trained with this aligned script.")

    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    model.to(device)
    return model, tokenizer, label2id, id2label, logit_masks



def _expand_modes(mode: str) -> list[str]:
    if mode == "all":
        return ["frozen", "finetune", "framework_cond"]
    if mode == "both":
        return ["frozen", "finetune"]
    return [mode]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--mode",
                   choices=["frozen", "finetune", "framework_cond", "both", "all"],
                   required=True,
                   help="'both' = frozen + finetune; 'all' = all three.")
    p.add_argument("--frozen-dir", default="results/xlmr_frozen_linear_results")
    p.add_argument("--finetune-dir", default="results/xlmr_finetune_results/best_model")
    p.add_argument("--framework-cond-dir", default="results/xlmr_framework_results/best_model",
                   help="Directory saved by finetune_xlm_roberta_framework_conditioned.py "
                        "(must contain the HF model + framework_conditioning.json).")
    p.add_argument("--fc-no-mask", action="store_true",
                   help="Disable output masking for the framework-conditioned model "
                        "(uses the stored map only for decoding reference).")
    p.add_argument("--data-dir", default="data_subset",
                   help="Root with per-corpus `<corpus>/<corpus>_<split>.rels` files.")
    p.add_argument("--split", default="test", choices=["train", "dev", "test"])
    p.add_argument("--mapping", default="disrpt_official/mapping_disrpt25.json")
    p.add_argument("--official-script", default="disrpt_official/disrpt_eval_2024.py")
    p.add_argument("--out-root", default="results/official_eval")
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--max-length", type=int, default=256,
                   help="Overrides the checkpoint-saved max_length when needed.")
    p.add_argument("--finetune-use-tsv-features", action="store_true",
                   help="Match training from finetune_xlm_roberta_tsv_features.py: prepend "
                        "dir / rel_type to unit1 (no orig_label; plain finetune = omit this).")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    mapping = load_mapping(Path(args.mapping))
    official_script = Path(args.official_script).resolve()
    data_dir = Path(args.data_dir)
    out_root = Path(args.out_root)
    device = torch.device(args.device)

    modes = _expand_modes(args.mode)

    if "frozen" in modes:
        print("\n=== Evaluating Frozen + linear head (official scorer) ===")
        encoder, tokenizer, head, label2id, id2label, max_length = load_frozen(Path(args.frozen_dir), device)

        def predict_rows(rows, corpus):
            return _predict_rows_frozen(
                rows, encoder=encoder, tokenizer=tokenizer, head=head,
                label2id=label2id, id2label=id2label, device=device,
                max_length=min(max_length, args.max_length), batch_size=args.batch_size,
            )

        evaluate_model_officially(
            model_tag="Frozen XLM-R + linear head",
            predict_rows=predict_rows,
            mapping=mapping,
            data_dir=data_dir,
            split=args.split,
            out_root=out_root / "frozen",
            official_script=official_script,
        )

    if "finetune" in modes:
        print("\n=== Evaluating Fine-tuned XLM-R (official scorer) ===")
        model, tokenizer, label2id, id2label = load_finetune(Path(args.finetune_dir), device)

        def predict_rows(rows, corpus):
            return _predict_rows_finetune(
                rows, model=model, tokenizer=tokenizer,
                label2id=label2id, id2label=id2label, device=device,
                max_length=args.max_length, batch_size=args.batch_size,
                use_tsv_features=args.finetune_use_tsv_features,
            )

        evaluate_model_officially(
            model_tag="Fine-tuned XLM-R"
            + (" + TSV features" if args.finetune_use_tsv_features else ""),
            predict_rows=predict_rows,
            mapping=mapping,
            data_dir=data_dir,
            split=args.split,
            out_root=out_root / "finetune",
            official_script=official_script,
        )

    if "framework_cond" in modes:
        print("\n=== Evaluating Framework-conditioned XLM-R (official scorer) ===")
        model, tokenizer, label2id, id2label, logit_masks = load_framework_cond(
            Path(args.framework_cond_dir), device
        )
        apply_mask = not args.fc_no_mask

        def predict_rows(rows, corpus):
            return _predict_rows_framework_cond(
                rows, corpus,
                model=model, tokenizer=tokenizer,
                label2id=label2id, id2label=id2label, device=device,
                max_length=args.max_length, batch_size=args.batch_size,
                logit_masks=logit_masks, apply_mask=apply_mask,
            )

        evaluate_model_officially(
            model_tag=f"Framework-conditioned XLM-R (mask={apply_mask})",
            predict_rows=predict_rows,
            mapping=mapping,
            data_dir=data_dir,
            split=args.split,
            out_root=out_root / "framework_cond",
            official_script=official_script,
        )


if __name__ == "__main__":
    main()
