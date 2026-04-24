"""Join per-dataset eval JSONs + optional TSV-feature metrics into comparison CSVs.

Default outputs: results/comparison/

* comparison_pooled_test.csv
* comparison_per_corpus_test.csv  (macro F1 + acc per corpus; 4th column may be empty without finetune_tsv eval)
* per_corpus_four_models_test.csv — one row per corpus, **all 4 models**: acc / macro F1 / weighted F1 + support
* per_corpus_three_models_test.csv — **only** 3 models (no TSV)

TSV-feature rows use, in order: (1) `finetune_tsv_test.json` from
`evaluate_per_dataset.py --mode finetune_tsv`, (2) `per_dataset_test_metrics.json` in the same
folder if its `model` field looks like the TSV run (e.g. contains "TSV"), e.g. from
`finetune_xlm_roberta_tsv_features.py` saving there. If neither, TSV pooled can fall back to
`results/xlmr_finetune_tsv_features_results/metrics.json` only, with per-corpus TSV left blank.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


def _pooled(m: dict) -> dict[str, Any]:
    p = m.get("pooled", {})
    return {
        "support": p.get("support", ""),
        "accuracy": p.get("accuracy", ""),
        "macro_f1": p.get("macro_f1", ""),
        "weighted_f1": p.get("weighted_f1", ""),
    }


def _pooled_from_train_metrics(mpath: Path) -> dict[str, Any]:
    m = json.loads(mpath.read_text(encoding="utf-8"))
    tm = m.get("test_metrics", {})
    rep = tm.get("classification_report", {})
    wf = float(rep.get("weighted avg", {}).get("f1-score", 0.0))
    sup = 0
    for k, v in rep.items():
        if k in ("accuracy", "macro avg", "weighted avg", "micro avg"):
            continue
        if isinstance(v, dict):
            sup += int(v.get("support", 0) or 0)
    return {
        "support": sup,
        "accuracy": float(tm.get("accuracy", 0.0)),
        "macro_f1": float(tm.get("macro_f1", 0.0)),
        "weighted_f1": wf,
    }


def corpus_sub(block: dict) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for k, v in block.items():
        if isinstance(v, dict) and "macro_f1" in v:
            out[k] = v
    return out


def tsv_feature_eval_path(ev: Path) -> Path | None:
    """Prefer `finetune_tsv_test.json`; else `per_dataset_test_metrics.json` if `model` looks TSV."""
    p1 = ev / "finetune_tsv_test.json"
    if p1.is_file():
        return p1
    p2 = ev / "per_dataset_test_metrics.json"
    if not p2.is_file():
        return None
    try:
        m = json.loads(p2.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if "tsv" in str(m.get("model", "")).lower():
        return p2
    return None


def load_tsv_feature_eval(ev: Path) -> tuple[dict | None, str | None]:
    p = tsv_feature_eval_path(ev)
    if p is None:
        return None, None
    m = json.loads(p.read_text(encoding="utf-8"))
    return m, p.name


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--per-dataset-dir", type=Path, default=Path("results/per_dataset_eval"))
    p.add_argument("--tsv-metrics", type=Path, default=Path("results/xlmr_finetune_tsv_features_results/metrics.json"))
    p.add_argument("--out-dir", type=Path, default=Path("results/comparison"))
    args = p.parse_args()
    ev = args.per_dataset_dir
    out = args.out_dir
    out.mkdir(parents=True, exist_ok=True)

    frozen = json.loads((ev / "frozen_test.json").read_text(encoding="utf-8"))
    finetune = json.loads((ev / "finetune_test.json").read_text(encoding="utf-8"))
    fw = json.loads((ev / "framework_cond_test.json").read_text(encoding="utf-8"))

    tsv_from_eval, tsv_filename = load_tsv_feature_eval(ev)

    tsv_pooled: dict[str, Any]
    if tsv_from_eval is not None:
        tsv_pooled = _pooled(tsv_from_eval)
    elif args.tsv_metrics.is_file():
        tsv_pooled = _pooled_from_train_metrics(args.tsv_metrics)
    else:
        tsv_pooled = {"support": "", "accuracy": "", "macro_f1": "", "weighted_f1": ""}

    tsv_corpus = corpus_sub(tsv_from_eval.get("per_corpus", {})) if tsv_from_eval else {}

    pcsv = out / "comparison_pooled_test.csv"
    with pcsv.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["model", "source", "support", "accuracy", "macro_f1", "weighted_f1"])
        w.writerow(["frozen", "per_dataset_eval", *_pooled(frozen).values()])
        w.writerow(["finetune", "per_dataset_eval", *_pooled(finetune).values()])
        w.writerow(["framework_cond", "per_dataset_eval", *_pooled(fw).values()])
        tsv_row_src = tsv_filename or (args.tsv_metrics.name if args.tsv_metrics.is_file() else "missing")
        w.writerow(["finetune_tsv", tsv_row_src, *tsv_pooled.values()])
    print(f"Wrote {pcsv}")

    c_fz = corpus_sub(frozen.get("per_corpus", {}))
    c_ft = corpus_sub(finetune.get("per_corpus", {}))
    c_fw = corpus_sub(fw.get("per_corpus", {}))
    all_c = sorted(set(c_fz) | set(c_ft) | set(c_fw) | set(tsv_corpus))

    corp_csv = out / "comparison_per_corpus_test.csv"
    with corp_csv.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "corpus",
            "frozen_macro_f1", "finetune_macro_f1", "framework_cond_macro_f1", "finetune_tsv_macro_f1",
            "frozen_acc", "finetune_acc", "framework_cond_acc", "finetune_tsv_acc",
        ])
        for c in all_c:
            def g(block: dict[str, Any], name: str, key: str) -> str:
                r = block.get(name) or {}
                v = r.get(key)
                if v is None or v == "":
                    return ""
                return f"{float(v):.6f}" if isinstance(v, (int, float)) else str(v)

            w.writerow([
                c,
                g(c_fz, c, "macro_f1"), g(c_ft, c, "macro_f1"), g(c_fw, c, "macro_f1"), g(tsv_corpus, c, "macro_f1"),
                g(c_fz, c, "accuracy"), g(c_ft, c, "accuracy"), g(c_fw, c, "accuracy"), g(tsv_corpus, c, "accuracy"),
            ])
    print(f"Wrote {corp_csv}")

    # --- 3-way per-corpus table (frozen / finetune / framework-cond only) ---
    three_path = out / "per_corpus_three_models_test.csv"
    all_three = sorted(set(c_fz) | set(c_ft) | set(c_fw))

    def g3(block: dict[str, Any], c: str, key: str) -> str:
        r = block.get(c) or {}
        v = r.get(key)
        if v is None or v == "":
            return ""
        return f"{float(v):.6f}" if isinstance(v, (int, float)) else str(v)

    with three_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "corpus",
            "acc_frozen", "acc_finetune", "acc_framework",
            "macro_f1_frozen", "macro_f1_finetune", "macro_f1_framework",
            "weighted_f1_frozen", "weighted_f1_finetune", "weighted_f1_framework",
            "support",
        ])
        for c in all_three:
            r_fz, r_ft, r_fw = c_fz.get(c, {}), c_ft.get(c, {}), c_fw.get(c, {})
            sup = r_fz.get("support") or r_ft.get("support") or r_fw.get("support")
            w.writerow([
                c,
                g3(c_fz, c, "accuracy"), g3(c_ft, c, "accuracy"), g3(c_fw, c, "accuracy"),
                g3(c_fz, c, "macro_f1"), g3(c_ft, c, "macro_f1"), g3(c_fw, c, "macro_f1"),
                g3(c_fz, c, "weighted_f1"), g3(c_ft, c, "weighted_f1"), g3(c_fw, c, "weighted_f1"),
                f"{int(sup)}" if sup not in (None, "") else "",
            ])
    print(f"Wrote {three_path}  (3 models: frozen, finetune, framework-cond)")

    # --- 4-way per-corpus: same shape as 3-way + finetune_tsv (TSV feature model) ---
    four_path = out / "per_corpus_four_models_test.csv"
    all_four = sorted(set(c_fz) | set(c_ft) | set(c_fw) | set(tsv_corpus))
    with four_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "corpus",
            "acc_frozen", "acc_finetune", "acc_framework", "acc_finetune_tsv",
            "macro_f1_frozen", "macro_f1_finetune", "macro_f1_framework", "macro_f1_finetune_tsv",
            "weighted_f1_frozen", "weighted_f1_finetune", "weighted_f1_framework", "weighted_f1_finetune_tsv",
            "support",
        ])
        for c in all_four:
            r_fz = c_fz.get(c, {})
            r_ft = c_ft.get(c, {})
            r_fw = c_fw.get(c, {})
            r_ts = tsv_corpus.get(c, {})
            sup = r_fz.get("support") or r_ft.get("support") or r_fw.get("support") or r_ts.get("support")
            w.writerow([
                c,
                g3(c_fz, c, "accuracy"), g3(c_ft, c, "accuracy"), g3(c_fw, c, "accuracy"), g3(tsv_corpus, c, "accuracy"),
                g3(c_fz, c, "macro_f1"), g3(c_ft, c, "macro_f1"), g3(c_fw, c, "macro_f1"), g3(tsv_corpus, c, "macro_f1"),
                g3(c_fz, c, "weighted_f1"), g3(c_ft, c, "weighted_f1"), g3(c_fw, c, "weighted_f1"), g3(tsv_corpus, c, "weighted_f1"),
                f"{int(sup)}" if sup not in (None, "") else "",
            ])
    if tsv_from_eval is not None:
        print(f"Wrote {four_path}  (4 models, TSV from {tsv_filename})")
    else:
        print(
            f"Wrote {four_path}  (4 models: finetune_tsv per-corpus empty; "
            "add finetune_tsv_test.json or TSV per_dataset_test_metrics.json, or xlmr metrics only)"
        )

    if not tsv_corpus and tsv_pooled.get("support") not in ("", None) and tsv_from_eval is None:
        print(
            "\n[info] finetune_tsv per-corpus cells are empty. "
            "Add results/per_dataset_eval/finetune_tsv_test.json (evaluate_per_dataset --mode finetune_tsv) "
            "or TSV per_dataset_test_metrics.json, or use training metrics.json for pooled TSV only."
        )


if __name__ == "__main__":
    main()
