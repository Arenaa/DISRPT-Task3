"""Render the four-way per-dataset breakdown as CSVs and PNG figures.

Consumes `per_dataset_test_metrics.json` produced by:
    * benchmark_frozen_xlm_roberta_linear.py              (frozen + linear head)
    * finetune_xlm_roberta.py                             (full fine-tune)
    * finetune_xlm_roberta_framework_conditioned.py       (framework-conditioned fine-tune)
    * evaluate_per_dataset.py                             (standalone re-eval; writes to results/per_dataset_eval/)

Outputs (under --out-dir, default `results/per_dataset_plots`):

    summary_per_corpus.csv
    summary_per_framework.csv
    summary_per_language.csv
    per_label_global.csv
    per_label_by_corpus.csv         (long-form)
    per_label_by_framework.csv      (long-form)
    per_label_by_language.csv       (long-form)

    corpus_metrics.png              (bar chart: acc / macro-F1 / weighted-F1)
    framework_metrics.png
    language_metrics.png
    per_label_global_f1.png         (horizontal bar)

Up to four models: --frozen-json / --finetune-json / --finetune-tsv-json /
--framework-cond-json. Any JSON that is missing is quietly skipped.

If `--finetune-tsv-json` does not exist, the same directory is searched for
`per_dataset_test_metrics.json` when its `model` field indicates the TSV-feature run
(see `tsv_feature_eval_path` in `export_model_comparison_tables.py`).
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from export_model_comparison_tables import tsv_feature_eval_path


MODEL_COLORS = {
    "frozen":         "#4C72B0",
    "finetune":       "#DD8452",
    "finetune_tsv":   "#9467BD",
    "framework_cond": "#55A868",
}
MODEL_LABELS = {
    "frozen":         "Frozen + linear",
    "finetune":       "Fine-tuned",
    "finetune_tsv":   "Fine-tuned + TSV",
    "framework_cond": "Framework-cond.",
}


def get_plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def load_report(path: Path | None) -> dict | None:
    if path is None:
        return None
    if not path.exists():
        print(f"[warn] {path} not found — skipping")
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def write_csv(path: Path, header: list[str], rows: list[list]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)


def _summary_rows(block: dict, model_name: str, group_key: str) -> list[list]:
    out = []
    for name, r in sorted(block.items()):
        out.append([
            model_name, group_key, name,
            r.get("support", 0),
            r.get("num_gold_labels", 0),
            r.get("accuracy", 0.0),
            r.get("macro_f1", 0.0),
            r.get("weighted_f1", 0.0),
            ",".join(r.get("corpora", [])) if "corpora" in r else "",
        ])
    return out


def _per_label_rows(block: dict, model_name: str, group_key: str) -> list[list]:
    out = []
    for group_name, r in sorted(block.items()):
        for lab, m in sorted(r.get("per_label", {}).items()):
            out.append([
                model_name, group_key, group_name, lab,
                m.get("support", 0),
                m.get("precision", 0.0),
                m.get("recall", 0.0),
                m.get("f1", 0.0),
            ])
    return out


def _bar_group(plt, ax, labels, values_by_model, title, ylabel):
    """values_by_model = {model_name: [v per label] or None}."""
    present = [(m, vs) for m, vs in values_by_model.items() if vs is not None]
    n = len(present)
    if n == 0:
        ax.set_visible(False)
        return
    x = np.arange(len(labels))
    total_w = 0.8
    w = total_w / n
    for i, (m, vs) in enumerate(present):
        offset = -total_w / 2 + w / 2 + i * w
        ax.bar(x + offset, vs, width=w, label=MODEL_LABELS[m], color=MODEL_COLORS[m])
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=9)
    ax.set_ylim(0, 1.02)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.axhline(1.0, color="#cccccc", lw=0.6, ls="--")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(loc="upper right", fontsize=8)


def plot_group(out_path: Path, blocks: dict[str, dict | None], group_key: str) -> None:
    plt = get_plt()
    all_names: set[str] = set()
    for b in blocks.values():
        if b:
            all_names |= set(b.keys())
    names = sorted(all_names)
    if not names:
        print(f"[skip] no data for {group_key}")
        return

    def col(block: dict | None, key: str) -> list[float] | None:
        if block is None:
            return None
        return [float(block.get(n, {}).get(key, 0.0)) for n in names]

    n_mod = len([1 for b in blocks.values() if b is not None])
    wscale = 1.0 + 0.12 * max(0, n_mod - 3)
    fig, axes = plt.subplots(1, 3, figsize=(max(12, len(names) * 1.1) * wscale, 5.2))
    for ax, metric, pretty in zip(
        axes,
        ("accuracy", "macro_f1", "weighted_f1"),
        ("Accuracy", "Macro F1", "Weighted F1"),
    ):
        vals = {m: col(b, metric) for m, b in blocks.items()}
        _bar_group(plt, ax, names, vals, f"{pretty} by {group_key}", pretty)
    fig.suptitle(f"Per-{group_key} test performance")
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out_path}")


def plot_per_label_global(out_path: Path, blocks: dict[str, dict | None]) -> None:
    plt = get_plt()
    labels_all: set[str] = set()
    for b in blocks.values():
        if b:
            labels_all |= set(b.keys())

    def get(block: dict | None, lab: str, key: str) -> float:
        if block is None:
            return 0.0
        return float(block.get(lab, {}).get(key, 0.0))

    rows: list[tuple] = []
    for lab in labels_all:
        sup = max(get(b, lab, "support") for b in blocks.values() if b is not None)
        if sup <= 0:
            continue
        f1s = {m: get(b, lab, "f1") for m, b in blocks.items()}
        sort_key = (
            f1s.get("finetune", 0.0),
            f1s.get("finetune_tsv", 0.0),
            f1s.get("frozen", 0.0),
            f1s.get("framework_cond", 0.0),
        )
        rows.append((lab, sup, f1s, sort_key))
    rows.sort(key=lambda r: r[3])

    if not rows:
        print("[skip] no labels with support")
        return

    labs = [r[0] for r in rows]
    present = [m for m, b in blocks.items() if b is not None]
    n = len(present)
    total_h = 0.8
    h = total_h / n

    y = np.arange(len(labs))
    fig, ax = plt.subplots(figsize=(10, max(6, len(labs) * 0.22)))
    for i, m in enumerate(present):
        offset = -total_h / 2 + h / 2 + i * h
        vs = [r[2][m] for r in rows]
        ax.barh(y + offset, vs, height=h, label=f"{MODEL_LABELS[m]} (F1)", color=MODEL_COLORS[m])
    ax.set_yticks(y)
    ax.set_yticklabels(labs, fontsize=8)
    ax.set_xlabel("F1")
    ax.set_xlim(0, 1.02)
    ax.set_title("Per-label global test F1")
    ax.legend(loc="lower right")
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out_path}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--frozen-json", default="results/per_dataset_eval/frozen_test.json")
    ap.add_argument("--finetune-json", default="results/per_dataset_eval/finetune_test.json")
    ap.add_argument(
        "--finetune-tsv-json",
        default="results/per_dataset_eval/finetune_tsv_test.json",
        help="TSV-feature eval JSON; if missing, per_dataset_eval/per_dataset_test_metrics.json is used when model mentions TSV.",
    )
    ap.add_argument("--framework-cond-json", default="results/per_dataset_eval/framework_cond_test.json")
    ap.add_argument("--out-dir", default="results/per_dataset_plots")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    tsv_path: Path | None = Path(args.finetune_tsv_json) if args.finetune_tsv_json else None
    if tsv_path is not None and not tsv_path.is_file():
        alt = tsv_feature_eval_path(tsv_path.parent)
        if alt is not None:
            tsv_path = alt
    reports: dict[str, dict | None] = {
        "frozen":         load_report(Path(args.frozen_json)),
        "finetune":       load_report(Path(args.finetune_json)),
        "finetune_tsv":   load_report(tsv_path) if tsv_path else None,
        "framework_cond": load_report(Path(args.framework_cond_json)),
    }
    if all(v is None for v in reports.values()):
        raise SystemExit(
            "None of the JSONs were found. Run the training scripts or evaluate_per_dataset.py first."
        )

    active = [k for k, v in reports.items() if v is not None]
    print(
        "Plotting models:",
        ", ".join(MODEL_LABELS[m] for m in active),
        f"({len(active)} / 4)",
    )
    if tsv_path is not None and reports.get("finetune_tsv"):
        print(f"  TSV eval file: {tsv_path}")

    summary_header = [
        "model", "group_key", "name", "support", "num_gold_labels",
        "accuracy", "macro_f1", "weighted_f1", "corpora",
    ]
    rows_corpus: list[list] = []
    rows_fw: list[list] = []
    rows_lang: list[list] = []
    for mname, block in reports.items():
        if block is None:
            continue
        rows_corpus += _summary_rows(block["per_corpus"], mname, "corpus")
        rows_fw += _summary_rows(block["per_framework"], mname, "framework")
        rows_lang += _summary_rows(block["per_language"], mname, "language")
    write_csv(out_dir / "summary_per_corpus.csv", summary_header, rows_corpus)
    write_csv(out_dir / "summary_per_framework.csv", summary_header, rows_fw)
    write_csv(out_dir / "summary_per_language.csv", summary_header, rows_lang)

    pl_header = ["model", "group_key", "group_name", "label", "support", "precision", "recall", "f1"]
    rows_pl_global: list[list] = []
    rows_pl_corpus: list[list] = []
    rows_pl_fw: list[list] = []
    rows_pl_lang: list[list] = []
    for mname, block in reports.items():
        if block is None:
            continue
        for lab, m in sorted(block.get("per_label_global", {}).items()):
            rows_pl_global.append([
                mname, "global", "ALL", lab,
                m.get("support", 0),
                m.get("precision", 0.0), m.get("recall", 0.0), m.get("f1", 0.0),
            ])
        rows_pl_corpus += _per_label_rows(block["per_corpus"], mname, "corpus")
        rows_pl_fw += _per_label_rows(block["per_framework"], mname, "framework")
        rows_pl_lang += _per_label_rows(block["per_language"], mname, "language")
    write_csv(out_dir / "per_label_global.csv", pl_header, rows_pl_global)
    write_csv(out_dir / "per_label_by_corpus.csv", pl_header, rows_pl_corpus)
    write_csv(out_dir / "per_label_by_framework.csv", pl_header, rows_pl_fw)
    write_csv(out_dir / "per_label_by_language.csv", pl_header, rows_pl_lang)

    plot_group(
        out_dir / "corpus_metrics.png",
        {m: (b["per_corpus"] if b else None) for m, b in reports.items()},
        "corpus",
    )
    plot_group(
        out_dir / "framework_metrics.png",
        {m: (b["per_framework"] if b else None) for m, b in reports.items()},
        "framework",
    )
    plot_group(
        out_dir / "language_metrics.png",
        {m: (b["per_language"] if b else None) for m, b in reports.items()},
        "language",
    )
    plot_per_label_global(
        out_dir / "per_label_global_f1.png",
        {m: (b["per_label_global"] if b else None) for m, b in reports.items()},
    )

    print(f"\nDone. Wrote CSVs and plots under {out_dir}")


if __name__ == "__main__":
    main()
