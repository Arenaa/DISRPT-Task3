"""Compare up to four XLM-R models: frozen, plain finetune, TSV-feature finetune, framework-cond.

Loads metrics.json from each results/*_results/ dir (when present) and writes PNGs to results/xlmr_comparison_plots/."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def get_pyplot():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def load_metrics(path: Path) -> dict:
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def _f1_support(
    test_rep: dict,
    label: str,
    idx: int,
) -> tuple[float, float]:
    """Get (f1, support) for a class; handles string label keys or str(idx) from sklearn."""
    row = test_rep.get(label)
    if row is None:
        row = test_rep.get(str(idx), {})
    if not row:
        return 0.0, 0.0
    return float(row.get("f1-score", 0.0)), float(row.get("support", 0.0))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--root",
        type=Path,
        default=Path(__file__).resolve().parent,
        help="Project root (default: this repo).",
    )
    p.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Output directory for PNGs (default: <root>/results/xlmr_comparison_plots).",
    )
    p.add_argument(
        "--frozen-metrics",
        type=Path,
        default=None,
        help="default: <root>/results/xlmr_frozen_linear_results/metrics.json",
    )
    p.add_argument(
        "--finetune-metrics",
        type=Path,
        default=None,
        help="Plain fine-tune metrics JSON (default: <root>/results/xlmr_finetune_results/metrics.json if present).",
    )
    p.add_argument(
        "--finetune-tsv-metrics",
        type=Path,
        default=None,
        help="TSV-feature fine-tune metrics (default: <root>/results/xlmr_finetune_tsv_features_results/metrics.json if present).",
    )
    p.add_argument(
        "--framework-metrics",
        type=Path,
        default=None,
        help="Framework-conditioned fine-tune metrics (default: <root>/results/xlmr_framework_results/metrics.json if present).",
    )
    args = p.parse_args()
    root = args.root
    out_dir = args.out_dir or (root / "results" / "xlmr_comparison_plots")
    out_dir.mkdir(parents=True, exist_ok=True)

    frozen_path = args.frozen_metrics or (root / "results" / "xlmr_frozen_linear_results" / "metrics.json")
    if not frozen_path.is_file():
        raise SystemExit(f"Missing frozen metrics: {frozen_path}")

    finetune_path = args.finetune_metrics or (root / "results" / "xlmr_finetune_results" / "metrics.json")
    tsv_path = args.finetune_tsv_metrics or (root / "results" / "xlmr_finetune_tsv_features_results" / "metrics.json")
    fw_path = args.framework_metrics or (root / "results" / "xlmr_framework_results" / "metrics.json")

    models: list[tuple[str, dict, str]] = []
    frozen = load_metrics(frozen_path)
    models.append(("Frozen + linear", frozen, "#4C72B0"))

    if finetune_path.is_file():
        models.append(("Fine-tuned", load_metrics(finetune_path), "#DD8452"))
    if tsv_path.is_file():
        models.append(("Fine-tuned + TSV features", load_metrics(tsv_path), "#9467BD"))
    if fw_path.is_file():
        models.append(("Framework-cond", load_metrics(fw_path), "#55A868"))

    if len(models) < 2:
        raise SystemExit(
            f"Need frozen plus at least one of:\n  {finetune_path}\n  {tsv_path}\n  {fw_path}\n"
            "or pass explicit paths to metrics JSON files."
        )

    label_set: list[str] = (
        frozen.get("label_set")
        or frozen.get("setup", {}).get("label_set")
        or []
    )
    for name, m, _ in models[1:]:
        other = m.get("setup", {}).get("label_set")
        if other and other != label_set:
            print(f"[warn] {name}: label_set differs from frozen — using frozen order")

    plt = get_pyplot()

    # --- Figure 1: overall dev/test bars (all available models) ---
    metrics = [
        ("Dev accuracy", "dev", "accuracy"),
        ("Test accuracy", "test", "accuracy"),
        ("Dev macro F1", "dev", "macro_f1"),
        ("Test macro F1", "test", "macro_f1"),
        ("Test weighted F1", "test", "weighted_cls"),
    ]
    n_metrics = len(metrics)
    n_m = len(models)
    w = 0.8 / n_m
    x = np.arange(n_metrics)
    fig, ax = plt.subplots(figsize=(max(9, n_metrics * 1.1), 4.5 + 0.3 * n_m))
    for mi, (mname, mdata, color) in enumerate(models):
        offset = -0.4 + w / 2 + mi * w
        vals: list[float] = []
        for title, split, key in metrics:
            if key == "weighted_cls":
                block = mdata.get(f"{split}_metrics", {})
                rep = block.get("classification_report", {})
                v = float(rep.get("weighted avg", {}).get("f1-score", 0.0))
            else:
                v = float(mdata.get(f"{split}_metrics", {}).get(key, 0.0))
            vals.append(v)
        ax.bar(
            x + offset,
            vals,
            width=w * 0.95,
            label=mname,
            color=color,
        )
    ax.set_xticks(x)
    ax.set_xticklabels([m[0] for m in metrics], rotation=15, ha="right")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Score")
    ax.set_title("XLM-RoBERTa-base: model comparison (dev / test)")
    ax.legend(loc="upper left", fontsize=8)
    ax.axhline(1.0, color="#bbbbbb", linewidth=0.8, linestyle="--")
    fig.tight_layout()
    fig.savefig(out_dir / "overall_metrics.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    # --- Figure 2: per-class test F1 (rows = labels, cols = models) ---
    rows_f1: list[list[float]] = []
    used_labels: list[str] = []
    for i, lab in enumerate(label_set):
        row: list[float] = []
        has_support = False
        for mname, mdata, _ in models:
            rep = mdata.get("test_metrics", {}).get("classification_report", {})
            f1, sup = _f1_support(rep, lab, i)
            row.append(f1)
            if sup > 0:
                has_support = True
        if has_support:
            used_labels.append(lab)
            rows_f1.append(row)

    if not used_labels:
        print("[skip] per-class F1: no classes with test support")
    else:
        if len(models) >= 2:
            # Sort by standard fine-tune gain vs frozen (index 1) when present
            ref = 1 if len(rows_f1[0]) > 1 else 0
            order = np.argsort([row[ref] - row[0] for row in rows_f1])
        else:
            order = np.argsort([row[0] for row in rows_f1])
        used_labels = [used_labels[i] for i in order]
        rows_f1 = [rows_f1[i] for i in order]

        y = np.arange(len(used_labels))
        fig, ax = plt.subplots(figsize=(10, max(6, len(used_labels) * 0.22)))
        n_m = len(models)
        h = 0.8 / max(n_m, 1)
        for mi, (mname, _, color) in enumerate(models):
            offset = (mi - (n_m - 1) / 2) * h
            vs = [row[mi] for row in rows_f1]
            ax.barh(y + offset, vs, height=h * 0.95, label=f"{mname} (test F1)", color=color)
        ax.set_yticks(y)
        ax.set_yticklabels(used_labels, fontsize=8)
        ax.set_xlabel("F1 score")
        ax.set_title("Per-relation test F1 (sorted by Δ vs frozen when ≥2 models)")
        ax.set_xlim(0, 1.02)
        ax.legend(loc="lower right", fontsize=7)
        fig.tight_layout()
        fig.savefig(out_dir / "per_class_test_f1.png", dpi=200, bbox_inches="tight")
        plt.close(fig)

        # --- Figure 3: delta vs frozen for each non-frozen model ---
        n_d = len(models) - 1
        if n_d > 0:
            w = min(4.5 * n_d, 16.0)
            fig, axes = plt.subplots(1, n_d, figsize=(w, max(5.5, len(used_labels) * 0.2)))
            ax_list: list = [axes] if n_d == 1 else list(axes)  # type: ignore[assignment]
            for di in range(n_d):
                dvals = [row[di + 1] - row[0] for row in rows_f1]
                colors = ["#55A868" if d >= 0 else "#C44E52" for d in dvals]
                ax = ax_list[di]
                ax.barh(y, dvals, color=colors)
                ax.set_yticks(y)
                ax.set_yticklabels(used_labels, fontsize=8)
                ax.axvline(0, color="#333333", linewidth=0.8)
                ax.set_xlabel("Δ test F1")
                ax.set_title(f"{models[di + 1][0]}\n− {models[0][0]}", fontsize=8)
            fig.suptitle("Per-class F1 change vs frozen baseline")
            fig.tight_layout()
            fig.savefig(out_dir / "per_class_f1_delta.png", dpi=200, bbox_inches="tight")
            plt.close(fig)

    # --- Figure 4: training curves ---
    ntrain = 0
    for _, mdata, _ in models:
        if mdata.get("head_training", {}).get("history") or mdata.get("training_history"):
            ntrain += 1
    if ntrain:
        n_m = len(models)
        if n_m == 1:
            fig, ax_one = plt.subplots(1, 1, figsize=(3.6, 4.0))
            axes_list = [ax_one]
        elif n_m == 2:
            fig, axs2 = plt.subplots(1, 2, figsize=(7.2, 4.0))
            axes_list = list(axs2)
        elif n_m == 3:
            fig, axs3 = plt.subplots(1, 3, figsize=(10.5, 4.0))
            axes_list = list(axs3)
        else:
            fig, axs4 = plt.subplots(2, 2, figsize=(8.5, 7.5))
            axes_list = list(axs4.ravel()[:n_m])
        for ax, (mname, mdata, color) in zip(axes_list, models):
            fh = mdata.get("head_training", {}).get("history", [])
            th = mdata.get("training_history", [])
            if fh:
                ep = [h["epoch"] for h in fh]
                ax.plot(ep, [h["accuracy"] for h in fh], marker="o", color=color, label="Dev acc")
                ax.set_title(f"{mname}\n(linear head)"[:60])
            elif th:
                ep = [h["epoch"] for h in th]
                ax.plot(ep, [h["dev_accuracy"] for h in th], marker="o", color=color, label="Dev acc")
                ax2 = ax.twinx()
                ax2.plot(ep, [h["train_loss"] for h in th], marker="s", color="#8172B2", ls="--", label="Train loss")
                ax2.set_ylabel("Train loss", color="#8172B2")
                ax.set_title(f"{mname}\n( fine-tune )"[:60])
            else:
                ax.set_visible(False)
            ax.set_xlabel("Epoch")
            ax.set_ylabel("Dev accuracy" if th or fh else "")
            if fh or th:
                ax.set_ylim(0, 1.0)
                ax.grid(True, alpha=0.3)
        fig.suptitle("Training dynamics (procedures differ across rows)")
        fig.tight_layout()
        fig.savefig(out_dir / "training_curves.png", dpi=200, bbox_inches="tight")
        plt.close(fig)

    print(f"Wrote plots to {out_dir}")


if __name__ == "__main__":
    main()
