"""Compare frozen XLMR + linear head vs full fine-tune metrics and save figures."""

from __future__ import annotations

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


def main() -> None:
    root = Path(__file__).resolve().parent
    frozen_path = root / "results" / "xlmr_frozen_linear_results" / "metrics.json"
    finetune_path = root / "results" / "xlmr_finetune_results" / "metrics.json"
    out_dir = root / "results" / "xlmr_comparison_plots"
    out_dir.mkdir(parents=True, exist_ok=True)

    frozen = load_metrics(frozen_path)
    ft = load_metrics(finetune_path)
    labels: list[str] = frozen["label_set"]
    assert ft["setup"]["label_set"] == labels

    plt = get_pyplot()

    # --- Figure 1: overall dev/test bars ---
    metrics = [
        ("Dev accuracy", frozen["dev_metrics"]["accuracy"], ft["dev_metrics"]["accuracy"]),
        ("Test accuracy", frozen["test_metrics"]["accuracy"], ft["test_metrics"]["accuracy"]),
        ("Dev macro F1", frozen["dev_metrics"]["macro_f1"], ft["dev_metrics"]["macro_f1"]),
        ("Test macro F1", frozen["test_metrics"]["macro_f1"], ft["test_metrics"]["macro_f1"]),
        (
            "Test weighted F1",
            frozen["test_metrics"]["classification_report"]["weighted avg"]["f1-score"],
            ft["test_metrics"]["classification_report"]["weighted avg"]["f1-score"],
        ),
    ]
    names = [m[0] for m in metrics]
    v_frozen = [m[1] for m in metrics]
    v_ft = [m[2] for m in metrics]

    x = np.arange(len(names))
    w = 0.36
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.bar(x - w / 2, v_frozen, width=w, label="Frozen + linear", color="#4C72B0")
    ax.bar(x + w / 2, v_ft, width=w, label="Fine-tuned", color="#DD8452")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=15, ha="right")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Score")
    ax.set_title("XLM-RoBERTa-base: frozen linear head vs full fine-tune")
    ax.legend(loc="upper left")
    ax.axhline(1.0, color="#bbbbbb", linewidth=0.8, linestyle="--")
    fig.tight_layout()
    fig.savefig(out_dir / "overall_metrics.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    # --- Figure 2: per-class test F1 (classes with support > 0 on test in either run) ---
    fr_rep = frozen["test_metrics"]["classification_report"]
    ft_rep = ft["test_metrics"]["classification_report"]

    f1_frozen: list[float] = []
    f1_ft: list[float] = []
    support: list[float] = []
    used_labels: list[str] = []
    for i, lab in enumerate(labels):
        fr_row = fr_rep.get(lab, {"f1-score": 0.0, "support": 0.0})
        ft_row = ft_rep.get(str(i), {"f1-score": 0.0, "support": 0.0})
        sup = max(float(fr_row.get("support", 0)), float(ft_row.get("support", 0)))
        if sup <= 0:
            continue
        f1_frozen.append(float(fr_row["f1-score"]))
        f1_ft.append(float(ft_row["f1-score"]))
        support.append(sup)
        used_labels.append(lab)

    delta = [b - a for a, b in zip(f1_frozen, f1_ft)]
    order = np.argsort(delta)
    used_labels = [used_labels[i] for i in order]
    f1_frozen = [f1_frozen[i] for i in order]
    f1_ft = [f1_ft[i] for i in order]
    delta = [delta[i] for i in order]

    y = np.arange(len(used_labels))
    fig, ax = plt.subplots(figsize=(10, max(6, len(used_labels) * 0.22)))
    ax.barh(y - 0.2, f1_frozen, height=0.38, label="Frozen + linear (test F1)", color="#4C72B0")
    ax.barh(y + 0.2, f1_ft, height=0.38, label="Fine-tuned (test F1)", color="#DD8452")
    ax.set_yticks(y)
    ax.set_yticklabels(used_labels, fontsize=8)
    ax.set_xlabel("F1 score")
    ax.set_title("Per-relation test F1 (sorted by fine-tune gain)")
    ax.set_xlim(0, 1.02)
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(out_dir / "per_class_test_f1.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    # --- Figure 3: delta only (compact) ---
    fig, ax = plt.subplots(figsize=(9, max(5.5, len(used_labels) * 0.2)))
    colors = ["#55A868" if d >= 0 else "#C44E52" for d in delta]
    ax.barh(y, delta, color=colors)
    ax.set_yticks(y)
    ax.set_yticklabels(used_labels, fontsize=8)
    ax.axvline(0, color="#333333", linewidth=0.8)
    ax.set_xlabel("Δ test F1 (fine-tuned − frozen)")
    ax.set_title("Where fine-tuning helps most (positive = gain)")
    fig.tight_layout()
    fig.savefig(out_dir / "per_class_f1_delta.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    # --- Figure 4: training curves where available ---
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    fh = frozen.get("head_training", {}).get("history", [])
    if fh:
        ep = [h["epoch"] for h in fh]
        axes[0].plot(ep, [h["accuracy"] for h in fh], marker="o", label="Dev accuracy")
        axes[0].set_title("Frozen: linear head training")
        axes[0].set_xlabel("Epoch")
        axes[0].set_ylabel("Accuracy")
        axes[0].set_ylim(0, 1.0)
        axes[0].grid(True, alpha=0.3)
    th = ft.get("training_history", [])
    if th:
        ep = [h["epoch"] for h in th]
        axes[1].plot(ep, [h["dev_accuracy"] for h in th], marker="o", color="#DD8452", label="Dev accuracy")
        ax2 = axes[1].twinx()
        ax2.plot(ep, [h["train_loss"] for h in th], marker="s", color="#8172B2", linestyle="--", label="Train loss")
        axes[1].set_title("Fine-tune: end-to-end")
        axes[1].set_xlabel("Epoch")
        axes[1].set_ylabel("Dev accuracy", color="#DD8452")
        ax2.set_ylabel("Train loss", color="#8172B2")
        axes[1].tick_params(axis="y", labelcolor="#DD8452")
        ax2.tick_params(axis="y", labelcolor="#8172B2")
        axes[1].set_ylim(0, 1.0)
        axes[1].grid(True, alpha=0.3)
        lines, labs = axes[1].get_legend_handles_labels()
        l2, la2 = ax2.get_legend_handles_labels()
        axes[1].legend(lines + l2, labs + la2, loc="center right")
    fig.suptitle("Training dynamics (not directly comparable: different procedures)")
    fig.tight_layout()
    fig.savefig(out_dir / "training_curves.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    print(f"Wrote plots to {out_dir}")


if __name__ == "__main__":
    main()
