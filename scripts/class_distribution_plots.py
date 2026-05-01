from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from label_verification import load_clean_rows_with_metadata


def get_pyplot():
    import matplotlib

    matplotlib.use("Agg")

    import matplotlib.pyplot as plt

    return plt


def split_dataset_name(dataset: str) -> tuple[str, str]:
    language, framework, *_ = dataset.split(".")
    return language, framework


def save_figure(path: Path) -> None:
    plt = get_pyplot()
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=220, bbox_inches="tight")
    plt.close()


def draw_heatmap(
    matrix: np.ndarray,
    row_labels: list[str],
    col_labels: list[str],
    title: str,
    colorbar_label: str,
    output_path: Path,
    annotate_with: np.ndarray,
) -> None:
    plt = get_pyplot()
    fig_width = max(11, len(col_labels) * 0.7)
    fig_height = max(4, len(row_labels) * 0.75)
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    image = ax.imshow(matrix, cmap="YlGnBu", aspect="auto")
    cbar = fig.colorbar(image, ax=ax)
    cbar.set_label(colorbar_label)

    ax.set_xticks(np.arange(len(col_labels)))
    ax.set_xticklabels(col_labels, rotation=45, ha="right")
    ax.set_yticks(np.arange(len(row_labels)))
    ax.set_yticklabels(row_labels)
    ax.set_title(title)

    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            ax.text(
                j,
                i,
                f"{int(annotate_with[i, j])}\n{matrix[i, j] * 100:.1f}%",
                ha="center",
                va="center",
                fontsize=8,
                color="black",
            )

    save_figure(output_path)


def build_plots(input_dir: Path, output_dir: Path) -> dict[str, object]:
    plt = get_pyplot()
    rows, totals = load_clean_rows_with_metadata(input_dir)

    label_counts = Counter(row["label"] for row in rows)
    label_order = [label for label, _ in label_counts.most_common()]

    framework_order = ["dep", "erst", "pdtb", "rst", "sdrt"]
    language_order = ["eng", "fas", "fra", "ita", "zho"]
    split_order = ["train", "dev", "test"]

    framework_counters: dict[str, Counter[str]] = defaultdict(Counter)
    language_counters: dict[str, Counter[str]] = defaultdict(Counter)
    split_counters: dict[str, Counter[str]] = defaultdict(Counter)

    for row in rows:
        label = row["label"]
        language, framework = split_dataset_name(row["dataset"])
        framework_counters[framework][label] += 1
        language_counters[language][label] += 1
        split_counters[row["split"]][label] += 1

    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Label Distribution
    plt.figure(figsize=(14, 6))
    plt.bar(label_order, [label_counts[label] for label in label_order], color="#4E79A7")
    plt.xticks(rotation=45, ha="right")
    plt.ylabel("Count")
    plt.title("Label Distribution (17 Normalized Labels)")
    save_figure(output_dir / "01_label_distribution.png")

    # 2. Label x Framework heatmap
    framework_counts = np.array(
        [[framework_counters[framework][label] for label in label_order] for framework in framework_order],
        dtype=float,
    )
    framework_percentages = framework_counts / framework_counts.sum(axis=1, keepdims=True).clip(min=1)
    draw_heatmap(
        matrix=framework_percentages,
        row_labels=framework_order,
        col_labels=label_order,
        title="Label x Framework Distribution",
        colorbar_label="Share within framework",
        output_path=output_dir / "02_label_by_framework_heatmap.png",
        annotate_with=framework_counts,
    )

    # 3. Label x Language heatmap
    language_counts = np.array(
        [[language_counters[language][label] for label in label_order] for language in language_order],
        dtype=float,
    )
    language_percentages = language_counts / language_counts.sum(axis=1, keepdims=True).clip(min=1)
    draw_heatmap(
        matrix=language_percentages,
        row_labels=language_order,
        col_labels=label_order,
        title="Label x Language Distribution",
        colorbar_label="Share within language",
        output_path=output_dir / "03_label_by_language_heatmap.png",
        annotate_with=language_counts,
    )

    # 4. Train / Dev / Test grouped bar chart
    x = np.arange(len(label_order))
    width = 0.25
    plt.figure(figsize=(16, 6))
    for index, split in enumerate(split_order):
        counter = split_counters[split]
        total = sum(counter.values())
        shares = [counter[label] / total if total else 0.0 for label in label_order]
        plt.bar(x + (index - 1) * width, shares, width=width, label=split)
    plt.xticks(x, label_order, rotation=45, ha="right")
    plt.ylabel("Share within split")
    plt.title("Train / Dev / Test Label Distribution Consistency")
    plt.legend()
    save_figure(output_dir / "04_train_dev_test_grouped_bar.png")

    # 5. Rare Label / Long-tail analysis
    long_tail_labels = [label for label, _ in label_counts.most_common()]
    long_tail_counts = [label_counts[label] for label in long_tail_labels]
    cumulative_share = np.cumsum(long_tail_counts) / sum(long_tail_counts)
    fig, ax1 = plt.subplots(figsize=(14, 6))
    ax1.bar(long_tail_labels, long_tail_counts, color="#59A14F")
    ax1.set_xticks(np.arange(len(long_tail_labels)))
    ax1.set_xticklabels(long_tail_labels, rotation=45, ha="right")
    ax1.set_ylabel("Count")
    ax1.set_title("Rare Label Distribution / Long-tail Analysis")
    ax2 = ax1.twinx()
    ax2.plot(np.arange(len(long_tail_labels)), cumulative_share, color="#E15759", marker="o")
    ax2.set_ylabel("Cumulative share")
    ax2.set_ylim(0, 1.05)
    save_figure(output_dir / "05_rare_label_long_tail.png")

    summary = {
        "input_dir": str(input_dir),
        "output_dir": str(output_dir),
        "num_unique_labels": len(label_order),
        "label_set": label_order,
        "label_normalization_applied": {
            "contingency.cause": "causal",
        },
        "num_rows_used_for_plots": len(rows),
        "totals": totals,
    }
    (output_dir / "plot_generation_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate plot-based class distribution EDA for DISRPT labels.")
    parser.add_argument("--input-dir", default="data_subset", help="Directory containing DISRPT .rels files.")
    parser.add_argument("--output-dir", default="results/eda_plots", help="Directory for EDA figure outputs.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = build_plots(
        input_dir=Path(args.input_dir),
        output_dir=Path(args.output_dir),
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
