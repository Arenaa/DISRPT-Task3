"""Grouped bar charts: macro-F1 by language and by framework, one series per model.

Reads each model's per_dataset_test_metrics.json (test split, per-language / per-framework pools).

Default inputs live under new_results/; override with --models name=path (repeat) or a single --glob."""

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


def load_per_dataset(path: Path) -> dict:
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def default_models(root: Path) -> list[tuple[str, Path]]:
    """(short label, path) for known new_results eval bundles."""
    base = root / "new_results"
    pairs = [
        ("Qwen3", "qwen3_finetune_results/per_dataset_test_metrics.json"),
        ("XLMR-FT", "xlmr_finetune_results/per_dataset_test_metrics.json"),
        ("XLMR-FW", "xlmr_framework_results/per_dataset_test_metrics.json"),
        ("XLMR-FL", "xlmr_frozen_linear_results/per_dataset_test_metrics.json"),
        ("FL+TSV", "xlmr_frozen_linear_tsv_features_results/per_dataset_test_metrics.json"),
        ("FT+TSV", "xlmr_finetune_tsv_features_results/per_dataset_test_metrics.json"),
    ]
    out: list[tuple[str, Path]] = []
    for label, rel in pairs:
        p = base / rel
        if p.is_file():
            out.append((label, p))
    return out


def extract_f1(
    data: dict,
    key: str,
    f1_key: str,
) -> tuple[list[str], list[float]]:
    """key = 'per_language' or 'per_framework'. Returns sorted keys and F1s in [0,1]."""
    block = data.get(key) or {}
    names = sorted(block.keys())
    values = [float(block[n].get(f1_key) or 0.0) for n in names]
    return names, values


def plot_grouped(
    plt,
    categories: list[str],
    model_names: list[str],
    values: list[list[float]],
    title: str,
    ylabel: str,
    out_path: Path,
) -> None:
    """
    values[i][j] = F1 for model j at category i (already 0..1 or use display scale).
    """
    n_c = len(categories)
    n_m = len(model_names)
    if n_c == 0 or n_m == 0:
        return
    x = np.arange(n_c, dtype=float)
    width = min(0.8 / n_m, 0.15)
    offset = (np.arange(n_m) - (n_m - 1) / 2.0) * width

    fig, ax = plt.subplots(figsize=(max(8.0, 0.5 + n_c * 1.1), 5.2))
    cmap = plt.get_cmap("tab10" if n_m <= 10 else "nipy_spectral")
    for j, name in enumerate(model_names):
        ys = [values[i][j] * 100.0 for i in range(n_c)]
        color = cmap(j / max(n_m - 1, 1)) if n_m > 1 else cmap(0)
        ax.bar(x + offset[j], ys, width, label=name, color=color, edgecolor="white", linewidth=0.3)

    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels(categories, rotation=18, ha="right")
    ax.set_ylim(0, 105)
    ax.grid(True, axis="y", alpha=0.28)
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.22),
        ncol=min(n_m, 3),
        fontsize=8.5,
        frameon=True,
    )
    fig.tight_layout()
    fig.subplots_adjust(bottom=0.25)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def parse_models_arg(specs: list[str], root: Path) -> list[tuple[str, Path]]:
    out: list[tuple[str, Path]] = []
    for s in specs:
        if "=" not in s:
            raise SystemExit(f"Bad --models entry (need name=path): {s}")
        name, p = s.split("=", 1)
        path = (root / p).resolve() if not Path(p).is_absolute() else Path(p)
        out.append((name.strip(), path))
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", type=Path, default=Path(__file__).resolve().parent, help="Project root")
    ap.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="PNG output directory (default: <root>/new_results/plots)",
    )
    ap.add_argument(
        "--f1",
        choices=("macro", "weighted"),
        default="macro",
        help="Which pooled F1 to plot per group (default: macro).",
    )
    ap.add_argument(
        "--models",
        action="append",
        default=[],
        metavar="NAME=PATH",
        help="Repeat for each model, relative to --root unless absolute. "
        "If none given, all existing default new_results/* paths are used.",
    )
    args = ap.parse_args()
    root = args.root
    out_dir = args.out_dir or (root / "new_results" / "plots")
    f1_key = "macro_f1" if args.f1 == "macro" else "weighted_f1"
    f1_label = "Macro-F1" if args.f1 == "macro" else "Weighted F1"

    if args.models:
        models = parse_models_arg(args.models, root)
    else:
        models = default_models(root)
    if not models:
        raise SystemExit(
            f"No per_dataset_test_metrics.json found under {root / 'new_results'}. "
            "Run evaluation or pass --models name=path."
        )

    # Union of all category names across models (so missing = 0)
    all_lang: set[str] = set()
    all_fw: set[str] = set()
    loaded: list[tuple[str, dict]] = []
    for label, path in models:
        if not path.is_file():
            print(f"[skip] missing {path}")
            continue
        d = load_per_dataset(path)
        loaded.append((label, d))
        pl, _ = extract_f1(d, "per_language", f1_key)
        all_lang |= set(pl)
        pf, _ = extract_f1(d, "per_framework", f1_key)
        all_fw |= set(pf)
    if not loaded:
        raise SystemExit("No valid metrics files loaded.")

    # Prefer sensible orders
    def sort_langs(keys: set[str]) -> list[str]:
        order = ["eng", "fas", "fra", "ita", "zho"]
        rest = sorted(k for k in keys if k not in order)
        return [k for k in order if k in keys] + rest

    def sort_fw(keys: set[str]) -> list[str]:
        order = ["dep", "pdtb", "rst", "sdrt"]
        rest = sorted(k for k in keys if k not in order)
        return [k for k in order if k in keys] + rest

    languages = sort_langs(all_lang)
    frameworks = sort_fw(all_fw)

    # Build matrices [category][model]
    def matrix(keys: list[str], block_key: str) -> list[list[float]]:
        rows: list[list[float]] = []
        for k in keys:
            row = []
            for _, d in loaded:
                block = (d.get(block_key) or {}).get(k) or {}
                row.append(float(block.get(f1_key) or 0.0))
            rows.append(row)
        return rows

    m_lang = matrix(languages, "per_language")
    m_fw = matrix(frameworks, "per_framework")
    model_labels = [t[0] for t in loaded]

    plt = get_pyplot()
    plot_grouped(
        plt,
        languages,
        model_labels,
        m_lang,
        title=f"Test {f1_label} by language (pooled per language)",
        ylabel=f"{f1_label} (%)",
        out_path=out_dir / f"f1_by_language_{args.f1}.png",
    )
    plot_grouped(
        plt,
        frameworks,
        model_labels,
        m_fw,
        title=f"Test {f1_label} by discourse framework (pooled per framework)",
        ylabel=f"{f1_label} (%)",
        out_path=out_dir / f"f1_by_framework_{args.f1}.png",
    )
    print(f"Wrote {out_dir / f'f1_by_language_{args.f1}.png'}")
    print(f"Wrote {out_dir / f'f1_by_framework_{args.f1}.png'}")
    print(f"Models ({len(loaded)}): {', '.join(model_labels)}")


if __name__ == "__main__":
    main()
