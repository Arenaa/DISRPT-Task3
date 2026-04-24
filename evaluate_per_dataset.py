"""Standalone per-dataset evaluator for DISRPT Task 3.

Loads a previously trained checkpoint (frozen-encoder + linear head, or a full
fine-tuned HF checkpoint) and reports test metrics broken down four ways:

    * per corpus (e.g. eng.rst.sts, zho.dep.scidtb, ...)
    * per framework (rst, pdtb, sdrt, dep; erst grouped with rst)
    * per language (eng, fas, fra, ita, zho)
    * per label (globally, and restricted inside each slice above)

Output: a single JSON with all four views, plus pretty-printed summary to stdout.
This script does **not** retrain; it reuses existing artifacts.

Artifacts expected:
    --mode frozen    : default `results/xlmr_frozen_linear_results/linear_head.pt`
    --mode finetune  : default `results/xlmr_finetune_results/best_model/`

Examples:

    python evaluate_per_dataset.py --mode frozen \
        --frozen-dir results/xlmr_frozen_linear_results

    python evaluate_per_dataset.py --mode finetune \
        --finetune-dir results/xlmr_finetune_results/best_model

    python evaluate_per_dataset.py --mode both \
        --frozen-dir results/xlmr_frozen_linear_results \
        --finetune-dir results/xlmr_finetune_results/best_model
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import (
    AutoModel,
    AutoModelForSequenceClassification,
    AutoTokenizer,
)

from benchmark_frozen_xlm_roberta_linear import (
    PairTextDataset,
    SplitData,
    aggregate_predictions as frozen_aggregate,
    compute_cls_pooled_embeddings,
    load_split,
    predict_per_corpus as frozen_predict_per_corpus,
    print_aggregate_summary,
)
from finetune_xlm_roberta import (
    PairDataset,
    _predict as ft_predict,
    aggregate_predictions as ft_aggregate,
    predict_per_corpus as ft_predict_per_corpus,
)
from finetune_xlm_roberta_framework_conditioned import (
    build_logit_masks as fw_build_logit_masks,
    predict_per_corpus as fw_predict_per_corpus,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mode",
                   choices=["frozen", "finetune", "framework_cond", "both", "all"],
                   required=True,
                   help="'both' = frozen + finetune; 'all' = all three.")
    p.add_argument("--frozen-dir", default="results/xlmr_frozen_linear_results")
    p.add_argument("--finetune-dir", default="results/xlmr_finetune_results/best_model")
    p.add_argument("--framework-cond-dir", default="results/xlmr_framework_results/best_model")
    p.add_argument("--fc-no-mask", action="store_true",
                   help="Disable output masking for the framework-conditioned model.")
    p.add_argument("--by-source-dir", default="results/processed_tsv/by_source_file")
    p.add_argument("--split", default="test", choices=["train", "dev", "test"])
    p.add_argument("--max-length", type=int, default=256)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--out-dir", default="results/per_dataset_eval")
    return p.parse_args()


def _expand_modes(mode: str) -> list[str]:
    if mode == "all":
        return ["frozen", "finetune", "framework_cond"]
    if mode == "both":
        return ["frozen", "finetune"]
    return [mode]


def eval_frozen(args) -> dict:
    head_path = Path(args.frozen_dir) / "linear_head.pt"
    if not head_path.exists():
        raise FileNotFoundError(
            f"No linear head found at {head_path}. Re-run benchmark_frozen_xlm_roberta_linear.py "
            f"(the patched version saves linear_head.pt automatically)."
        )
    ckpt = torch.load(head_path, map_location="cpu")
    label_set = ckpt["label_set"]
    label2id = ckpt["label2id"]
    id2label = {i: lab for lab, i in label2id.items()}
    model_name = ckpt["model_name"]
    max_length = ckpt.get("max_length", args.max_length)

    device = torch.device(args.device)
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    encoder = AutoModel.from_pretrained(model_name)
    for p in encoder.parameters():
        p.requires_grad = False
    encoder.to(device)

    sd = ckpt["state_dict"]
    hidden = sd["weight"].shape[1]
    head = nn.Linear(hidden, len(label_set)).to(device)
    head.load_state_dict(sd)
    head.eval()

    print(f"\n[frozen] evaluating per-corpus {args.split} from {args.by_source_dir}")
    preds = frozen_predict_per_corpus(
        by_source_dir=Path(args.by_source_dir),
        encoder=encoder,
        tokenizer=tokenizer,
        head=head,
        label2id=label2id,
        device=device,
        max_length=max_length,
        emb_batch_size=args.batch_size,
        emb_num_workers=0,
        split=args.split,
    )
    agg = frozen_aggregate(preds, id2label)
    print_aggregate_summary(agg)
    return {
        "model": "Frozen XLM-RoBERTa + linear head",
        "model_name": model_name,
        "max_length": max_length,
        "label_set": label_set,
        **agg,
    }


def eval_finetune(args) -> dict:
    model_dir = Path(args.finetune_dir)
    if not model_dir.exists():
        raise FileNotFoundError(
            f"No fine-tuned model found at {model_dir}. Re-run finetune_xlm_roberta.py "
            f"(it auto-saves best_model by default after the patch)."
        )
    device = torch.device(args.device)
    tokenizer = AutoTokenizer.from_pretrained(model_dir, use_fast=True)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    model.to(device)
    label2id = {str(k): int(v) for k, v in model.config.label2id.items()}
    id2label = {int(v): k for k, v in label2id.items()}
    max_length = args.max_length

    print(f"\n[finetune] evaluating per-corpus {args.split} from {args.by_source_dir}")
    preds = ft_predict_per_corpus(
        by_source_dir=Path(args.by_source_dir),
        model=model,
        tokenizer=tokenizer,
        label2id=label2id,
        device=device,
        max_length=max_length,
        eval_batch_size=args.batch_size,
        split=args.split,
    )
    agg = ft_aggregate(preds, id2label)
    print_aggregate_summary(agg)
    return {
        "model": "Fine-tuned XLM-RoBERTa",
        "model_name": str(model_dir),
        "max_length": max_length,
        "label_set": sorted(label2id, key=lambda k: label2id[k]),
        **agg,
    }


def eval_framework_cond(args) -> dict:
    model_dir = Path(args.framework_cond_dir)
    if not model_dir.exists():
        raise FileNotFoundError(
            f"No framework-conditioned model at {model_dir}. Re-run "
            f"finetune_xlm_roberta_framework_conditioned.py first."
        )
    aux_path = model_dir / "framework_conditioning.json"
    if not aux_path.exists():
        raise FileNotFoundError(
            f"{aux_path} missing; retrain with the aligned framework-conditioned "
            f"script so this side-car JSON is written."
        )
    aux = json.loads(aux_path.read_text(encoding="utf-8"))
    label_set = aux["label_set"]
    label2id = {str(k): int(v) for k, v in aux["label2id"].items()}
    id2label = {v: k for k, v in label2id.items()}
    framework_label_map = {fw: set(int(i) for i in ids) for fw, ids in aux["framework_label_map"].items()}
    logit_masks = fw_build_logit_masks(framework_label_map, len(label_set))
    apply_mask = not args.fc_no_mask

    device = torch.device(args.device)
    tokenizer = AutoTokenizer.from_pretrained(model_dir, use_fast=True)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    model.to(device)

    print(f"\n[framework_cond] evaluating per-corpus {args.split} from {args.by_source_dir} "
          f"(apply_mask={apply_mask})")
    preds = fw_predict_per_corpus(
        model=model,
        tokenizer=tokenizer,
        label2id=label2id,
        device=device,
        logit_masks=logit_masks,
        max_length=args.max_length,
        eval_batch_size=args.batch_size,
        by_source_dir=Path(args.by_source_dir),
        split=args.split,
        apply_mask=apply_mask,
    )
    agg = ft_aggregate(preds, id2label)
    print_aggregate_summary(agg)
    return {
        "model": f"Framework-conditioned XLM-R (mask={apply_mask})",
        "model_name": str(model_dir),
        "max_length": args.max_length,
        "label_set": label_set,
        "framework_tokens": aux.get("framework_tokens"),
        "output_masking": apply_mask,
        **agg,
    }


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    modes = _expand_modes(args.mode)

    if "frozen" in modes:
        res = eval_frozen(args)
        (out_dir / f"frozen_{args.split}.json").write_text(
            json.dumps(res, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        print(f"\n[frozen] saved → {out_dir / f'frozen_{args.split}.json'}")

    if "finetune" in modes:
        res = eval_finetune(args)
        (out_dir / f"finetune_{args.split}.json").write_text(
            json.dumps(res, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        print(f"\n[finetune] saved → {out_dir / f'finetune_{args.split}.json'}")

    if "framework_cond" in modes:
        res = eval_framework_cond(args)
        (out_dir / f"framework_cond_{args.split}.json").write_text(
            json.dumps(res, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        print(f"\n[framework_cond] saved → {out_dir / f'framework_cond_{args.split}.json'}")


if __name__ == "__main__":
    main()
