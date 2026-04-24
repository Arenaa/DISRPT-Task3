"""Helpers to prepend TSV / gold `.rels` fields `dir`, `rel_type`, `orig_label` to unit1 for encoding."""

from __future__ import annotations

from typing import Dict, List


def disrpt_feature_prefix_from_tsv_row(r: Dict[str, str]) -> str:
    d = (r.get("dir") or "").strip()
    rt = (r.get("rel_type") or "").strip()
    ol = (r.get("orig_label") or "").strip()
    if not d and not rt and not ol:
        return ""
    return f"Direction: {d} | Rel type: {rt} | Orig label: {ol}\n"


def disrpt_feature_prefix_from_rels_row(r: List[str]) -> str:
    if len(r) < 14:
        return ""
    d, rt, ol = r[11].strip(), r[12].strip(), r[13].strip()
    if not d and not rt and not ol:
        return ""
    return f"Direction: {d} | Rel type: {rt} | Orig label: {ol}\n"
