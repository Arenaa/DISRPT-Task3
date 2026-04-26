"""Helpers to prepend TSV / gold `.rels` fields `dir` and `rel_type` to unit1 (not `orig_label` — gold)."""

from __future__ import annotations

from typing import Dict, List


def disrpt_feature_prefix_from_tsv_row(r: Dict[str, str]) -> str:
    d = (r.get("dir") or "").strip()
    rt = (r.get("rel_type") or "").strip()
    if not d and not rt:
        return ""
    return f"Direction: {d} | Rel type: {rt}\n"


def disrpt_feature_prefix_from_rels_row(r: List[str]) -> str:
    if len(r) < 13:
        return ""
    d, rt = r[11].strip(), r[12].strip()
    if not d and not rt:
        return ""
    return f"Direction: {d} | Rel type: {rt}\n"
