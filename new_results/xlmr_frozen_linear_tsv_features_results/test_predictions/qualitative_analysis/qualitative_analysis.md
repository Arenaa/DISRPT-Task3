# Qualitative Analysis

- Predictions: `new_results/xlmr_frozen_linear_tsv_features_results/test_predictions/test_gold_vs_pred.tsv`
- Total examples: `10380`

## Overall

- Accuracy: `0.3394`  Macro-F1: `0.1795`  Weighted-F1: `0.2752`  Errors: `6857`

## Error Cases

### Top Overall Confusions

| Gold | Pred | Count |
|---|---|---:|
| conjunction | elaboration | 682 |
| elaboration | conjunction | 459 |
| frame | elaboration | 291 |
| comment | elaboration | 287 |
| temporal | conjunction | 243 |
| purpose | elaboration | 241 |
| temporal | elaboration | 230 |
| causal | elaboration | 228 |
| causal | conjunction | 227 |
| contrast | elaboration | 225 |

## By Language

### eng

- Support: `8578`  Accuracy: `0.3528`  Macro-F1: `0.1862`  Errors: `5552`

| Gold | Pred | Count |
|---|---|---:|
| conjunction | elaboration | 495 |
| elaboration | conjunction | 352 |
| comment | elaboration | 246 |
| frame | elaboration | 222 |
| purpose | elaboration | 198 |
| temporal | conjunction | 189 |
| causal | conjunction | 188 |
| attribution | elaboration | 177 |
| mode | elaboration | 168 |
| contrast | elaboration | 167 |

### fas

- Support: `592`  Accuracy: `0.2905`  Macro-F1: `0.0593`  Errors: `420`

| Gold | Pred | Count |
|---|---|---:|
| conjunction | elaboration | 60 |
| elaboration | conjunction | 47 |
| contrast | elaboration | 30 |
| explanation | elaboration | 25 |
| contrast | conjunction | 23 |
| causal | elaboration | 22 |
| explanation | conjunction | 20 |
| comment | elaboration | 20 |
| attribution | conjunction | 20 |
| attribution | elaboration | 17 |

### fra

- Support: `621`  Accuracy: `0.2979`  Macro-F1: `0.0746`  Errors: `436`

| Gold | Pred | Count |
|---|---|---:|
| elaboration | conjunction | 58 |
| conjunction | elaboration | 53 |
| frame | conjunction | 45 |
| frame | elaboration | 43 |
| temporal | elaboration | 32 |
| temporal | conjunction | 31 |
| causal | conjunction | 18 |
| explanation | elaboration | 16 |
| contrast | elaboration | 16 |
| elaboration | temporal | 14 |

### ita

- Support: `374`  Accuracy: `0.1257`  Macro-F1: `0.0651`  Errors: `327`

| Gold | Pred | Count |
|---|---|---:|
| reformulation | elaboration | 46 |
| causal | elaboration | 44 |
| conjunction | elaboration | 44 |
| temporal | elaboration | 26 |
| purpose | elaboration | 14 |
| concession | elaboration | 14 |
| condition | elaboration | 12 |
| temporal | conjunction | 11 |
| concession | conjunction | 10 |
| purpose | conjunction | 10 |

### zho

- Support: `215`  Accuracy: `0.4326`  Macro-F1: `0.1104`  Errors: `122`

| Gold | Pred | Count |
|---|---|---:|
| conjunction | elaboration | 30 |
| purpose | elaboration | 20 |
| comment | elaboration | 13 |
| frame | elaboration | 12 |
| causal | elaboration | 6 |
| contrast | elaboration | 5 |
| attribution | frame | 4 |
| attribution | elaboration | 4 |
| elaboration | frame | 4 |
| mode | frame | 3 |

## By Framework

### dep

- Support: `2117`  Accuracy: `0.4360`  Macro-F1: `0.0899`  Errors: `1194`

| Gold | Pred | Count |
|---|---|---:|
| comment | elaboration | 189 |
| conjunction | elaboration | 180 |
| frame | elaboration | 136 |
| purpose | elaboration | 134 |
| mode | elaboration | 116 |
| attribution | elaboration | 96 |
| contrast | elaboration | 72 |
| organization | elaboration | 51 |
| causal | elaboration | 47 |
| condition | elaboration | 29 |

### pdtb

- Support: `2050`  Accuracy: `0.2610`  Macro-F1: `0.1074`  Errors: `1515`

| Gold | Pred | Count |
|---|---|---:|
| conjunction | elaboration | 142 |
| causal | conjunction | 131 |
| elaboration | conjunction | 95 |
| temporal | conjunction | 89 |
| causal | elaboration | 84 |
| temporal | elaboration | 73 |
| conjunction | causal | 64 |
| concession | conjunction | 56 |
| purpose | conjunction | 52 |
| reformulation | elaboration | 48 |

### rst

- Support: `4478`  Accuracy: `0.2925`  Macro-F1: `0.1376`  Errors: `3168`

| Gold | Pred | Count |
|---|---|---:|
| conjunction | elaboration | 306 |
| elaboration | conjunction | 300 |
| explanation | elaboration | 149 |
| temporal | conjunction | 123 |
| frame | elaboration | 112 |
| explanation | conjunction | 110 |
| contrast | elaboration | 107 |
| attribution | elaboration | 102 |
| temporal | elaboration | 101 |
| organization | elaboration | 90 |

### sdrt

- Support: `1735`  Accuracy: `0.4352`  Macro-F1: `0.1342`  Errors: `980`

| Gold | Pred | Count |
|---|---|---:|
| comment | query | 119 |
| conjunction | query | 112 |
| elaboration | query | 91 |
| elaboration | conjunction | 59 |
| contrast | query | 55 |
| conjunction | elaboration | 54 |
| frame | conjunction | 45 |
| frame | elaboration | 43 |
| temporal | elaboration | 32 |
| temporal | conjunction | 31 |

## By Dataset

### eng.dep.scidtb

- Support: `1902`  Accuracy: `0.4364`  Macro-F1: `0.0827`  Errors: `1072`

| Gold | Pred | Count |
|---|---|---:|
| comment | elaboration | 176 |
| conjunction | elaboration | 150 |
| frame | elaboration | 124 |
| mode | elaboration | 116 |
| purpose | elaboration | 114 |
| attribution | elaboration | 92 |
| contrast | elaboration | 67 |
| organization | elaboration | 48 |
| causal | elaboration | 41 |
| condition | elaboration | 29 |

### eng.erst.gum

- Support: `3558`  Accuracy: `0.2915`  Macro-F1: `0.1423`  Errors: `2521`

| Gold | Pred | Count |
|---|---|---:|
| elaboration | conjunction | 241 |
| conjunction | elaboration | 227 |
| temporal | conjunction | 110 |
| explanation | elaboration | 101 |
| explanation | conjunction | 90 |
| temporal | elaboration | 84 |
| attribution | elaboration | 83 |
| frame | elaboration | 76 |
| organization | elaboration | 73 |
| causal | conjunction | 63 |

### eng.pdtb.gum

- Support: `1676`  Accuracy: `0.2912`  Macro-F1: `0.1324`  Errors: `1188`

| Gold | Pred | Count |
|---|---|---:|
| causal | conjunction | 122 |
| conjunction | elaboration | 98 |
| elaboration | conjunction | 95 |
| temporal | conjunction | 78 |
| conjunction | causal | 56 |
| temporal | elaboration | 47 |
| concession | conjunction | 46 |
| conjunction | temporal | 45 |
| contrast | conjunction | 43 |
| condition | conjunction | 43 |

### eng.rst.sts

- Support: `328`  Accuracy: `0.3079`  Macro-F1: `0.0565`  Errors: `227`

| Gold | Pred | Count |
|---|---|---:|
| contrast | elaboration | 23 |
| explanation | elaboration | 23 |
| frame | elaboration | 22 |
| conjunction | elaboration | 19 |
| causal | elaboration | 17 |
| concession | elaboration | 17 |
| elaboration | conjunction | 12 |
| temporal | elaboration | 10 |
| comment | elaboration | 9 |
| organization | elaboration | 9 |

### eng.sdrt.stac

- Support: `1114`  Accuracy: `0.5117`  Macro-F1: `0.0993`  Errors: `544`

| Gold | Pred | Count |
|---|---|---:|
| comment | query | 119 |
| conjunction | query | 112 |
| elaboration | query | 89 |
| contrast | query | 55 |
| causal | query | 27 |
| explanation | query | 26 |
| alternation | query | 19 |
| query | comment | 16 |
| condition | query | 14 |
| temporal | query | 13 |

### fas.rst.prstc

- Support: `592`  Accuracy: `0.2905`  Macro-F1: `0.0593`  Errors: `420`

| Gold | Pred | Count |
|---|---|---:|
| conjunction | elaboration | 60 |
| elaboration | conjunction | 47 |
| contrast | elaboration | 30 |
| explanation | elaboration | 25 |
| contrast | conjunction | 23 |
| causal | elaboration | 22 |
| explanation | conjunction | 20 |
| comment | elaboration | 20 |
| attribution | conjunction | 20 |
| attribution | elaboration | 17 |

### fra.sdrt.annodis

- Support: `621`  Accuracy: `0.2979`  Macro-F1: `0.0746`  Errors: `436`

| Gold | Pred | Count |
|---|---|---:|
| elaboration | conjunction | 58 |
| conjunction | elaboration | 53 |
| frame | conjunction | 45 |
| frame | elaboration | 43 |
| temporal | elaboration | 32 |
| temporal | conjunction | 31 |
| causal | conjunction | 18 |
| explanation | elaboration | 16 |
| contrast | elaboration | 16 |
| elaboration | temporal | 14 |

### ita.pdtb.luna

- Support: `374`  Accuracy: `0.1257`  Macro-F1: `0.0651`  Errors: `327`

| Gold | Pred | Count |
|---|---|---:|
| reformulation | elaboration | 46 |
| causal | elaboration | 44 |
| conjunction | elaboration | 44 |
| temporal | elaboration | 26 |
| purpose | elaboration | 14 |
| concession | elaboration | 14 |
| condition | elaboration | 12 |
| temporal | conjunction | 11 |
| concession | conjunction | 10 |
| purpose | conjunction | 10 |

### zho.dep.scidtb

- Support: `215`  Accuracy: `0.4326`  Macro-F1: `0.1104`  Errors: `122`

| Gold | Pred | Count |
|---|---|---:|
| conjunction | elaboration | 30 |
| purpose | elaboration | 20 |
| comment | elaboration | 13 |
| frame | elaboration | 12 |
| causal | elaboration | 6 |
| contrast | elaboration | 5 |
| attribution | frame | 4 |
| attribution | elaboration | 4 |
| elaboration | frame | 4 |
| mode | frame | 3 |

## Cross-Framework Inconsistencies

| Label | Overall Support | F1 Range | F1 Std | Frameworks >= Threshold |
|---|---:|---:|---:|---:|
| query | 672 | 0.5461 | 0.2352 | 3 |
| elaboration | 2408 | 0.4708 | 0.1700 | 4 |
| conjunction | 1657 | 0.3818 | 0.1415 | 4 |
| temporal | 764 | 0.3806 | 0.1495 | 4 |
| attribution | 423 | 0.3587 | 0.1500 | 3 |
| organization | 318 | 0.3514 | 0.1757 | 2 |
| frame | 501 | 0.2756 | 0.1172 | 3 |
| causal | 639 | 0.2672 | 0.1091 | 4 |
| comment | 543 | 0.2612 | 0.1209 | 3 |
| purpose | 435 | 0.1429 | 0.0575 | 4 |

## Rare Labels

- Rare threshold: labels with support <= `50`

| Label | Support | Precision | Recall | F1 |
|---|---:|---:|---:|---:|

## Framework-Internal Rare Labels

- Rare threshold inside each framework: support <= `50`

### dep

| Label | Support | Precision | Recall | F1 |
|---|---:|---:|---:|---:|
| concession | 23 | 0.0000 | 0.0000 | 0.0000 |
| temporal | 24 | 0.0000 | 0.0000 | 0.0000 |
| explanation | 28 | 0.0000 | 0.0000 | 0.0000 |
| condition | 34 | 0.0000 | 0.0000 | 0.0000 |
| causal | 47 | 0.0000 | 0.0000 | 0.0000 |

### pdtb

| Label | Support | Precision | Recall | F1 |
|---|---:|---:|---:|---:|
| explanation | 1 | 0.0000 | 0.0000 | 0.0000 |
| organization | 3 | 0.0000 | 0.0000 | 0.0000 |
| alternation | 34 | 0.0000 | 0.0000 | 0.0000 |

### rst

| Label | Support | Precision | Recall | F1 |
|---|---:|---:|---:|---:|
| alternation | 36 | 0.0000 | 0.0000 | 0.0000 |

### sdrt

| Label | Support | Precision | Recall | F1 |
|---|---:|---:|---:|---:|
| attribution | 14 | 0.0000 | 0.0000 | 0.0000 |
| purpose | 20 | 0.0000 | 0.0000 | 0.0000 |
| alternation | 24 | 0.0000 | 0.0000 | 0.0000 |
| condition | 24 | 0.0000 | 0.0000 | 0.0000 |

