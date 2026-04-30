# Qualitative Analysis

- Predictions: `new_results/xlmr_frozen_linear_results/test_predictions/test_gold_vs_pred.tsv`
- Total examples: `10380`

## Overall

- Accuracy: `0.3326`  Macro-F1: `0.1690`  Weighted-F1: `0.2664`  Errors: `6928`

## Error Cases

### Top Overall Confusions

| Gold | Pred | Count |
|---|---|---:|
| conjunction | elaboration | 662 |
| elaboration | conjunction | 421 |
| frame | elaboration | 357 |
| comment | elaboration | 280 |
| temporal | elaboration | 244 |
| contrast | elaboration | 242 |
| purpose | elaboration | 232 |
| attribution | elaboration | 227 |
| causal | conjunction | 220 |
| temporal | conjunction | 216 |

## By Language

### eng

- Support: `8578`  Accuracy: `0.3409`  Macro-F1: `0.1712`  Errors: `5654`

| Gold | Pred | Count |
|---|---|---:|
| conjunction | elaboration | 497 |
| elaboration | conjunction | 328 |
| frame | elaboration | 261 |
| comment | elaboration | 235 |
| attribution | elaboration | 193 |
| contrast | elaboration | 190 |
| temporal | elaboration | 189 |
| purpose | elaboration | 188 |
| mode | elaboration | 187 |
| causal | conjunction | 182 |

### fas

- Support: `592`  Accuracy: `0.2889`  Macro-F1: `0.0649`  Errors: `421`

| Gold | Pred | Count |
|---|---|---:|
| elaboration | conjunction | 63 |
| conjunction | elaboration | 41 |
| contrast | conjunction | 29 |
| attribution | conjunction | 26 |
| contrast | elaboration | 23 |
| explanation | conjunction | 23 |
| explanation | elaboration | 20 |
| comment | elaboration | 20 |
| causal | elaboration | 17 |
| frame | conjunction | 17 |

### fra

- Support: `621`  Accuracy: `0.3398`  Macro-F1: `0.0774`  Errors: `410`

| Gold | Pred | Count |
|---|---|---:|
| conjunction | elaboration | 74 |
| frame | elaboration | 51 |
| temporal | elaboration | 37 |
| frame | conjunction | 31 |
| elaboration | conjunction | 26 |
| causal | elaboration | 20 |
| temporal | conjunction | 19 |
| explanation | elaboration | 18 |
| contrast | elaboration | 18 |
| purpose | elaboration | 13 |

### ita

- Support: `374`  Accuracy: `0.1872`  Macro-F1: `0.0762`  Errors: `304`

| Gold | Pred | Count |
|---|---|---:|
| causal | elaboration | 30 |
| conjunction | elaboration | 20 |
| temporal | causal | 19 |
| reformulation | causal | 18 |
| concession | causal | 17 |
| conjunction | causal | 17 |
| reformulation | elaboration | 15 |
| reformulation | conjunction | 14 |
| temporal | elaboration | 14 |
| temporal | conjunction | 13 |

### zho

- Support: `215`  Accuracy: `0.3535`  Macro-F1: `0.0599`  Errors: `139`

| Gold | Pred | Count |
|---|---|---:|
| frame | elaboration | 31 |
| conjunction | elaboration | 30 |
| purpose | elaboration | 21 |
| comment | elaboration | 13 |
| attribution | elaboration | 10 |
| contrast | elaboration | 6 |
| causal | elaboration | 6 |
| elaboration | conjunction | 4 |
| mode | elaboration | 4 |
| organization | elaboration | 2 |

## By Framework

### dep

- Support: `2117`  Accuracy: `0.4166`  Macro-F1: `0.0784`  Errors: `1235`

| Gold | Pred | Count |
|---|---|---:|
| comment | elaboration | 188 |
| conjunction | elaboration | 176 |
| frame | elaboration | 168 |
| purpose | elaboration | 131 |
| mode | elaboration | 121 |
| attribution | elaboration | 110 |
| contrast | elaboration | 76 |
| causal | elaboration | 47 |
| organization | elaboration | 47 |
| condition | elaboration | 32 |

### pdtb

- Support: `2050`  Accuracy: `0.2678`  Macro-F1: `0.1136`  Errors: `1501`

| Gold | Pred | Count |
|---|---|---:|
| conjunction | elaboration | 141 |
| causal | conjunction | 119 |
| elaboration | conjunction | 94 |
| causal | elaboration | 79 |
| temporal | conjunction | 79 |
| temporal | elaboration | 71 |
| conjunction | causal | 64 |
| conjunction | temporal | 53 |
| purpose | conjunction | 47 |
| concession | conjunction | 46 |

### rst

- Support: `4478`  Accuracy: `0.2729`  Macro-F1: `0.1244`  Errors: `3256`

| Gold | Pred | Count |
|---|---|---:|
| elaboration | conjunction | 293 |
| conjunction | elaboration | 267 |
| explanation | elaboration | 160 |
| frame | elaboration | 138 |
| temporal | conjunction | 116 |
| temporal | elaboration | 114 |
| organization | elaboration | 112 |
| contrast | elaboration | 112 |
| explanation | conjunction | 106 |
| attribution | elaboration | 105 |

### sdrt

- Support: `1735`  Accuracy: `0.4605`  Macro-F1: `0.1397`  Errors: `936`

| Gold | Pred | Count |
|---|---|---:|
| conjunction | query | 96 |
| elaboration | query | 88 |
| comment | query | 78 |
| conjunction | elaboration | 78 |
| contrast | query | 54 |
| frame | elaboration | 51 |
| temporal | elaboration | 37 |
| query | comment | 32 |
| frame | conjunction | 31 |
| elaboration | conjunction | 28 |

## By Dataset

### eng.dep.scidtb

- Support: `1902`  Accuracy: `0.4238`  Macro-F1: `0.0802`  Errors: `1096`

| Gold | Pred | Count |
|---|---|---:|
| comment | elaboration | 175 |
| conjunction | elaboration | 146 |
| frame | elaboration | 137 |
| mode | elaboration | 117 |
| purpose | elaboration | 110 |
| attribution | elaboration | 100 |
| contrast | elaboration | 70 |
| organization | elaboration | 45 |
| causal | elaboration | 41 |
| condition | elaboration | 31 |

### eng.erst.gum

- Support: `3558`  Accuracy: `0.2687`  Macro-F1: `0.1276`  Errors: `2602`

| Gold | Pred | Count |
|---|---|---:|
| elaboration | conjunction | 219 |
| conjunction | elaboration | 208 |
| explanation | elaboration | 118 |
| temporal | elaboration | 101 |
| organization | elaboration | 98 |
| temporal | conjunction | 98 |
| frame | elaboration | 95 |
| attribution | elaboration | 91 |
| explanation | conjunction | 82 |
| attribution | conjunction | 64 |

### eng.pdtb.gum

- Support: `1676`  Accuracy: `0.2858`  Macro-F1: `0.1188`  Errors: `1197`

| Gold | Pred | Count |
|---|---|---:|
| conjunction | elaboration | 121 |
| causal | conjunction | 108 |
| elaboration | conjunction | 94 |
| temporal | conjunction | 66 |
| temporal | elaboration | 57 |
| conjunction | temporal | 50 |
| causal | elaboration | 49 |
| conjunction | causal | 47 |
| purpose | conjunction | 40 |
| concession | conjunction | 39 |

### eng.rst.sts

- Support: `328`  Accuracy: `0.2896`  Macro-F1: `0.0632`  Errors: `233`

| Gold | Pred | Count |
|---|---|---:|
| frame | elaboration | 29 |
| contrast | elaboration | 26 |
| explanation | elaboration | 22 |
| concession | elaboration | 20 |
| conjunction | elaboration | 18 |
| organization | elaboration | 12 |
| elaboration | conjunction | 11 |
| causal | elaboration | 11 |
| temporal | elaboration | 10 |
| comment | elaboration | 8 |

### eng.sdrt.stac

- Support: `1114`  Accuracy: `0.5278`  Macro-F1: `0.1139`  Errors: `526`

| Gold | Pred | Count |
|---|---|---:|
| conjunction | query | 96 |
| elaboration | query | 87 |
| comment | query | 78 |
| contrast | query | 54 |
| query | comment | 32 |
| causal | query | 23 |
| explanation | query | 20 |
| alternation | query | 19 |
| condition | query | 14 |
| conjunction | comment | 14 |

### fas.rst.prstc

- Support: `592`  Accuracy: `0.2889`  Macro-F1: `0.0649`  Errors: `421`

| Gold | Pred | Count |
|---|---|---:|
| elaboration | conjunction | 63 |
| conjunction | elaboration | 41 |
| contrast | conjunction | 29 |
| attribution | conjunction | 26 |
| contrast | elaboration | 23 |
| explanation | conjunction | 23 |
| explanation | elaboration | 20 |
| comment | elaboration | 20 |
| causal | elaboration | 17 |
| frame | conjunction | 17 |

### fra.sdrt.annodis

- Support: `621`  Accuracy: `0.3398`  Macro-F1: `0.0774`  Errors: `410`

| Gold | Pred | Count |
|---|---|---:|
| conjunction | elaboration | 74 |
| frame | elaboration | 51 |
| temporal | elaboration | 37 |
| frame | conjunction | 31 |
| elaboration | conjunction | 26 |
| causal | elaboration | 20 |
| temporal | conjunction | 19 |
| explanation | elaboration | 18 |
| contrast | elaboration | 18 |
| purpose | elaboration | 13 |

### ita.pdtb.luna

- Support: `374`  Accuracy: `0.1872`  Macro-F1: `0.0762`  Errors: `304`

| Gold | Pred | Count |
|---|---|---:|
| causal | elaboration | 30 |
| conjunction | elaboration | 20 |
| temporal | causal | 19 |
| reformulation | causal | 18 |
| concession | causal | 17 |
| conjunction | causal | 17 |
| reformulation | elaboration | 15 |
| reformulation | conjunction | 14 |
| temporal | elaboration | 14 |
| temporal | conjunction | 13 |

### zho.dep.scidtb

- Support: `215`  Accuracy: `0.3535`  Macro-F1: `0.0599`  Errors: `139`

| Gold | Pred | Count |
|---|---|---:|
| frame | elaboration | 31 |
| conjunction | elaboration | 30 |
| purpose | elaboration | 21 |
| comment | elaboration | 13 |
| attribution | elaboration | 10 |
| contrast | elaboration | 6 |
| causal | elaboration | 6 |
| elaboration | conjunction | 4 |
| mode | elaboration | 4 |
| organization | elaboration | 2 |

## Cross-Framework Inconsistencies

| Label | Overall Support | F1 Range | F1 Std | Frameworks >= Threshold |
|---|---:|---:|---:|---:|
| query | 672 | 0.6098 | 0.2731 | 3 |
| comment | 543 | 0.4482 | 0.2038 | 3 |
| elaboration | 2408 | 0.4265 | 0.1529 | 4 |
| conjunction | 1657 | 0.3483 | 0.1292 | 4 |
| temporal | 764 | 0.3169 | 0.1197 | 4 |
| causal | 639 | 0.2807 | 0.1146 | 4 |
| attribution | 423 | 0.2195 | 0.0925 | 3 |
| purpose | 435 | 0.1556 | 0.0629 | 4 |
| reformulation | 148 | 0.1416 | 0.0708 | 2 |
| condition | 272 | 0.1301 | 0.0581 | 4 |

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
| temporal | 24 | 1.0000 | 0.0417 | 0.0800 |
| explanation | 28 | 0.0000 | 0.0000 | 0.0000 |
| condition | 34 | 0.0000 | 0.0000 | 0.0000 |
| causal | 47 | 0.0000 | 0.0000 | 0.0000 |

### pdtb

| Label | Support | Precision | Recall | F1 |
|---|---:|---:|---:|---:|
| explanation | 1 | 0.0000 | 0.0000 | 0.0000 |
| organization | 3 | 0.0000 | 0.0000 | 0.0000 |
| alternation | 34 | 0.5000 | 0.0294 | 0.0556 |

### rst

| Label | Support | Precision | Recall | F1 |
|---|---:|---:|---:|---:|
| alternation | 36 | 0.5000 | 0.0556 | 0.1000 |

### sdrt

| Label | Support | Precision | Recall | F1 |
|---|---:|---:|---:|---:|
| attribution | 14 | 0.0000 | 0.0000 | 0.0000 |
| purpose | 20 | 0.0000 | 0.0000 | 0.0000 |
| alternation | 24 | 0.0000 | 0.0000 | 0.0000 |
| condition | 24 | 0.0000 | 0.0000 | 0.0000 |

