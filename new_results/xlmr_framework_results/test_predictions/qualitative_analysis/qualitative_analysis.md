# Qualitative Analysis

- Predictions: `new_results/xlmr_framework_results/test_predictions/test_gold_vs_pred.tsv`
- Total examples: `10380`

## Overall

- Accuracy: `0.6328`  Macro-F1: `0.6225`  Weighted-F1: `0.6296`  Errors: `3812`

## Error Cases

### Top Overall Confusions

| Gold | Pred | Count |
|---|---|---:|
| conjunction | elaboration | 200 |
| elaboration | conjunction | 194 |
| temporal | conjunction | 136 |
| conjunction | temporal | 122 |
| frame | elaboration | 109 |
| causal | conjunction | 102 |
| comment | elaboration | 95 |
| explanation | elaboration | 93 |
| elaboration | comment | 89 |
| contrast | concession | 87 |

## By Language

### eng

- Support: `8578`  Accuracy: `0.6450`  Macro-F1: `0.6252`  Errors: `3045`

| Gold | Pred | Count |
|---|---|---:|
| conjunction | elaboration | 156 |
| elaboration | conjunction | 148 |
| temporal | conjunction | 108 |
| causal | conjunction | 88 |
| frame | elaboration | 82 |
| conjunction | temporal | 81 |
| comment | elaboration | 78 |
| contrast | concession | 78 |
| explanation | elaboration | 74 |
| elaboration | comment | 64 |

### fas

- Support: `592`  Accuracy: `0.5389`  Macro-F1: `0.4042`  Errors: `273`

| Gold | Pred | Count |
|---|---|---:|
| elaboration | comment | 13 |
| elaboration | conjunction | 13 |
| conjunction | elaboration | 13 |
| contrast | conjunction | 10 |
| comment | elaboration | 10 |
| explanation | causal | 9 |
| conjunction | explanation | 9 |
| frame | conjunction | 9 |
| contrast | elaboration | 8 |
| causal | explanation | 8 |

### fra

- Support: `621`  Accuracy: `0.5459`  Macro-F1: `0.4864`  Errors: `282`

| Gold | Pred | Count |
|---|---|---:|
| conjunction | elaboration | 27 |
| conjunction | temporal | 27 |
| elaboration | conjunction | 25 |
| temporal | conjunction | 17 |
| frame | temporal | 15 |
| frame | elaboration | 14 |
| explanation | elaboration | 12 |
| frame | conjunction | 11 |
| temporal | elaboration | 9 |
| elaboration | temporal | 9 |

### ita

- Support: `374`  Accuracy: `0.6230`  Macro-F1: `0.3991`  Errors: `141`

| Gold | Pred | Count |
|---|---|---:|
| conjunction | temporal | 13 |
| causal | reformulation | 10 |
| conjunction | reformulation | 9 |
| conjunction | causal | 9 |
| reformulation | causal | 7 |
| reformulation | temporal | 7 |
| contrast | concession | 7 |
| temporal | causal | 5 |
| temporal | conjunction | 5 |
| condition | reformulation | 4 |

### zho

- Support: `215`  Accuracy: `0.6698`  Macro-F1: `0.4691`  Errors: `71`

| Gold | Pred | Count |
|---|---|---:|
| elaboration | comment | 11 |
| elaboration | conjunction | 8 |
| elaboration | frame | 5 |
| frame | elaboration | 5 |
| conjunction | elaboration | 4 |
| elaboration | contrast | 4 |
| elaboration | purpose | 3 |
| purpose | elaboration | 3 |
| causal | elaboration | 2 |
| organization | conjunction | 2 |

## By Framework

### dep

- Support: `2117`  Accuracy: `0.7624`  Macro-F1: `0.6033`  Errors: `503`

| Gold | Pred | Count |
|---|---|---:|
| elaboration | comment | 47 |
| comment | elaboration | 40 |
| elaboration | purpose | 38 |
| organization | conjunction | 29 |
| contrast | concession | 25 |
| elaboration | frame | 22 |
| purpose | elaboration | 16 |
| conjunction | elaboration | 16 |
| frame | elaboration | 15 |
| elaboration | conjunction | 15 |

### pdtb

- Support: `2050`  Accuracy: `0.6337`  Macro-F1: `0.5730`  Errors: `751`

| Gold | Pred | Count |
|---|---|---:|
| conjunction | temporal | 63 |
| causal | conjunction | 57 |
| temporal | conjunction | 55 |
| elaboration | conjunction | 46 |
| conjunction | elaboration | 43 |
| conjunction | causal | 39 |
| contrast | concession | 30 |
| causal | elaboration | 29 |
| elaboration | causal | 28 |
| causal | temporal | 17 |

### rst

- Support: `4478`  Accuracy: `0.5710`  Macro-F1: `0.5518`  Errors: `1921`

| Gold | Pred | Count |
|---|---|---:|
| elaboration | conjunction | 107 |
| conjunction | elaboration | 99 |
| frame | elaboration | 79 |
| explanation | elaboration | 69 |
| temporal | conjunction | 59 |
| explanation | causal | 53 |
| explanation | conjunction | 53 |
| frame | conjunction | 47 |
| comment | elaboration | 42 |
| organization | elaboration | 41 |

### sdrt

- Support: `1735`  Accuracy: `0.6329`  Macro-F1: `0.5730`  Errors: `637`

| Gold | Pred | Count |
|---|---|---:|
| query | comment | 54 |
| conjunction | query | 44 |
| conjunction | elaboration | 42 |
| conjunction | temporal | 27 |
| elaboration | conjunction | 26 |
| temporal | conjunction | 22 |
| conjunction | comment | 20 |
| elaboration | comment | 19 |
| explanation | elaboration | 18 |
| frame | elaboration | 15 |

## By Dataset

### eng.dep.scidtb

- Support: `1902`  Accuracy: `0.7729`  Macro-F1: `0.6096`  Errors: `432`

| Gold | Pred | Count |
|---|---|---:|
| comment | elaboration | 39 |
| elaboration | comment | 36 |
| elaboration | purpose | 35 |
| organization | conjunction | 27 |
| contrast | concession | 24 |
| elaboration | frame | 17 |
| purpose | elaboration | 13 |
| mode | elaboration | 12 |
| elaboration | causal | 12 |
| conjunction | elaboration | 12 |

### eng.erst.gum

- Support: `3558`  Accuracy: `0.5899`  Macro-F1: `0.5671`  Errors: `1459`

| Gold | Pred | Count |
|---|---|---:|
| conjunction | elaboration | 82 |
| elaboration | conjunction | 78 |
| frame | elaboration | 58 |
| temporal | conjunction | 52 |
| explanation | elaboration | 51 |
| explanation | conjunction | 45 |
| explanation | causal | 43 |
| frame | conjunction | 37 |
| organization | elaboration | 35 |
| temporal | elaboration | 33 |

### eng.pdtb.gum

- Support: `1676`  Accuracy: `0.6360`  Macro-F1: `0.5610`  Errors: `610`

| Gold | Pred | Count |
|---|---|---:|
| causal | conjunction | 56 |
| temporal | conjunction | 50 |
| conjunction | temporal | 50 |
| elaboration | conjunction | 46 |
| conjunction | elaboration | 43 |
| conjunction | causal | 30 |
| causal | elaboration | 29 |
| elaboration | causal | 28 |
| contrast | concession | 23 |
| contrast | conjunction | 15 |

### eng.rst.sts

- Support: `328`  Accuracy: `0.4238`  Macro-F1: `0.3594`  Errors: `189`

| Gold | Pred | Count |
|---|---|---:|
| elaboration | conjunction | 16 |
| frame | elaboration | 13 |
| explanation | elaboration | 11 |
| elaboration | explanation | 10 |
| elaboration | concession | 6 |
| contrast | elaboration | 6 |
| organization | concession | 5 |
| explanation | conjunction | 4 |
| comment | elaboration | 4 |
| elaboration | causal | 4 |

### eng.sdrt.stac

- Support: `1114`  Accuracy: `0.6813`  Macro-F1: `0.4272`  Errors: `355`

| Gold | Pred | Count |
|---|---|---:|
| query | comment | 54 |
| conjunction | query | 43 |
| conjunction | comment | 20 |
| elaboration | comment | 18 |
| conjunction | elaboration | 15 |
| elaboration | query | 14 |
| comment | query | 14 |
| contrast | query | 13 |
| contrast | comment | 13 |
| query | elaboration | 12 |

### fas.rst.prstc

- Support: `592`  Accuracy: `0.5389`  Macro-F1: `0.4042`  Errors: `273`

| Gold | Pred | Count |
|---|---|---:|
| elaboration | comment | 13 |
| elaboration | conjunction | 13 |
| conjunction | elaboration | 13 |
| contrast | conjunction | 10 |
| comment | elaboration | 10 |
| explanation | causal | 9 |
| conjunction | explanation | 9 |
| frame | conjunction | 9 |
| contrast | elaboration | 8 |
| causal | explanation | 8 |

### fra.sdrt.annodis

- Support: `621`  Accuracy: `0.5459`  Macro-F1: `0.4864`  Errors: `282`

| Gold | Pred | Count |
|---|---|---:|
| conjunction | elaboration | 27 |
| conjunction | temporal | 27 |
| elaboration | conjunction | 25 |
| temporal | conjunction | 17 |
| frame | temporal | 15 |
| frame | elaboration | 14 |
| explanation | elaboration | 12 |
| frame | conjunction | 11 |
| temporal | elaboration | 9 |
| elaboration | temporal | 9 |

### ita.pdtb.luna

- Support: `374`  Accuracy: `0.6230`  Macro-F1: `0.3991`  Errors: `141`

| Gold | Pred | Count |
|---|---|---:|
| conjunction | temporal | 13 |
| causal | reformulation | 10 |
| conjunction | reformulation | 9 |
| conjunction | causal | 9 |
| reformulation | causal | 7 |
| reformulation | temporal | 7 |
| contrast | concession | 7 |
| temporal | causal | 5 |
| temporal | conjunction | 5 |
| condition | reformulation | 4 |

### zho.dep.scidtb

- Support: `215`  Accuracy: `0.6698`  Macro-F1: `0.4691`  Errors: `71`

| Gold | Pred | Count |
|---|---|---:|
| elaboration | comment | 11 |
| elaboration | conjunction | 8 |
| elaboration | frame | 5 |
| frame | elaboration | 5 |
| conjunction | elaboration | 4 |
| elaboration | contrast | 4 |
| elaboration | purpose | 3 |
| purpose | elaboration | 3 |
| causal | elaboration | 2 |
| organization | conjunction | 2 |

## Cross-Framework Inconsistencies

| Label | Overall Support | F1 Range | F1 Std | Frameworks >= Threshold |
|---|---:|---:|---:|---:|
| frame | 501 | 0.6877 | 0.2808 | 3 |
| organization | 318 | 0.6462 | 0.3231 | 2 |
| comment | 543 | 0.4178 | 0.1785 | 3 |
| attribution | 423 | 0.3601 | 0.1470 | 3 |
| causal | 639 | 0.3380 | 0.1257 | 4 |
| conjunction | 1657 | 0.3372 | 0.1239 | 4 |
| condition | 272 | 0.3330 | 0.1222 | 4 |
| elaboration | 2408 | 0.3181 | 0.1128 | 4 |
| alternation | 94 | 0.2867 | 0.1171 | 3 |
| concession | 307 | 0.2648 | 0.1084 | 3 |

## Rare Labels

- Rare threshold: labels with support <= `50`

| Label | Support | Precision | Recall | F1 |
|---|---:|---:|---:|---:|

## Framework-Internal Rare Labels

- Rare threshold inside each framework: support <= `50`

### dep

| Label | Support | Precision | Recall | F1 |
|---|---:|---:|---:|---:|
| concession | 23 | 0.2917 | 0.6087 | 0.3944 |
| temporal | 24 | 0.4889 | 0.9167 | 0.6377 |
| explanation | 28 | 0.7143 | 0.1786 | 0.2857 |
| condition | 34 | 1.0000 | 0.3235 | 0.4889 |
| causal | 47 | 0.3148 | 0.3617 | 0.3366 |

### pdtb

| Label | Support | Precision | Recall | F1 |
|---|---:|---:|---:|---:|
| explanation | 1 | 0.0000 | 0.0000 | 0.0000 |
| organization | 3 | 0.0000 | 0.0000 | 0.0000 |
| alternation | 34 | 0.8000 | 0.7059 | 0.7500 |

### rst

| Label | Support | Precision | Recall | F1 |
|---|---:|---:|---:|---:|
| alternation | 36 | 0.6452 | 0.5556 | 0.5970 |

### sdrt

| Label | Support | Precision | Recall | F1 |
|---|---:|---:|---:|---:|
| attribution | 14 | 0.6154 | 0.5714 | 0.5926 |
| purpose | 20 | 0.7727 | 0.8500 | 0.8095 |
| condition | 24 | 0.6818 | 0.6250 | 0.6522 |
| alternation | 24 | 1.0000 | 0.7917 | 0.8837 |

