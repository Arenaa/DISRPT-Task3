# Qualitative Analysis

- Predictions: `new_results/xlmr_finetune_results/test_predictions/test_gold_vs_pred.tsv`
- Total examples: `10380`

## Overall

- Accuracy: `0.6357`  Macro-F1: `0.6228`  Weighted-F1: `0.6322`  Errors: `3781`

## Error Cases

### Top Overall Confusions

| Gold | Pred | Count |
|---|---|---:|
| elaboration | conjunction | 199 |
| conjunction | elaboration | 183 |
| conjunction | temporal | 124 |
| temporal | conjunction | 118 |
| causal | conjunction | 101 |
| frame | elaboration | 100 |
| contrast | concession | 84 |
| explanation | elaboration | 78 |
| conjunction | causal | 76 |
| comment | elaboration | 75 |

## By Language

### eng

- Support: `8578`  Accuracy: `0.6490`  Macro-F1: `0.6255`  Errors: `3011`

| Gold | Pred | Count |
|---|---|---:|
| elaboration | conjunction | 153 |
| conjunction | elaboration | 142 |
| causal | conjunction | 87 |
| temporal | conjunction | 87 |
| conjunction | temporal | 86 |
| frame | elaboration | 72 |
| contrast | concession | 70 |
| comment | elaboration | 62 |
| causal | elaboration | 62 |
| conjunction | causal | 58 |

### fas

- Support: `592`  Accuracy: `0.5321`  Macro-F1: `0.4378`  Errors: `277`

| Gold | Pred | Count |
|---|---|---:|
| elaboration | comment | 18 |
| elaboration | conjunction | 14 |
| conjunction | elaboration | 11 |
| explanation | elaboration | 9 |
| conjunction | causal | 9 |
| contrast | comment | 8 |
| explanation | causal | 8 |
| elaboration | explanation | 7 |
| contrast | elaboration | 7 |
| conjunction | comment | 6 |

### fra

- Support: `621`  Accuracy: `0.5217`  Macro-F1: `0.3478`  Errors: `297`

| Gold | Pred | Count |
|---|---|---:|
| conjunction | elaboration | 26 |
| elaboration | conjunction | 23 |
| conjunction | temporal | 21 |
| frame | elaboration | 18 |
| temporal | conjunction | 18 |
| explanation | elaboration | 13 |
| frame | temporal | 11 |
| temporal | elaboration | 9 |
| frame | conjunction | 9 |
| causal | temporal | 8 |

### ita

- Support: `374`  Accuracy: `0.6257`  Macro-F1: `0.4401`  Errors: `140`

| Gold | Pred | Count |
|---|---|---:|
| conjunction | temporal | 15 |
| conjunction | reformulation | 11 |
| temporal | conjunction | 8 |
| contrast | concession | 8 |
| reformulation | temporal | 8 |
| reformulation | causal | 7 |
| causal | reformulation | 7 |
| conjunction | causal | 6 |
| causal | temporal | 5 |
| concession | temporal | 5 |

### zho

- Support: `215`  Accuracy: `0.7395`  Macro-F1: `0.6175`  Errors: `56`

| Gold | Pred | Count |
|---|---|---:|
| elaboration | conjunction | 9 |
| elaboration | frame | 5 |
| purpose | elaboration | 4 |
| frame | elaboration | 4 |
| conjunction | elaboration | 4 |
| elaboration | comment | 3 |
| elaboration | purpose | 2 |
| elaboration | contrast | 2 |
| causal | elaboration | 2 |
| frame | purpose | 2 |

## By Framework

### dep

- Support: `2117`  Accuracy: `0.7813`  Macro-F1: `0.5938`  Errors: `463`

| Gold | Pred | Count |
|---|---|---:|
| comment | elaboration | 32 |
| elaboration | purpose | 32 |
| elaboration | comment | 28 |
| organization | conjunction | 27 |
| contrast | concession | 25 |
| elaboration | frame | 25 |
| purpose | elaboration | 19 |
| elaboration | conjunction | 18 |
| mode | elaboration | 14 |
| frame | elaboration | 13 |

### pdtb

- Support: `2050`  Accuracy: `0.6298`  Macro-F1: `0.4715`  Errors: `759`

| Gold | Pred | Count |
|---|---|---:|
| conjunction | temporal | 72 |
| causal | conjunction | 57 |
| conjunction | elaboration | 50 |
| temporal | conjunction | 47 |
| elaboration | conjunction | 44 |
| conjunction | causal | 32 |
| causal | elaboration | 27 |
| contrast | concession | 26 |
| causal | temporal | 22 |
| elaboration | causal | 21 |

### rst

- Support: `4478`  Accuracy: `0.5719`  Macro-F1: `0.5573`  Errors: `1917`

| Gold | Pred | Count |
|---|---|---:|
| elaboration | conjunction | 110 |
| conjunction | elaboration | 80 |
| frame | elaboration | 68 |
| explanation | conjunction | 56 |
| explanation | elaboration | 53 |
| temporal | conjunction | 47 |
| explanation | causal | 40 |
| organization | elaboration | 37 |
| conjunction | causal | 36 |
| conjunction | frame | 34 |

### sdrt

- Support: `1735`  Accuracy: `0.6300`  Macro-F1: `0.4415`  Errors: `642`

| Gold | Pred | Count |
|---|---|---:|
| conjunction | query | 47 |
| conjunction | elaboration | 40 |
| query | comment | 29 |
| elaboration | conjunction | 27 |
| temporal | conjunction | 24 |
| comment | query | 22 |
| conjunction | temporal | 21 |
| frame | elaboration | 19 |
| explanation | elaboration | 19 |
| elaboration | query | 18 |

## By Dataset

### eng.dep.scidtb

- Support: `1902`  Accuracy: `0.7860`  Macro-F1: `0.5937`  Errors: `407`

| Gold | Pred | Count |
|---|---|---:|
| comment | elaboration | 31 |
| elaboration | purpose | 30 |
| organization | conjunction | 25 |
| elaboration | comment | 25 |
| contrast | concession | 24 |
| elaboration | frame | 20 |
| purpose | elaboration | 15 |
| mode | elaboration | 13 |
| elaboration | causal | 12 |
| organization | causal | 10 |

### eng.erst.gum

- Support: `3558`  Accuracy: `0.5964`  Macro-F1: `0.5766`  Errors: `1436`

| Gold | Pred | Count |
|---|---|---:|
| elaboration | conjunction | 81 |
| conjunction | elaboration | 64 |
| frame | elaboration | 51 |
| explanation | conjunction | 48 |
| temporal | conjunction | 42 |
| explanation | elaboration | 35 |
| organization | elaboration | 32 |
| explanation | causal | 31 |
| conjunction | frame | 30 |
| contrast | conjunction | 29 |

### eng.pdtb.gum

- Support: `1676`  Accuracy: `0.6307`  Macro-F1: `0.4612`  Errors: `619`

| Gold | Pred | Count |
|---|---|---:|
| conjunction | temporal | 57 |
| causal | conjunction | 56 |
| conjunction | elaboration | 50 |
| elaboration | conjunction | 44 |
| temporal | conjunction | 39 |
| causal | elaboration | 26 |
| conjunction | causal | 26 |
| elaboration | causal | 21 |
| contrast | concession | 18 |
| causal | temporal | 17 |

### eng.rst.sts

- Support: `328`  Accuracy: `0.3780`  Macro-F1: `0.3199`  Errors: `204`

| Gold | Pred | Count |
|---|---|---:|
| elaboration | conjunction | 15 |
| frame | elaboration | 11 |
| elaboration | explanation | 10 |
| explanation | elaboration | 9 |
| elaboration | concession | 8 |
| elaboration | organization | 7 |
| contrast | concession | 6 |
| elaboration | causal | 6 |
| organization | concession | 5 |
| frame | concession | 5 |

### eng.sdrt.stac

- Support: `1114`  Accuracy: `0.6903`  Macro-F1: `0.3916`  Errors: `345`

| Gold | Pred | Count |
|---|---|---:|
| conjunction | query | 46 |
| query | comment | 29 |
| comment | query | 22 |
| elaboration | query | 18 |
| conjunction | elaboration | 14 |
| query | conjunction | 14 |
| contrast | query | 13 |
| conjunction | comment | 12 |
| elaboration | comment | 12 |
| query | elaboration | 11 |

### fas.rst.prstc

- Support: `592`  Accuracy: `0.5321`  Macro-F1: `0.4378`  Errors: `277`

| Gold | Pred | Count |
|---|---|---:|
| elaboration | comment | 18 |
| elaboration | conjunction | 14 |
| conjunction | elaboration | 11 |
| explanation | elaboration | 9 |
| conjunction | causal | 9 |
| contrast | comment | 8 |
| explanation | causal | 8 |
| elaboration | explanation | 7 |
| contrast | elaboration | 7 |
| conjunction | comment | 6 |

### fra.sdrt.annodis

- Support: `621`  Accuracy: `0.5217`  Macro-F1: `0.3478`  Errors: `297`

| Gold | Pred | Count |
|---|---|---:|
| conjunction | elaboration | 26 |
| elaboration | conjunction | 23 |
| conjunction | temporal | 21 |
| frame | elaboration | 18 |
| temporal | conjunction | 18 |
| explanation | elaboration | 13 |
| frame | temporal | 11 |
| temporal | elaboration | 9 |
| frame | conjunction | 9 |
| causal | temporal | 8 |

### ita.pdtb.luna

- Support: `374`  Accuracy: `0.6257`  Macro-F1: `0.4401`  Errors: `140`

| Gold | Pred | Count |
|---|---|---:|
| conjunction | temporal | 15 |
| conjunction | reformulation | 11 |
| temporal | conjunction | 8 |
| contrast | concession | 8 |
| reformulation | temporal | 8 |
| reformulation | causal | 7 |
| causal | reformulation | 7 |
| conjunction | causal | 6 |
| causal | temporal | 5 |
| concession | temporal | 5 |

### zho.dep.scidtb

- Support: `215`  Accuracy: `0.7395`  Macro-F1: `0.6175`  Errors: `56`

| Gold | Pred | Count |
|---|---|---:|
| elaboration | conjunction | 9 |
| elaboration | frame | 5 |
| purpose | elaboration | 4 |
| frame | elaboration | 4 |
| conjunction | elaboration | 4 |
| elaboration | comment | 3 |
| elaboration | purpose | 2 |
| elaboration | contrast | 2 |
| causal | elaboration | 2 |
| frame | purpose | 2 |

## Cross-Framework Inconsistencies

| Label | Overall Support | F1 Range | F1 Std | Frameworks >= Threshold |
|---|---:|---:|---:|---:|
| frame | 501 | 0.6338 | 0.2588 | 3 |
| organization | 318 | 0.5723 | 0.2862 | 2 |
| comment | 543 | 0.4699 | 0.1952 | 3 |
| attribution | 423 | 0.3878 | 0.1584 | 3 |
| conjunction | 1657 | 0.3422 | 0.1246 | 4 |
| elaboration | 2408 | 0.3308 | 0.1192 | 4 |
| alternation | 94 | 0.2689 | 0.1100 | 3 |
| causal | 639 | 0.2663 | 0.1070 | 4 |
| concession | 307 | 0.2614 | 0.1067 | 3 |
| temporal | 764 | 0.2453 | 0.1008 | 4 |

## Rare Labels

- Rare threshold: labels with support <= `50`

| Label | Support | Precision | Recall | F1 |
|---|---:|---:|---:|---:|

## Framework-Internal Rare Labels

- Rare threshold inside each framework: support <= `50`

### dep

| Label | Support | Precision | Recall | F1 |
|---|---:|---:|---:|---:|
| concession | 23 | 0.2955 | 0.5652 | 0.3881 |
| temporal | 24 | 0.5676 | 0.8750 | 0.6885 |
| explanation | 28 | 0.6154 | 0.2857 | 0.3902 |
| condition | 34 | 0.9474 | 0.5294 | 0.6792 |
| causal | 47 | 0.2903 | 0.3830 | 0.3303 |

### pdtb

| Label | Support | Precision | Recall | F1 |
|---|---:|---:|---:|---:|
| explanation | 1 | 0.0000 | 0.0000 | 0.0000 |
| organization | 3 | 0.0000 | 0.0000 | 0.0000 |
| alternation | 34 | 0.7742 | 0.7059 | 0.7385 |

### rst

| Label | Support | Precision | Recall | F1 |
|---|---:|---:|---:|---:|
| alternation | 36 | 0.6250 | 0.5556 | 0.5882 |

### sdrt

| Label | Support | Precision | Recall | F1 |
|---|---:|---:|---:|---:|
| attribution | 14 | 0.5714 | 0.5714 | 0.5714 |
| purpose | 20 | 0.7727 | 0.8500 | 0.8095 |
| condition | 24 | 0.7083 | 0.7083 | 0.7083 |
| alternation | 24 | 1.0000 | 0.7500 | 0.8571 |

