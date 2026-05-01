# Qualitative Analysis

- Predictions: `new_results/xlmr_finetune_tsv_features_results/test_predictions/test_gold_vs_pred.tsv`
- Total examples: `10380`

## Overall

- Accuracy: `0.6815`  Macro-F1: `0.6712`  Weighted-F1: `0.6782`  Errors: `3306`

## Error Cases

### Top Overall Confusions

| Gold | Pred | Count |
|---|---|---:|
| conjunction | elaboration | 198 |
| elaboration | conjunction | 182 |
| temporal | conjunction | 133 |
| causal | conjunction | 96 |
| explanation | elaboration | 93 |
| comment | elaboration | 91 |
| elaboration | comment | 84 |
| conjunction | temporal | 80 |
| conjunction | causal | 67 |
| temporal | elaboration | 62 |

## By Language

### eng

- Support: `8578`  Accuracy: `0.6965`  Macro-F1: `0.6752`  Errors: `2603`

| Gold | Pred | Count |
|---|---|---:|
| conjunction | elaboration | 153 |
| elaboration | conjunction | 135 |
| temporal | conjunction | 91 |
| causal | conjunction | 86 |
| comment | elaboration | 76 |
| explanation | elaboration | 74 |
| elaboration | comment | 66 |
| conjunction | causal | 52 |
| temporal | elaboration | 51 |
| elaboration | causal | 48 |

### fas

- Support: `592`  Accuracy: `0.5929`  Macro-F1: `0.5135`  Errors: `241`

| Gold | Pred | Count |
|---|---|---:|
| elaboration | conjunction | 15 |
| elaboration | comment | 13 |
| conjunction | elaboration | 13 |
| explanation | elaboration | 10 |
| comment | elaboration | 10 |
| temporal | conjunction | 9 |
| explanation | comment | 7 |
| explanation | causal | 7 |
| contrast | elaboration | 7 |
| conjunction | comment | 7 |

### fra

- Support: `621`  Accuracy: `0.5572`  Macro-F1: `0.4726`  Errors: `275`

| Gold | Pred | Count |
|---|---|---:|
| conjunction | elaboration | 29 |
| conjunction | temporal | 21 |
| elaboration | conjunction | 21 |
| temporal | conjunction | 20 |
| elaboration | temporal | 13 |
| frame | elaboration | 12 |
| causal | temporal | 10 |
| frame | temporal | 10 |
| frame | conjunction | 9 |
| explanation | elaboration | 9 |

### ita

- Support: `374`  Accuracy: `0.6551`  Macro-F1: `0.4665`  Errors: `129`

| Gold | Pred | Count |
|---|---|---:|
| temporal | conjunction | 13 |
| conjunction | temporal | 11 |
| conjunction | reformulation | 11 |
| reformulation | temporal | 8 |
| conjunction | causal | 6 |
| reformulation | causal | 5 |
| contrast | concession | 5 |
| causal | reformulation | 5 |
| causal | condition | 5 |
| purpose | reformulation | 4 |

### zho

- Support: `215`  Accuracy: `0.7302`  Macro-F1: `0.4896`  Errors: `58`

| Gold | Pred | Count |
|---|---|---:|
| elaboration | conjunction | 11 |
| purpose | elaboration | 6 |
| elaboration | comment | 4 |
| purpose | conjunction | 3 |
| causal | elaboration | 3 |
| elaboration | contrast | 3 |
| mode | frame | 3 |
| organization | conjunction | 3 |
| elaboration | purpose | 2 |
| conjunction | elaboration | 2 |

## By Framework

### dep

- Support: `2117`  Accuracy: `0.8059`  Macro-F1: `0.6304`  Errors: `411`

| Gold | Pred | Count |
|---|---|---:|
| elaboration | comment | 39 |
| elaboration | purpose | 32 |
| comment | elaboration | 30 |
| purpose | elaboration | 24 |
| organization | conjunction | 22 |
| elaboration | frame | 16 |
| conjunction | elaboration | 15 |
| elaboration | conjunction | 15 |
| causal | frame | 11 |
| organization | elaboration | 11 |

### pdtb

- Support: `2050`  Accuracy: `0.6863`  Macro-F1: `0.5379`  Errors: `643`

| Gold | Pred | Count |
|---|---|---:|
| causal | conjunction | 53 |
| elaboration | conjunction | 43 |
| conjunction | temporal | 40 |
| conjunction | elaboration | 40 |
| temporal | conjunction | 35 |
| conjunction | causal | 32 |
| elaboration | causal | 29 |
| causal | elaboration | 21 |
| contrast | concession | 21 |
| causal | temporal | 18 |

### rst

- Support: `4478`  Accuracy: `0.6362`  Macro-F1: `0.6206`  Errors: `1629`

| Gold | Pred | Count |
|---|---|---:|
| conjunction | elaboration | 100 |
| elaboration | conjunction | 99 |
| explanation | elaboration | 73 |
| temporal | conjunction | 71 |
| comment | elaboration | 52 |
| temporal | elaboration | 45 |
| explanation | causal | 43 |
| causal | conjunction | 36 |
| frame | elaboration | 35 |
| elaboration | comment | 32 |

### sdrt

- Support: `1735`  Accuracy: `0.6409`  Macro-F1: `0.5868`  Errors: `623`

| Gold | Pred | Count |
|---|---|---:|
| conjunction | query | 45 |
| conjunction | elaboration | 43 |
| temporal | conjunction | 26 |
| comment | query | 25 |
| query | comment | 25 |
| elaboration | conjunction | 25 |
| conjunction | temporal | 21 |
| elaboration | query | 20 |
| contrast | query | 16 |
| frame | elaboration | 13 |

## By Dataset

### eng.dep.scidtb

- Support: `1902`  Accuracy: `0.8144`  Macro-F1: `0.6376`  Errors: `353`

| Gold | Pred | Count |
|---|---|---:|
| elaboration | comment | 35 |
| elaboration | purpose | 30 |
| comment | elaboration | 29 |
| organization | conjunction | 19 |
| purpose | elaboration | 18 |
| elaboration | frame | 14 |
| conjunction | elaboration | 13 |
| causal | frame | 11 |
| organization | elaboration | 10 |
| mode | elaboration | 10 |

### eng.erst.gum

- Support: `3558`  Accuracy: `0.6551`  Macro-F1: `0.6353`  Errors: `1227`

| Gold | Pred | Count |
|---|---|---:|
| conjunction | elaboration | 74 |
| elaboration | conjunction | 73 |
| temporal | conjunction | 62 |
| explanation | elaboration | 46 |
| temporal | elaboration | 37 |
| explanation | causal | 35 |
| comment | elaboration | 35 |
| causal | conjunction | 28 |
| frame | elaboration | 26 |
| explanation | frame | 25 |

### eng.pdtb.gum

- Support: `1676`  Accuracy: `0.6933`  Macro-F1: `0.5676`  Errors: `514`

| Gold | Pred | Count |
|---|---|---:|
| causal | conjunction | 52 |
| elaboration | conjunction | 43 |
| conjunction | elaboration | 39 |
| conjunction | temporal | 29 |
| elaboration | causal | 29 |
| conjunction | causal | 26 |
| temporal | conjunction | 22 |
| causal | elaboration | 21 |
| causal | temporal | 16 |
| contrast | concession | 16 |

### eng.rst.sts

- Support: `328`  Accuracy: `0.5091`  Macro-F1: `0.3829`  Errors: `161`

| Gold | Pred | Count |
|---|---|---:|
| explanation | elaboration | 17 |
| conjunction | elaboration | 13 |
| elaboration | conjunction | 11 |
| elaboration | explanation | 7 |
| comment | elaboration | 7 |
| contrast | concession | 5 |
| frame | elaboration | 5 |
| contrast | frame | 5 |
| concession | frame | 5 |
| organization | elaboration | 5 |

### eng.sdrt.stac

- Support: `1114`  Accuracy: `0.6876`  Macro-F1: `0.4554`  Errors: `348`

| Gold | Pred | Count |
|---|---|---:|
| conjunction | query | 44 |
| comment | query | 25 |
| query | comment | 25 |
| elaboration | query | 20 |
| contrast | query | 16 |
| conjunction | elaboration | 14 |
| query | elaboration | 13 |
| query | conjunction | 13 |
| conjunction | comment | 12 |
| comment | conjunction | 12 |

### fas.rst.prstc

- Support: `592`  Accuracy: `0.5929`  Macro-F1: `0.5135`  Errors: `241`

| Gold | Pred | Count |
|---|---|---:|
| elaboration | conjunction | 15 |
| elaboration | comment | 13 |
| conjunction | elaboration | 13 |
| explanation | elaboration | 10 |
| comment | elaboration | 10 |
| temporal | conjunction | 9 |
| explanation | comment | 7 |
| explanation | causal | 7 |
| contrast | elaboration | 7 |
| conjunction | comment | 7 |

### fra.sdrt.annodis

- Support: `621`  Accuracy: `0.5572`  Macro-F1: `0.4726`  Errors: `275`

| Gold | Pred | Count |
|---|---|---:|
| conjunction | elaboration | 29 |
| conjunction | temporal | 21 |
| elaboration | conjunction | 21 |
| temporal | conjunction | 20 |
| elaboration | temporal | 13 |
| frame | elaboration | 12 |
| causal | temporal | 10 |
| frame | temporal | 10 |
| frame | conjunction | 9 |
| explanation | elaboration | 9 |

### ita.pdtb.luna

- Support: `374`  Accuracy: `0.6551`  Macro-F1: `0.4665`  Errors: `129`

| Gold | Pred | Count |
|---|---|---:|
| temporal | conjunction | 13 |
| conjunction | temporal | 11 |
| conjunction | reformulation | 11 |
| reformulation | temporal | 8 |
| conjunction | causal | 6 |
| reformulation | causal | 5 |
| contrast | concession | 5 |
| causal | reformulation | 5 |
| causal | condition | 5 |
| purpose | reformulation | 4 |

### zho.dep.scidtb

- Support: `215`  Accuracy: `0.7302`  Macro-F1: `0.4896`  Errors: `58`

| Gold | Pred | Count |
|---|---|---:|
| elaboration | conjunction | 11 |
| purpose | elaboration | 6 |
| elaboration | comment | 4 |
| purpose | conjunction | 3 |
| causal | elaboration | 3 |
| elaboration | contrast | 3 |
| mode | frame | 3 |
| organization | conjunction | 3 |
| elaboration | purpose | 2 |
| conjunction | elaboration | 2 |

## Cross-Framework Inconsistencies

| Label | Overall Support | F1 Range | F1 Std | Frameworks >= Threshold |
|---|---:|---:|---:|---:|
| organization | 318 | 0.6204 | 0.3102 | 2 |
| comment | 543 | 0.3775 | 0.1548 | 3 |
| attribution | 423 | 0.3732 | 0.1554 | 3 |
| conjunction | 1657 | 0.3636 | 0.1325 | 4 |
| elaboration | 2408 | 0.3261 | 0.1191 | 4 |
| temporal | 764 | 0.3125 | 0.1200 | 4 |
| frame | 501 | 0.3117 | 0.1400 | 3 |
| causal | 639 | 0.2780 | 0.1059 | 4 |
| query | 672 | 0.2713 | 0.1118 | 3 |
| condition | 272 | 0.2565 | 0.0973 | 4 |

## Rare Labels

- Rare threshold: labels with support <= `50`

| Label | Support | Precision | Recall | F1 |
|---|---:|---:|---:|---:|

## Framework-Internal Rare Labels

- Rare threshold inside each framework: support <= `50`

### dep

| Label | Support | Precision | Recall | F1 |
|---|---:|---:|---:|---:|
| concession | 23 | 0.6316 | 0.5217 | 0.5714 |
| temporal | 24 | 0.6333 | 0.7917 | 0.7037 |
| explanation | 28 | 0.7500 | 0.3214 | 0.4500 |
| condition | 34 | 0.8750 | 0.4118 | 0.5600 |
| causal | 47 | 0.4186 | 0.3830 | 0.4000 |

### pdtb

| Label | Support | Precision | Recall | F1 |
|---|---:|---:|---:|---:|
| explanation | 1 | 0.0000 | 0.0000 | 0.0000 |
| organization | 3 | 0.0000 | 0.0000 | 0.0000 |
| alternation | 34 | 0.7500 | 0.7059 | 0.7273 |

### rst

| Label | Support | Precision | Recall | F1 |
|---|---:|---:|---:|---:|
| alternation | 36 | 0.6452 | 0.5556 | 0.5970 |

### sdrt

| Label | Support | Precision | Recall | F1 |
|---|---:|---:|---:|---:|
| attribution | 14 | 0.6154 | 0.5714 | 0.5926 |
| purpose | 20 | 0.7778 | 0.7000 | 0.7368 |
| condition | 24 | 0.7083 | 0.7083 | 0.7083 |
| alternation | 24 | 0.9474 | 0.7500 | 0.8372 |

