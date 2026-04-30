# Qualitative Analysis

- Predictions: `new_results/qwen3_finetune_results/test_predictions/test_gold_vs_pred.tsv`
- Total examples: `10380`

## Overall

- Accuracy: `0.5224`  Macro-F1: `0.4780`  Weighted-F1: `0.5095`  Errors: `4957`

## Error Cases

### Top Overall Confusions

| Gold | Pred | Count |
|---|---|---:|
| elaboration | conjunction | 287 |
| conjunction | elaboration | 276 |
| frame | elaboration | 195 |
| temporal | conjunction | 175 |
| comment | elaboration | 169 |
| causal | conjunction | 161 |
| explanation | elaboration | 130 |
| organization | elaboration | 105 |
| temporal | elaboration | 103 |
| causal | elaboration | 102 |

## By Language

### eng

- Support: `8578`  Accuracy: `0.5462`  Macro-F1: `0.5010`  Errors: `3893`

| Gold | Pred | Count |
|---|---|---:|
| elaboration | conjunction | 218 |
| conjunction | elaboration | 200 |
| comment | elaboration | 141 |
| frame | elaboration | 137 |
| temporal | conjunction | 127 |
| causal | conjunction | 123 |
| explanation | elaboration | 98 |
| organization | elaboration | 96 |
| contrast | concession | 82 |
| causal | elaboration | 80 |

### fas

- Support: `592`  Accuracy: `0.4155`  Macro-F1: `0.2604`  Errors: `346`

| Gold | Pred | Count |
|---|---|---:|
| elaboration | conjunction | 25 |
| conjunction | elaboration | 22 |
| contrast | elaboration | 16 |
| explanation | elaboration | 16 |
| causal | conjunction | 14 |
| temporal | conjunction | 13 |
| frame | conjunction | 12 |
| contrast | conjunction | 12 |
| attribution | elaboration | 12 |
| comment | elaboration | 11 |

### fra

- Support: `621`  Accuracy: `0.3977`  Macro-F1: `0.2148`  Errors: `374`

| Gold | Pred | Count |
|---|---|---:|
| conjunction | elaboration | 42 |
| frame | elaboration | 40 |
| elaboration | conjunction | 31 |
| temporal | conjunction | 23 |
| frame | temporal | 20 |
| temporal | elaboration | 17 |
| conjunction | temporal | 17 |
| explanation | elaboration | 15 |
| frame | conjunction | 15 |
| causal | conjunction | 12 |

### ita

- Support: `374`  Accuracy: `0.3556`  Macro-F1: `0.1541`  Errors: `241`

| Gold | Pred | Count |
|---|---|---:|
| reformulation | causal | 14 |
| conjunction | causal | 13 |
| temporal | conjunction | 12 |
| reformulation | conjunction | 12 |
| reformulation | temporal | 10 |
| concession | causal | 10 |
| condition | conjunction | 10 |
| condition | temporal | 9 |
| temporal | causal | 9 |
| purpose | conjunction | 9 |

### zho

- Support: `215`  Accuracy: `0.5209`  Macro-F1: `0.2823`  Errors: `103`

| Gold | Pred | Count |
|---|---|---:|
| elaboration | conjunction | 13 |
| frame | elaboration | 9 |
| purpose | elaboration | 8 |
| comment | elaboration | 7 |
| conjunction | elaboration | 6 |
| elaboration | purpose | 5 |
| elaboration | frame | 4 |
| contrast | concession | 3 |
| conjunction | purpose | 3 |
| elaboration | causal | 3 |

## By Framework

### dep

- Support: `2117`  Accuracy: `0.6703`  Macro-F1: `0.4049`  Errors: `698`

| Gold | Pred | Count |
|---|---|---:|
| comment | elaboration | 85 |
| frame | elaboration | 58 |
| elaboration | comment | 39 |
| organization | conjunction | 32 |
| elaboration | purpose | 30 |
| elaboration | conjunction | 29 |
| mode | elaboration | 26 |
| contrast | concession | 26 |
| purpose | elaboration | 24 |
| conjunction | elaboration | 22 |

### pdtb

- Support: `2050`  Accuracy: `0.5244`  Macro-F1: `0.3613`  Errors: `975`

| Gold | Pred | Count |
|---|---|---:|
| causal | conjunction | 84 |
| temporal | conjunction | 66 |
| elaboration | conjunction | 62 |
| conjunction | temporal | 51 |
| conjunction | elaboration | 49 |
| conjunction | causal | 41 |
| causal | elaboration | 34 |
| contrast | concession | 27 |
| elaboration | causal | 26 |
| temporal | causal | 21 |

### rst

- Support: `4478`  Accuracy: `0.4576`  Macro-F1: `0.4210`  Errors: `2429`

| Gold | Pred | Count |
|---|---|---:|
| elaboration | conjunction | 158 |
| conjunction | elaboration | 154 |
| explanation | elaboration | 99 |
| frame | elaboration | 96 |
| organization | elaboration | 91 |
| temporal | conjunction | 82 |
| frame | conjunction | 64 |
| comment | elaboration | 63 |
| contrast | elaboration | 60 |
| explanation | conjunction | 59 |

### sdrt

- Support: `1735`  Accuracy: `0.5072`  Macro-F1: `0.3075`  Errors: `855`

| Gold | Pred | Count |
|---|---|---:|
| query | comment | 65 |
| conjunction | query | 61 |
| conjunction | elaboration | 51 |
| elaboration | query | 44 |
| frame | elaboration | 41 |
| comment | query | 41 |
| elaboration | conjunction | 38 |
| temporal | conjunction | 26 |
| causal | conjunction | 23 |
| comment | elaboration | 21 |

## By Dataset

### eng.dep.scidtb

- Support: `1902`  Accuracy: `0.6872`  Macro-F1: `0.4164`  Errors: `595`

| Gold | Pred | Count |
|---|---|---:|
| comment | elaboration | 78 |
| frame | elaboration | 49 |
| elaboration | comment | 37 |
| organization | conjunction | 29 |
| mode | elaboration | 26 |
| elaboration | purpose | 25 |
| contrast | concession | 23 |
| causal | elaboration | 18 |
| elaboration | frame | 18 |
| purpose | elaboration | 16 |

### eng.erst.gum

- Support: `3558`  Accuracy: `0.4750`  Macro-F1: `0.4324`  Errors: `1868`

| Gold | Pred | Count |
|---|---|---:|
| conjunction | elaboration | 126 |
| elaboration | conjunction | 104 |
| organization | elaboration | 80 |
| explanation | elaboration | 77 |
| temporal | conjunction | 68 |
| frame | elaboration | 67 |
| temporal | elaboration | 49 |
| frame | conjunction | 49 |
| explanation | conjunction | 48 |
| comment | elaboration | 48 |

### eng.pdtb.gum

- Support: `1676`  Accuracy: `0.5621`  Macro-F1: `0.3964`  Errors: `734`

| Gold | Pred | Count |
|---|---|---:|
| causal | conjunction | 75 |
| elaboration | conjunction | 62 |
| temporal | conjunction | 54 |
| conjunction | elaboration | 43 |
| conjunction | temporal | 42 |
| causal | elaboration | 31 |
| conjunction | causal | 28 |
| contrast | concession | 26 |
| elaboration | causal | 25 |
| contrast | conjunction | 18 |

### eng.rst.sts

- Support: `328`  Accuracy: `0.3445`  Macro-F1: `0.2271`  Errors: `215`

| Gold | Pred | Count |
|---|---|---:|
| elaboration | conjunction | 29 |
| frame | elaboration | 20 |
| contrast | elaboration | 10 |
| elaboration | causal | 7 |
| temporal | elaboration | 7 |
| contrast | concession | 6 |
| elaboration | concession | 6 |
| conjunction | elaboration | 6 |
| explanation | elaboration | 6 |
| explanation | concession | 5 |

### eng.sdrt.stac

- Support: `1114`  Accuracy: `0.5682`  Macro-F1: `0.2592`  Errors: `481`

| Gold | Pred | Count |
|---|---|---:|
| query | comment | 65 |
| conjunction | query | 60 |
| elaboration | query | 44 |
| comment | query | 41 |
| elaboration | comment | 19 |
| query | conjunction | 18 |
| comment | conjunction | 16 |
| contrast | query | 14 |
| conjunction | comment | 14 |
| contrast | comment | 14 |

### fas.rst.prstc

- Support: `592`  Accuracy: `0.4155`  Macro-F1: `0.2604`  Errors: `346`

| Gold | Pred | Count |
|---|---|---:|
| elaboration | conjunction | 25 |
| conjunction | elaboration | 22 |
| contrast | elaboration | 16 |
| explanation | elaboration | 16 |
| causal | conjunction | 14 |
| temporal | conjunction | 13 |
| frame | conjunction | 12 |
| contrast | conjunction | 12 |
| attribution | elaboration | 12 |
| comment | elaboration | 11 |

### fra.sdrt.annodis

- Support: `621`  Accuracy: `0.3977`  Macro-F1: `0.2148`  Errors: `374`

| Gold | Pred | Count |
|---|---|---:|
| conjunction | elaboration | 42 |
| frame | elaboration | 40 |
| elaboration | conjunction | 31 |
| temporal | conjunction | 23 |
| frame | temporal | 20 |
| temporal | elaboration | 17 |
| conjunction | temporal | 17 |
| explanation | elaboration | 15 |
| frame | conjunction | 15 |
| causal | conjunction | 12 |

### ita.pdtb.luna

- Support: `374`  Accuracy: `0.3556`  Macro-F1: `0.1541`  Errors: `241`

| Gold | Pred | Count |
|---|---|---:|
| reformulation | causal | 14 |
| conjunction | causal | 13 |
| temporal | conjunction | 12 |
| reformulation | conjunction | 12 |
| reformulation | temporal | 10 |
| concession | causal | 10 |
| condition | conjunction | 10 |
| condition | temporal | 9 |
| temporal | causal | 9 |
| purpose | conjunction | 9 |

### zho.dep.scidtb

- Support: `215`  Accuracy: `0.5209`  Macro-F1: `0.2823`  Errors: `103`

| Gold | Pred | Count |
|---|---|---:|
| elaboration | conjunction | 13 |
| frame | elaboration | 9 |
| purpose | elaboration | 8 |
| comment | elaboration | 7 |
| conjunction | elaboration | 6 |
| elaboration | purpose | 5 |
| elaboration | frame | 4 |
| contrast | concession | 3 |
| conjunction | purpose | 3 |
| elaboration | causal | 3 |

## Cross-Framework Inconsistencies

| Label | Overall Support | F1 Range | F1 Std | Frameworks >= Threshold |
|---|---:|---:|---:|---:|
| attribution | 423 | 0.5872 | 0.2398 | 3 |
| frame | 501 | 0.4830 | 0.2205 | 3 |
| concession | 307 | 0.4478 | 0.1993 | 3 |
| condition | 272 | 0.4263 | 0.1593 | 4 |
| organization | 318 | 0.3824 | 0.1912 | 2 |
| causal | 639 | 0.3638 | 0.1350 | 4 |
| elaboration | 2408 | 0.3608 | 0.1322 | 4 |
| comment | 543 | 0.3433 | 0.1439 | 3 |
| conjunction | 1657 | 0.3216 | 0.1234 | 4 |
| query | 672 | 0.2947 | 0.1382 | 3 |

## Rare Labels

- Rare threshold: labels with support <= `50`

| Label | Support | Precision | Recall | F1 |
|---|---:|---:|---:|---:|

## Framework-Internal Rare Labels

- Rare threshold inside each framework: support <= `50`

### dep

| Label | Support | Precision | Recall | F1 |
|---|---:|---:|---:|---:|
| concession | 23 | 0.0488 | 0.0870 | 0.0625 |
| temporal | 24 | 0.4194 | 0.5417 | 0.4727 |
| explanation | 28 | 0.1250 | 0.0357 | 0.0556 |
| condition | 34 | 0.8333 | 0.1471 | 0.2500 |
| causal | 47 | 0.2857 | 0.1702 | 0.2133 |

### pdtb

| Label | Support | Precision | Recall | F1 |
|---|---:|---:|---:|---:|
| explanation | 1 | 0.0000 | 0.0000 | 0.0000 |
| organization | 3 | 0.0000 | 0.0000 | 0.0000 |
| alternation | 34 | 0.6667 | 0.3529 | 0.4615 |

### rst

| Label | Support | Precision | Recall | F1 |
|---|---:|---:|---:|---:|
| alternation | 36 | 0.6190 | 0.3611 | 0.4561 |

### sdrt

| Label | Support | Precision | Recall | F1 |
|---|---:|---:|---:|---:|
| attribution | 14 | 0.3636 | 0.2857 | 0.3200 |
| purpose | 20 | 0.7059 | 0.6000 | 0.6486 |
| condition | 24 | 0.6667 | 0.4167 | 0.5128 |
| alternation | 24 | 0.9286 | 0.5417 | 0.6842 |

