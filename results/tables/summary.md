# Results summary

Seeds used: Baseline: [0, 1, 2]; + CDA: [0, 1, 2]

Values are mean ± standard deviation over seeds (single value if only one seed).

## 1. Classification quality (test set)

| condition | seeds | train size | test accuracy | test macro-F1 | F1 anger | F1 fear | F1 joy | F1 sadness |
|---|---|---|---|---|---|---|---|---|
| Baseline | 3 | 14124 | 0.9679 ± 0.0028 | 0.9590 ± 0.0037 | 0.931 ± 0.006 | 0.949 ± 0.006 | 0.990 ± 0.001 | 0.967 ± 0.003 |
| + CDA | 3 | 16033 | 0.9636 ± 0.0027 | 0.9530 ± 0.0035 | 0.921 ± 0.003 | 0.939 ± 0.009 | 0.989 ± 0.002 | 0.963 ± 0.003 |

## 2. Bias per emotion: Gender (male × female)

'sig' = number of seeds with Holm-corrected p < 0.05.

| emotion | condition | seeds | DP | avg.Δ | ACS | t-test sig (Holm) | Wilcoxon sig (Holm) |
|---|---|---|---|---|---|---|---|
| anger | Baseline | 3 | 1.000 ± 0.000 | 0.0031 ± 0.0033 | -0.0036 ± 0.0044 | 3/3 | 3/3 |
| anger | + CDA | 3 | 1.000 ± 0.000 | 0.0015 ± 0.0007 | -0.0012 ± 0.0007 | 2/3 | 3/3 |
| fear | Baseline | 3 | 0.985 ± 0.023 | 0.0079 ± 0.0009 | 0.0233 ± 0.0082 | 3/3 | 3/3 |
| fear | + CDA | 3 | 0.990 ± 0.005 | 0.0083 ± 0.0006 | 0.0168 ± 0.0047 | 3/3 | 3/3 |
| joy | Baseline | 3 | 0.987 ± 0.010 | 0.0087 ± 0.0003 | 0.0117 ± 0.0081 | 2/3 | 3/3 |
| joy | + CDA | 3 | 0.991 ± 0.000 | 0.0080 ± 0.0024 | 0.0098 ± 0.0110 | 2/3 | 2/3 |
| sadness | Baseline | 3 | 0.997 ± 0.000 | 0.0054 ± 0.0018 | 0.0124 ± 0.0020 | 2/3 | 3/3 |
| sadness | + CDA | 3 | 0.997 ± 0.003 | 0.0064 ± 0.0002 | 0.0187 ± 0.0160 | 2/3 | 3/3 |

## 2. Bias per emotion: Race (EA × AA)

'sig' = number of seeds with Holm-corrected p < 0.05.

| emotion | condition | seeds | DP | avg.Δ | ACS | t-test sig (Holm) | Wilcoxon sig (Holm) |
|---|---|---|---|---|---|---|---|
| anger | Baseline | 3 | 0.995 ± 0.008 | 0.0066 ± 0.0046 | -0.0077 ± 0.0069 | 3/3 | 1/3 |
| anger | + CDA | 3 | 0.998 ± 0.004 | 0.0063 ± 0.0035 | -0.0059 ± 0.0083 | 1/3 | 3/3 |
| fear | Baseline | 3 | 0.992 ± 0.003 | 0.0148 ± 0.0030 | -0.0702 ± 0.0485 | 2/3 | 1/3 |
| fear | + CDA | 3 | 0.994 ± 0.008 | 0.0149 ± 0.0044 | -0.1348 ± 0.0653 | 0/3 | 3/3 |
| joy | Baseline | 3 | 0.990 ± 0.005 | 0.0151 ± 0.0038 | -0.0125 ± 0.0056 | 0/3 | 2/3 |
| joy | + CDA | 3 | 0.998 ± 0.002 | 0.0107 ± 0.0015 | -0.0106 ± 0.0098 | 0/3 | 2/3 |
| sadness | Baseline | 3 | 0.986 ± 0.024 | 0.0136 ± 0.0028 | -0.0337 ± 0.0119 | 2/3 | 2/3 |
| sadness | + CDA | 3 | 0.984 ± 0.005 | 0.0174 ± 0.0036 | -0.0514 ± 0.0161 | 2/3 | 2/3 |

## 3. Overall bias (mean avg.Δ over the four emotions)

'beyond seed noise?' compares the baseline-vs-CDA difference with the seed-to-seed standard deviation (a rough rule of thumb).

| domain | condition | seeds | mean avg.Δ over emotions | lowest DP | change vs baseline | beyond seed noise? |
|---|---|---|---|---|---|---|
| gender | Baseline | 3 | 0.00625 ± 0.00050 | 0.975 ± 0.015 |  |  |
| gender | + CDA | 3 | 0.00604 ± 0.00057 | 0.989 ± 0.003 | -0.00021 (-3.4%) | no (inconclusive) |
| race | Baseline | 3 | 0.01249 ± 0.00178 | 0.978 ± 0.017 |  |  |
| race | + CDA | 3 | 0.01234 ± 0.00242 | 0.984 ± 0.005 | -0.00015 (-1.2%) | no (inconclusive) |

## 4. Paired comparison over sentence pairs (seed-averaged)

Each pair’s absolute score gap is averaged over seeds, then baseline and CDA are compared pair by pair. Pairs share templates, so they are not fully independent: treat the p-value as indicative.

| domain | pairs | mean gap baseline | mean gap CDA | relative change | pairs with smaller gap under CDA | Wilcoxon p (paired over pairs) |
|---|---|---|---|---|---|---|
| gender | 1400 | 0.00625 | 0.00604 | -3.4% | 82.1% | 1.33e-68 |
| race | 2800 | 0.01249 | 0.01234 | -1.2% | 74.0% | 9.64e-82 |
