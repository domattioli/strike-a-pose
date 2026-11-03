## val

| model | head mm | MSE | MSE / L2 | pixel (≤ L2 +5 %) | vs R0 64-d +5 % | prior valid | RT margin pts |
|---|---|---|---|---|---|---|---|
| V1 | 66.8 | 0.00171248 | 1.0219 | PASS | FAIL | 0.416 | 15.1 |
| V2 | n/a | 0.00116767 | 0.6968 | PASS | PASS | 0.460 | 26.5 |
| R2b-v3 | n/a | 0.00160545 | 0.9580 | PASS | PASS | 0.590 | 24.9 |

## test

| model | head mm | MSE | MSE / L2 | pixel (≤ L2 +5 %) | vs R0 64-d +5 % | prior valid | RT margin pts |
|---|---|---|---|---|---|---|---|
| V1 | 64.8 | 0.00169166 | 1.0231 | PASS | FAIL | 0.416 | 15.1 |
| V2 | n/a | 0.00115654 | 0.6995 | PASS | PASS | 0.460 | 26.5 |
| R2b-v3 | n/a | 0.00158037 | 0.9558 | PASS | PASS | 0.590 | 24.9 |

V2 vs V1 prior: HURTS (diff -0.044, sd 0.022)
VERDICT PARTIAL failing=valid next=head on z
