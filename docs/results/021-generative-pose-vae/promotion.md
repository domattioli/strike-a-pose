| Variant | Val MSE vs R0 | Probe MPJPE val (mm) | Δ mm | Verdict (015) | Reason | Pose track | Pixel cost |
|---|---|---|---|---|---|---|---|
| V1 | +8.6 % | 68.0 | -63.0 | DROPPED | +8.6 % MSE (gap 0.6 %); -63.0 mm probe (gap 3.3); MSE > +5 % | PROMOTED (pose) | +8.4 % |
| V2 | -26.0 % | 127.4 | -3.6 | WIN-pixel | -26.0 % MSE (gap 0.6 %); -3.6 mm probe (gap 3.3) | PROMOTED (pose) | -26.1 % |
| L1 | -28.9 % | 120.5 | -10.5 | WIN-pixel | -28.9 % MSE (gap 0.6 %); -10.5 mm probe (gap 3.3) | PROMOTED (pose) | -29.0 % |
| L2 | +5.8 % | 67.1 | -63.9 | DROPPED | +5.8 % MSE (gap 0.6 %); -63.9 mm probe (gap 3.3); MSE > +5 % | PROMOTED (pose) | +5.7 % |
| L2-seed1 | +6.7 % | 67.0 | -64.0 | DROPPED | +6.7 % MSE (gap 0.6 %); -64.0 mm probe (gap 3.3); MSE > +5 % | PROMOTED (pose) | +6.5 % |

| Variant | Test MSE vs R0 | Probe MPJPE test (mm) | Δ mm | Verdict (015) | Reason | Pose track | Pixel cost |
|---|---|---|---|---|---|---|---|
| V1 | +8.8 % | 66.1 | -61.3 | DROPPED | +8.8 % MSE (gap 0.5 %); -61.3 mm probe (gap 3.0); MSE > +5 % | PROMOTED (pose) | +8.5 % |
| V2 | -25.6 % | 123.9 | -3.6 | WIN-pixel | -25.6 % MSE (gap 0.5 %); -3.6 mm probe (gap 3.0) | PROMOTED (pose) | -25.8 % |
| L1 | -28.7 % | 117.3 | -10.2 | WIN-pixel | -28.7 % MSE (gap 0.5 %); -10.2 mm probe (gap 3.0) | PROMOTED (pose) | -28.9 % |
| L2 | +5.9 % | 64.9 | -62.5 | DROPPED | +5.9 % MSE (gap 0.5 %); -62.5 mm probe (gap 3.0); MSE > +5 % | PROMOTED (pose) | +5.7 % |
| L2-seed1 | +6.7 % | 65.2 | -62.2 | DROPPED | +6.7 % MSE (gap 0.5 %); -62.2 mm probe (gap 3.0); MSE > +5 % | PROMOTED (pose) | +6.5 % |

| Variant | Verdict val | Verdict test | Agree | Pose track val / test |
|---|---|---|---|---|
| V1 | DROPPED | DROPPED | yes | PROMOTED (pose) (PROMOTED (pose) / PROMOTED (pose)) |
| V2 | WIN-pixel | WIN-pixel | yes | PROMOTED (pose) (PROMOTED (pose) / PROMOTED (pose)) |
| L1 | WIN-pixel | WIN-pixel | yes | PROMOTED (pose) (PROMOTED (pose) / PROMOTED (pose)) |
| L2 | DROPPED | DROPPED | yes | PROMOTED (pose) (PROMOTED (pose) / PROMOTED (pose)) |
| L2-seed1 | DROPPED | DROPPED | yes | PROMOTED (pose) (PROMOTED (pose) / PROMOTED (pose)) |
