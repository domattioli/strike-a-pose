<!-- provenance: author=domattioli model=claude-opus-5-5 effort=max date=2026-10-07 skill=speckit-analyze repo=strike-a-pose session=session_012P6L2vQy2nq18wTJ6TLzTC -->
# Data Model: Kill-Test MVP for Calibrated Multi-View Body-Measurement Uncertainty

**Branch**: `001-kill-test-mvp` | **Date**: 2026-10-07 | **Spec**: [spec.md](spec.md) | **Plan**: [plan.md](plan.md)

Units: lengths in metres inside the body models and cameras, in centimetres in every measurement, interval, and table. Angles in degrees in configuration and tables, in radians inside code. Vertical axis `y`. File layouts and column lists are in [contracts/artifacts.md](contracts/artifacts.md); this file defines the entities, their fields, validation rules, and state transitions.

## Entities

### BodySample
One synthetic subject (spec: Body Sample).

| Field | Type | Rule |
|---|---|---|
| body_id | int | unique, 0-based, contiguous within a run |
| shard | int | `body_id // data.shard_size` |
| split | enum train, cal, test | contiguous ranges by body_id (DataSplit) |
| betas | float[10] | standard normal, clipped to `[-body.beta_clip, body.beta_clip]` |
| pose_root | float[3] | axis-angle root orientation; zero (an upright body) for both pose sources, because AMASS root orientations are discarded (research R2) and the rig's uniform azimuth supplies the viewing direction |
| pose_body | float[63] | 21 joints by 3 axis-angle; passed the PoseFilter |
| pose_source | enum amass, limits | from configuration |
| pose_rejections | int | draws rejected before this pose was accepted |
| measurements_true | float[5] | height, chest, waist, hip, thigh in cm, canonical pose (FR-004); finite |
| flags | set of enum empty_mask, out_of_frame, slice_nan | non-empty flags exclude the body from training and count it as skipped in evaluation; `out_of_frame` is set when any of the four views touches the border, so the body rate is up to four times the per-view rate (DataSplit sets the floor on unflagged bodies) |

### CameraPlacement
One view of one body (spec: Camera Placement).

| Field | Type | Rule |
|---|---|---|
| body_id, view_index | int, int in 0..3 | four cameras per body in the dataset |
| K | float[3,3] | fixed intrinsics from `camera.image_size` and `camera.focal_px` |
| R_true, t_true | float[3,3], float[3] | world-to-camera; look-at the center of the posed mesh's axis-aligned bounding box plus `camera.lookat_jitter_m` jitter per axis, so raised arms and wide poses stay in frame |
| azimuth_deg, distance_m, height_m | float | within `camera.*` ranges; azimuths of one rig at least `camera.min_separation_deg` apart |
| noise_axis | float[3] | unit vector, drawn once per (body, view) at generation and stored; every cell rotates about it by its own `noise_deg`; training draws the angle only, about the stored axis |
| R_given(noise_deg) | derived | `R_noise(noise_deg, noise_axis) @ R_true`; `t` unchanged (FR-005) |

### Silhouette
| Field | Type | Rule |
|---|---|---|
| body_id, view_index | int | key |
| mask | uint8[H, W] in {0, 1} | `H = W = camera.image_size`; stored bit-packed in the shard |
| area_fraction | float | `> 0`, else flag empty_mask |
| touches_border | bool | true flags out_of_frame |

### PoseFilter result
| Field | Type | Rule |
|---|---|---|
| joint_angle_ok | bool | every joint rotation angle within `pose.limits_deg[joint]` |
| self_intersection_ok | bool | no non-adjacent capsule pair overlaps by more than `pose.capsule_overlap_cm` (default 1 cm) |
| accepted | bool | both true |

### PerViewPosterior and FusedPosterior
| Field | Type | Rule |
|---|---|---|
| mu_v, logvar_v | float[L] each, `L = model.latent_dim` | the per-view posterior: the product of the prior expert `N(0, I)` and the encoder expert of view v (encoder input: mask, `R_given`, `t`); with one view the fused posterior equals it exactly |
| mu, logvar | float[L] | product of experts with the `N(0, I)` prior expert; `exp(logvar) <= exp(logvar_v)` for every contributing v (FR-007, SC-004) |

### MeasurementInterval
Per (body, cell, measurement).

| Field | Type | Rule |
|---|---|---|
| m_median | float cm | median over `predict.n_samples` decoded samples |
| spread | float cm | 1.4826 times the median absolute deviation; floored at `calibrate.spread_floor_cm` |
| q_hat | float | from CalibrationQuantile of the same cell and measurement |
| lower, upper | float cm | `m_median -/+ q_hat * spread`; lower clipped at 0 with `clipped = true` |
| width | float cm | `upper - lower` |
| covered | bool | `lower <= m_true <= upper`, inclusive |

### DataSplit
| Field | Type | Rule |
|---|---|---|
| n_train, n_cal, n_test | int | from `data.*`; `n_cal >= calibrate.min_cal` else calibration refuses (FR-011) |
| ranges | train `[0, n_train)`, cal `[n_train, n_train + n_cal)`, test `[n_train + n_cal, n_train + n_cal + n_test)` | disjoint by construction; asserted by test |
| monitor | the last `n_monitor = ceil(0.05 n_train)` bodies of the train range | `split` stays `train` in the manifest; loss logged as `monitor_loss` in `history.csv`; no gradient step; no selection reads it |
| n_unflagged_cal, n_unflagged_test | int | bodies without flags; generation exits 4 when either is below `data.min_unflagged` (full 2,000, SC-002) and names both counts |

### ExperimentCondition (cell)
| Field | Type | Rule |
|---|---|---|
| views | int in `evaluate.views` (default 1, 2, 4) | uses cameras `0..views-1` of each rig |
| noise_deg | float in `evaluate.noise_deg` (default 0, 2, 5) | rotation angle of `R_noise` |
| cell_id | string `v{views}_n{noise_deg}` | key in file names |

### CalibrationQuantile
| Field | Type | Rule |
|---|---|---|
| cell_id, measurement | key | 9 by 5 rows |
| n_cal | int | bodies without flags in the cal split |
| alpha | float | `calibrate.alpha` (0.10) |
| q_hat | float | `ceil((n_cal + 1)(1 - alpha))`-th smallest normalized score |

### ResultCell
| Field | Type | Rule |
|---|---|---|
| cell_id, measurement | key | 9 by 5 rows in `results.csv` |
| n_test, n_cal | int | counts used |
| coverage | float in [0, 1] | mean of `covered` over test bodies |
| median_width_cm | float | median of `width` |
| mae_cm | float | mean of `abs(m_median - m_true)` (secondary metric) |
| mean_signed_error_cm | float | mean of `m_median - m_true` |
| clipped_count | int | intervals clipped at 0 |
| in_band | bool | inclusive (FR-013), compared as integers: `100 * covered_count >= 87 * n_test` and `100 * covered_count <= 93 * n_test` |
| config_hash, seed, code_version, hardware_class | provenance | copied from the RunRecord into every row (FR-025) |

### KillVerdict (FR-014)
| Field | Type | Rule |
|---|---|---|
| noise_deg | 0 | fixed |
| ratios | map circumference to float | `median_width(v4) / median_width(v1)` for chest, waist, hip, thigh |
| median_ratio | float | `numpy.median` of the four ratios: the mean of the 2nd and 3rd smallest |
| threshold | 0.70 | constant in `verdict.py`; a configuration that differs exits 2 |
| cells_in_band | bool | true when all eight compared cells (v1 and v4, four circumferences, 0 degrees) have `in_band = true`. `false` is a legitimate KILL, not an error: `out_of_band_cells` lists the failing compared cells, the verdict line ends with `; comparison invalid for cells <ids>`, the command exits 0, and `sap run` continues to `report` |
| out_of_band_cells | list of string | compared cells with `in_band = false`, written as `<cell_id>:<measurement>` (for example `v4_n0:waist`); empty when `cells_in_band` is true |
| verdict | enum PASS, KILL | PASS iff `median_ratio <= threshold and cells_in_band and not invalid_comparison and sc004_violations == 0`; an SC-004 violation therefore never leaves a PASS on disk |
| invalid_comparison | bool | true only when one of the four width ratios is non-finite or a compared cell is missing from `results.csv`; the verdict is then KILL, and the command writes `verdict.json` and `verdict.md` first, then exits 4. A compared cell outside the band does not set this field; it sets `cells_in_band = false` |
| sc004_violations | int | count from `evaluate/sc004.json`; above 0, the command writes `verdict.json` and `verdict.md` first, then exits 4 |
| reported_only | height row and the 2 and 5 degree rows | shown next to the verdict, never used in it |
| run_record | RunRecord | attached |

### RunRecord (FR-025)
| Field | Type | Rule |
|---|---|---|
| config_hash | hex string | SHA-256 of the canonical YAML dump |
| seed | int | from configuration |
| code_version | string | package version plus git commit when available |
| hardware_class | enum cpu, gpu | with device name |
| versions | map | python, numpy, torch, opencv, smplx, sam2 when installed |
| timings | map stage to seconds | written at stage end |

### RealImageSubject (FR-017 to FR-020)
| Field | Type | Rule |
|---|---|---|
| dataset, subject_id | enum bodym_testA, bodym_testB, ssp3d; string | key |
| views | list of (mask, nominal placement) | BodyM: front, side; SSP-3D: one |
| mask_source | enum provided, sam2 | per configuration |
| mask_status | enum ok, skipped_no_mask, skipped_unusable, skipped_multi_person | skipped rows are counted, never dropped; decided in this order: no mask gives `skipped_no_mask`; two connected components that each hold at least 10% of the mask area give `skipped_multi_person`; a mask that fails the usability rule gives `skipped_unusable`; otherwise `ok` |
| measurements_true | float[5] cm | BodyM: tape values; SSP-3D: from the SMPL mesh in canonical pose with the FR-004 rules |
| intervals | MeasurementInterval[5] | using the CalibrationQuantile of the matching cell (`v2_n0` for BodyM, `v1_n0` for SSP-3D) |

## State transitions

### Stage markers (FR-029)
Each stage directory holds `DONE.json` once complete.

```text
absent --(stage runs)--> running --(outputs written, marker written atomically)--> done
done --(config_hash or an input marker differs)--> stale --(--resume re-runs the stage)--> running
running --(time budget reached or session ends)--> partial --(--resume continues from the last shard, epoch, or cell)--> running
```

Stage order and inputs: generate (none) -> train (generate) -> predict (generate, train) -> calibrate (predict) -> evaluate (calibrate) -> verdict (evaluate) -> report (evaluate, verdict). `real-eval` depends on train and calibrate.

### Body flags
`ok -> flagged(empty_mask | out_of_frame | slice_nan)` at generation; a flagged body is excluded from training and counted as skipped in calibration and evaluation; counts appear in `data/summary.json` and in `results.csv` (`n_test` excludes them). Generation exits 4 when fewer than `data.min_unflagged` unflagged calibration or test bodies remain.

## Validation rules that tests assert

- Splits are disjoint by `body_id` and cover exactly `n_train + n_cal + n_test` bodies.
- One view with 0 degrees noise: the fused posterior equals that view's posterior (spec edge case).
- Fused variance is at most every contributing view's variance, and adding a view never increases it (SC-004).
- Calibration refuses `n_cal < calibrate.min_cal` with a message that names the minimum (FR-011).
- The verdict read back from `verdict.json` equals the verdict recomputed from the predict outputs (SC-003).
- Every table and record carries `config_hash`, `seed`, `code_version`, `hardware_class` (FR-025).
- Every body id yielded by the training sampler lies in `[0, n_train - n_monitor)` (the monitor slice takes no gradient step); calibration refuses a predict file whose `split` is not `cal`; evaluation refuses one whose `split` is not `test`.
- On run outputs, per test body and noise level, `latent_var_mean` never increases over the view counts of `evaluate.views` in ascending order (`v4 <= v2 <= v1` by default), within a relative tolerance of 1e-5 of the fewer-view value (SC-004). Per-view posteriors are encoded once and reused across the view-count cells, so a correct run has no violation in floating point. Violations are counted in `evaluate/sc004.json`; the verdict command writes the verdict files, then exits 4.
- A compared cell outside the band gives KILL with `cells_in_band = false` and exit 0; only a non-finite ratio or a missing compared cell sets `invalid_comparison` (exit 4 after the verdict files are written).
- Generation exits 4 when fewer than `data.min_unflagged` unflagged calibration or test bodies remain, and names both counts.
- `verdict.py` refuses a configuration whose FR-014 keys differ from its constants (exit 2).
