# Data model: slice 016

## RoundTripConfig (`eval/roundtrip_config.py`, YAML)

| key | type | default | rule |
|---|---|---|---|
| sample_run | str | required | dir with `config.yaml`, `samples.json` (015 sample run) |
| probe_run | str | required | dir with `config.yaml`, `metrics.json` (006 probe run of the same VAE) |
| smpl_model_path | str | `assets/smpl/SMPL_NEUTRAL.npz` | file must exist at run time |
| run_name | str | `roundtrip` | `RUN_NAME_RE` |
| run_root | str | `runs` | non-empty |
| seed | int | 0 | ≥ 0 (control stream) |
| calibration_percentile | float | 5.0 | in (0, 50] |
| control | bool | true | shuffled-pose control on / off |
| figure_samples | int | 8 | 1..32 |

Unknown keys → `ConfigError`. `with_overrides(sample_run, probe_run, smpl_model_path, run_root)`
ignores `None` values (as `SampleConfig`).

## samples.json (augmented copy)

Source keys unchanged and in place, plus:

```
"round_trip_version": 1,
"source_sample_run": {"path", "samples_sha256"},
"round_trip_criteria": {
  "silhouette_threshold": <source criteria.iou_threshold>,
  "calibration": {"set": "real_test", "split": "test", "percentile": 5.0,
                  "method": "lower", "n": int, "tau": float},
  "probe": {"run": path, "metrics_sha256", "checkpoint_sha256",
            "weight_decay": {"smpl_pose": float, "yaw": float},
            "refit": {"split": str | null, "angle_deg": float | null, "recorded_angle_deg",
                      "accuracy": float | null, "recorded_accuracy", "matches": bool | null}},
  "smpl_model": {"path", "sha256", "manifest_sha256", "matches_manifest": bool},
  "render": "SMPLModel.forward(predicted 72-d pose, zero betas); ring camera of the predicted yaw class",
  "control": {"enabled": bool, "stream": 16, "train_poses": int}},
"sets": {<each source set>: {... 015 fields ..., "round_trip": RT},
         "real_test": {"n", "valid_fraction", "fail_empty", "fail_area", "fail_components",
                      "nn_mse_mean": null, "nn_mse_median": null, "foreground_px_mean",
                      "round_trip": RT}}
RT = {"n", "iou_mean", "iou_quantiles": {"p5", "p25", "p50", "p75", "p95"},
      "plausible_fraction", "plausible_and_valid_fraction", "empty_silhouette",
      "yaw_counts": [8 ints], "control_iou_median": float | null,
      "control_plausible_fraction": float | null}
```

Set order everywhere: `vae_prior`, `ae_prior` (if in source), `vae_recon_val`, `real_test`.
Floats go through `write_json_exact` (`round_sig`, 6 significant digits). Empty set (n = 0):
`iou_mean`, quantiles, fractions null.

## round_trip_per_image.json

`{"sets": {<set>: {"iou": [float], "yaw": [int], "valid": [bool], "control_iou": [float] | null}}}`

## run_info.json

`argv, cwd, seed, git, env, sample_run {path, samples_sha256}, probe_run {path, metrics_sha256},
checkpoint {path, sha256}, reference_checkpoint {path, sha256} | null, smpl_model {path, sha256},
cache {path, hit}, started_utc, finished_utc, wall_s, peak_rss_mb`.
