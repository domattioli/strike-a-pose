# Licensed data assets: SMPL and AMASS

The default pose source is procedural (spec 001 FR-002, amended 2026-10-20),
so only SMPL is required. AMASS is needed only with `pose_source: amass`.

strike-a-pose never downloads these. Both require free registration and are
licensed for **non-commercial research only** (compatible with this repo's
PolyForm Noncommercial license). Read and accept the terms on each site. Do not
commit them: `assets/` and `data/` are git-ignored.

## SMPL body model

1. Register at <https://smpl.is.tue.mpg.de> and accept the SMPL license.
2. Download the SMPL Python model (v1.1.0, 10 shape PCs) and take the
   **neutral** model.
3. Put it at `assets/smpl/SMPL_NEUTRAL.npz` (or `.pkl`; set `smpl_model_path`).

The loader needs these arrays: `v_template, shapedirs, posedirs, J_regressor,
weights, kintree_table, f`. A `.pkl` that only unpickles with `chumpy` must be
converted once to `.npz` on a machine with chumpy installed:

```python
import pickle, numpy as np

d = pickle.load(open("SMPL_NEUTRAL.pkl", "rb"), encoding="latin1")
dense = lambda x: x.toarray() if hasattr(x, "toarray") else np.asarray(x)
np.savez(
    "SMPL_NEUTRAL.npz",
    **{
        k: dense(d[k])
        for k in [
            "v_template",
            "shapedirs",
            "posedirs",
            "J_regressor",
            "weights",
            "kintree_table",
            "f",
        ]
    },
)
```

## AMASS poses (optional, `pose_source: amass`)

1. Register at <https://amass.is.tue.mpg.de> and accept the AMASS license (and
   the licenses of each original sub-collection).
2. Download one or two small sub-collections in the **SMPL+H G** format
   (default config: `BMLmovi`, `CMU`).
3. Extract so each sub-collection is a directory under `assets/amass/`, e.g.
   `assets/amass/CMU/01/01_01_poses.npz`. Files must contain a `poses` array
   (frames × 156). `shape.npz` files are ignored.

## Generate and validate

```bash
uv sync                                     # or: pip install -e .
strike-a-pose-data generate --config configs/default.yaml --seed 0
strike-a-pose-data validate data/generated/smpl_proc_v1
```

If an asset is missing, `generate` exits with code 3, names the missing path,
and writes nothing.

## Committed fixture (operator)

```bash
strike-a-pose-data generate --config configs/fixture.yaml --seed 0 --output-root tests/fixtures
strike-a-pose-data validate tests/fixtures/smpl_fixture
pytest tests/test_fixture.py
git add tests/fixtures/smpl_fixture
```
