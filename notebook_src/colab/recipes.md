## Recipes (Colab)

Every recipe: run the install cell (skip for reports), **Runtime > Restart session**, edit **Settings**, then **Run after** from Settings.

**1. Real-backbone smoke test (default):** `RUN_MODE='smoke'`, choose `BACKBONE`/`TRAINING_POLICY`. Uses real clips, the real checkpoint and the GPU; writes `smoke_report.json` and `timing_estimate.json` in the run folder.

**2. Development experiment (bounded):** `RUN_MODE='train'`, `EVALUATION_MODE='development'`, a new `EXPERIMENT_NAME` per idea, one change at a time (e.g. `TRAINING_POLICY='frozen_probe'` vs `'finetune'`). Same saved development split for every idea (`OUTPUT_ROOT/splits/dev_*.json`). Results are labeled DEVELOPMENT ONLY and are never a submission or a full OOF table. After a disconnect: same settings with `RUN_MODE='resume'`.

**3. Full nested evaluation (deliberate):** `RUN_MODE='train'`, `EVALUATION_MODE='nested_cv'` (optionally `PREDICT_PRIVATE=True`). Long runs belong on HPC; here they resume epoch by epoch from Drive.

**4. Inference from a saved run or an HPC bundle (no training):** copy the bundle folder to Drive or `/content`, then `RUN_MODE='inference'`, `TRAINED_RUN_DIR=<bundle>`, `INFERENCE_INPUT='private'` (needs the ZIP; writes `submission.csv`), `'public_oof'` (held-out fold per clip), or a CSV `clip,path,preferred_foot,gender` (no ZIP, no labels). Bundles are checksum-verified first; outputs go to `OUTPUT_ROOT/inference/colab/<run>/`, never into the bundle.

**5. Reports only:** `RUN_MODE='reports'`, `TRAINED_RUN_DIR=<run or bundle>`. Skip the install cell. Only pandas/numpy are used; Drive is mounted only if the path is on Drive; the notebook stops after the summary.

**Keep on Drive:** `OUTPUT_ROOT/runs/colab/...` (checkpoints, reports), `OUTPUT_ROOT/splits/`, `OUTPUT_ROOT/diagnostic_splits/`, detector caches. `/content` (SCRATCH_ROOT) is disposable. Never point Colab at an HPC run folder that is still training; use an exported bundle.

**Validation status:** see `docs/VALIDATION.md`. CPU synthetic tests pass; real-checkpoint/GPU smoke, pilot and full evaluation have not been run by the author of these notebooks.
