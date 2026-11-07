## Recipes (Colab)

Every recipe: run the install cell (skip for reports), **Runtime > Restart session**, edit **Settings**, then **Run after** from Settings.

**1. Real-backbone smoke test:** `RUN_MODE='smoke'`, choose `BACKBONE`/`TRAINING_POLICY`. Uses real clips, the real checkpoint and the GPU; writes `smoke_report.json` and `timing_estimate.json` in the run folder.

**2. Development backbone sweep (no smoke test):** `RUN_MODE='train'`, `EVALUATION_MODE='development'`, keep `PREDICT_PRIVATE=True`. Every backbone in `DEVELOPMENT_BACKBONES` (VideoMAE, V-JEPA 2.1, V-JEPA 2 by default) is trained in turn on the same saved development split and seeds, each in its own run folder. With `PREDICT_PRIVATE=True` each backbone writes a development submission. The sweep ends with `OUTPUT_ROOT/comparisons/<name>__<split>__<attempt>/`: `backbone_comparison.csv` (best inner-validation loss, best epoch, accuracies, time, peak GPU memory, submission path, rank), `backbone_validation_curves.png` and `submission_<backbone>.csv`. A backbone that fails is listed as failed and the others continue. These are development results (half the sources, one seed, `DEV_EPOCHS` epochs, checkpoint chosen on the same validation subset) and are not compute-matched. After a disconnect, run the same settings again: finished epochs and backbones resume from their checkpoints.

**3. Full nested evaluation (the default settings):** `RUN_MODE='train'`, `EVALUATION_MODE='nested_cv'`, `PREDICT_PRIVATE=True` (writes `submission.csv`). Long runs belong on HPC; here they resume epoch by epoch from Drive.

**4. Inference from a saved run or an HPC bundle (no training):** copy the bundle folder to Drive or `/content`, then `RUN_MODE='inference'`, `TRAINED_RUN_DIR=<bundle>`, `INFERENCE_INPUT='private'` (needs the ZIP; writes `submission.csv`), `'public_oof'` (held-out fold per clip), or a CSV `clip,path,preferred_foot,gender` (no ZIP, no labels). Bundles are checksum-verified first; outputs go to `OUTPUT_ROOT/inference/colab/<run>/`, never into the bundle.

**5. Reports only:** `RUN_MODE='reports'`, `TRAINED_RUN_DIR=<run or bundle>`. Skip the install cell. Only pandas/numpy are used; Drive is mounted only if the path is on Drive; the notebook stops after the summary.

**6. Refine a finished run (no video retraining):** `RUN_MODE='refine'`, `TRAINED_RUN_DIR=<OUTPUT_ROOT>/runs/colab/final_videomae_v1__learned_<id>`. Reuses all saved weights; adds a class-weighted pose/motion tree, reselects each task's ensemble on inner predictions, and fits a center-class offset. Writes `<run>__refine_<id>/submission.csv`, `refine_comparison.csv` and the Priority-0 audits; the original run is untouched. A GPU speeds up the inference passes. See `docs/REFINEMENT.md`.

**Keep on Drive:** `OUTPUT_ROOT/runs/colab/...` (checkpoints, reports), `OUTPUT_ROOT/splits/`, `OUTPUT_ROOT/diagnostic_splits/`, detector caches. `/content` (SCRATCH_ROOT) is disposable. Never point Colab at an HPC run folder that is still training; use an exported bundle.

**Validation status:** see `docs/VALIDATION.md`. CPU synthetic tests pass; real-checkpoint/GPU smoke, pilot and full evaluation have not been run by the author of these notebooks.
