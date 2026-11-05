# Inspection record and mapping (before editing)

## Input

The input is `reference/Penalty_Kick_Colab_input.ipynb`, kept unchanged. It is the notebook from the previous session, which the user stored at `...\HW3\HW3\Penalty_Kick_Colab.ipynb`.
- Companion tests from that session are in `tests/` (ported).
- Its docs were folded into `docs/`.
- The HPC codelab is `Codelab_attention.ipynb`. Only its first markdown cell concerns HPC; the rest is a ViT/GradCAM tutorial.

## Verified current behavior of the input (and issues reproduced)

1. **Prediction cache key omitted metadata.**
   - `run_saved_inference` used one key, `clip_cache_key(clip, content, preprocessing)`, for both crops and predictions.
   - Changing `preferred_foot` or `gender`, which feed the numeric MLP and the CatBoost trees, therefore reused stale predictions.
   - Fixed with a separate `prediction_cache_key(crop key, metadata, fitted artifact id, feature signature)`. Crop caches stay metadata-independent. Test: `test_changed_metadata_changes_prediction_identity_and_recomputes`.
2. **Setup always used Colab, Drive and `/content`.**
   - The guard/import cell imported `google.colab`, mounted Drive and used `/content/penalty_data` in every mode.
   - Fixed by moving everything platform-specific into the `platform_helpers` and `environment` cells. Drive is mounted only for paths that need it. The HPC notebook contains no such code (static check).
3. **Custom-video inference needed the competition ZIP.**
   - Custom inference went through ZIP hashing/extraction and the `len(train)==1300`/`len(test)==900` assertions.
   - Fixed: `NEEDS_DATASET` is false for CSV inference, and the dataset cell creates empty tables instead. Test: `test_custom_inference_needs_no_competition_files_or_labels`.
4. **Reports mode was not lightweight.**
   - It still imported torch, ultralytics and transformers, and built a tiny model.
   - Fixed: reports mode runs only Settings, platform helpers, validation, environment, artifacts, constants/summary and the reports cell, then stops (`StopNotebook`, or `SystemExit(0)` in batch). Tests run both notebooks in fresh processes and assert that none of torch, cv2, av, transformers, ultralytics, catboost or sklearn are imported.
5. **The V-JEPA tests do not prove real checkpoint/GPU operation.** This is unchanged and stated in `docs/VALIDATION.md`.

## Input cells mapped to outputs

| Input cell (marker) | Output | Edit |
|---|---|---|
| install | Colab `install` | unchanged |
| run_settings | `colab/settings.py`, `hpc/settings.py` | Rewritten to the spec's keys: `RUN_MODE` smoke/train/resume/inference/reports, paths, `EXPERIMENT_NAME`, `FOLD_SELECTION`, `DEV_*`, utilities. `QUICK_SMOKE` became `RUN_MODE='smoke'`. |
| restart_guard_and_imports | `colab/environment.py`, `hpc/environment.py`, shared `heavy_imports` | Guard and Drive moved to Colab only; HPC version checks and Slurm checks added; heavy imports and GPU visibility check split out |
| dataset_loading | shared | Conditional on `NEEDS_DATASET`; extraction into `SCRATCH_ROOT` via a partial folder and rename |
| backbone_adapters | shared | unchanged |
| inference_helpers | shared | Metadata in prediction key; `output_root`; crop reuse checks existence (bundles); unique temp names |
| diagnostics_helpers | shared | Smoke output moved to `SCRATCH_ROOT` |
| experiment_configuration | shared | Runs under `runs/<platform>/`, experiment identity, attempt record, optimizer in identity (non-legacy), development preset, ZIP-free inference, bundle verification |
| extraction, model_and_training | shared | Unique temp names for concurrent writers; `TRAINING_MODES` |
| roles_and_crops | shared | unchanged |
| nested_orchestration | shared | Split into `run_nested_fold` (isolated `folds/fold<k>/`, fold lock, completion marker) and `aggregate_nested` (identity check, aggregate lock); `run_nested_diagnostics(fold_selection=...)` |
| inference_glue | shared | Outputs outside the trained run; custom-CSV metadata validation and relative paths |
| smoke_holdout_dispatch | shared | Adds development preset; routing with writer locks; fold-job routing |
| validation report, heatmaps | shared | Run only after a complete aggregation |
| submission | shared | Browser download moved to the Colab platform helper |
| pipeline_summary_helpers / _run | shared | `development` status; partial fold results; `output_dir`; constants moved here |
| *(new)* run_artifacts | shared | Atomic writes, writer locks, attempts, bundles, releases |
| *(new)* reports_mode | shared | Lightweight early exit |
| *(new)* batch_files_and_release, run_penalty_kick.py, penalty_kick.sbatch | HPC | Batch execution and frozen releases |

## Cluster facts used

These are listed in `hpc/ENVIRONMENT.md`, together with the unresolved cluster-specific details.
