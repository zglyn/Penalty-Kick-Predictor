# Baseline releases, bundles and artifact migration

## What "freeze the baseline" means

Freezing a baseline saves an immutable snapshot of code, configuration, dependencies and splits, and runs that snapshot without editing it. This is unrelated to `TRAINING_POLICY='frozen_probe'`, which freezes encoder **weights**. A frozen release can still fine-tune every parameter.

1. **Validate first.** Pass the validation ladder levels A–D as far as the allocation allows (see `docs/VALIDATION.md`).
2. **Set the freeze switches** in the HPC notebook: the baseline settings, plus `FREEZE_RELEASE=True` and `NOTEBOOK_PATH='<path of this .ipynb>'`.
3. **Run the cells up to the release cell.** Run Settings, platform helpers, validation, environment and artifacts, then `batch_files_and_release`, which stops after freezing. It writes:
   - `run_penalty_kick.py`, `penalty_kick.sbatch` and `<EXPERIMENT_NAME>__<RUN_MODE>.json`, next to the notebook.
   - `OUTPUT_ROOT/releases/<EXPERIMENT_NAME>__<hash>/`, containing the notebook, runner, sbatch script, job JSON and `settings.json`. All files are read-only.
   - `RELEASE.json` in that folder, holding the release id, sha256 of every file, the shared-code fingerprint, the environment (Python, host, Slurm), the backbone and the training policy.
4. **Launch from the release folder.** The runner re-verifies every checksum before it runs and refuses a modified release.
5. **Record the Git commit too.** If you keep the notebook in Git (this repository), note the commit or tag: the code fingerprint in `RELEASE.json` and every `attempt.json` ties results to the generated shared code.

The run folder then records the rest of the experiment identity:
- `attempts/*/attempt.json` holds settings, versions, hardware and the code fingerprint.
- `experiment_manifest.json` holds the dataset hash, split ID, seeds, class mapping and configuration (backbone checkpoint and revision, preprocessing, features, policy, optimizer).
- `stages/*/backbone_serialization.json` records the resolved Hugging Face revision or the V-JEPA 2.1 checkpoint sha256.

## Bundles: moving finished artifacts between platforms

- **What goes in.** `export_bundle(run, destination)`, or `EXPORT_BUNDLE_TO`, copies the inference manifest and every file it lists, the reports, and the per-seed loss histories. It never copies `*_latest.pt` optimizer checkpoints.
- **How it is written.** Paths are relative and checksummed. Files are staged in `.<bundle>.partial/`, each copy is verified, `BUNDLE_COMPLETE.json` is written last, and the folder is renamed.
- **Checking a copy.** `verify_bundle(path)`, which inference and reports run automatically when given a bundle, rejects:
  - a missing marker (an interrupted transfer)
  - missing files
  - size or checksum changes
- **Relocation.** Bundles reload from any root. Inference then rebuilds evaluation crops label-free, because bundles omit the large training-time crop caches.
- **Credentials.** Use Open OnDemand's file browser or your own tools to move bundles. No credentials are stored in the notebooks.

## Artifact compatibility

| Artifact | Status |
|---|---|
| Pre-update Colab runs at `OUTPUT_ROOT/learned_<id>/` | Found automatically on Colab for the same configuration and resumed in place. New runs use `runs/<platform>/...`. |
| Inference manifests (schema 1) | Unchanged. Prediction caches made before this update are **not reused**, because the prediction key now includes metadata and fitted identity. They are recomputed, which is intended. |
| Runs trained before inference manifests | `MIGRATE_LEGACY_RUN=True` once. It only adds `class_prior.json` and `inference_manifest.json`. |
| Stage folders | The same fitted-stage identity is reused across prediction lists, and legacy-named folders are reused when their fit manifests match. |
| Nested runs from the previous notebook version | `fold_results.json` and `component_predictions_fold<k>.npz` at the run root are still read by the summary and the exporter. New runs write per-fold files under `folds/fold<k>/`, then pool them into `fold_results.json` and `predictions.npz`. Resuming an in-progress old nested run re-runs each fold through the new per-fold path; existing stage checkpoints are reused. |
| Checkpoints | Format unchanged. Resume checkpoints without `model_identity` are accepted as VideoMAE fine-tuning only. |
| Split files | `OUTPUT_ROOT/diagnostic_splits/<split id>.json` (nested) is unchanged. The new development splits are in `OUTPUT_ROOT/splits/dev_<id>.json`. |
| Detector caches | The `detector_cache_<id>` key is unchanged. Writes now use unique temporary names, so concurrent jobs are safe. |
| Experiment identity | Non-VideoMAE or frozen-probe configurations now include the optimizer settings, so their identities differ from the previous notebook's. The default VideoMAE fine-tuning identity is unchanged. |
