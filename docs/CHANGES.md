# Change summary

## New

- **Two generated notebooks from one source.** `notebooks/Penalty_Kick_HPC.ipynb` and `notebooks/Penalty_Kick_Colab_VideoMAE_Full.ipynb` are both built from `notebook_src/` by `tools/build_notebooks.py`.
  - Shared cells are byte-identical. `tools/check_consistency.py` detects drift and any Colab code in the HPC notebook.
  - Each notebook still works from its documented uploads: all helper code is embedded.
- **One Settings cell per platform**, with the same keys on both: `RUN_MODE` (smoke/train/resume/inference/reports), `EXPERIMENT_NAME`, `TRAINED_RUN_DIR`, `DATASET_PATH`, `SCRATCH_ROOT`, `OUTPUT_ROOT`, `BACKBONE`, `TRAINING_POLICY`, `PREDICT_PRIVATE`, `RUN_ABLATIONS`, `FOLD_SELECTION`, `DEV_*`, and utilities.
  - Defaults (user choice): `RUN_MODE='train'` (smoke off), `EVALUATION_MODE='nested_cv'`, `PREDICT_PRIVATE=True`, ablations off. The spec proposed smoke-first defaults; use `RUN_MODE='smoke'` explicitly for a cheap check.
  - Incompatible combinations fail before any import, decoding or download.
- **Experiment identity is separate from execution.**
  - Platform, paths and Slurm or attempt IDs never change the identity. Model, feature, preprocessing, optimizer and protocol changes do.
  - Run folders are `runs/<platform>/<name>__learned_<id>/`, with one immutable `attempts/<id>/attempt.json` per execution.
- **Writer isolation.** Per-scope locks with heartbeats prevent two writers in one run folder. They are taken over safely after a Slurm requeue or a dead process. All shared files are written atomically with unique temporary names.
- **HPC batch execution.**
  - `hpc/run_penalty_kick.py` runs the notebook's own code cells with JSON overrides, so batch and interactive code cannot drift. Unknown settings are rejected and frozen releases are checksum-verified.
  - `hpc/penalty_kick.sbatch` is grounded in the codelab: the account, partitions, `--requeue` and `--comment="gpu_mps=yes"`, with a Singularity overlay. Cluster values the codelab does not give are marked `UNVERIFIED`.
  - Example job JSONs are in `hpc/jobs/`.
- **Fold-parallel nested evaluation.**
  - `FOLD_SELECTION=[k]` (or `--fold-from-array`) runs one fold's inner selection and then its outer refit into an isolated `folds/fold<k>/`.
  - Pooling happens once every fold is complete with the same identity. Mixed configurations are refused.
- **Bounded development preset** (`EVALUATION_MODE='development'`).
  - One seed, few epochs, and a fixed, label-free, source-disjoint subset whose split file is saved.
  - Results are labeled development-only and are never treated as full evaluation. Submissions from this mode (opt-in via `PREDICT_PRIVATE=True`) are labeled development submissions trained on the subset.
- **Development backbone sweep.** In development mode, `DEVELOPMENT_BACKBONES` (Colab default: all three backbones; HPC default: `None`) trains each backbone on the same saved split and seeds. With `PREDICT_PRIVATE=True` it writes one labeled development submission per backbone, and it finishes with a best-inner-validation-loss comparison (`backbone_comparison.csv` and a curves plot). Failures are reported per backbone, never substituted.
- **Frozen releases** (`create_release`, `verify_release`) and **relocatable bundles** (`export_bundle`, `verify_bundle`). Interrupted transfers are rejected.
- **Durability guards.**
  - Training refuses ephemeral `OUTPUT_ROOT` (Colab `/content`, HPC `/tmp` or `$TMPDIR`).
  - HPC training refuses to run outside a Slurm allocation (login node).
  - A GPU must be visible before any model loads.

## Fixed issues from the prior review

1. **Prediction caches ignored metadata.** The prediction key now includes the metadata vector, the fitted-artifact ID and the feature signature. Crop caches stay geometry-only.
2. **Unconditional Colab, Drive and `/content` setup.** This is now split into platform cells. Drive is mounted only when a used path needs it; the HPC notebook has none of it.
3. **Custom-video inference needed the competition ZIP.** It no longer does. The ZIP, labels, clip-count assertions and template are needed only for private or public modes.
4. **Heavyweight reports mode.** Reports mode now uses only json, pandas and numpy, and stops early. It is verified in fresh processes for both notebooks, including through the HPC batch runner.
5. **V-JEPA validation status.** It is stated honestly: no real-checkpoint or GPU run was performed (`docs/VALIDATION.md`).

Inference and report outputs are also no longer written into the trained run or bundle. They go to `OUTPUT_ROOT/inference/<platform>/...` and `OUTPUT_ROOT/reports/...`.

## Remaining limitations

- **Nothing ran on real hardware.** No HPC job, Slurm submission, real checkpoint download, real video, GPU run, pilot or full evaluation was executed in authoring. All tests are CPU tests with synthetic data and tiny models.
- **Cluster values are missing.** The wall time, `--gres`, CPU/memory, overlay/image names, scratch purge policy and the "preempt script" are not in the codelab and remain placeholders.
- **Epoch-boundary recovery only.** A preempted epoch repeats.
- **Parallel granularity is per fold.** Seeds within a fold run sequentially.
- **Interactive OOD use** assumes a Jupyter kernel that sees the overlay environment. How to set that up on Cloud Bursting is not covered by the codelab.
- **V-JEPA 2.1 is untested against Meta's real code and weights.** Only a stand-in with the same interface was tested.
- **OpenCV builds differ.** Colab pins `opencv-python==4.12.0.88`; the HPC check accepts headless or regular OpenCV ≥4.10. Minor decode or resize differences between builds are possible, so same-platform comparisons are safest.
