# Shared workflow: HPC and Colab

Both notebooks are generated from `notebook_src/` by `python tools/build_notebooks.py`. Shared cells are byte-identical, and `python tools/check_consistency.py` checks this. For identical experiment settings, both notebooks use the same model, preprocessing and evaluation code, and the same experiment identity. Only the defaults and the environment cells differ.

| | `notebooks/Penalty_Kick_HPC.ipynb` | `notebooks/Penalty_Kick_Colab_VideoMAE_Full.ipynb` |
|---|---|---|
| Purpose | Validated baselines, full source-grouped nested evaluation, selected ablations, durable artifacts | Real-model smoke tests, bounded development experiments, visual audits, inference, report review |
| Default settings | `RUN_MODE='train'`, `EVALUATION_MODE='nested_cv'`, `PREDICT_PRIVATE=True` | same as HPC |
| Durable storage | `/scratch/$NETID/penalty_kick/outputs` (codelab) | Google Drive `MyDrive/Fall2026/CV/penalty_kick_outputs` |
| Ephemeral staging | `/scratch/$NETID/penalty_kick/staging` (regenerable) | `/content/penalty_scratch` (lost with the runtime) |
| Long runs | Batch jobs: `hpc/run_penalty_kick.py` + `hpc/penalty_kick.sbatch` | Interactive only; each run resumes epoch by epoch from Drive |
| Never does | Colab imports, Drive mounts, package installs, Colab paths, training on a login node | Forcing Drive authentication for reports on local files |

## Settings

Each notebook has one **Settings** cell, with the same keys on both platforms.

**Execution settings** do not change what is computed:
- `RUN_MODE`: `smoke` | `train` | `resume` | `inference` | `reports`
- `TRAINED_RUN_DIR`, `INFERENCE_INPUT`
- `FOLD_SELECTION`, `EXPORT_BUNDLE_TO`
- paths: `DATASET_PATH`, `OUTPUT_ROOT`, `SCRATCH_ROOT`
- report/audit toggles

**Experiment settings** define the computation:
- `EXPERIMENT_NAME`, `EVALUATION_MODE`
- `BACKBONE`, `BACKBONE_REVISION`, `TRAINING_POLICY`
- `PREDICT_PRIVATE`, `RUN_ABLATIONS`
- `DEV_*`

**Validation before anything expensive.** Incompatible combinations fail before any import, decoding or download. Examples: inference without a run; `PREDICT_PRIVATE` with development or holdout; `FOLD_SELECTION` outside nested training; ephemeral output for training; an unknown backbone.

**Experiment identity** is a hash of the computational configuration: dataset hash, split, frames, resolution, detector, tracking, backbone checkpoint/revision, feature variant, training policy, optimizer, epochs, seeds and evaluation protocol.
- Paths, platform, Slurm job IDs and attempt IDs never change it.
- The run folder is `OUTPUT_ROOT/runs/<platform>/<EXPERIMENT_NAME>__learned_<identity>/`.
- Every execution adds `attempts/<attempt id>/attempt.json`, with settings, host, Slurm variables, versions and the code fingerprint.
- The default VideoMAE fine-tuning configuration keeps its original identity, so pre-update Colab runs (`OUTPUT_ROOT/learned_<id>`) are found and resumed.

**Writers.** One writer per run directory and scope (`run`, `fold<k>`, `aggregate`). Locks carry a heartbeat:
- A second live writer fails immediately.
- A requeued Slurm job (same `SLURM_JOB_ID`) or a lock with no heartbeat for 10 minutes is taken over.

HPC and Colab never share a run folder, even for equivalent configurations.

## Concurrent working pattern

1. **Validate the baseline on HPC.** Run smoke, then the pilot: `RUN_MODE='smoke'`, then `jobs/pilot.json`. Read memory and throughput from `timing_estimate.json`, and check crop audits and finite losses.
2. **Freeze the baseline.** Set `FREEZE_RELEASE=True` and `NOTEBOOK_PATH=...`, then run the Settings cell through the release cell, which stops after freezing. This creates the read-only `OUTPUT_ROOT/releases/<EXPERIMENT_NAME>__<hash>/`, containing the notebook, runner, sbatch script, job JSON, checksums and environment.
3. **Start the immutable baseline on HPC** from the release folder: `sbatch penalty_kick.sbatch <job>.json`, or with `--array=0-4 ... --fold-from-array`. Bug fixes or experiment changes make a new release; the running one is never edited.
4. **Copy the release notebook settings to Colab** under a different `EXPERIMENT_NAME`. Any computational change produces a new identity and run folder automatically.
5. **Screen ideas in Colab** with `EVALUATION_MODE='development'`, using the same saved development split. In Colab this trains every backbone in `DEVELOPMENT_BACKBONES` and writes a best-validation-loss comparison plus a development submission per backbone (`OUTPUT_ROOT/comparisons/`). Change one thing at a time, select on inner validation only, and inspect crop/tracking audits and loss curves.
6. **Promote a promising change** into a new release and a full HPC experiment.

**Sharing.** Dataset hashes, class mappings and split assignments (`OUTPUT_ROOT/diagnostic_splits/`, `OUTPUT_ROOT/splits/`) are immutable and keyed by identity. Never point one platform at the other's live run folder. To move a finished run, export a bundle (below).

**Comparing runs.** Only compare runs with the same dataset, split and target identities. `pipeline_summary.json` records them. Never mix development tables with full OOF tables. Repeated outer-fold inspection is exploratory development feedback.

## Recipes

Each notebook ends with its own recipe cell. The essentials:

**Train / resume.**
- HPC: `sbatch penalty_kick.sbatch jobs/baseline_nested.json`. After preemption, Slurm requeues the job, which resumes at the last completed epoch. To resubmit by hand, use the same release, job JSON and `RUN_MODE='resume'`.
- Colab: `RUN_MODE='train'`; after a disconnect, use the same settings with `RUN_MODE='resume'`.

**Inference without training.** Set `RUN_MODE='inference'` and `TRAINED_RUN_DIR=<run or bundle>`. Then choose `INFERENCE_INPUT`:
- `'private'` needs the ZIP and writes `submission.csv`.
- `'public_oof'` routes each clip through its held-out fold only.
- A CSV with `clip,path,preferred_foot,gender` needs no ZIP and no labels.

Outputs go to `OUTPUT_ROOT/inference/<platform>/<run>/`, never into the trained run.

**Reports.** Set `RUN_MODE='reports'` and `TRAINED_RUN_DIR=<run or bundle>`. Only json/pandas/numpy are imported, and the notebook stops after writing `pipeline_summary.*` to `OUTPUT_ROOT/reports/...` (on Colab, local sources stay in `SCRATCH_ROOT`). On HPC: `sbatch penalty_kick.sbatch jobs/reports.json`, or run it interactively.

**Refine without retraining.** Set `RUN_MODE='refine'` and `TRAINED_RUN_DIR=<finished nested_cv run>`. Saved video weights are reused (inference only); a class-weighted CatBoost is refit on the CPU; ensembles and a center-class offset are reselected on inner predictions. Outputs go to `<run>__refine_<id>/`. Details: `docs/REFINEMENT.md`.

**Transfer HPC → Colab.**
1. Set `EXPORT_BUNDLE_TO=OUTPUT_ROOT/'bundles'` in a reports or inference run, or after training.
2. Download the bundle folder with the OOD file browser and upload it to Drive.
3. In Colab, set `TRAINED_RUN_DIR` to it. The bundle is verified automatically: an interrupted copy has no `BUNDLE_COMPLETE.json` and is rejected.

**Parallel folds on HPC.** `sbatch --array=0-4 penalty_kick.sbatch jobs/baseline_nested.json --fold-from-array`.
- Each task runs its fold's inner selection and then that fold's outer refit, writing only `folds/fold<k>/` and that fold's stages.
- The task that completes the last fold pools the reports, after checking that every fold has the same identity and private-prediction setting.
- `FOLD_SELECTION=[]` pools without training.
