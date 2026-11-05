## Recipes (HPC, NYU Cloud Bursting)

One-time setup (environment overlay, data upload, log folder): `hpc/ENVIRONMENT.md`. Long runs use batch jobs; the notebook is for interactive smoke tests and inspection in an Open OnDemand Jupyter session on a GPU partition.

**1. Smoke test on the target GPU:** interactive `RUN_MODE='smoke'`, or `sbatch penalty_kick.sbatch jobs/smoke.json`. Record peak memory and throughput (`timing_estimate.json`).

**2. Bounded pilot (validation level D):** `sbatch penalty_kick.sbatch jobs/pilot.json` (`EVALUATION_MODE='development'`, fixed saved split). Check finite losses, crop audits, time and storage before scheduling all folds.

**3. Freeze and launch the baseline:** set the baseline settings, `FREEZE_RELEASE=True`, `NOTEBOOK_PATH=<this .ipynb>`, run Settings through the release cell (it stops there). Then `cd OUTPUT_ROOT/releases/<id>` and `sbatch penalty_kick.sbatch <job>.json` (all folds in one job) or `sbatch --array=0-4 penalty_kick.sbatch <job>.json --fold-from-array` (one fold per task; the last finished fold pools the reports). Never edit a release; changes make a new one.

**4. Resume / preemption:** jobs carry `--requeue`; a requeued or resubmitted job resumes from the last completed epoch (the unfinished epoch repeats). To resubmit by hand use the same release and job JSON (`RUN_MODE='resume'` refuses to start a different experiment).

**5. Inference:** `RUN_MODE='inference'`, `TRAINED_RUN_DIR=runs/hpc/<experiment>` or a bundle, `INFERENCE_INPUT` = `'private'`, `'public_oof'` or a CSV (no ZIP needed). Outputs go to `OUTPUT_ROOT/inference/hpc/<run>/`.

**6. Reports / transfer to Colab:** `RUN_MODE='reports'` (no GPU needed; CPU partition or OOD session). `EXPORT_BUNDLE_TO=OUTPUT_ROOT/'bundles'` writes a checksummed, relocatable bundle; download it with the OOD file browser and load it in Colab with `TRAINED_RUN_DIR`.

**Monitoring:** `squeue -u $USER`, `scancel <jobid>`, logs in `/scratch/$USER/penalty_kick/logs/`, checkpoints in `OUTPUT_ROOT/runs/hpc/<experiment>/stages/*/video_seed*_latest.pt`, attempts in `.../attempts/`.

**Validation status:** see `docs/VALIDATION.md`. No cluster job, real checkpoint or GPU run has been executed by the author of these notebooks.
