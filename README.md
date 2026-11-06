# Penalty-kick direction prediction (CS-GY 6643)

This project predicts two three-class targets from penalty-kick video clips:

- **kick direction**: left / middle / right
- **goalkeeper direction**: left / stands in the middle / right

The score is `0.70 * kick_accuracy + 0.30 * goalkeeper_accuracy`. Weighted cross-entropy is reported separately as a loss.

## Notebooks

| Notebook | Use it for |
|---|---|
| [`notebooks/Penalty_Kick_HPC.ipynb`](notebooks/Penalty_Kick_HPC.ipynb) | Validated baselines, full source-grouped nested evaluation and selected ablations on NYU HPC Cloud Bursting (Slurm). Batch runs go through `hpc/run_penalty_kick.py` and `hpc/penalty_kick.sbatch`. |
| [`notebooks/Penalty_Kick_Colab_VideoMAE_Full.ipynb`](notebooks/Penalty_Kick_Colab_VideoMAE_Full.ipynb) | Colab Pro+ smoke tests with real backbones, bounded development experiments, crop/tracking audits, inference from saved runs or HPC bundles, and report review. |

Both notebooks share the same model, preprocessing and evaluation code. They differ only in their environment cells and defaults.

**Modes.** Both notebooks support `RUN_MODE` = `smoke` | `train` (default: full nested evaluation with `PREDICT_PRIVATE=True`) | `resume` | `inference` | `reports`.

**Backbones.**
- VideoMAE is the baseline.
- V-JEPA 2.1 (`vjepa2_1_vitb_384`) and V-JEPA 2 (`vjepa2_vitl_256`) are explicit alternatives, each with its own pinned loader.
- `TRAINING_POLICY` is `finetune` or `frozen_probe`.

## Documentation

- [`docs/WORKFLOW.md`](docs/WORKFLOW.md): the shared workflow, platform setup, and launch/resume/inference/report recipes.
- [`hpc/ENVIRONMENT.md`](hpc/ENVIRONMENT.md): HPC facts taken from the codelab, unresolved cluster values, and Singularity/conda setup.
- [`docs/VALIDATION.md`](docs/VALIDATION.md): the validation ladder, split into passed, failed and not-run checks.
- [`docs/RELEASES_AND_MIGRATION.md`](docs/RELEASES_AND_MIGRATION.md): baseline releases, bundles and artifact compatibility.
- [`docs/CHANGES.md`](docs/CHANGES.md): fixed issues and remaining limitations.
- [`docs/INSPECTION.md`](docs/INSPECTION.md): a mapping from the input notebook's cells to the two outputs.
- [`reference/Penalty_Kick_Colab_input.ipynb`](reference/Penalty_Kick_Colab_input.ipynb): the input notebook, kept unchanged for reference.

## Editing the notebooks

Edit `notebook_src/` rather than the generated notebooks: `shared/` holds the common code, and `colab/` and `hpc/` hold the platform cells. Then run:

```bash
python tools/build_notebooks.py
```

```bash
python tools/check_consistency.py
```

```bash
pip install -r requirements-test.txt
```

```bash
python -m pytest tests
```

The tests run on a CPU with synthetic data and tiny models. They check the plumbing, not model performance.
