# Penalty direction prediction: Colab, full VideoMAE nested_cv run

**Preconfigured:** `RUN_MODE='train'`, `EVALUATION_MODE='nested_cv'`, `BACKBONE='videomae'`, `TRAINING_POLICY='finetune'`, `PREDICT_PRIVATE=True`, `EXPERIMENT_NAME='final_videomae_v1'`. Run the install cell, restart, then **Run after** from Settings; rerun with the same settings after a disconnect. The submission is written to `OUTPUT_ROOT/runs/colab/final_videomae_v1__learned_<id>/submission.csv` once all 5 folds finish.

Two direction heads (kick 70%, goalkeeper 30%), source-cross-fitted player roles, camera compensation, VideoMAE / V-JEPA 2.1 / V-JEPA 2 video backbones, numeric specialist and ensembling. Goal outcome is not a target. Upload this notebook and, for training or competition inference, the dataset ZIP; all helper code is embedded and shared byte-for-byte with `Penalty_Kick_HPC.ipynb` (checked by `tools/check_consistency.py`).

**Use Colab for:** real-backbone smoke tests, the bounded development preset, crop/tracking audits, inference from saved models or HPC bundles, and report review. Long, validated baselines and full nested evaluations run on HPC. Colab Pro+ is not guaranteed uninterrupted compute: `/content` is ephemeral; checkpoints go to Drive, and training resumes at the last completed epoch.

Modes (`RUN_MODE`): **smoke** (cheap check), **train** (default: full nested_cv with a private submission), **resume**, **inference** (no fitting; no ZIP for custom videos), **reports** (pandas only; stops early). Recipes are at the end; the full workflow is in `docs/WORKFLOW.md`.
