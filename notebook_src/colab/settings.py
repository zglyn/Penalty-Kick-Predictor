# CELL: settings
# ============================== SETTINGS: the only cell you normally edit ==============================
# Plain Python (no NumPy/torch), safe right after the restart. Same experiment keys as the HPC notebook.
from pathlib import Path
PLATFORM = 'colab'
SHARED_CODE_FINGERPRINT = '{{SHARED_CODE_FINGERPRINT}}'   # filled by tools/build_notebooks.py

# ---- Execution (does not change experiment identity) ---------------------------------------------
RUN_MODE = 'train'            # 'smoke' | 'train' | 'resume' | 'inference' | 'reports' | 'refine' (improve a finished run, no video retraining)
TRAINED_RUN_DIR = None        # run folder or verified bundle (inference/reports). Relative = under OUTPUT_ROOT
INFERENCE_INPUT = 'private'   # 'private' | 'public_oof' | CSV with clip,path,preferred_foot,gender (no ZIP needed)
FOLD_SELECTION = None         # None = all folds then aggregate; [k] = only fold k; [] = aggregate only
MIGRATE_LEGACY_RUN = False    # inference on a run trained before inference manifests existed
VERIFY_CHECKSUMS = True
RUN_TIMING_BENCHMARK = True   # smoke mode only
GENERATE_HEATMAPS = False
GENERATE_TRACKING_AUDITS = True
ALLOW_EPHEMERAL_OUTPUT = False  # True only for throwaway tests: /content disappears with the runtime

# ---- Utilities (do not change experiment identity) ---------------------------------------------
EXPORT_BUNDLE_TO = None   # e.g. OUTPUT_ROOT/'bundles': checksummed, relocatable copy of a finished run

# ---- Environment: Colab paths. /content is EPHEMERAL staging; Drive is durable ----------------------
DATASET_PATH = Path('/content/drive/MyDrive/Fall2026/CV/penalty-kick-prediction-cs-gy-6643.zip')
OUTPUT_ROOT = Path('/content/drive/MyDrive/Fall2026/CV/penalty_kick_outputs')   # checkpoints, reports, bundles
SCRATCH_ROOT = Path('/content/penalty_scratch')   # extracted videos; lost when the runtime is replaced

# ---- Experiment (identity-defining: changing any of these creates a new experiment) ---------------
EXPERIMENT_NAME = 'final_videomae_v1'      # full nested_cv VideoMAE run; the hash of the configuration is the identity
EVALUATION_MODE = 'nested_cv'     # 'nested_cv' (full) | 'development' (bounded preset) | 'holdout'
BACKBONE = 'videomae'             # 'videomae' | 'vjepa2_1_vitb_384' | 'vjepa2_vitl_256'
BACKBONE_REVISION = None
TRAINING_POLICY = 'finetune'      # 'finetune' | 'frozen_probe' (freezes encoder WEIGHTS; unrelated to releases)
PREDICT_PRIVATE = True        # predict the 900 private clips and write submission.csv
RUN_ABLATIONS = False
# Bounded development preset: one seed, few epochs, fixed source-disjoint subset chosen without labels.
# Development mode trains EVERY backbone listed here (same saved split and seeds), writes one development
# submission per backbone (PREDICT_PRIVATE=True), and ends with a best-validation-loss comparison. None = BACKBONE only.
# RUN_MODE='refine' (TRAINED_RUN_DIR = finished nested_cv run): reuses all saved models; writes a new run folder.
REFINE_TREE_CLASS_WEIGHTS = True   # add a pose/motion CatBoost refit with capped inverse-sqrt class weights (CPU)
REFINE_WEIGHT_CAP = 3.0            # cap on per-sample class weight (mean weight = 1)
REFINE_CLASS_OFFSETS = True        # inner-fitted center-class log-probability offset, adopted only if inner accuracy rises
REFINE_CENTER_OFFSETS = [0.0, 0.25, 0.5, 0.75, 1.0]
REFINE_AUDITS = True               # label-mapping, temporal-input, crop-geometry audits and training-only label samples
DEVELOPMENT_BACKBONES = ['videomae', 'vjepa2_1_vitb_384', 'vjepa2_vitl_256']
DEV_SEEDS = [42]
DEV_EPOCHS = 3
DEV_TRAIN_SOURCE_FRACTION = 0.5    # of public sources, for training
DEV_VALIDATION_SOURCE_FRACTION = 0.2  # disjoint sources for inner validation (development-only metric)
