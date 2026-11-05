# CELL: settings
# ============================== SETTINGS: the only cell you normally edit ==============================
# Plain Python (no NumPy/torch). Same experiment keys as the Colab notebook. Batch jobs override these
# from a JSON file (hpc/run_penalty_kick.py --settings ...), so the notebook itself is never edited per job.
import os
from pathlib import Path
PLATFORM = 'hpc'
SHARED_CODE_FINGERPRINT = '{{SHARED_CODE_FINGERPRINT}}'   # filled by tools/build_notebooks.py
NETID = os.environ.get('USER', 'NETID')

# ---- Execution (does not change experiment identity) ---------------------------------------------
RUN_MODE = 'smoke'            # 'smoke' | 'train' | 'resume' | 'inference' | 'reports'
TRAINED_RUN_DIR = None        # run folder or verified bundle (inference/reports). Relative = under OUTPUT_ROOT
INFERENCE_INPUT = 'private'   # 'private' | 'public_oof' | CSV with clip,path,preferred_foot,gender (no ZIP needed)
FOLD_SELECTION = None         # None = all folds then aggregate; [k] = only fold k (one job per fold); [] = aggregate
MIGRATE_LEGACY_RUN = False
VERIFY_CHECKSUMS = True
RUN_TIMING_BENCHMARK = True   # smoke mode only
GENERATE_HEATMAPS = False
GENERATE_TRACKING_AUDITS = True
ALLOW_EPHEMERAL_OUTPUT = False   # never put the only checkpoint in node-local/temporary storage
ALLOW_OUTSIDE_SLURM = False      # training/smoke refuse to run outside a Slurm allocation (login node)

# ---- Utilities (do not change experiment identity) ---------------------------------------------
EXPORT_BUNDLE_TO = None   # e.g. OUTPUT_ROOT/'bundles': checksummed, relocatable copy of a finished run
WRITE_BATCH_FILES = False  # write run_penalty_kick.py + penalty_kick.sbatch + job JSON next to this notebook
FREEZE_RELEASE = False     # copy notebook + runner + sbatch + settings into OUTPUT_ROOT/releases/<id> (read-only)
NOTEBOOK_PATH = None       # path of this .ipynb (needed for FREEZE_RELEASE / WRITE_BATCH_FILES in Jupyter)

# ---- Environment: NYU Cloud Bursting (codelab: checkpoints go to /scratch/$NETID) -------------------
DATASET_PATH = Path(f'/scratch/{NETID}/penalty_kick/data/penalty-kick-prediction-cs-gy-6643.zip')
OUTPUT_ROOT = Path(f'/scratch/{NETID}/penalty_kick/outputs')     # checkpoints, logs, reports, bundles, releases
SCRATCH_ROOT = Path(f'/scratch/{NETID}/penalty_kick/staging')    # extracted videos (regenerable from the ZIP)

# ---- Experiment (identity-defining: changing any of these creates a new experiment) ---------------
EXPERIMENT_NAME = 'baseline_videomae_v1'
EVALUATION_MODE = 'nested_cv'     # 'nested_cv' (full) | 'development' (bounded pilot preset) | 'holdout'
BACKBONE = 'videomae'             # 'videomae' | 'vjepa2_1_vitb_384' | 'vjepa2_vitl_256'
BACKBONE_REVISION = None
TRAINING_POLICY = 'finetune'      # 'finetune' | 'frozen_probe' (freezes encoder WEIGHTS; unrelated to releases)
PREDICT_PRIVATE = False
RUN_ABLATIONS = False
DEV_SEEDS = [42]
DEV_EPOCHS = 3
DEV_TRAIN_SOURCE_FRACTION = 0.5
DEV_VALIDATION_SOURCE_FRACTION = 0.2
