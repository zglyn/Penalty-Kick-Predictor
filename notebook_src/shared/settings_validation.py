# CELL: settings_validation
# Fails fast on incompatible settings, before any decoding, download or model loading. Pure Python.
import os
from pathlib import Path

RUN_MODES = {'smoke', 'train', 'resume', 'inference', 'reports', 'refine'}
EVALUATION_MODES = {'nested_cv', 'holdout', 'development'}
KNOWN_BACKBONES = {'videomae', 'vjepa2_1_vitb_384', 'vjepa2_vitl_256'}
KNOWN_POLICIES = {'finetune', 'frozen_probe'}


def _problems():
    p = []
    if RUN_MODE not in RUN_MODES: p.append(f'RUN_MODE must be one of {sorted(RUN_MODES)}')
    if EVALUATION_MODE not in EVALUATION_MODES: p.append(f'EVALUATION_MODE must be one of {sorted(EVALUATION_MODES)}')
    if BACKBONE not in KNOWN_BACKBONES: p.append(f'BACKBONE must be one of {sorted(KNOWN_BACKBONES)}')
    if TRAINING_POLICY not in KNOWN_POLICIES: p.append(f'TRAINING_POLICY must be one of {sorted(KNOWN_POLICIES)}')
    if RUN_MODE in {'inference', 'reports', 'refine'} and TRAINED_RUN_DIR is None:
        p.append(f"RUN_MODE='{RUN_MODE}' needs TRAINED_RUN_DIR (a run folder or a verified bundle).")
    if RUN_ABLATIONS and (EVALUATION_MODE != 'nested_cv' or RUN_MODE not in {'train', 'resume'} or FOLD_SELECTION is not None):
        p.append("RUN_ABLATIONS needs EVALUATION_MODE='nested_cv', RUN_MODE train/resume and FOLD_SELECTION=None.")
    if PREDICT_PRIVATE and EVALUATION_MODE == 'holdout' and RUN_MODE in {'train', 'resume'}:
        p.append("PREDICT_PRIVATE is available with EVALUATION_MODE='nested_cv' (outer-fold ensemble) or 'development' "
                 '(one labeled development submission per backbone), not holdout.')
    if FOLD_SELECTION is not None and (EVALUATION_MODE != 'nested_cv' or RUN_MODE not in {'train', 'resume'}):
        p.append("FOLD_SELECTION applies to nested_cv training (one job per fold).")
    if EVALUATION_MODE == 'development':
        if not (0 < DEV_TRAIN_SOURCE_FRACTION < 1 and 0 < DEV_VALIDATION_SOURCE_FRACTION < 1
                and DEV_TRAIN_SOURCE_FRACTION + DEV_VALIDATION_SOURCE_FRACTION <= 1):
            p.append('DEV_*_SOURCE_FRACTION values must be in (0,1) and sum to at most 1.')
        if not DEV_SEEDS or DEV_EPOCHS < 1: p.append('DEV_SEEDS must be nonempty and DEV_EPOCHS >= 1.')
        if DEVELOPMENT_BACKBONES is not None:
            names = list(DEVELOPMENT_BACKBONES)
            if not names or len(set(names)) != len(names) or set(names) - KNOWN_BACKBONES:
                p.append(f'DEVELOPMENT_BACKBONES must be a nonempty list of distinct names from {sorted(KNOWN_BACKBONES)}.')
    if RUN_MODE == 'inference' and INFERENCE_INPUT not in {'private', 'public_oof'} and not str(INFERENCE_INPUT).endswith('.csv'):
        p.append("INFERENCE_INPUT must be 'private', 'public_oof' or a .csv path.")
    if REFINE_WEIGHT_CAP < 1 or any(o < 0 for o in REFINE_CENTER_OFFSETS) or 0.0 not in REFINE_CENTER_OFFSETS:
        p.append('REFINE_WEIGHT_CAP must be >= 1 and REFINE_CENTER_OFFSETS must be non-negative and include 0.0.')
    return p


_issues = _problems()
if RUN_MODE in {'smoke', 'train', 'resume'} and is_ephemeral_path(OUTPUT_ROOT) and not ALLOW_EPHEMERAL_OUTPUT:
    _issues.append(f'OUTPUT_ROOT={OUTPUT_ROOT} is ephemeral on {PLATFORM}; checkpoints would be lost. Use durable '
                   'storage (Colab: Drive; HPC: /scratch/$NETID) or set ALLOW_EPHEMERAL_OUTPUT=True for a throwaway test.')
if _issues:
    raise ValueError('Settings problems (nothing expensive has run):\n- ' + '\n- '.join(_issues))

QUICK_SMOKE = RUN_MODE == 'smoke'
if TRAINED_RUN_DIR is not None:
    TRAINED_RUN_DIR = Path(TRAINED_RUN_DIR)
    if not TRAINED_RUN_DIR.is_absolute(): TRAINED_RUN_DIR = Path(OUTPUT_ROOT) / TRAINED_RUN_DIR
CUSTOM_INFERENCE = RUN_MODE == 'inference' and INFERENCE_INPUT not in {'private', 'public_oof'}
# The competition ZIP is needed for training and for private/public inference only; custom-video
# inference and reports never open it.
NEEDS_DATASET = RUN_MODE in {'smoke', 'train', 'resume', 'refine'} or (RUN_MODE == 'inference' and not CUSTOM_INFERENCE)
EXECUTION_SETTINGS = dict(RUN_MODE=RUN_MODE, TRAINED_RUN_DIR=str(TRAINED_RUN_DIR) if TRAINED_RUN_DIR else None,
    INFERENCE_INPUT=str(INFERENCE_INPUT), FOLD_SELECTION=FOLD_SELECTION, DATASET_PATH=str(DATASET_PATH),
    OUTPUT_ROOT=str(OUTPUT_ROOT), SCRATCH_ROOT=str(SCRATCH_ROOT), PLATFORM=PLATFORM)
EXPERIMENT_SETTINGS = dict(EXPERIMENT_NAME=EXPERIMENT_NAME, EVALUATION_MODE=EVALUATION_MODE, BACKBONE=BACKBONE,
    BACKBONE_REVISION=BACKBONE_REVISION, TRAINING_POLICY=TRAINING_POLICY, PREDICT_PRIVATE=PREDICT_PRIVATE,
    RUN_ABLATIONS=RUN_ABLATIONS, REFINE_TREE_CLASS_WEIGHTS=REFINE_TREE_CLASS_WEIGHTS, REFINE_WEIGHT_CAP=REFINE_WEIGHT_CAP,
    REFINE_CLASS_OFFSETS=REFINE_CLASS_OFFSETS, REFINE_CENTER_OFFSETS=REFINE_CENTER_OFFSETS, DEV_SEEDS=DEV_SEEDS, DEV_EPOCHS=DEV_EPOCHS, DEVELOPMENT_BACKBONES=DEVELOPMENT_BACKBONES,
    DEV_TRAIN_SOURCE_FRACTION=DEV_TRAIN_SOURCE_FRACTION, DEV_VALIDATION_SOURCE_FRACTION=DEV_VALIDATION_SOURCE_FRACTION)
print(f'[{PLATFORM}] RUN_MODE={RUN_MODE} | experiment={EXPERIMENT_NAME} ({EVALUATION_MODE}, {BACKBONE}/{TRAINING_POLICY}) | '
      f'PREDICT_PRIVATE={PREDICT_PRIVATE} | dataset needed={NEEDS_DATASET}' + (f' | run={TRAINED_RUN_DIR}' if TRAINED_RUN_DIR else ''))
