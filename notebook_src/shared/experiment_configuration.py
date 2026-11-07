# CELL: experiment_configuration
from ultralytics import YOLO
from transformers import VideoMAEForVideoClassification, VideoMAEImageProcessor
from catboost import CatBoostClassifier
from sklearn.model_selection import GroupKFold, GroupShuffleSplit
from sklearn.pipeline import make_pipeline
from sklearn.dummy import DummyClassifier
from PIL import Image, ImageEnhance
import cv2, copy, itertools, gc, pickle, subprocess, time
from collections import deque

# User-facing switches are in Settings. These are fixed protocol constants (part of the experiment identity).
TRAINING_MODES = {'smoke', 'train', 'resume'}
VALIDATION_FRACTION = .20
TEST_FRACTION = .05
TIMING_WARMUP_BATCHES = 8
TIMING_MEASURED_BATCHES = 24  # Three full accumulation groups at the default settings.
TIMING_INFERENCE_CLIPS = 8
POSE_WEIGHTS = 'yolo11m-pose.pt'
DETECTOR_CONF = .08  # Retain small/distant keepers; learned roles and tracking filter candidates.
DETECTOR_IMGSZ = 1280
WINDOW = 1.6
VIDEO_EPOCHS = 6  # Maximum epochs; inner validation selects outer epochs.
VIDEO_SEEDS = [42, 137]
OUTER_FOLDS = 5
ROLE_FOLDS = 3
GRAD_ACCUM = 8
ROLE_MIN_PROB = .35
TRACK_MAX_GAP_SECONDS = .5
TRACK_MAX_JUMP_HEIGHTS = 1.25
TRACK_SPEED_HEIGHTS_PER_SECOND = 6.0
AUDIT_HIGH_LOSS = 3
AUDIT_CORRECT = 2
ABLATION_VARIANTS = ['A_context','B_context_kicker','C_context_keeper','D_all_visual',
                     'E_plus_pose','F_raw_motion','G_compensated_motion','H_plus_metadata']
HEATMAP_TOP_K = 4
HEATMAP_GRID = 5
HEATMAP_FILL = 'blur'  # Compare with 'mean' to check masking sensitivity.
HEATMAP_SEEDS = None  # None explains the full video-seed ensemble.
SMOKE_MAX_BATCHES = 4
CAMERA_CACHE_VERSION = 'similarity-jpeg-v1'

# Backbone: frame count and crop resolution come from the adapter, not VideoMAE defaults.
BACKBONE_SPEC = backbone_spec(BACKBONE, BACKBONE_REVISION)
LEGACY_IDENTITY = backbone_identity(backbone_spec('videomae'), 'finetune')
SAVED_INFERENCE_MANIFEST = None
OUTPUT = Path(OUTPUT_ROOT)   # historical name used by split/cache helpers
SAVED_SETTINGS_MODES = {'inference', 'refine'}   # these modes rebuild the SAVED run's settings, never notebook defaults
if RUN_MODE in SAVED_SETTINGS_MODES:
    # Inference/refinement reconstruct the SAVED model identity; current notebook defaults never redefine it.
    if not TRAINED_RUN_DIR.is_dir(): raise FileNotFoundError(f'TRAINED_RUN_DIR not found: {TRAINED_RUN_DIR}')
    if is_bundle(TRAINED_RUN_DIR):
        verify_bundle(TRAINED_RUN_DIR, VERIFY_CHECKSUMS); print('Bundle verified:', TRAINED_RUN_DIR)
    _manifest_path = TRAINED_RUN_DIR/'inference_manifest.json'
    if _manifest_path.exists():
        SAVED_INFERENCE_MANIFEST = json.loads(_manifest_path.read_text())
        _saved_cfg = SAVED_INFERENCE_MANIFEST['config']
        BACKBONE_SPEC = SAVED_INFERENCE_MANIFEST['backbone']['spec']
        TRAINING_POLICY = SAVED_INFERENCE_MANIFEST['backbone']['training_policy']
    else:
        _source = TRAINED_RUN_DIR/'experiment_manifest.json'
        if not _source.exists():
            raise FileNotFoundError(f'{_manifest_path} not found and {_source} missing; inference needs a finished '
                                    'nested_cv run (or export a holdout run while it is the active training RUN).')
        _saved_cfg = json.loads(_source.read_text())['config']
        if 'backbone' in _saved_cfg:
            BACKBONE_SPEC = backbone_spec(_saved_cfg['backbone']['name'], _saved_cfg['backbone']['revision'])
            TRAINING_POLICY = _saved_cfg['training_policy']
        else:
            BACKBONE_SPEC = dict(backbone_spec('videomae'), checkpoint=_saved_cfg['video']); TRAINING_POLICY = 'finetune'
    N_FRAMES, WINDOW, SIZE = int(_saved_cfg['frames']), float(_saved_cfg['window']), int(_saved_cfg['size'])
    DETECTOR_CONF, DETECTOR_IMGSZ, POSE_WEIGHTS = _saved_cfg['detector_conf'], _saved_cfg['detector_imgsz'], _saved_cfg['pose']
    TRACK_MAX_GAP_SECONDS = _saved_cfg['tracking']['max_gap']; TRACK_MAX_JUMP_HEIGHTS = _saved_cfg['tracking']['max_jump']
    TRACK_SPEED_HEIGHTS_PER_SECOND = _saved_cfg['tracking']['speed']; ROLE_MIN_PROB = _saved_cfg['role_min_prob']
    VIDEO_SEEDS = list(_saved_cfg['seeds'])
    if BACKBONE != BACKBONE_SPEC['name']:
        print(f'NOTE: Settings say BACKBONE={BACKBONE!r}, but the saved run uses {BACKBONE_SPEC["name"]!r}. '
              'The saved run wins; inference never converts a run to another backbone.')
    print('Loaded saved preprocessing/model settings from', TRAINED_RUN_DIR)
if EVALUATION_MODE == 'development' and RUN_MODE not in SAVED_SETTINGS_MODES:
    VIDEO_SEEDS = list(DEV_SEEDS); VIDEO_EPOCHS = int(DEV_EPOCHS)
FULL_TRAIN_SETTINGS=dict(epochs=VIDEO_EPOCHS,seeds=list(VIDEO_SEEDS),accumulation=GRAD_ACCUM)
if QUICK_SMOKE:
    VIDEO_EPOCHS = 1
    VIDEO_SEEDS = [42]
    ROLE_FOLDS = 2
    GRAD_ACCUM = 2
if RUN_MODE in SAVED_SETTINGS_MODES and DEVICE != 'cuda':
    print('WARNING: no GPU. Inference will work but uncached detection/video encoding will be slow.')


def image_normalization(spec):
    """VideoMAE reads its checkpoint processor (small config, not weights); V-JEPA uses official ImageNet values."""
    if spec['family'] == 'videomae':
        processor = VideoMAEImageProcessor.from_pretrained(spec['checkpoint'])
        return np.array(processor.image_mean, dtype=np.float32), np.array(processor.image_std, dtype=np.float32)
    return np.array(IMAGENET_MEAN, dtype=np.float32), np.array(IMAGENET_STD, dtype=np.float32)


def optimizer_settings(policy):
    return dict(name='AdamW', weight_decay=.05, policy=policy,
                learning_rates=dict(finetune=dict(backbone=1e-5, heads=1e-4), frozen_probe=dict(heads=1e-3))[policy],
                gradient_clip=1.0, amp='float16')


def configure_training_run(backbone, revision=None, policy=None, label=None, record_attempt=True):
    """Set every backbone-dependent global (spec, frames, crop size, normalization, CFG, identity, RUN, STAGES)
    for one training experiment. Used for the configured backbone and for each backbone of a development sweep,
    so a sweep member is exactly the experiment a single run with that BACKBONE would be."""
    global BACKBONE_SPEC, VIDEO_ID, N_FRAMES, SIZE, BACKBONE_IDENTITY, LEGACY_DEFAULT_BACKBONE, OPTIMIZER_SETTINGS
    global CFG, EXPERIMENT_ID, RUN, STAGES, IMAGE_MEAN, IMAGE_STD
    policy = TRAINING_POLICY if policy is None else policy
    BACKBONE_SPEC = backbone_spec(backbone, revision)
    VIDEO_ID = BACKBONE_SPEC['checkpoint']; N_FRAMES = BACKBONE_SPEC['frames']; SIZE = BACKBONE_SPEC['size']
    BACKBONE_IDENTITY = backbone_identity(BACKBONE_SPEC, policy)
    # The original VideoMAE fine-tuning configuration keeps its exact CFG, so old runs still resume.
    LEGACY_DEFAULT_BACKBONE = BACKBONE_IDENTITY == LEGACY_IDENTITY
    OPTIMIZER_SETTINGS = optimizer_settings(policy)
    CFG = dict(dataset=DATA_ID,video=VIDEO_ID,pose=POSE_WEIGHTS,frames=N_FRAMES,
               detector_conf=DETECTOR_CONF,detector_imgsz=DETECTOR_IMGSZ,
               window=WINDOW,size=SIZE,epochs=VIDEO_EPOCHS,seeds=VIDEO_SEEDS,
               outer_folds=OUTER_FOLDS,role_folds=ROLE_FOLDS,role_min_prob=ROLE_MIN_PROB,
               version='nested-components-v6',gradient_accumulation=GRAD_ACCUM,evaluation_mode=EVALUATION_MODE,
               validation_fraction=VALIDATION_FRACTION,test_fraction=TEST_FRACTION,split_seed=SEED,targets=TARGETS,metric_weights=WEIGHTS.tolist(),
               tracking=dict(max_gap=TRACK_MAX_GAP_SECONDS,max_jump=TRACK_MAX_JUMP_HEIGHTS,speed=TRACK_SPEED_HEIGHTS_PER_SECOND),
               smoke=QUICK_SMOKE,smoke_max_batches=SMOKE_MAX_BATCHES)
    if not LEGACY_DEFAULT_BACKBONE:
        # New backbones/policies get distinct runs, crop caches (size/frames) and checkpoints.
        CFG.update(backbone=BACKBONE_IDENTITY, training_policy=policy, optimizer=OPTIMIZER_SETTINGS)
    if EVALUATION_MODE == 'development':
        CFG.update(development=dict(train_source_fraction=DEV_TRAIN_SOURCE_FRACTION,
            validation_source_fraction=DEV_VALIDATION_SOURCE_FRACTION, seeds=list(DEV_SEEDS), epochs=int(DEV_EPOCHS),
            scope='development inner-validation subset; not complete OOF performance'))
    # Identity = computation only. Platform/paths/attempts choose WHERE it is written, never WHAT it is.
    EXPERIMENT_ID = hashlib.sha256(json.dumps(CFG,sort_keys=True).encode()).hexdigest()[:16]
    RUN = OUTPUT/'runs'/PLATFORM/f'{label or EXPERIMENT_NAME}__learned_{EXPERIMENT_ID}'
    legacy = OUTPUT/f'learned_{EXPERIMENT_ID}'          # layout before platform-separated runs (Colab)
    if PLATFORM == 'colab' and legacy.is_dir() and not RUN.exists():
        RUN = legacy; print('Using the existing pre-update run folder', RUN)
    if RUN_MODE == 'resume':
        if TRAINED_RUN_DIR is not None and TRAINED_RUN_DIR.resolve() != RUN.resolve():
            saved = TRAINED_RUN_DIR/'experiment_manifest.json'
            hint = ''
            if saved.exists():
                old = json.loads(saved.read_text())['config']
                hint = ' Differing settings: ' + ', '.join(sorted(k for k in set(old)|set(CFG)
                    if k in CFG and old.get(k) != CFG.get(k)))
            raise ValueError(f'Current settings map to {RUN.name}, not {TRAINED_RUN_DIR.name}.{hint} '
                             'Restore the original settings to resume (backbone/policy changes are new runs).')
        if not (RUN/'stages').is_dir():
            raise ValueError(f'RUN_MODE="resume" but {RUN} has no saved stages. Use RUN_MODE="train" to start it.')
    RUN.mkdir(parents=True, exist_ok=True)
    STAGES = RUN/'stages'; STAGES.mkdir(exist_ok=True)
    IMAGE_MEAN, IMAGE_STD = image_normalization(BACKBONE_SPEC)
    if record_attempt:
        write_attempt_record(RUN, dict(execution=EXECUTION_SETTINGS, experiment=dict(EXPERIMENT_SETTINGS, BACKBONE=backbone),
                                       experiment_identity=EXPERIMENT_ID, config=CFG), ENVIRONMENT_INFO)
    return RUN


if RUN_MODE in SAVED_SETTINGS_MODES:
    VIDEO_ID = BACKBONE_SPEC['checkpoint']
    BACKBONE_IDENTITY = backbone_identity(BACKBONE_SPEC, TRAINING_POLICY)
    LEGACY_DEFAULT_BACKBONE = BACKBONE_IDENTITY == LEGACY_IDENTITY
    OPTIMIZER_SETTINGS = optimizer_settings(TRAINING_POLICY)
    CFG = _saved_cfg; RUN = TRAINED_RUN_DIR; STAGES = RUN/'stages'
    EXPERIMENT_ID = stable_hash(CFG)
    if SAVED_INFERENCE_MANIFEST is not None:
        IMAGE_MEAN=np.array(SAVED_INFERENCE_MANIFEST['preprocessing']['image_mean'],dtype=np.float32)
        IMAGE_STD=np.array(SAVED_INFERENCE_MANIFEST['preprocessing']['image_std'],dtype=np.float32)
    else:
        IMAGE_MEAN, IMAGE_STD = image_normalization(BACKBONE_SPEC)
    if DATA_ID is not None and _saved_cfg.get('dataset') != DATA_ID:
        print('NOTE: dataset ZIP differs from the training ZIP; training-time crop/detector caches will not be reused.')
else:
    configure_training_run(BACKBONE, BACKBONE_REVISION, TRAINING_POLICY, label=EXPERIMENT_NAME)
print('RUN:', RUN, '| experiment identity:', EXPERIMENT_ID, '| attempt:', ATTEMPT_ID)
rows=pd.concat([train,test],ignore_index=True)
if NEEDS_DATASET:
    # Shared detector cache excludes learned role/model parameters and target labels.
    # Click timestamps affect only separate annotation-nearest detector frames, never sampled model inputs.
    _click_table=pd.read_csv(find_one({'public_position_clicks.csv'}))
    _time_columns=['clip','kicker_click_time_seconds','keeper_click_time_seconds']
    _click_times_hash=hashlib.sha256(_click_table[_time_columns].sort_values('clip').to_csv(index=False).encode()).hexdigest()
    clicks=_click_table.set_index('clip')
    assert clicks.index.is_unique
else:
    _click_times_hash=None
    clicks=pd.DataFrame(columns=['kicker_click_time_seconds','keeper_click_time_seconds','kicker_x_norm','kicker_y_norm',
                                 'keeper_x_norm','keeper_y_norm']).rename_axis('clip')
RAW_CFG=dict(dataset=DATA_ID,pose=POSE_WEIGHTS,conf=DETECTOR_CONF,imgsz=DETECTOR_IMGSZ,
    frames=N_FRAMES,window=WINDOW,click_timestamps=_click_times_hash,camera_cache_version=CAMERA_CACHE_VERSION)
RAW=OUTPUT/('detector_cache_'+hashlib.sha256(json.dumps(RAW_CFG,sort_keys=True).encode()).hexdigest()[:16])
if NEEDS_DATASET:
    RAW.mkdir(parents=True, exist_ok=True)
    atomic_write_text(RAW/'manifest.json',json.dumps(RAW_CFG,indent=2))

# Choose annotated examples by source only, without selecting on target outcomes.
# Four training sources x four clips; two held-out sources x two clips.
if QUICK_SMOKE:
    eligible=train[train['clip'].isin(clicks.index)]
    sources=eligible.groupby('source_id').size()
    sources=sources[sources>=4].index.to_numpy()
    if len(sources)<6: raise ValueError('Smoke test needs six sources with at least four clicked clips each.')
    selected=np.random.default_rng(SEED).permutation(sources)[:6]
    smoke_train=np.concatenate([eligible[eligible.source_id==g].index.to_numpy()[:4] for g in selected[:4]])
    smoke_hold=np.concatenate([eligible[eligible.source_id==g].index.to_numpy()[:2] for g in selected[4:]])
    assert not set(train.source_id.iloc[smoke_train]) & set(train.source_id.iloc[smoke_hold])
    print('Smoke subset:',len(smoke_train),'training clips,',len(smoke_hold),'held-out clips; 6 disjoint sources')

def make_dataset_splits(frame, validation_fraction, test_fraction, seed):
    # Split whole sources; never choose a partition by validation/test outcomes.
    if not 0 < test_fraction < validation_fraction < 1-test_fraction:
        raise ValueError('Require 0 < test fraction < validation fraction and a positive training fraction.')
    if frame.source_id.isna().any(): raise ValueError('Missing source_id prevents safe splitting.')
    ids=np.arange(len(frame))
    remaining,held=next(GroupShuffleSplit(n_splits=1,test_size=test_fraction,random_state=seed).split(
        ids,groups=frame.source_id))
    a,b=next(GroupShuffleSplit(n_splits=1,test_size=validation_fraction/(1-test_fraction),
        random_state=seed+1).split(remaining,groups=frame.source_id.iloc[remaining]))
    parts={'training':remaining[a], 'validation':remaining[b], 'test':held}
    assert len(np.unique(np.concatenate(list(parts.values()))))==len(frame)
    for left,right in itertools.combinations(parts.values(),2):
        assert not set(frame.source_id.iloc[left]) & set(frame.source_id.iloc[right])
    if not len(parts['test']) < len(parts['validation']) < len(parts['training']):
        raise ValueError('Source sizes prevent training > validation > test. Adjust fractions before training.')
    return parts

if EVALUATION_MODE=='holdout' and RUN_MODE in {'train', 'resume'}:
    split_indices=make_dataset_splits(train,VALIDATION_FRACTION,TEST_FRACTION,SEED)
    for j,classes in enumerate(CLASSES):
        if len(np.unique(Y[split_indices['training'],j]))!=len(classes):
            raise ValueError(f'Training split is missing a class for {TARGETS[j]}; adjust split settings.')
    split_manifest=train[['clip','source_id']].copy()
    for name,ids in split_indices.items(): split_manifest.loc[ids,'split']=name
    split_manifest.to_csv(RUN/'dataset_splits.csv',index=False)
    split_summary=pd.DataFrame([dict(split=name,clips=len(ids),sources=train.source_id.iloc[ids].nunique(),
        fraction=len(ids)/len(train)) for name,ids in split_indices.items()])
    display(split_summary)
    print('Fractions apply to source groups, so actual clip percentages may differ.')
    print('The labeled test split is separate from the 900 unlabeled competition clips.')

print(f'Backbone {BACKBONE_SPEC["name"]}: {N_FRAMES} frames, {SIZE}px crops, policy={TRAINING_POLICY}')
# The pose detector loads lazily (get_pose_model) only when an uncached clip must be decoded.
