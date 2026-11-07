"""Execute real notebook cells on CPU with tiny synthetic data (no videos, detector, GPU or downloads).

Cells are located by their '# CELL: <name>' marker, so tests follow the notebook rather than cell numbers.
"""
import contextlib, hashlib, json, os, pickle, sys, types, warnings
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
NOTEBOOKS = dict(colab=REPO / 'notebooks' / 'Penalty_Kick_Colab_VideoMAE_Full.ipynb', hpc=REPO / 'notebooks' / 'Penalty_Kick_HPC.ipynb')
NOTEBOOK = Path(os.environ.get('PENALTY_NOTEBOOK', NOTEBOOKS['colab']))
WRAPPER = os.environ.get('PENALTY_WRAPPER', 'colab')   # which notebook's cells the harness executes
TARGETS = ['kick_direction', 'goalkeeper_direction']
CLASSES = [['left', 'middle', 'right'], ['left', 'stands in the middle', 'right']]


def notebook_cells(platform=None):
    nb = json.loads((NOTEBOOKS[platform] if platform else NOTEBOOKS[WRAPPER]).read_text())
    return nb, [''.join(c['source']) for c in nb['cells']]


def cell_source(name, platform=None):
    _, cells = notebook_cells(platform)
    found = [c for c in cells if c.startswith(f'# CELL: {name}\n')]
    if len(found) != 1: raise KeyError(f'{name}: {len(found)} cells')
    return found[0]


def run_cell(name, ns, platform=None):
    exec(compile(cell_source(name, platform), f'<notebook:{name}>', 'exec'), ns)
    return ns


class CPUGradScaler:
    """Stand-in for torch.amp.GradScaler('cuda') on a CPU-only machine (a disabled real scaler cannot
    load_state_dict). Same interface; scale is always 1."""
    def __init__(self, *a, **k): pass
    def scale(self, loss): return loss
    def unscale_(self, optimizer): pass
    def step(self, optimizer): optimizer.step()
    def update(self): pass
    def get_scale(self): return 1.0
    def state_dict(self): return {'scale': 1.0}
    def load_state_dict(self, state): assert state == {'scale': 1.0}


def synthetic_view(clip, frames, size):
    rng = np.random.default_rng(int(hashlib.sha256(clip.encode()).hexdigest()[:8], 16))
    return dict(video=rng.integers(0, 256, size=(3, frames, 3, size, size), dtype=np.uint8),
                numeric=rng.normal(size=(frames, 299)).astype(np.float32),
                traces=np.concatenate([rng.uniform(.2, .8, size=(frames, 2, 2)), np.ones((frames, 2, 1))], 2).astype(np.float32),
                track_ids=np.zeros((frames, 2), dtype=int), anchor_frame=frames - 1)


def make_dataset(root, n_public=24, n_private=6, seed=0):
    rng = np.random.default_rng(seed)
    sources = [f'src{i:02d}' for i in range(12)]
    train = pd.DataFrame(dict(clip=[f'pub{i:03d}.mp4' for i in range(n_public)],
                              source_id=[sources[i % 12] for i in range(n_public)],
                              preferred_foot=rng.choice(['left', 'right'], n_public),
                              gender=rng.choice(['F', 'M'], n_public)))
    for t, cs in zip(TARGETS, CLASSES):
        values = np.array([cs[i % 3] for i in range(n_public)]); rng.shuffle(values); train[t] = values
    test = pd.DataFrame(dict(clip=[f'priv{i:03d}.mp4' for i in range(n_private)],
                             source_id=[f'p{i}' for i in range(n_private)],
                             preferred_foot=rng.choice(['left', 'right'], n_private), gender=rng.choice(['F', 'M'], n_private)))
    videos = Path(root) / 'videos'; videos.mkdir(parents=True, exist_ok=True)
    paths = {}
    for c in list(train['clip']) + list(test['clip']):
        p = videos / c; p.write_bytes(hashlib.sha256(c.encode()).digest() * 64); paths[c] = p
    return train, test, paths


def base_namespace(tmp, backbone='videomae', policy='finetune', run_mode='train', n_public=24, n_private=6):
    """Namespace equivalent to the notebook globals after the configuration cell, scaled down."""
    import torch, joblib, itertools, copy, gc, random, time, hashlib as hl, cv2
    from collections import deque
    from tqdm.auto import tqdm
    from catboost import CatBoostClassifier
    from sklearn.model_selection import GroupKFold, GroupShuffleSplit, StratifiedGroupKFold
    from sklearn.metrics import accuracy_score
    from sklearn.dummy import DummyClassifier
    from transformers import VideoMAEConfig, VideoMAEForVideoClassification, VJEPA2Config, VJEPA2Model
    from PIL import Image
    tmp = Path(tmp)
    train, test, paths = make_dataset(tmp, n_public, n_private)
    Y = np.column_stack([train[t].map({c: i for i, c in enumerate(cs)}).to_numpy() for t, cs in zip(TARGETS, CLASSES)]).astype(int)
    vm_config = VideoMAEConfig(image_size=4, patch_size=2, num_frames=2, tubelet_size=2, hidden_size=8,
                               intermediate_size=16, num_hidden_layers=1, num_attention_heads=2, num_labels=3)
    jepa_config = VJEPA2Config(crop_size=4, patch_size=2, frames_per_clip=2, tubelet_size=2, hidden_size=8,
                               num_attention_heads=2, num_hidden_layers=1, pred_hidden_size=8,
                               pred_num_hidden_layers=1, pred_num_attention_heads=2)

    class TinyVideoMAE(VideoMAEForVideoClassification):
        downloads = 0
        @classmethod
        def from_pretrained(cls, name, **kwargs):  # stands in for the hub download: deterministic random init
            TinyVideoMAE.downloads += 1
            torch.manual_seed(0); return cls(vm_config)

    class TinyVJEPA2(VJEPA2Model):
        downloads = 0
        @classmethod
        def from_pretrained(cls, name, **kwargs):
            TinyVJEPA2.downloads += 1
            torch.manual_seed(0); return cls(jepa_config)

    ns = dict(__name__='__notebook__', np=np, pd=pd, torch=torch, json=json, os=os, random=random, time=time, gc=gc,
              copy=copy, itertools=itertools, hashlib=hl, Path=Path, joblib=joblib, tqdm=tqdm, pickle=pickle, deque=deque,
              cv2=cv2, Image=Image, CatBoostClassifier=CatBoostClassifier, GroupKFold=GroupKFold,
              GroupShuffleSplit=GroupShuffleSplit, StratifiedGroupKFold=StratifiedGroupKFold,
              accuracy_score=accuracy_score, DummyClassifier=DummyClassifier, display=lambda *a, **k: None,
              YOLO=None, VideoMAEForVideoClassification=TinyVideoMAE, TinyVJEPA2=TinyVJEPA2,
              RUN_MODE=run_mode, train=train, test=test, Y=Y, TARGETS=TARGETS, CLASSES=CLASSES,
              WEIGHTS=np.array([.7, .3], dtype=np.float32), rows=pd.concat([train, test], ignore_index=True),
              clicks=pd.DataFrame(columns=['kicker_click_time_seconds']).rename_axis('clip'),
              video_paths=paths, DATA_ID='synthdata0001', OUTPUT=tmp / 'outputs', SEED=42,
              VIDEO_EPOCHS=2, VIDEO_SEEDS=[42, 137], OUTER_FOLDS=2, ROLE_FOLDS=2, GRAD_ACCUM=2, QUICK_SMOKE=False,
              SMOKE_MAX_BATCHES=4, DEVICE='cpu', N_FRAMES=2, SIZE=4, WINDOW=1.6, EVALUATION_MODE='nested_cv',
              GENERATE_TRACKING_AUDITS=False, PREDICT_PRIVATE=True, RUN_ABLATIONS=False, ABLATION_VARIANTS=[],
              CAMERA_CACHE_VERSION='similarity-jpeg-v1', template=test[['clip']].assign(kick_direction='left', goalkeeper_direction='left'),
              TRAINED_RUN_DIR=None, INFERENCE_INPUT='private', VERIFY_CHECKSUMS=True, MIGRATE_LEGACY_RUN=False,
              IMAGE_MEAN=np.array([.485, .456, .406], dtype=np.float32), IMAGE_STD=np.array([.229, .224, .225], dtype=np.float32))
    ns['OUTPUT'].mkdir(parents=True, exist_ok=True)
    ns['find_one'] = lambda names: None
    ns['RUN_MODE'] = 'inference'  # cells execute definitions only (no CPU smoke, no bulk extraction)
    ns.update(PLATFORM=WRAPPER, SCRATCH_ROOT=tmp / 'scratch', OUTPUT_ROOT=tmp / 'outputs', EXPERIMENT_NAME='test',
              ENVIRONMENT_INFO=dict(platform=WRAPPER), EXECUTION_SETTINGS={}, EXPERIMENT_SETTINGS={},
              SHARED_CODE_FINGERPRINT='test', TRAINING_MODES={'smoke', 'train', 'resume'}, EXPORT_BUNDLE_TO=None,
              DEV_SEEDS=[42], DEV_EPOCHS=2, DEV_TRAIN_SOURCE_FRACTION=.5, DEV_VALIDATION_SOURCE_FRACTION=.25,
              REFINE_TREE_CLASS_WEIGHTS=True, REFINE_WEIGHT_CAP=3.0, REFINE_CLASS_OFFSETS=True,
              REFINE_CENTER_OFFSETS=[0.0, 0.25, 0.5, 0.75, 1.0], REFINE_AUDITS=True)
    for name in ['platform_helpers', 'run_artifacts', 'constants_and_summary_helpers']:
        run_cell(name, ns)
    ns['ATTEMPT_ID'] = ns['make_attempt_id'](WRAPPER)
    for name in ['backbone_adapters', 'inference_helpers', 'diagnostics_helpers']:
        run_cell(name, ns)
    spec = ns['backbone_spec'](backbone)
    spec.update(frames=2, size=4, feature_dim=8, checkpoint=f'tiny-{backbone}')
    ns.update(BACKBONE_SPEC=spec, TRAINING_POLICY=policy, BACKBONE_IDENTITY=ns['backbone_identity'](spec, policy),
              VIDEO_ID=spec['checkpoint'])
    # Scaled-down world: tiny VideoMAE fine-tuning plays the role of the original (legacy-default) configuration.
    legacy = ns['backbone_spec']('videomae'); legacy.update(frames=2, size=4, feature_dim=8, checkpoint='tiny-videomae')
    ns['LEGACY_IDENTITY'] = ns['backbone_identity'](legacy, 'finetune')
    if backbone == 'vjepa2_vitl_256':
        import transformers
        ns['_vjepa2_patch'] = TinyVJEPA2
    cfg = dict(dataset=ns['DATA_ID'], video=ns['VIDEO_ID'], pose='yolo11m-pose.pt', frames=2, detector_conf=.08,
               detector_imgsz=1280, window=1.6, size=4, epochs=2, seeds=[42, 137], outer_folds=2, role_folds=2,
               role_min_prob=.35, version='nested-components-v6', gradient_accumulation=2, evaluation_mode='nested_cv',
               validation_fraction=.2, test_fraction=.05, split_seed=42, targets=TARGETS, metric_weights=[.7, .3],
               tracking=dict(max_gap=.5, max_jump=1.25, speed=6.0), smoke=False, smoke_max_batches=4)
    if ns['BACKBONE_IDENTITY'] != ns['LEGACY_IDENTITY']:
        cfg.update(backbone=ns['BACKBONE_IDENTITY'], training_policy=policy)
    run = ns['OUTPUT'] / ('learned_' + hashlib.sha256(json.dumps(cfg, sort_keys=True).encode()).hexdigest()[:16])
    run.mkdir(exist_ok=True)
    raw_cfg = dict(dataset=ns['DATA_ID'], pose='yolo11m-pose.pt', conf=.08, imgsz=1280, frames=2, window=1.6,
                   click_timestamps='none', camera_cache_version='similarity-jpeg-v1')
    ns.update(CFG=cfg, RUN=run, STAGES=run / 'stages', RAW=ns['OUTPUT'] / 'detector_cache_test', RAW_CFG=raw_cfg,
              EXPERIMENT_ID=hashlib.sha256(json.dumps(cfg, sort_keys=True).encode()).hexdigest()[:16],
              OUTPUT=ns['OUTPUT'])
    ns['STAGES'].mkdir(exist_ok=True); ns['RAW'].mkdir(exist_ok=True)
    for name in ['extraction', 'roles_and_crops', 'model_and_training', 'nested_orchestration', 'inference_glue', 'refinement']:
        run_cell(name, ns)
    install_label_free_stubs(ns)
    ns['RUN_MODE'] = run_mode
    return ns


def install_label_free_stubs(ns):
    """Replace video decoding/detection/role fitting with deterministic synthetic equivalents.
    make_views output depends only on the clip, so training-time and inference-time crops agree."""
    calls = ns.setdefault('_calls', dict(extract_raw=0, make_views=0, fit_roles=0))

    def extract_raw(clip, path=None, cache_path=None):
        calls['extract_raw'] += 1
        return dict(clip=clip)

    def make_views(raw, selector):
        calls['make_views'] += 1
        return synthetic_view(raw['clip'], ns['N_FRAMES'], ns['SIZE'])

    def fit_roles(indices, path):
        calls['fit_roles'] += 1
        result = dict(models=[], sources=sorted(set(ns['rows'].source_id.iloc[indices])), counts=[])
        import joblib; joblib.dump(result, path); return result

    def collect_tracking_diagnostics(indices, mapping, fold, output_dir=None, fit_sources=None):
        rows = ns['rows']; q = []
        for i in map(int, indices):
            seed = int(hashlib.sha256(rows['clip'].iloc[i].encode()).hexdigest()[:6], 16)
            frac = (seed % 10) / 10
            q.append(dict(clip=rows['clip'].iloc[i], source_id=rows.source_id.iloc[i], fold=int(fold),
                          kicker_tracked_fraction=frac, keeper_tracked_fraction=1.0,
                          kicker_tracking_quality='mostly_tracked' if frac >= .8 else 'partial',
                          keeper_tracking_quality='mostly_tracked', camera_valid_fraction=1.0))
        q, f, c = pd.DataFrame(q), pd.DataFrame([dict(clip='x')]), pd.DataFrame(columns=['role', 'timestamp_eligible'])
        if output_dir is not None:   # same files as the real collector (aggregation reads them)
            out = Path(output_dir); out.mkdir(parents=True, exist_ok=True)
            q.to_csv(out / 'tracking_quality.csv', index=False); f.to_csv(out / 'tracking_frames.csv', index=False)
            c.to_csv(out / 'role_click_audit.csv', index=False)
        return q, f, c

    ns.update(extract_raw=extract_raw, make_views=make_views, fit_roles=fit_roles,
              collect_tracking_diagnostics=collect_tracking_diagnostics)
    if '_vjepa2_patch' in ns:
        import transformers
        ns['_orig_vjepa2'] = transformers.VJEPA2Model
        transformers.VJEPA2Model = ns['_vjepa2_patch']


@contextlib.contextmanager
def cpu_training_patches():
    """torch.cuda.synchronize -> no-op; GradScaler('cuda') -> CPUGradScaler; autocast('cuda') disables itself."""
    import torch
    saved = (torch.cuda.synchronize, torch.amp.GradScaler)
    torch.cuda.synchronize = lambda *a, **k: None
    torch.amp.GradScaler = CPUGradScaler
    try:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            yield
    finally:
        torch.cuda.synchronize, torch.amp.GradScaler = saved


def train_nested(ns):
    with cpu_training_patches():
        result = ns['run_nested_diagnostics']('production', private_inference=True)
    return result


FIT_ENTRY_POINTS = ['fit_video', 'fit_roles', 'diagnostic_stage', 'run_nested_diagnostics', 'fit_feature_scaler',
                    'select_diagnostic_mixture', 'select_reference_components', 'select_mixture', 'fit_predict_stage',
                    'build_stage', 'run_holdout', 'run_smoke']


@contextlib.contextmanager
def forbid_fitting(ns):
    """Every training/fitting/selection entry point raises while inside this context."""
    import torch, sklearn.preprocessing, catboost
    def boom(name):
        def f(*a, **k): raise AssertionError(f'inference called forbidden fitting entry point: {name}')
        return f
    saved_ns = {k: ns.get(k) for k in FIT_ENTRY_POINTS}
    for k in FIT_ENTRY_POINTS: ns[k] = boom(k)
    patches = [(torch.optim.Optimizer, '__init__'), (sklearn.preprocessing.StandardScaler, 'fit'),
               (sklearn.preprocessing.StandardScaler, 'fit_transform'), (catboost.CatBoostClassifier, 'fit')]
    originals = [getattr(o, a) for o, a in patches]
    for (o, a) in patches: setattr(o, a, boom(f'{o.__name__}.{a}'))
    try: yield
    finally:
        for (o, a), v in zip(patches, originals): setattr(o, a, v)
        ns.update(saved_ns)


def sha_tree(run):
    run = Path(run)
    return {str(p.relative_to(run)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(run.rglob('*')) if p.is_file() and 'inference' not in p.relative_to(run).parts[:1]}
