"""Train tiny synthetic nested runs with the notebook's real orchestration (CPU), then test inference-only use.

Real cells: build_stage/resolve_stage_folder, fit_video, predict_video, diagnostic_stage, run_nested_diagnostics,
ComponentOOF export, inference manifest export, run_saved_inference and its notebook bindings, migration.
Synthetic stand-ins: video decoding, YOLO detection, role CatBoost fitting and crops (deterministic per clip),
tracking audits, and tiny randomly initialized VideoMAE / V-JEPA 2 models in place of hub downloads.
"""
import json, shutil, subprocess, sys, textwrap
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent))
import nb_harness as h


@pytest.fixture(scope='module')
def trained(tmp_path_factory):
    root = tmp_path_factory.mktemp('videomae_run')
    ns = h.base_namespace(root)
    result = h.train_nested(ns)
    run = Path(result['run'])
    ns.update(TRAINED_RUN_DIR=run, RUN_MODE='inference', INFERENCE_RAW=root / 'inference_raw')
    ns['INFERENCE_RAW'].mkdir()
    return dict(ns=ns, run=run, root=root, result=result)


def infer(ns, spec, **kw):
    frame, routing = ns['resolve_inference_clips'](spec)
    with h.forbid_fitting(ns):
        return ns['run_saved_inference'](ns['TRAINED_RUN_DIR'], frame, hooks=ns['INFERENCE_HOOKS'], routing=routing,
                                         device='cpu', dataset_id=ns['DATA_ID'], **kw)


def test_training_exports_priors_serialization_and_manifest(trained):
    ns, run = trained['ns'], trained['run']
    manifest = json.loads((run / 'inference_manifest.json').read_text())
    assert [m['member_id'] for m in manifest['members']] == ['fold0', 'fold1']
    y = ns['Y']; clips = ns['train']['clip'].tolist()
    for m in manifest['members']:
        stage = run / m['stage_dir']
        assert (stage / 'class_prior.json').exists() and (stage / 'backbone_serialization.json').exists()
        idx = [clips.index(c) for c in m['training_clips']]
        for j in range(2):
            np.testing.assert_allclose(m['class_prior'][j], np.bincount(y[idx, j], minlength=3) / len(idx))
        assert all('latest' not in s['weights'] for s in m['seeds'])            # inference weights, not resume state
        assert {s['weight_kind'] for s in m['seeds']} == {'selected_epoch_final_outer_refit'}
        assert set(m['heldout_public_clips']).isdisjoint(m['training_clips'])
        assert m['class_prior_source'] == 'training_labels_at_fit'
    assert manifest['backbone']['identity']['name'] == 'videomae'
    assert manifest['preprocessing']['frames'] == 2 and manifest['preprocessing']['size'] == 4
    for rel, info in manifest['files'].items():
        assert info['sha256'] == h.hashlib.sha256((run / rel).read_bytes()).hexdigest()


def test_public_oof_replay_uses_only_heldout_fold_and_reproduces_saved_oof(trained):
    ns, run = trained['ns'], trained['run']
    downloads = ns['VideoMAEForVideoClassification'].downloads
    r = infer(ns, 'public_oof')
    folds = pd.read_csv(run / 'folds.csv').set_index('clip').fold
    assert (r['table'].members == ['fold%d' % folds[c] for c in r['table']['clip']]).all()
    assert set(r['table'].exposure) == {'out_of_fold'}
    assert max(r['record']['max_abs_difference_vs_saved_oof'].values()) <= 1e-6
    assert ns['VideoMAEForVideoClassification'].downloads == downloads          # no pretrained load at inference
    assert r['record']['labels_used'] is False and r['record']['fitting_performed'] is False


def test_private_inference_matches_training_time_predictions_and_writes_submission(trained):
    ns, run = trained['ns'], trained['run']
    before = h.sha_tree(run / 'stages')
    r = infer(ns, 'private')
    saved = np.load(run / 'predictions.npz')
    for j in range(2):
        np.testing.assert_allclose(r['final'][j], saved[f'private_{j}'], atol=1e-6, rtol=0)
        np.testing.assert_allclose(r['final'][j].sum(1), 1, atol=1e-6)
    submission = ns['write_submission'](r, ns['template'], r['output_dir'] / 'submission.csv')
    assert list(submission.columns) == ['clip', 'kick_direction', 'goalkeeper_direction'] and len(submission) == len(ns['test'])
    assert h.sha_tree(run / 'stages') == before                                 # fitted artifacts untouched


def test_changed_clip_set_reuses_fitted_identity_and_only_adds_prediction_caches(trained, tmp_path):
    ns, run = trained['ns'], trained['run']
    manifest = json.loads((run / 'inference_manifest.json').read_text())
    ids = [m['fitted_artifact_id'] for m in manifest['members']]
    external = []
    for k in range(2):  # brand-new videos outside the dataset: decoded/detected/cropped from scratch
        p = tmp_path / f'new{k}.mp4'; p.write_bytes(bytes([k + 1]) * 999)
        external.append(dict(clip=f'new{k}.mp4', path=str(p), preferred_foot='left', gender='F'))
    private = ns['test'].iloc[:3]
    rows = pd.DataFrame(external + [dict(clip=c, path=str(ns['video_paths'][c]), preferred_foot=f, gender=g)
                                     for c, f, g in zip(private['clip'], private.preferred_foot, private.gender)])
    csv = tmp_path / 'clips.csv'; rows.to_csv(csv, index=False)
    calls = dict(ns['_calls'])
    r1 = infer(ns, str(csv))
    stats = r1['record']['cache_statistics']
    assert stats['crops_built'] == 2 * 2                                           # 2 new clips x 2 fold members
    assert ns['_calls']['make_views'] - calls['make_views'] == 4
    assert stats['predictions_computed'] == 4 and stats['predictions_reused'] == 3 * 2   # private clips cached earlier
    assert list(r1['record']['fitted_artifact_ids'].values()) == ids
    r2 = infer(ns, str(csv))
    assert r2['record']['cache_statistics']['predictions_computed'] == 0
    for j in range(2): np.testing.assert_allclose(r1['final'][j], r2['final'][j], atol=0)
    assert json.loads((run / 'inference_manifest.json').read_text())['files'] == manifest['files']


def test_ensemble_routing_marks_public_clips_in_sample(trained):
    ns = trained['ns']
    frame, _ = ns['resolve_inference_clips']('public_oof')
    with h.forbid_fitting(ns):
        r = ns['run_saved_inference'](ns['TRAINED_RUN_DIR'], frame.iloc[:4], hooks=ns['INFERENCE_HOOKS'],
                                      routing='ensemble', device='cpu', dataset_id=ns['DATA_ID'])
    assert set(r['table'].exposure) == {'in_sample'}


def test_missing_or_modified_artifact_fails_clearly(trained):
    ns, run = trained['ns'], trained['run']
    manifest = json.loads((run / 'inference_manifest.json').read_text())
    weights = run / manifest['members'][1]['seeds'][1]['weights']
    moved = weights.with_suffix('.bak'); weights.rename(moved)
    try:
        with pytest.raises(ns['InferenceArtifactError'], match=str(weights).replace('\\', '\\\\')):
            ns['load_inference_manifest'](run)
    finally:
        moved.rename(weights)
    scaler = run / manifest['members'][0]['numeric_scaler']
    original = scaler.read_bytes()
    try:
        scaler.write_bytes(original[:-1] + bytes([original[-1] ^ 1]))
        with pytest.raises(ns['InferenceArtifactError'], match='checksum changed'):
            ns['load_inference_manifest'](run)
    finally:
        scaler.write_bytes(original)
    ns['load_inference_manifest'](run)


def test_incomplete_run_is_never_exported(trained, tmp_path):
    ns, run = trained['ns'], trained['run']
    copy = tmp_path / run.name; shutil.copytree(run, copy)
    results = json.loads((copy / 'fold_results.json').read_text())
    (copy / 'fold_results.json').write_text(json.dumps(results[:1]))
    pre = json.loads((run / 'inference_manifest.json').read_text())['preprocessing']
    with pytest.raises(ns['InferenceArtifactError'], match='incomplete'):
        ns['export_inference_manifest'](copy, preprocessing=pre)


def test_legacy_videomae_run_migrates_and_predicts_identically(trained, tmp_path, monkeypatch):
    ns, run = trained['ns'], trained['run']
    legacy = tmp_path / run.name; shutil.copytree(run, legacy)
    shutil.rmtree(legacy / 'inference', ignore_errors=True)
    (legacy / 'inference_manifest.json').unlink()
    serialization = json.loads(next(legacy.glob('stages/fold0_outer*/backbone_serialization.json')).read_text())
    for pattern in ['stages/*/class_prior.json', 'stages/*/backbone_serialization.json', 'stages/*/stage_identity.json']:
        for p in legacy.glob(pattern): p.unlink()
    # Old experiment configs had no backbone key; the migration fetches only the architecture config.
    import transformers
    tiny = transformers.VideoMAEConfig.from_dict(serialization['architecture']['config'])
    monkeypatch.setattr(transformers.VideoMAEConfig, 'from_pretrained', classmethod(lambda cls, name, **k: tiny))
    pre = json.loads((run / 'inference_manifest.json').read_text())['preprocessing']
    with pytest.raises(ns['InferenceArtifactError'], match='class_prior.json is missing'):
        ns['export_inference_manifest'](legacy, preprocessing=pre, backbone=serialization)
    assert not (legacy / 'inference_manifest.json').exists()
    old_files = {p: p.read_bytes() for p in legacy.rglob('*.pt')}
    ns['migrate_legacy_run'](legacy)                                    # real notebook migration function
    manifest = json.loads((legacy / 'inference_manifest.json').read_text())
    assert {m['class_prior_source'] for m in manifest['members']} == {'migrated_from_training_labels'}
    original = json.loads((run / 'inference_manifest.json').read_text())
    for a, b in zip(manifest['members'], original['members']):
        np.testing.assert_allclose(a['class_prior'], b['class_prior'])
        assert a['fitted_artifact_id'] == b['fitted_artifact_id']
    assert {p: p.read_bytes() for p in legacy.rglob('*.pt')} == old_files   # old folder preserved
    ns_legacy = dict(ns, TRAINED_RUN_DIR=legacy)
    frame, routing = ns['resolve_inference_clips']('private')
    with h.forbid_fitting(ns):
        r = ns['run_saved_inference'](legacy, frame, hooks=ns['INFERENCE_HOOKS'], routing=routing, device='cpu',
                                      dataset_id=ns['DATA_ID'])
    saved = np.load(run / 'predictions.npz')
    for j in range(2): np.testing.assert_allclose(r['final'][j], saved[f'private_{j}'], atol=1e-6)


def test_fresh_process_inference_with_every_fit_entry_point_disabled(trained, tmp_path):
    root = trained['root']
    new_video = tmp_path / 'fresh_new.mp4'; new_video.write_bytes(b'\x07' * 777)
    csv = tmp_path / 'fresh.csv'
    pd.DataFrame([dict(clip='fresh_new.mp4', path=str(new_video), preferred_foot='right', gender='M')] +
                 [dict(clip=c, path=str(trained['ns']['video_paths'][c]), preferred_foot='left', gender='F')
                  for c in trained['ns']['test']['clip'][:2]]).to_csv(csv, index=False)
    script = textwrap.dedent(f'''
        import sys, json; sys.path.insert(0, {str(Path(__file__).parent)!r})
        import nb_harness as h
        from pathlib import Path
        ns = h.base_namespace({str(root)!r}, run_mode='inference')
        run = Path({str(trained['run'])!r}); ns.update(TRAINED_RUN_DIR=run, INFERENCE_RAW=Path({str(tmp_path)!r}))
        with h.forbid_fitting(ns):
            for spec in ['private', {str(csv)!r}]:
                frame, routing = ns['resolve_inference_clips'](spec)
                r = ns['run_saved_inference'](run, frame, hooks=ns['INFERENCE_HOOKS'], routing=routing, device='cpu',
                                              dataset_id=ns['DATA_ID'])
                print(json.dumps(dict(spec=spec, n=len(frame), stats=r['record']['cache_statistics'])))
    ''')
    out = subprocess.run([sys.executable, '-c', script], capture_output=True, text=True)
    assert out.returncode == 0, out.stderr[-3000:]
    lines = [json.loads(l) for l in out.stdout.splitlines() if l.startswith('{"spec"')]
    assert len(lines) == 2 and lines[1]['stats']['crops_built'] >= 2


@pytest.fixture(scope='module')
def trained_vjepa2_frozen(tmp_path_factory):
    root = tmp_path_factory.mktemp('vjepa2_run')
    ns = h.base_namespace(root, backbone='vjepa2_vitl_256', policy='frozen_probe')
    try:
        result = h.train_nested(ns)
        ns.update(TRAINED_RUN_DIR=Path(result['run']), RUN_MODE='inference', INFERENCE_RAW=root / 'raw')
        ns['INFERENCE_RAW'].mkdir()
        yield dict(ns=ns, run=Path(result['run']))
    finally:
        import transformers
        if '_orig_vjepa2' in ns: transformers.VJEPA2Model = ns['_orig_vjepa2']


def test_vjepa2_frozen_probe_run_trains_heads_only_and_reloads_for_inference(trained_vjepa2_frozen):
    ns, run = trained_vjepa2_frozen['ns'], trained_vjepa2_frozen['run']
    import torch
    manifest = json.loads((run / 'inference_manifest.json').read_text())
    assert manifest['backbone']['identity']['name'] == 'vjepa2_vitl_256'
    assert manifest['backbone']['training_policy'] == 'frozen_probe'
    assert manifest['config']['backbone']['training_policy'] == 'frozen_probe'
    a, b = [torch.load(run / s['weights'], weights_only=True)['model'] for s in manifest['members'][0]['seeds']]
    backbone = [k for k in a if k.startswith('backbone.')]
    assert backbone and all(torch.equal(a[k], b[k]) for k in backbone)     # frozen pretrained encoder (same init)
    assert not torch.equal(a['heads.0.weight'], b['heads.0.weight'])
    serial = json.loads((run / manifest['members'][0]['stage_dir'] / 'backbone_serialization.json').read_text())
    assert serial['parameters']['backbone']['trainable'] == 0 and serial['parameters']['heads']['trainable'] > 0
    downloads = ns['TinyVJEPA2'].downloads
    r = infer(ns, 'public_oof')
    assert max(r['record']['max_abs_difference_vs_saved_oof'].values()) <= 1e-6
    assert ns['TinyVJEPA2'].downloads == downloads


PATH_KEYS = {'DATASET_PATH', 'OUTPUT_ROOT', 'SCRATCH_ROOT', 'TRAINED_RUN_DIR', 'EXPORT_BUNDLE_TO'}


def settings_source(platform, **settings):
    import re
    src = h.cell_source('settings', platform)
    for key, value in settings.items():
        text = f'Path({str(value)!r})' if key in PATH_KEYS and value is not None else repr(value)
        src, n = re.subn(rf'^{key} = .*$', f'{key} = {text}', src, count=1, flags=re.M)
        assert n == 1, key
    return src


def config_namespace(trained, monkeypatch, platform='colab', output_root=None, **settings):
    """Execute the real Settings, platform helpers, validation, artifact, reports-gate and configuration cells."""
    import types, transformers
    monkeypatch.setitem(sys.modules, 'ultralytics', types.SimpleNamespace(YOLO=lambda *a, **k: None))
    monkeypatch.setattr(transformers.VideoMAEImageProcessor, 'from_pretrained',
                        classmethod(lambda cls, name, **k: types.SimpleNamespace(image_mean=[.5] * 3, image_std=[.25] * 3)))
    base = trained['ns']
    clicks = trained['root'] / 'clicks.csv'
    pd.DataFrame(dict(clip=['pub000.mp4'], kicker_click_time_seconds=[1.0], keeper_click_time_seconds=[1.0])).to_csv(clicks, index=False)
    settings = dict(dict(OUTPUT_ROOT=output_root or base['OUTPUT'], SCRATCH_ROOT=trained['root'] / f'scratch_{platform}',
                         DATASET_PATH=trained['root'] / 'videos' / 'pub000.mp4', ALLOW_EPHEMERAL_OUTPUT=True,
                         RUN_MODE='train'), **settings)
    if platform == 'hpc': settings.setdefault('ALLOW_OUTSIDE_SLURM', True)
    ns = {k: base[k] for k in ['np', 'pd', 'json', 'hashlib', 'Path', 'train', 'test', 'Y', 'DATA_ID', 'SEED', 'display']}
    ns.update(DEVICE='cuda', find_one=lambda names: clicks, ENVIRONMENT_INFO=dict(platform=platform))
    exec(settings_source(platform, **settings), ns)
    for name in ['platform_helpers', 'settings_validation', 'run_artifacts', 'constants_and_summary_helpers',
                 'reports_mode', 'backbone_adapters', 'experiment_configuration']:
        h.run_cell(name, ns, platform)
    return ns


def test_configuration_cell_inference_uses_saved_run_settings(trained, monkeypatch, capsys):
    run = trained['run']
    ns = config_namespace(trained, monkeypatch, RUN_MODE='inference', TRAINED_RUN_DIR=run, BACKBONE='vjepa2_1_vitb_384')
    saved = json.loads((run / 'inference_manifest.json').read_text())
    assert ns['RUN'] == run and ns['CFG'] == saved['config']
    assert (ns['N_FRAMES'], ns['SIZE']) == (2, 4) and ns['BACKBONE_SPEC'] == saved['backbone']['spec']
    assert ns['TRAINING_POLICY'] == 'finetune' and ns['VIDEO_SEEDS'] == saved['config']['seeds']
    np.testing.assert_allclose(ns['IMAGE_MEAN'], saved['preprocessing']['image_mean'])
    assert 'The saved run wins' in capsys.readouterr().out


def test_configuration_cell_resume_refuses_changed_settings(trained, monkeypatch):
    with pytest.raises(ValueError, match='Differing settings'):
        config_namespace(trained, monkeypatch, RUN_MODE='resume', TRAINED_RUN_DIR=trained['run'], EVALUATION_MODE='nested_cv')


def test_configuration_cell_default_videomae_keeps_legacy_run_identity(trained, monkeypatch):
    default = config_namespace(trained, monkeypatch, EVALUATION_MODE='nested_cv')
    assert 'backbone' not in default['CFG'] and default['LEGACY_DEFAULT_BACKBONE'] and default['SIZE'] == 224
    jepa = config_namespace(trained, monkeypatch, BACKBONE='vjepa2_1_vitb_384', EVALUATION_MODE='nested_cv')
    assert jepa['CFG']['backbone']['name'] == 'vjepa2_1_vitb_384' and (jepa['N_FRAMES'], jepa['SIZE']) == (16, 384)
    assert jepa['RUN'] != default['RUN'] and np.allclose(jepa['IMAGE_MEAN'], [.485, .456, .406])


def test_identity_ignores_platform_and_paths_but_runs_are_isolated(trained, monkeypatch, tmp_path):
    colab = config_namespace(trained, monkeypatch, 'colab', output_root=tmp_path / 'drive', EVALUATION_MODE='nested_cv')
    hpc = config_namespace(trained, monkeypatch, 'hpc', output_root=tmp_path / 'scratch_netid', EVALUATION_MODE='nested_cv')
    assert colab['CFG'] == hpc['CFG'] and colab['EXPERIMENT_ID'] == hpc['EXPERIMENT_ID']
    assert colab['RUN'] != hpc['RUN'] and 'colab' in colab['RUN'].parts and 'hpc' in hpc['RUN'].parts
    same_root = config_namespace(trained, monkeypatch, 'hpc', output_root=tmp_path / 'drive', EVALUATION_MODE='nested_cv')
    assert same_root['RUN'] != colab['RUN']                    # same root, equivalent config: still separate writers
    assert list((colab['RUN'] / 'attempts').iterdir()) and list((hpc['RUN'] / 'attempts').iterdir())
    changed = config_namespace(trained, monkeypatch, 'hpc', output_root=tmp_path / 'scratch_netid',
                               EVALUATION_MODE='nested_cv', TRAINING_POLICY='frozen_probe')
    assert changed['EXPERIMENT_ID'] != hpc['EXPERIMENT_ID'] and changed['CFG']['optimizer']['policy'] == 'frozen_probe'
    dev = config_namespace(trained, monkeypatch, 'colab', output_root=tmp_path / 'drive', EVALUATION_MODE='development',
                           PREDICT_PRIVATE=False)
    assert dev['CFG']['evaluation_mode'] == 'development' and dev['VIDEO_SEEDS'] == [42] and dev['VIDEO_EPOCHS'] == 3


@pytest.mark.parametrize('platform', ['colab', 'hpc'])
def test_incompatible_settings_fail_before_anything_expensive(trained, monkeypatch, platform):
    for bad in [dict(RUN_MODE='inference'), dict(PREDICT_PRIVATE=True, EVALUATION_MODE='holdout'), dict(EVALUATION_MODE='development', DEVELOPMENT_BACKBONES=['videomae', 'vjepa3']),
                dict(FOLD_SELECTION=[0], EVALUATION_MODE='holdout'), dict(RUN_ABLATIONS=True, EVALUATION_MODE='development'),
                dict(ALLOW_EPHEMERAL_OUTPUT=False, OUTPUT_ROOT='/tmp/x' if platform == 'hpc' else '/content/out'),
                dict(BACKBONE='vjepa3')]:
        with pytest.raises(ValueError, match='Settings problems'):
            config_namespace(trained, monkeypatch, platform, **bad)
