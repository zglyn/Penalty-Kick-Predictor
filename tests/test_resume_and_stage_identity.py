"""Epoch resume under each training policy (real fit_video on CPU) and fitted-stage identity (real build_stage)."""
import json, sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).parent))
import nb_harness as h


class Interrupt(Exception):
    pass


def setup(tmp, policy, backbone='videomae'):
    ns = h.base_namespace(tmp, backbone=backbone, policy=policy)
    ns['set_active_ablation']('production')
    indices = np.arange(8)
    folder = ns['STAGES'] / 'manual'; folder.mkdir(parents=True)
    mapping = {}
    for i in indices:
        path = folder / f'{i}.pkl'
        ns['atomic_pickle'](h.synthetic_view(ns['rows']['clip'].iloc[i], 2, 4), path); mapping[int(i)] = path
    return ns, indices, mapping


def fit(ns, indices, mapping, folder, interrupt_after=None):
    folder.mkdir(exist_ok=True)
    real_save = ns['atomic_torch_save']; saves = []
    def save(state, path):
        real_save(state, path); saves.append(Path(path).name)
        if interrupt_after and Path(path).name.endswith('_latest.pt') and state['completed_epochs'] == interrupt_after:
            raise Interrupt()
    ns['atomic_torch_save'] = save
    try:
        with h.cpu_training_patches():
            model, scaler = ns['fit_video'](indices, mapping, folder, 42)
        return model
    finally:
        ns['atomic_torch_save'] = real_save


@pytest.mark.parametrize('policy,backbone', [('finetune', 'videomae'), ('frozen_probe', 'videomae'), ('finetune', 'vjepa2_vitl_256'),
                                             ('frozen_probe', 'vjepa2_vitl_256')])
def test_interrupted_training_resumes_to_identical_weights(tmp_path, policy, backbone, capsys):
    ns, indices, mapping = setup(tmp_path, policy, backbone)
    try:
        straight = fit(ns, indices, mapping, ns['STAGES'] / 'straight')
        with pytest.raises(Interrupt):
            fit(ns, indices, mapping, ns['STAGES'] / 'resumed', interrupt_after=1)
        saved = torch.load(ns['STAGES'] / 'resumed' / 'video_seed42_latest.pt', weights_only=True)
        assert saved['completed_epochs'] == 1 and saved['model_identity'] == ns['BACKBONE_IDENTITY']
        assert set(saved) >= {'optimizer', 'amp', 'rng', 'history'}
        resumed = fit(ns, indices, mapping, ns['STAGES'] / 'resumed')
        assert 'Restored seed 42: 1/2 epochs complete. Resuming at epoch 2.' in capsys.readouterr().out
        a, b = straight.state_dict(), resumed.state_dict()
        for k in a: torch.testing.assert_close(a[k], b[k], rtol=0, atol=0)
        history = torch.load(ns['STAGES'] / 'resumed' / 'video_seed42_latest.pt', weights_only=True)['history']
        assert [r['epoch'] for r in history] == [1, 2]
        if policy == 'frozen_probe':
            final = torch.load(ns['STAGES'] / 'resumed' / 'video_seed42.pt', weights_only=True)['model']
            fresh = ns['MultiViewVideo'](len(ns['numeric_feature_manifest']())).state_dict()
            assert all(torch.equal(final[k], fresh[k]) for k in final if k.startswith('backbone.'))
    finally:
        import transformers
        if '_orig_vjepa2' in ns: transformers.VJEPA2Model = ns['_orig_vjepa2']


def test_backbone_or_policy_mismatch_on_resume_is_rejected(tmp_path):
    ns, indices, mapping = setup(tmp_path, 'finetune')
    folder = ns['STAGES'] / 'mismatch'
    with pytest.raises(Interrupt):
        fit(ns, indices, mapping, folder, interrupt_after=1)
    # Same CFG/fit manifest, different training policy -> refused instead of silently loading.
    ns['TRAINING_POLICY'] = 'frozen_probe'; ns['BACKBONE_IDENTITY'] = ns['backbone_identity'](ns['BACKBONE_SPEC'], 'frozen_probe')
    with pytest.raises(ValueError, match='Backbone/policy changes must start a new run'):
        fit(ns, indices, mapping, folder)
    # A pre-update checkpoint (no model_identity field) is accepted only as legacy VideoMAE fine-tuning.
    ns['TRAINING_POLICY'] = 'finetune'; ns['BACKBONE_IDENTITY'] = ns['LEGACY_IDENTITY']
    path = folder / 'video_seed42_latest.pt'
    state = torch.load(path, weights_only=True); state.pop('model_identity'); torch.save(state, path)
    fit(ns, indices, mapping, folder)


def test_prediction_list_does_not_change_fitted_stage_and_legacy_folder_is_reused(tmp_path):
    ns = h.base_namespace(tmp_path)
    # Source-disjoint public split (source = index % 12): training sources 0-7, evaluation sources 8-11.
    training = np.array([i for i in range(24) if i % 12 < 8]); heldout = np.array([i for i in range(24) if i % 12 >= 8])
    first = heldout[:4]; second = np.r_[heldout[2:], np.arange(24, 27)]        # overlapping list + private clips
    selection = heldout[:4]
    f1, m1, _ = ns['build_stage'](training, first, 'fold9_inner', selection=selection)
    calls = ns['_calls']['fit_roles']
    f2, m2, _ = ns['build_stage'](training, second, 'fold9_inner', selection=selection)
    assert f1 == f2 and set(m2) >= set(second)
    prov = json.loads((f2 / 'provenance.json').read_text())
    evaluation = [e for e in prov if e['kind'] == 'evaluation']
    assert len(evaluation) == 1 and evaluation[0]['prediction_indices'] == sorted(set(first) | set(second))
    f3, _, _ = ns['build_stage'](training, first, 'fold9_inner', selection=None)  # different selection -> new fit
    assert f3 != f1
    # Legacy folder (old tag hashed the evaluation list) with matching fit manifests is reused.
    legacy = ns['STAGES'] / 'fold8_outer_abcdef123456'; legacy.mkdir()
    (legacy / 'provenance.json').write_text(json.dumps([dict(kind='training_crossfit', fit_sources=[], prediction_indices=training.tolist()),
                                                         dict(kind='evaluation', fit_sources=[], prediction_indices=[int(heldout[0])])]))
    (legacy / 'video_seed42_fit_manifest.json').write_text(json.dumps(dict(config=ns['CFG'], training_indices=training.tolist(),
                                                                          validation_indices=[])))
    f4, _, _ = ns['build_stage'](training, second, 'fold8_outer')
    assert f4 == legacy and json.loads((legacy / 'stage_identity.json').read_text())['legacy_folder'] is True
    (legacy / 'video_seed42_fit_manifest.json').write_text(json.dumps(dict(config=dict(ns['CFG'], epochs=99),
                                                                          training_indices=training.tolist(), validation_indices=[])))
    other = ns['STAGES'] / 'fold7_outer_abcdef123456'; other.mkdir()
    (other / 'provenance.json').write_text((legacy / 'provenance.json').read_text())
    (other / 'video_seed42_fit_manifest.json').write_text((legacy / 'video_seed42_fit_manifest.json').read_text())
    f5, _, _ = ns['build_stage'](training, second, 'fold7_outer')
    assert f5 != other                                                      # different config: never reused


@pytest.mark.parametrize('policy', ['finetune', 'frozen_probe'])
def test_quick_smoke_update_checks_for_each_policy(tmp_path, policy):
    ns, indices, mapping = setup(tmp_path, policy)
    ns['QUICK_SMOKE'] = True
    folder = ns['STAGES'] / f'smoke_{policy}'
    fit(ns, indices, mapping, folder)
    checks = json.loads((folder / 'video_training_checks.json').read_text())
    assert checks['training_policy'] == policy and checks['probe_max_change'] > 0 and checks['optimizer_updates'] > 0
    if policy == 'finetune':
        assert checks['probe_parameter'].startswith('backbone.') and checks['frozen_backbone_max_change'] is None
    else:
        assert not checks['probe_parameter'].startswith('backbone.') and checks['frozen_backbone_max_change'] == 0.0
        assert checks['trainable_parameters']['backbone']['trainable'] == 0


def test_holdout_run_exports_inference_manifest_and_predicts(tmp_path):
    ns = h.base_namespace(tmp_path)
    ns['EVALUATION_MODE'] = 'holdout'
    src = ns['train'].source_id
    ns['split_indices'] = dict(training=np.flatnonzero(src.isin([f'src{i:02d}' for i in range(8)])),
                               validation=np.flatnonzero(src.isin(['src08', 'src09', 'src10'])),
                               test=np.flatnonzero(src == 'src11'))
    ns['train'][['clip', 'source_id']].assign(split=np.select(
        [src.isin([f'src{i:02d}' for i in range(8)]), src == 'src11'], ['training', 'test'], 'validation')).to_csv(ns['RUN'] / 'dataset_splits.csv', index=False)
    mode = ns['RUN_MODE']; ns['RUN_MODE'] = 'reports'           # dispatch cell: definitions only
    h.run_cell('smoke_holdout_dispatch', ns); ns['RUN_MODE'] = mode
    with h.cpu_training_patches():
        ns['run_holdout']()
    run = ns['RUN']
    manifest = json.loads((run / 'inference_manifest.json').read_text())
    assert manifest['evaluation_mode'] == 'holdout' and len(manifest['members']) == 1
    member = manifest['members'][0]
    assert member['ensemble']['components'] == ['video_ensemble'] and member['class_prior'] is None
    assert {s['weight_kind'] for s in member['seeds']} == {'best_validation'}
    ns.update(TRAINED_RUN_DIR=run, INFERENCE_RAW=tmp_path / 'raw'); ns['INFERENCE_RAW'].mkdir()
    test_clips = ns['train'].iloc[ns['split_indices']['test']][['clip', 'source_id', 'preferred_foot', 'gender']]
    frame = test_clips.assign(path=[str(ns['video_paths'][c]) for c in test_clips['clip']], in_dataset=True)
    with h.forbid_fitting(ns):
        r = ns['run_saved_inference'](run, frame, hooks=ns['INFERENCE_HOOKS'], routing='ensemble', device='cpu',
                                      dataset_id=ns['DATA_ID'])
    saved = np.load(run / 'holdout_predictions.npz')
    for j in range(2): np.testing.assert_allclose(r['final'][j], saved[f'test_{j}'], atol=1e-6)
    assert set(r['table'].exposure) == {'not_in_training'}
