"""Refinement of a finished run without video retraining, and the Priority-0 diagnostics.

Uses the notebook's real cells on a tiny synthetic nested run (CPU). Synthetic numbers carry no performance meaning.
"""
import json, shutil, sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent))
import nb_harness as h


@pytest.fixture(scope='module')
def refined(tmp_path_factory):
    root = tmp_path_factory.mktemp('refine_run')
    ns = h.base_namespace(root)
    result = h.train_nested(ns)
    parent = Path(result['run'])
    before = h.sha_tree(parent)
    full_before = {str(p.relative_to(parent)): p.read_bytes() for p in parent.rglob('*') if p.is_file()}
    trained_calls = dict(ns['_calls'])
    with h.cpu_training_patches():
        refine = ns['run_refinement'](parent)
    return dict(ns=ns, parent=parent, derived=Path(refine['run']), refine=refine, before=before,
                full_before=full_before, calls_before=trained_calls)


def test_parent_run_is_untouched_and_outputs_are_separate(refined):
    parent, derived = refined['parent'], refined['derived']
    after = {str(p.relative_to(parent)): p.read_bytes() for p in parent.rglob('*') if p.is_file()}
    assert after == refined['full_before']
    assert derived.parent == parent.parent and derived.name.startswith(parent.name + '__refine_')
    for name in ['submission.csv', 'refine_comparison.csv', 'refine_candidates.csv', 'refinement_manifest.json',
                 'inference_manifest.json', 'label_mapping_audit.json', 'temporal_input_audit.csv',
                 'component_reports/component_metrics.csv', 'predictions.npz']:
        assert (derived / name).exists(), name


def test_no_video_retraining_and_inner_predictions_reproduce_original_choices(refined):
    ns, derived = refined['ns'], refined['derived']
    assert ns['_calls'] == refined['calls_before']            # no crops built, no role selectors fitted
    records = json.loads((derived / 'fold_results.json').read_text())
    assert all(r['original_inner_selection']['reproduced'] for r in records)
    parent_records = json.loads((refined['parent'] / 'fold_results.json').read_text())
    for r, p in zip(records, parent_records):
        assert r['original_inner_selection']['names'] == p['ensemble_components']
        np.testing.assert_allclose(r['original_inner_selection']['weights'], p['weights'])
        assert r['selected_epochs'] == p['selected_epochs']
    assert not list(derived.glob('**/*.pt'))                   # no new video weights anywhere


def test_selection_uses_inner_rule_and_baseline_is_preserved(refined):
    derived, parent = refined['derived'], refined['parent']
    cands = pd.read_csv(derived / 'refine_candidates.csv')
    for (fold, task), part in cands.groupby(['fold', 'task']):
        chosen = part[part.selected]
        assert len(chosen) == 1
        best = part.sort_values(['inner_accuracy', 'inner_cross_entropy'], ascending=[False, True]).iloc[0]
        assert chosen.inner_accuracy.iloc[0] == best.inner_accuracy
    metrics = pd.read_csv(derived / 'component_reports' / 'component_metrics.csv')
    assert {'baseline_ensemble', 'ensemble', 'pose_motion_tree', 'pose_motion_tree_weighted'} <= set(metrics.component)
    base = metrics[(metrics.component == 'baseline_ensemble') & (metrics.fold == 'pooled')].iloc[0]
    orig = pd.read_csv(parent / 'component_reports' / 'component_metrics.csv')
    orig = orig[(orig.component == 'ensemble') & (orig.fold == 'pooled')].iloc[0]
    assert base.weighted_accuracy == pytest.approx(orig.weighted_accuracy)
    comp = pd.read_csv(derived / 'refine_comparison.csv')
    pooled = comp[comp.fold == 'pooled'].set_index('component')
    for col in ['kick_direction_macro_recall', 'kick_direction_center_recall', 'kick_direction_predicted__middle',
                'weighted_accuracy_minus_baseline_ensemble', 'weighted_accuracy_minus_pose_motion_tree']:
        assert col in pooled
    assert pooled.loc['baseline_ensemble', 'weighted_accuracy_minus_baseline_ensemble'] == 0


def test_fresh_inference_replays_refined_ensemble_without_reselection(refined, tmp_path):
    ns, derived = refined['ns'], refined['derived']
    saved = np.load(derived / 'predictions.npz')
    ns['INFERENCE_RAW'] = tmp_path / 'raw'; ns['INFERENCE_RAW'].mkdir()
    for spec, key in [('private', 'private'), ('public_oof', 'oof')]:
        frame, routing = ns['resolve_inference_clips'](spec)
        with h.forbid_fitting(ns):
            r = ns['run_saved_inference'](derived, frame, hooks=ns['INFERENCE_HOOKS'], routing=routing, device='cpu',
                                          dataset_id=ns['DATA_ID'], output_root=tmp_path / spec)
        for j in range(2):
            np.testing.assert_allclose(r['final'][j], saved[f'{key}_{j}'], atol=1e-6)
    manifest = json.loads((derived / 'inference_manifest.json').read_text())
    assert all('pose_motion_tree_weighted' in m['extra_trees'] for m in manifest['members'])
    assert all(len(m['class_offsets']) == 2 for m in manifest['members'])
    with pytest.raises(ValueError, match='refinement run'):
        ns['export_bundle'](derived, tmp_path / 'bundles')


def test_refined_summary_leads_with_class_health(refined):
    ns, derived = refined['ns'], refined['derived']
    summary, text = ns['write_pipeline_summary'](derived, output_dir=derived)
    assert summary['status'] == 'complete' and text.index('## Measured headlines') < text.index('## Investigation priorities')
    health = pd.read_csv(derived / 'class_health.csv')
    assert {'support', 'predicted_support', 'precision', 'recall', 'zero_predictions'} <= set(health)
    assert (derived / 'training_history_report.csv').exists()


def test_weights_are_capped_normalized_and_stage_local(refined):
    ns = refined['ns']
    y = np.array([0] * 80 + [1] * 5 + [2] * 15)
    w = ns['capped_inverse_sqrt_weights'](y, 3, 3.0)
    assert w.mean() == pytest.approx(1) and w.max() <= 3.0 + 1e-9 and w[y == 1].mean() > w[y == 0].mean()
    w2 = ns['capped_inverse_sqrt_weights'](np.array([0] * 9 + [2]), 3, 3.0)   # absent class: no division by zero
    assert np.isfinite(w2).all()


def test_label_mapping_audit_passes_and_detects_a_swapped_decoder(refined, tmp_path):
    ns = refined['ns']
    report = ns['label_mapping_audit'](tmp_path / 'audit.json')
    assert report['passed'] and len(report['checks']) == 6 and report['augmentation_policy']['horizontal_flip'] is False
    real = ns['decode_labels']
    ns['decode_labels'] = lambda p, task: np.asarray(ns['CLASSES'][task])[::-1][np.asarray(p).argmax(1)]
    try:
        with pytest.raises(AssertionError, match='Label mapping audit failed'):
            ns['label_mapping_audit']()
    finally:
        ns['decode_labels'] = real
    with pytest.raises(ValueError, match='Unknown kick_direction'):
        ns['encode_labels'](pd.DataFrame(dict(kick_direction=['center'], goalkeeper_direction=['left'])))


def test_training_history_records_class_metrics_and_seen_counts(tmp_path):
    ns = h.base_namespace(tmp_path)
    ns['set_active_ablation']('production')
    idx = np.arange(8); val = np.arange(12, 16)
    folder = ns['STAGES'] / 'hist'; folder.mkdir(parents=True)
    mapping = {}
    for i in np.r_[idx, val]:
        p = folder / f'{i}.pkl'; ns['atomic_pickle'](h.synthetic_view(ns['rows']['clip'].iloc[i], 2, 4), p); mapping[int(i)] = p
    with h.cpu_training_patches():
        ns['fit_video'](idx, mapping, folder, 42, validation_indices=val)
    hist = pd.read_csv(folder / 'loss_history_seed42.csv')
    seen = hist[[f'seen_kick_direction_{k}' for k in range(3)]].sum(axis=1)
    assert (seen == len(idx)).all()
    pred = hist[[f'validation_kick_direction_predicted_{k}' for k in range(3)]].sum(axis=1)
    assert (pred == len(val)).all() and 'validation_goalkeeper_direction_recall_1' in hist


def test_class_health_alerts_and_role_crosstab_arithmetic():
    ns = dict(__name__='__ch__', RUN_MODE='reports', Path=Path)
    h.run_cell('constants_and_summary_helpers', ns)
    errors = pd.DataFrame(dict(component='ensemble', fold=[0, 0, 1, 1, 1],
        kick_direction_true=['left', 'middle', 'right', 'middle', 'left'],
        kick_direction_predicted=['left', 'left', 'left', 'right', 'left'],
        goalkeeper_direction_true=['left', 'right', 'right', 'left', 'stands in the middle'],
        goalkeeper_direction_predicted=['left', 'right', 'right', 'left', 'left']))
    table = ns['class_health_table'](errors)
    pooled = table[(table.fold == 'pooled') & (table.target == 'kick_direction')].set_index('class_label')
    assert pooled.loc['middle', 'support'] == 2 and pooled.loc['middle', 'predicted_support'] == 0
    assert np.isnan(pooled.loc['middle', 'precision']) and pooled.loc['middle', 'recall'] == 0
    assert pooled.loc['left', 'precision'] == pytest.approx(2 / 4) and pooled.loc['__macro__', 'recall'] == pytest.approx((1 + 0 + 0) / 3)
    texts = [a['text'] for a in ns['class_health_alerts'](table)]
    assert any('"middle" recall 0/2' in t and 'never predicted' in t for t in texts)
    assert any('"stands in the middle" recall 0/1' in t for t in texts)
    audits = pd.DataFrame(dict(clip=['a', 'b', 'c', 'a'], role=['kicker', 'kicker', 'kicker', 'keeper'], fold=[0, 0, 1, 0],
        failure_category=['click_inside_selected_box_proxy', 'no_click_containing_detection_proxy',
                          'click_inside_selected_box_proxy', 'role_missing_with_candidate']))
    frames = pd.DataFrame(dict(clip=['a', 'a', 'c', 'c'], role='kicker', selected=[True, False, True, True]))
    quality = pd.DataFrame(dict(clip=['a', 'b', 'c', 'd'], fold=[0, 0, 1, 1]))
    cross = ns['role_failure_crosstab'](audits, frames, quality).set_index(['role', 'fold'])
    assert cross.loc[('kicker', 0), 'selected_at_audit_but_gaps_elsewhere'] == 1
    assert cross.loc[('kicker', 1), 'click_inside_selected_box_proxy'] == 1
    assert cross.loc[('kicker', 1), 'no_annotation'] == 1 and cross.loc[('kicker', 0), 'annotated_clips'] == 2
