"""Pipeline diagnosis summary: synthetic failure cases built with the notebook's own report exporters."""
import json, subprocess, sys, textwrap
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent))
from nb_harness import CLASSES, TARGETS, cell_source, run_cell, NOTEBOOK


@pytest.fixture(scope='module')
def ns():
    namespace = dict(__name__='__summary_test__', RUN_MODE='reports', Path=Path, SCRATCH_ROOT=Path('/tmp/penalty_test_scratch'))
    run_cell('diagnostics_helpers', namespace)      # ComponentOOF, compare_ablation_metrics (real exporters)
    run_cell('constants_and_summary_helpers', namespace)
    return namespace


def make_run(ns, root, *, n=60, folds=3, harm=False, bad_tracking=False, ablation=None, seed=0):
    """ablation: None | 'negative' | 'positive'. Returns (run, truth dict)."""
    rng = np.random.default_rng(seed)
    run = Path(root); run.mkdir(parents=True, exist_ok=True)
    sources = [f's{i:02d}' for i in range(folds * 4)]
    frame = pd.DataFrame(dict(clip=[f'c{i:03d}.mp4' for i in range(n)], source_id=[sources[i % len(sources)] for i in range(n)],
                              gender=rng.choice(['F', 'M'], n)))
    y = np.column_stack([rng.integers(0, 3, n), rng.integers(0, 3, n)])
    fold_ids = np.array([sources.index(s) % folds for s in frame.source_id])
    plan = dict(split_id='split123', identity=dict(clips=frame['clip'].tolist()),
                folds=[dict(fold=f, validation=np.flatnonzero(fold_ids == f).tolist(),
                            training=np.flatnonzero(fold_ids != f).tolist()) for f in range(folds)])
    config = dict(dataset='data1', outer_folds=folds, metric_weights=[.7, .3], epochs=6, ablation='production',
                  evaluation_mode='nested_cv', split_id='split123', video='MCG-NJU/videomae-base-finetuned-kinetics')
    (run / 'experiment_manifest.json').write_text(json.dumps(dict(config=config, split_plan=plan,
        classes=dict(zip(TARGETS, CLASSES)), seeds=[42, 137])))
    frame[['clip', 'source_id']].assign(fold=fold_ids).to_csv(run / 'folds.csv', index=False)
    # Reference correct on 70% of clips for each target.
    correct = rng.random((n, 2)) < .7
    def probs(ok):
        out = []
        for j in range(2):
            pred = np.where(ok[:, j], y[:, j], (y[:, j] + 1) % 3)
            out.append(np.eye(3)[pred] * .7 + .1)
        return out
    reference = probs(correct)
    ens_ok = correct.copy()
    if harm:
        flip = np.flatnonzero(correct[:, 0])[:12]; ens_ok[flip, 0] = False   # 12 correct -> wrong
        fix = np.flatnonzero(~correct[:, 0])[:3]; ens_ok[fix, 0] = True      # 3 wrong -> correct
    ensemble = probs(ens_ok)
    components = dict(video_seed42=reference, video_ensemble=reference, class_prior=[np.full((n, 3), 1 / 3)] * 2,
                      pose_motion_tree=probs(rng.random((n, 2)) < .5), ensemble=ensemble)
    registry = ns['ComponentOOF'](frame, y, fold_ids, TARGETS, CLASSES, [.7, .3])
    for name, p in components.items():
        registry.add(name, np.arange(n), p, frame['clip'], CLASSES)
    ens_wrong = ~ens_ok
    if bad_tracking:
        low = (ens_wrong[:, 0] & (rng.random(n) < .9)) | (rng.random(n) < .1)
    else:
        low = rng.random(n) < .3
    quality = pd.DataFrame(dict(clip=frame['clip'], source_id=frame.source_id, fold=fold_ids,
                                kicker_tracked_fraction=np.where(low, .3, 1.), kicker_tracking_quality=np.where(low, 'partial', 'mostly_tracked'),
                                keeper_tracked_fraction=1., keeper_tracking_quality='mostly_tracked', camera_valid_fraction=1.))
    quality.to_csv(run / 'tracking_quality.csv', index=False)
    references = {f: ['video_ensemble', 'video_ensemble'] for f in range(folds)}
    exported = registry.export(run / 'component_reports', quality=quality.drop(columns=['source_id', 'fold']), references=references)
    (run / 'fold_results.json').write_text(json.dumps([dict(fold=f, seeds=[42, 137], stage_folder=f'stages/fold{f}_outer',
        ensemble_components=['video_ensemble', 'pose_motion_tree', 'class_prior'], weights=[[.5, .25, .25], [.5, .25, .25]],
        selected_epochs={'42': 3, '137': 4}) for f in range(folds)]))
    truth = dict(n=n, harmed=int((correct[:, 0] & ~ens_ok[:, 0]).sum()), corrected=int((~correct[:, 0] & ens_ok[:, 0]).sum()),
                 kick_wrong=int(ens_wrong[:, 0].sum()), keeper_wrong=int(ens_wrong[:, 1].sum()), low=low,
                 weighted_error=.7 * ens_wrong[:, 0] + .3 * ens_wrong[:, 1], fold_ids=fold_ids)
    if ablation:
        results = {}
        for variant, shift in [('F_raw_motion', 0), ('G_compensated_motion', -1 if ablation == 'negative' else 1)]:
            reg = ns['ComponentOOF'](frame, y, fold_ids, TARGETS, CLASSES, [.7, .3])
            ok = correct.copy()
            idx = np.flatnonzero(correct[:, 0] if shift < 0 else ~correct[:, 0])[:9]
            if shift: ok[idx, 0] = shift > 0
            reg.add('ensemble', np.arange(n), probs(ok), frame['clip'], CLASSES)
            results[variant] = dict(metrics=reg.metric_table())
            vdir = run / 'ablations' / variant; vdir.mkdir(parents=True, exist_ok=True)
            (vdir / 'experiment_manifest.json').write_text(json.dumps(dict(config=dict(config, ablation=variant),
                split_plan=plan, seeds=[42, 137], classes=dict(zip(TARGETS, CLASSES)))))
        ns['compare_ablation_metrics'](results, run, component='ensemble')
        truth['ablation_delta_pp'] = 100 * (results['G_compensated_motion']['metrics'].query("fold=='pooled'").weighted_accuracy.iloc[0]
                                            - results['F_raw_motion']['metrics'].query("fold=='pooled'").weighted_accuracy.iloc[0])
    return run, truth


def find(summary, kind=None, component=None):
    return [p for p in summary['priority_details'] if (kind is None or p['evidence_type'] == kind)
            and (component is None or component in p['suspected_component'])]


def test_baseline_has_no_injected_findings(ns, tmp_path):
    run, _ = make_run(ns, tmp_path / 'base')
    s = ns['build_pipeline_summary'](run)
    assert s['status'] == 'complete'
    assert not find(s, component='ensemble weight') and not find(s, 'retrained_comparison')
    assert len(s['priorities']) <= 5


def test_harmful_ensemble_counts_units_and_denominator(ns, tmp_path):
    run, t = make_run(ns, tmp_path / 'harm', harm=True)
    s = ns['build_pipeline_summary'](run)
    f = find(s, 'predictive_comparison', 'ensemble weight')
    assert len(f) == 1
    f = f[0]
    assert f['sample_count'] == t['n']
    assert f['effect_value'] == pytest.approx(round(100 * (t['corrected'] - t['harmed']) / t['n'], 3))
    assert f['weighted_impact_pp'] == pytest.approx(100 * .7 * (t['corrected'] - t['harmed']) / t['n'])
    assert 'percentage points' in f['effect_units'] and 'folds harmed > corrected' in f['fold_support']
    assert f"changed {t['harmed']} correct" in f['finding'] and f"corrected {t['corrected']}" in f['finding']
    assert 'Outer labels must not be used' in f['limitations']


def test_bad_tracking_association(ns, tmp_path):
    run, t = make_run(ns, tmp_path / 'track', bad_tracking=True)
    s = ns['build_pipeline_summary'](run)
    f = find(s, 'quality_association', 'kicker')
    assert len(f) == 1
    low = t['low']; expected = 100 * (t['weighted_error'][low].mean() - t['weighted_error'][~low].mean())
    assert f[0]['effect_value'] == pytest.approx(round(expected, 3))
    assert f[0]['sample_count'] == int(low.sum())
    assert 'not a causal' in f[0]['limitations'].lower()


def test_negative_ablation_ranks_as_retrained_evidence(ns, tmp_path):
    run, t = make_run(ns, tmp_path / 'abl', ablation='negative')
    s = ns['build_pipeline_summary'](run)
    f = find(s, 'retrained_comparison')
    assert len(f) == 1 and f[0]['priority'] == 1
    assert f[0]['effect_value'] == pytest.approx(round(t['ablation_delta_pp'], 3)) and t['ablation_delta_pp'] < 0
    assert 'camera' in f[0]['suggested_change'].lower()   # compensated vs raw guidance
    not_eval = {r['comparison'] for r in s['not_evaluated']}
    assert 'G_compensated_motion vs F_raw_motion' not in not_eval and 'E_plus_pose vs D_all_visual' in not_eval


def test_positive_ablation_is_not_a_weakness(ns, tmp_path):
    run, _ = make_run(ns, tmp_path / 'pos', ablation='positive')
    assert not find(ns['build_pipeline_summary'](run), 'retrained_comparison')


def test_missing_ablations_are_not_evaluated_not_zero(ns, tmp_path):
    run, _ = make_run(ns, tmp_path / 'noabl')
    s = ns['build_pipeline_summary'](run)
    assert all(r['status'] == 'not evaluated' for r in s['not_evaluated'])
    assert len([r for r in s['not_evaluated'] if ' vs ' in r['comparison']]) == 8
    assert not find(s, 'retrained_comparison')


def test_target_contributions_use_full_population(ns, tmp_path):
    run, t = make_run(ns, tmp_path / 'contrib')
    c = ns['build_pipeline_summary'](run)['overview']['target_contributions']
    assert c['kick_direction']['weighted_error_contribution'] == pytest.approx(.7 * t['kick_wrong'] / t['n'])
    assert c['goalkeeper_direction']['weighted_error_contribution'] == pytest.approx(.3 * t['keeper_wrong'] / t['n'])


def test_incomplete_folds_suppress_conclusions(ns, tmp_path):
    run, _ = make_run(ns, tmp_path / 'inc', harm=True, bad_tracking=True)
    results = json.loads((run / 'fold_results.json').read_text())
    (run / 'fold_results.json').write_text(json.dumps(results[:2]))
    import shutil; shutil.rmtree(run / 'component_reports')
    summary, text = ns['write_pipeline_summary'](run)
    assert summary['status'] == 'incomplete'
    assert [p['evidence_type'] for p in summary['priorities']] == ['insufficient_evidence']
    assert 'cannot yet be determined' in text and 'fold(s) [2]' in summary['priorities'][0]['next_controlled_test']
    assert any('Missing outer folds [2]' in w for w in summary['warnings'])


@pytest.mark.parametrize('corruption', ['class_order', 'fold_assignment', 'ablation_split', 'duplicate_rows', 'seeds'])
def test_conflicting_reports_are_rejected(ns, tmp_path, corruption):
    run, _ = make_run(ns, tmp_path / corruption, ablation='negative')
    if corruption == 'class_order':
        m = json.loads((run / 'component_reports' / 'component_manifest.json').read_text())
        m['classes'][1] = m['classes'][1][::-1]
        (run / 'component_reports' / 'component_manifest.json').write_text(json.dumps(m))
    elif corruption == 'fold_assignment':
        f = pd.read_csv(run / 'folds.csv'); f.loc[0, 'fold'] = (f.loc[0, 'fold'] + 1) % 3; f.to_csv(run / 'folds.csv', index=False)
    elif corruption == 'ablation_split':
        p = run / 'ablations' / 'F_raw_motion' / 'experiment_manifest.json'
        m = json.loads(p.read_text()); m['split_plan']['split_id'] = 'other'; p.write_text(json.dumps(m))
    elif corruption == 'duplicate_rows':
        p = run / 'component_reports' / 'component_clip_errors.csv'
        e = pd.read_csv(p); pd.concat([e, e.iloc[:1]]).to_csv(p, index=False)
    elif corruption == 'seeds':
        r = json.loads((run / 'fold_results.json').read_text()); r[0]['seeds'] = [1]
        (run / 'fold_results.json').write_text(json.dumps(r))
    with pytest.raises(ns['MixedRunError']):
        ns['load_run_reports'](run)


def test_outputs_written_with_exact_columns(ns, tmp_path):
    run, _ = make_run(ns, tmp_path / 'out', harm=True, ablation='negative')
    summary, text = ns['write_pipeline_summary'](run)
    table = pd.read_csv(run / 'improvement_priorities.csv')
    assert list(table.columns) == ns['PRIORITY_COLUMNS'] and 1 <= len(table) <= 5
    assert set(table.evidence_type) <= set(ns['EVIDENCE_TYPES'])
    data = json.loads((run / 'pipeline_summary.json').read_text())
    assert data['rules']['small_slice_clips'] == 30 and 'ranking_rule' in data['rules']
    assert data['identity']['run_id'] == run.name and data['identity']['split_id'] == 'split123'
    assert text.startswith('# Pipeline diagnosis and next improvements')


def test_inference_only_directory_reports_no_accuracy(ns, tmp_path):
    run = tmp_path / 'inf'; run.mkdir()
    (run / 'inference_run.json').write_text(json.dumps(dict(backbone=dict(name='videomae'), input=dict(count=3))))
    summary, text = ns['write_pipeline_summary'](run)
    assert summary['status'] == 'inference_only' and summary['priorities'] == []
    assert 'No accuracy' in text


def test_reports_only_in_fresh_process_without_torch(ns, tmp_path):
    run, _ = make_run(ns, tmp_path / 'fresh', harm=True)
    script = textwrap.dedent(f'''
        import json, sys
        from pathlib import Path
        nb = json.loads(Path({str(NOTEBOOK)!r}).read_text())
        src = [''.join(c['source']) for c in nb['cells'] if ''.join(c['source']).startswith('# CELL: constants_and_summary_helpers')][0]
        ns = {{}}; exec(src, ns)
        summary, text = ns['write_pipeline_summary'](Path({str(run)!r}))
        heavy = [m for m in ('torch', 'cv2', 'av', 'transformers', 'ultralytics', 'catboost') if m in sys.modules]
        print(json.dumps(dict(status=summary['status'], heavy=heavy, n=len(summary['priorities']))))
    ''')
    out = subprocess.run([sys.executable, '-c', script], capture_output=True, text=True, check=True).stdout
    result = json.loads(out.strip().splitlines()[-1])
    assert result == dict(status='complete', heavy=[], n=result['n']) and result['n'] >= 1
