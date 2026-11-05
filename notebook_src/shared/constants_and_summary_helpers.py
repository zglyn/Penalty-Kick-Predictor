# CELL: constants_and_summary_helpers
# Competition constants (both notebooks, every mode) and the report-only pipeline summary.
import numpy as np
TARGETS = ['kick_direction', 'goalkeeper_direction']
CLASSES = [['left', 'middle', 'right'], ['left', 'stands in the middle', 'right']]
WEIGHTS = np.array([0.70, 0.30], dtype=np.float32)
"""Pipeline diagnosis and next improvements: report-only summary of a saved run directory.

Pure numpy/pandas/json. It never imports torch, decodes videos, loads models or downloads weights.
Structure: load_run_reports (loading + validation) -> analyze_reports (numbers) -> render_summary_markdown
(rendering) -> write_pipeline_summary (files). Each stage is independently testable.
"""
import json, math
from pathlib import Path
import numpy as np
import pandas as pd

SUMMARY_SCHEMA_VERSION = 1
EXPECTED_TARGETS = ['kick_direction', 'goalkeeper_direction']
EXPECTED_CLASSES = [['left', 'middle', 'right'], ['left', 'stands in the middle', 'right']]
EXPECTED_WEIGHTS = [0.7, 0.3]
PRIORITY_COLUMNS = ['priority', 'suspected_component', 'finding', 'evidence_files', 'component_or_comparison',
                    'sample_count', 'fold_support', 'effect_value', 'effect_units', 'evidence_type',
                    'suggested_change', 'next_controlled_test', 'limitations']
EVIDENCE_TYPES = ('retrained_comparison', 'predictive_comparison', 'quality_association',
                  'visual_audit_needed', 'insufficient_evidence')
SUMMARY_RULES = dict(
    small_slice_clips=30,
    max_priorities=5,
    detector_unmatched_rate_flag=0.20,
    selector_failure_rate_flag=0.20,
    low_tracking_definition="{role}_tracking_quality != 'mostly_tracked' (tracked fraction < 0.8 in the exporter)",
    low_camera_definition='camera_valid_fraction < 0.5',
    overfit_min_validation_rise=0.02,
    target_dominance_ratio=2.0,
    association_requires_fold_majority=True,
    evidence_strength=dict(retrained_comparison=1.0, predictive_comparison=0.8, quality_association=0.5,
                           visual_audit_needed=0.4, insufficient_evidence=0.0),
    ranking_rule=('Investigation priority, not a causal ranking. Findings with a quantified weighted-score effect '
                  '(percentage points of the 70/30 score) rank first by |effect| x evidence strength x fold-support '
                  'fraction, halved when the supporting sample is below small_slice_clips. Unquantified findings '
                  'follow, ordered by evidence strength then sample count. At most max_priorities rows.'),
    interpretation=('Outer-fold diagnosis is exploratory development feedback, not an untouched estimate. A '
                    'recommendation is a proposed future experiment; do not tune against the same outer labels '
                    'and report the result as untouched.'),
)


class MixedRunError(ValueError):
    """Report files disagree about the run they belong to."""


# ----------------------------------------------------------------------------- loading and validation
def _read_csv(path):
    if not Path(path).exists(): return None
    try: return pd.read_csv(path)
    except pd.errors.EmptyDataError: return pd.DataFrame()  # exporter writes header-less files for empty audits


def _read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8')) if Path(path).exists() else None


def _fold_key(value):
    text = str(value)
    return 'pooled' if text == 'pooled' else str(int(float(text)))


def load_run_reports(run_dir):
    """Load every known report file from an explicit run directory and validate cross-file consistency.

    Returns a dict bundle: data frames/json, coverage table, warnings, status. Raises MixedRunError on
    conflicting run identity, class order, fold assignment or duplicated clip/component rows.
    """
    run = Path(run_dir)
    if not run.is_dir():
        raise FileNotFoundError(f'Run directory not found: {run}')
    files = {
        'experiment_manifest': run / 'experiment_manifest.json', 'folds': run / 'folds.csv',
        'fold_results': run / 'fold_results.json',
        'component_metrics': run / 'component_reports' / 'component_metrics.csv',
        'component_manifest': run / 'component_reports' / 'component_manifest.json',
        'ensemble_comparison': run / 'component_reports' / 'ensemble_comparison.csv',
        'component_clip_errors': run / 'component_reports' / 'component_clip_errors.csv',
        'component_slices': run / 'component_reports' / 'component_slices.csv',
        'component_confusions': run / 'component_reports' / 'component_confusions.csv',
        'tracking_quality': run / 'tracking_quality.csv', 'role_click_audit': run / 'role_click_audit.csv',
        'role_click_summary': run / 'role_click_summary.csv',
        'ablation_comparisons': run / 'ablation_comparisons.csv',
        'inference_manifest': run / 'inference_manifest.json',
        'inference_run': run / 'inference_run.json',
        'holdout_metrics': run / 'holdout_metrics.csv', 'holdout_loss_history': run / 'loss_history.csv',
        'development_manifest': run / 'development_manifest.json',
    }
    bundle = dict(run_dir=str(run), files={k: str(v.relative_to(run)) for k, v in files.items()}, warnings=[])
    loaders = {k: (_read_json if str(v).endswith('.json') else _read_csv) for k, v in files.items()}
    for key, path in files.items(): bundle[key] = loaders[key](path)
    bundle['coverage'] = [dict(report=key, file=bundle['files'][key],
                               status='present' if Path(path).exists() else 'missing') for key, path in files.items()]
    bundle['loss_histories'] = _load_loss_histories(run)
    bundle['ablation_manifests'] = {p.parent.name: _read_json(p)
                                    for p in sorted((run / 'ablations').glob('*/experiment_manifest.json'))}
    _validate(bundle)
    return bundle


def _load_loss_histories(run):
    frames = []
    for path in sorted((run / 'stages').glob('*/loss_history_seed*.csv')):
        frame = pd.read_csv(path)
        seed = path.stem.replace('loss_history_seed', '')
        frames.append(frame.assign(stage=path.parent.name, seed=seed, file=str(path.relative_to(run))))
    return pd.concat(frames, ignore_index=True) if frames else None


def _validate(bundle):
    warn = bundle['warnings'].append
    manifest = bundle['experiment_manifest']
    if bundle['inference_run'] is not None and manifest is None:
        bundle['status'] = 'inference_only'
        warn('Inference-only output: there are no labels, so no accuracy or loss can be reported.')
        return
    if manifest is None and bundle.get('development_manifest') is not None:
        bundle['status'] = 'development'
        warn('Development preset: bounded subset, one seed, few epochs. Inner-validation results only; this is '
             'not complete OOF performance and must not be compared with full nested tables.')
        return
    if manifest is None:
        bundle['status'] = 'holdout' if bundle['holdout_metrics'] is not None else 'empty'
        if bundle['status'] == 'empty':
            warn('No experiment_manifest.json: this is not a nested-diagnostics run directory.')
        return
    classes = manifest.get('classes', {})
    if list(classes) != EXPECTED_TARGETS or [list(v) for v in classes.values()] != EXPECTED_CLASSES:
        raise MixedRunError(f'Unexpected target/class order in experiment_manifest.json: {classes}')
    config = manifest.get('config', {})
    weights = config.get('metric_weights')
    if weights is not None and not np.allclose(weights, EXPECTED_WEIGHTS):
        raise MixedRunError(f'Metric weights {weights} are not the 70/30 competition weights')
    plan = manifest.get('split_plan', {})
    expected_folds = int(config.get('outer_folds', len(plan.get('folds', [])) or 0))
    bundle['expected_folds'] = expected_folds
    bundle['split_id'] = plan.get('split_id', config.get('split_id'))
    component_manifest = bundle['component_manifest']
    if component_manifest is not None:
        if component_manifest.get('targets') != EXPECTED_TARGETS or component_manifest.get('classes') != EXPECTED_CLASSES:
            raise MixedRunError('component_manifest.json target/class order differs from the experiment manifest')
    # Fold identities ---------------------------------------------------------------------------------
    folds = bundle['folds']; fold_map = None
    if folds is not None:
        if not {'clip', 'source_id', 'fold'} <= set(folds) or not folds['clip'].is_unique:
            raise MixedRunError('folds.csv must have unique clip rows with source_id and fold')
        if (folds.groupby('source_id').fold.nunique() > 1).any():
            raise MixedRunError('folds.csv places one source in multiple outer folds')
        identity_clips = plan.get('identity', {}).get('clips')
        if identity_clips is not None and list(folds['clip']) != list(identity_clips):
            raise MixedRunError('folds.csv clip order differs from the saved split identity')
        if plan.get('folds'):
            for item in plan['folds']:
                saved = set(folds['clip'].iloc[item['validation']])
                if saved != set(folds.loc[folds.fold == item['fold'], 'clip']):
                    raise MixedRunError(f'folds.csv fold {item["fold"]} differs from the saved split plan')
        fold_map = folds.set_index('clip').fold
    bundle['fold_map'] = fold_map
    results = bundle['fold_results']
    if results is None:  # fold jobs still running: read each completed fold's own result
        run = Path(bundle['run_dir'])
        results = [json.loads((d / 'result.json').read_text()) for d in sorted(run.glob('folds/fold*'))
                   if (d / 'COMPLETE').exists()]
    done = sorted({int(r['fold']) for r in results})
    if len(done) != len(results): raise MixedRunError('fold_results.json repeats a fold')
    for r in results:
        if bundle['split_id'] is not None and r.get('seeds') is not None and list(r['seeds']) != list(manifest.get('seeds', r['seeds'])):
            raise MixedRunError(f'Fold {r["fold"]} used seeds {r["seeds"]}, manifest lists {manifest.get("seeds")}')
    bundle['completed_folds'] = done
    missing = sorted(set(range(expected_folds)) - set(done))
    bundle['missing_folds'] = missing
    complete = not missing and bundle['component_clip_errors'] is not None and bundle['component_metrics'] is not None
    bundle['status'] = 'complete' if complete else 'incomplete'
    if missing: warn(f'Missing outer folds {missing}: complete-OOF conclusions are suppressed.')
    elif not complete: warn('All folds are recorded but pooled component reports are missing; rerun the export.')
    # Clip/component reports --------------------------------------------------------------------------
    errors = bundle['component_clip_errors']
    if errors is not None:
        if errors.duplicated(['component', 'clip']).any():
            raise MixedRunError('component_clip_errors.csv has duplicated clip/component rows')
        if fold_map is not None:
            if set(errors['clip']) - set(fold_map.index):
                raise MixedRunError('component_clip_errors.csv contains clips absent from folds.csv')
            if not (errors['fold'].to_numpy() == fold_map.loc[errors['clip']].to_numpy()).all():
                raise MixedRunError('component_clip_errors.csv fold assignment differs from folds.csv')
            for name, part in errors.groupby('component'):
                if len(part) != len(fold_map):
                    bundle['status'] = 'incomplete'
                    warn(f'Component {name} covers {len(part)}/{len(fold_map)} clips; pooled conclusions suppressed.')
        if component_manifest is not None and set(component_manifest.get('components', [])) != set(errors['component']):
            raise MixedRunError('component_clip_errors.csv components differ from component_manifest.json')
        if component_manifest is not None and fold_map is not None and component_manifest.get('clips') != list(fold_map.index):
            raise MixedRunError('component_manifest.json clip order differs from folds.csv')
    metrics = bundle['component_metrics']
    if metrics is not None:
        metrics['fold'] = metrics['fold'].map(_fold_key)
        if errors is not None and set(metrics['component']) != set(errors['component']):
            raise MixedRunError('component_metrics.csv components differ from component_clip_errors.csv')
        if metrics.duplicated(['component', 'fold']).any():
            raise MixedRunError('component_metrics.csv repeats a component/fold row')
        pooled = metrics[metrics.fold == 'pooled']
        if fold_map is not None and not (pooled['clips'] == len(fold_map)).all():
            raise MixedRunError('Pooled component metrics do not use the full evaluated population')
    for key in ('ensemble_comparison',):
        frame = bundle[key]
        if frame is not None: frame['fold'] = frame['fold'].map(_fold_key)
    quality = bundle['tracking_quality']
    if quality is not None and fold_map is not None:
        if not quality['clip'].is_unique or set(quality['clip']) - set(fold_map.index):
            raise MixedRunError('tracking_quality.csv clips are duplicated or absent from folds.csv')
        if 'fold' in quality and not (quality['fold'].to_numpy() == fold_map.loc[quality['clip']].to_numpy()).all():
            raise MixedRunError('tracking_quality.csv fold assignment differs from folds.csv')
    # Ablations: same folds/seeds/backbone; feature groups differ by design ----------------------------
    reference_backbone = config.get('backbone', 'videomae (legacy default)')
    for variant, vm in bundle['ablation_manifests'].items():
        vc = vm.get('config', {})
        if vm.get('split_plan', {}).get('split_id') != bundle['split_id']:
            raise MixedRunError(f'Ablation {variant} uses a different saved split')
        if vm.get('seeds') != manifest.get('seeds'):
            raise MixedRunError(f'Ablation {variant} uses different seeds')
        if vc.get('backbone', 'videomae (legacy default)') != reference_backbone or vc.get('epochs') != config.get('epochs'):
            raise MixedRunError(f'Ablation {variant} uses a different backbone/epoch policy')
    comparisons = bundle['ablation_comparisons']
    if comparisons is not None:
        comparisons['fold'] = comparisons['fold'].map(_fold_key)
        for variant in set(comparisons['variant']) | set(comparisons['reference']):
            if variant != 'production' and variant not in bundle['ablation_manifests']:
                warn(f'Ablation comparison references {variant} but ablations/{variant}/experiment_manifest.json is missing.')


# ----------------------------------------------------------------------------- numerical analysis
def _finding(**values):
    row = {k: values.get(k) for k in PRIORITY_COLUMNS if k != 'priority'}
    row['weighted_impact_pp'] = values.get('weighted_impact_pp')
    row['small_sample'] = bool(values.get('small_sample', False))
    row['fold_support_fraction'] = values.get('fold_support_fraction')
    assert row['evidence_type'] in EVIDENCE_TYPES
    return row


def target_contributions(errors, component='ensemble'):
    part = errors[errors.component == component]
    n = len(part)
    out = {}
    for target, weight in zip(EXPECTED_TARGETS, EXPECTED_WEIGHTS):
        wrong = int((~part[target + '_correct'].astype(bool)).sum())
        out[target] = dict(wrong=wrong, clips=n, accuracy=1 - wrong / n,
                           weighted_error_contribution=weight * wrong / n)
    return out


def _per_fold_support(frame, value_column, predicate):
    rows = frame[frame.fold != 'pooled']
    hits = int(predicate(rows[value_column]).sum())
    return hits, len(rows), rows[['fold', value_column]].to_dict('records')


def analyze_reports(bundle, rules=None):
    rules = dict(SUMMARY_RULES, **(rules or {}))
    findings = []; overview = dict(status=bundle['status'])
    status = bundle['status']
    if status in ('inference_only', 'empty'):
        return dict(overview=overview, findings=[], not_evaluated=[], rules=rules)
    complete = status == 'complete'
    errors = bundle.get('component_clip_errors'); metrics = bundle.get('component_metrics')
    n = len(bundle['fold_map']) if bundle.get('fold_map') is not None else None
    if complete:
        pooled = metrics[metrics.fold == 'pooled'].set_index('component')
        overview['components'] = {c: dict(weighted_accuracy_pct=100 * r.weighted_accuracy,
            kick_accuracy_pct=100 * r.kick_direction_accuracy, keeper_accuracy_pct=100 * r.goalkeeper_direction_accuracy,
            weighted_cross_entropy=r.weighted_cross_entropy) for c, r in pooled.iterrows()}
        if 'ensemble' in set(errors.component):
            contrib = target_contributions(errors)
            overview['target_contributions'] = contrib
            findings += _target_dominance(bundle, contrib, rules)
        findings += _ensemble_harm(bundle, rules, n)
        findings += _numeric_specialist(bundle, rules, n)
        findings += _tracking_association(bundle, rules, n)
    findings += _click_audit(bundle, rules)   # audit proxies; valid per evaluated fold
    findings += _ablations(bundle, rules, complete)
    findings += _overfitting(bundle, rules)
    not_evaluated = _not_evaluated(bundle)
    return dict(overview=overview, findings=findings, not_evaluated=not_evaluated, rules=rules)


def _target_dominance(bundle, contrib, rules):
    k, g = (contrib[t]['weighted_error_contribution'] for t in EXPECTED_TARGETS)
    if min(k, g) <= 0 or max(k, g) / min(k, g) < rules['target_dominance_ratio']:
        return []
    target = EXPECTED_TARGETS[int(g > k)]
    conf = bundle.get('component_confusions')
    worst = ''
    if conf is not None:
        off = conf[(conf.component == 'ensemble') & (conf.target == target) & (conf.true_label != conf.predicted_label)]
        if len(off):
            top = off.sort_values('clips', ascending=False).iloc[0]
            worst = f' Largest confusion: true {top.true_label!r} predicted {top.predicted_label!r} ({int(top.clips)} clips).'
    c = contrib[target]
    return [_finding(suspected_component=f'{target} head / inputs', evidence_type='predictive_comparison',
        finding=(f'{target} accounts for {100 * c["weighted_error_contribution"]:.1f} pp of weighted score loss vs '
                 f'{100 * min(k, g):.1f} pp for the other target ({c["wrong"]} wrong of {c["clips"]}).' + worst),
        evidence_files='component_reports/component_clip_errors.csv; component_reports/component_confusions.csv',
        component_or_comparison='ensemble: kick vs keeper weighted error contribution (denominator = all evaluated clips)',
        sample_count=c['clips'], fold_support='pooled only', effect_value=round(100 * c['weighted_error_contribution'], 3),
        effect_units='percentage points of weighted score lost (0.70/0.30 x wrong / N)',
        suggested_change=f'Inspect {target} class confusions and the crops/features most relevant to that target.',
        next_controlled_test='Change one input/head element for this target and compare on the same saved folds with inner-only selection.',
        limitations='Descriptive measurement; does not identify which stage causes the errors.')]


def _ensemble_harm(bundle, rules, n):
    comp = bundle.get('ensemble_comparison')
    if comp is None: return []
    out = []
    for target, weight in zip(EXPECTED_TARGETS, EXPECTED_WEIGHTS):
        rows = comp[comp.target == target]; pooled = rows[rows.fold == 'pooled']
        if pooled.empty: continue
        p = pooled.iloc[0]
        if p.correct_changed_to_mistake <= p.mistakes_corrected: continue
        folds = rows[rows.fold != 'pooled']
        hits = int((folds.correct_changed_to_mistake > folds.mistakes_corrected).sum())
        detail = ', '.join(f'fold {r.fold}: {100 * r.accuracy_delta:+.1f} pp' for r in folds.itertuples())
        out.append(_finding(suspected_component='ensemble weight selection', evidence_type='predictive_comparison',
            finding=(f'For {target}, the frozen ensemble changed {int(p.correct_changed_to_mistake)} correct reference '
                     f'predictions into mistakes and corrected {int(p.mistakes_corrected)} '
                     f'({100 * p.accuracy_delta:+.2f} pp target accuracy; per fold {detail}).'),
            evidence_files='component_reports/ensemble_comparison.csv; fold_results.json',
            component_or_comparison=f'ensemble vs fold-specific inner-selected reference ({target})',
            sample_count=int(p.clips), fold_support=f'{hits}/{len(folds)} folds harmed > corrected',
            fold_support_fraction=hits / max(len(folds), 1), effect_value=round(100 * p.accuracy_delta, 3),
            effect_units=f'percentage points of {target} accuracy', weighted_impact_pp=100 * p.weighted_score_delta,
            suggested_change='Investigate inner-validation weight selection (grid granularity, tie-breaks) or a simpler mixture.',
            next_controlled_test='Compare current mixture with reference-only and equal-weight mixtures selected on inner validation, same saved folds.',
            limitations='Outer labels must not be used to choose the replacement; inner validation sets are small.'))
    return out


def _numeric_specialist(bundle, rules, n):
    metrics = bundle['component_metrics']
    names = set(metrics.component)
    if not {'pose_motion_tree', 'class_prior'} <= names: return []
    out = []
    weights = {}
    for r in bundle.get('fold_results') or []:
        if 'pose_motion_tree' in r.get('ensemble_components', []):
            k = r['ensemble_components'].index('pose_motion_tree')
            for j, t in enumerate(EXPECTED_TARGETS): weights.setdefault(t, []).append(r['weights'][j][k])
    for target in EXPECTED_TARGETS:
        col = target + '_accuracy'
        tree = metrics[metrics.component == 'pose_motion_tree'].set_index('fold')[col]
        prior = metrics[metrics.component == 'class_prior'].set_index('fold')[col]
        delta = (tree - prior).dropna()
        if delta.get('pooled', 0) > 0: continue
        folds = delta.drop('pooled', errors='ignore')
        hits = int((folds <= 0).sum())
        w = weights.get(target)
        out.append(_finding(suspected_component='numeric pose/motion CatBoost specialist', evidence_type='predictive_comparison',
            finding=(f'Standalone {target} accuracy of the numeric specialist is {100 * delta["pooled"]:+.2f} pp vs the '
                     f'training-only class prior' + (f'; mean inner-selected ensemble weight {np.mean(w):.2f}.' if w else '.')),
            evidence_files='component_reports/component_metrics.csv; fold_results.json',
            component_or_comparison=f'pose_motion_tree vs class_prior ({target})', sample_count=n,
            fold_support=f'{hits}/{len(folds)} folds tree <= prior', fold_support_fraction=hits / max(len(folds), 1),
            effect_value=round(100 * delta['pooled'], 3), effect_units=f'percentage points of {target} accuracy',
            suggested_change='Inspect feature scale, missingness, dimensionality and overfitting of the numeric specialist.',
            next_controlled_test='Retrain without the specialist (or with a reduced feature group) on the same folds; check whether it still adds ensemble diversity.',
            limitations='Weak standalone accuracy does not show it is harmful inside the ensemble.'))
    return out


def _group_gap(errors, mask, rules, n):
    low = errors[mask]; high = errors[~mask]
    if not len(low) or not len(high): return None
    gap = low.weighted_error.mean() - high.weighted_error.mean()
    folds = []
    for fold, part in errors.groupby('fold'):
        m = mask.loc[part.index]
        if m.any() and (~m).any():
            folds.append(dict(fold=int(fold), low_minus_high_pp=100 * (part[m].weighted_error.mean() - part[~m].weighted_error.mean()),
                              low_clips=int(m.sum())))
    hits = sum(f['low_minus_high_pp'] > 0 for f in folds)
    excess = (gap * len(low)) / n  # error mass attributable if the low group matched the other group (associational)
    return dict(low=len(low), gap=gap, folds=folds, hits=hits, excess=excess)


def _tracking_association(bundle, rules, n):
    errors = bundle['component_clip_errors']
    part = errors[errors.component == 'ensemble'].reset_index(drop=True)
    out = []
    checks = []
    for role in ('kicker', 'keeper'):
        col = role + '_tracking_quality'
        if col in part: checks.append((f'{role} tracking', part[col].astype(str) != 'mostly_tracked',
                                       f'{role} crop coverage / role tracking', col))
    if 'camera_valid_fraction' in part:
        checks.append(('camera estimate', part['camera_valid_fraction'] < .5, 'camera-motion compensation', 'camera_valid_fraction'))
    for label, mask, component, col in checks:
        g = _group_gap(part, mask, rules, n)
        if g is None or g['gap'] <= 0: continue
        if rules['association_requires_fold_majority'] and not g['hits'] > len(g['folds']) / 2:
            continue  # sign not consistent across a majority of folds: not reported as a priority
        small = g['low'] < rules['small_slice_clips']
        per = ', '.join(f"fold {f['fold']}: {f['low_minus_high_pp']:+.1f} pp (n={f['low_clips']})" for f in g['folds'])
        out.append(_finding(suspected_component=component, evidence_type='quality_association',
            finding=(f'Clips with low {label} have {100 * g["gap"]:.1f} pp higher weighted error rate '
                     f'({g["low"]} low-quality clips{"; SMALL slice" if small else ""}). Per fold: {per}.'),
            evidence_files='component_reports/component_clip_errors.csv; tracking_quality.csv',
            component_or_comparison=f'ensemble errors split by {col}', sample_count=g['low'],
            fold_support=f'{g["hits"]}/{len(g["folds"])} folds with both groups show higher error',
            fold_support_fraction=g['hits'] / max(len(g['folds']), 1), small_sample=small,
            effect_value=round(100 * g['gap'], 3), effect_units='percentage points of weighted error rate (low minus other clips)',
            weighted_impact_pp=100 * g['excess'],
            suggested_change='Review fallback crops and sampled visual audits for these clips; then retrain a targeted change.',
            next_controlled_test='Retrain a targeted tracking/camera change and compare on the same saved folds with inner-only selection.',
            limitations='Association only: hard clips may also be hard to track. Not a causal component effect.'))
    return out


def _click_audit(bundle, rules):
    summary = bundle.get('role_click_summary'); audits = bundle.get('role_click_audit')
    out = []
    if summary is not None and len(summary) and 'role' in summary:
        for r in summary.itertuples():
            rate = getattr(r, 'unmatched_detection_proxy_rate', np.nan)
            if pd.notna(rate) and rate >= rules['detector_unmatched_rate_flag']:
                out.append(_finding(suspected_component=f'person detector ({r.role})', evidence_type='visual_audit_needed',
                    finding=(f'{100 * rate:.1f}% of timestamp-eligible {r.role} clicks have no detection containing the '
                             f'click at the nearest sampled frame ({int(r.timestamp_eligible_clips)} eligible clips).'),
                    evidence_files='role_click_summary.csv; role_click_audit.csv',
                    component_or_comparison=f'{r.role} click-containment detector proxy',
                    sample_count=int(r.timestamp_eligible_clips), fold_support='pooled audit rows',
                    small_sample=r.timestamp_eligible_clips < rules['small_slice_clips'],
                    effect_value=round(100 * rate, 3), effect_units='percent of timestamp-eligible annotated clips',
                    suggested_change='Inspect detector coverage at eligible timestamps (confidence threshold, image size).',
                    next_controlled_test='Re-run detection with a changed detector setting on a sampled audit set, then retrain on the same folds.',
                    limitations='Click containment is a localization proxy, not identity ground truth.'))
    if audits is not None and len(audits) and 'failure_category' in audits:
        eligible = audits[audits.timestamp_eligible.astype(bool)]
        for role, part in eligible.groupby('role'):
            bad = part.failure_category.isin(['role_missing_with_candidate', 'selected_other_box_proxy'])
            rate = float(bad.mean()) if len(part) else np.nan
            if pd.notna(rate) and rate >= rules['selector_failure_rate_flag']:
                folds = part.assign(bad=bad).groupby('fold').bad.mean()
                out.append(_finding(suspected_component=f'role selector / tracker ({role})', evidence_type='visual_audit_needed',
                    finding=(f'{100 * rate:.1f}% of eligible {role} clicks fall inside a candidate box while the selected '
                             f'role is missing or a different box (proxy).'),
                    evidence_files='role_click_audit.csv', component_or_comparison=f'{role} selector failure proxy',
                    sample_count=len(part), fold_support=f'{int((folds >= rules["selector_failure_rate_flag"]).sum())}/{len(folds)} folds at or above flag',
                    small_sample=len(part) < rules['small_slice_clips'],
                    effect_value=round(100 * rate, 3), effect_units='percent of timestamp-eligible annotated clips',
                    suggested_change='Inspect role training matches, selector confidence and track continuity.',
                    next_controlled_test='Change the selector/tracker, regenerate crops, retrain on the same folds.',
                    limitations='Proxy for localization; possible identity switches need visual review, not an automatic rate.'))
    return out


def _ablations(bundle, rules, complete):
    comp = bundle.get('ablation_comparisons')
    if comp is None or comp.empty: return []
    out = []
    for (variant, reference), part in comp.groupby(['variant', 'reference'], sort=False):
        pooled = part[part.fold == 'pooled']
        if pooled.empty: continue
        p = pooled.iloc[0]; folds = part[part.fold != 'pooled']
        delta = p.weighted_accuracy_delta
        if delta >= 0: continue
        hits = int((folds.weighted_accuracy_delta < 0).sum())
        per = ', '.join(f'fold {r.fold}: {100 * r.weighted_accuracy_delta:+.1f} pp' for r in folds.itertuples())
        compensation = variant == 'G_compensated_motion' and reference == 'F_raw_motion'
        out.append(_finding(suspected_component=f'feature/stream change {reference} -> {variant}', evidence_type='retrained_comparison',
            finding=(f'Adding {variant} over {reference} changed pooled weighted accuracy by {100 * delta:+.2f} pp '
                     f'(cross-entropy {p.weighted_cross_entropy_delta:+.4f}; per fold {per}).'),
            evidence_files=f'ablation_comparisons.csv; ablations/{variant}/experiment_manifest.json',
            component_or_comparison=f'{variant} vs {reference} (ensemble, matched folds)', sample_count=int(p.clips),
            fold_support=f'{hits}/{len(folds)} folds negative', fold_support_fraction=hits / max(len(folds), 1),
            effect_value=round(100 * delta, 3), effect_units='percentage points of weighted accuracy (variant minus reference)',
            weighted_impact_pp=100 * delta,
            suggested_change=('Inspect invalid camera estimates and camera quality; keep raw motion as a control.' if compensation
                              else 'Remove or revise that feature group and repeat the controlled comparison.'),
            next_controlled_test='Repeat the matched-fold comparison with a revised feature group and inner-only selection.',
            limitations='Measured under this training policy and seed list; outer-fold reuse makes this exploratory.'))
    return out


def _overfitting(bundle, rules):
    hist = bundle.get('loss_histories')
    if hist is None or not {'train_loss', 'validation_loss'} <= set(hist): return []
    flagged = []; total = 0
    for (stage, seed), h in hist.groupby(['stage', 'seed']):
        h = h.dropna(subset=['train_loss', 'validation_loss']).sort_values('epoch')
        if len(h) < 2: continue
        total += 1
        best = h.loc[h.validation_loss.idxmin()]; last = h.iloc[-1]
        if last.validation_loss - best.validation_loss >= rules['overfit_min_validation_rise'] and last.train_loss < best.train_loss:
            flagged.append(dict(stage=stage, seed=seed, rise=float(last.validation_loss - best.validation_loss),
                                best_epoch=int(best.epoch), last_epoch=int(last.epoch)))
    if not flagged: return []
    rises = [f['rise'] for f in flagged]
    return [_finding(suspected_component='video fine-tuning (capacity/regularization)', evidence_type='predictive_comparison',
        finding=(f'{len(flagged)}/{total} inner-validated fits keep lowering training loss while inner validation loss rises '
                 f'(mean rise {np.mean(rises):.3f}; e.g. {flagged[0]["stage"]} seed {flagged[0]["seed"]}: best epoch '
                 f'{flagged[0]["best_epoch"]} of {flagged[0]["last_epoch"]}).'),
        evidence_files='stages/*/loss_history_seed*.csv', component_or_comparison='eval-mode train vs inner-validation weighted cross-entropy',
        sample_count=len(flagged), fold_support=f'{len(flagged)}/{total} fits', fold_support_fraction=len(flagged) / total,
        effect_value=round(float(np.mean(rises)), 4), effect_units='weighted cross-entropy rise after best epoch (loss, not accuracy)',
        suggested_change='Consider capacity/regularization/augmentation changes selected on inner validation; best-epoch selection already limits impact.',
        next_controlled_test='Compare a regularization change with the same inner splits and seeds; select epochs on inner validation only.',
        limitations='Loss curves; possible overfitting, not a measured accuracy effect.')]


def _not_evaluated(bundle):
    rows = []
    comparisons = bundle.get('ablation_comparisons')
    evaluated = set() if comparisons is None else set(map(tuple, comparisons[['variant', 'reference']].drop_duplicates().to_numpy()))
    pairs = [('B_context_kicker', 'A_context'), ('C_context_keeper', 'A_context'), ('D_all_visual', 'A_context'),
             ('E_plus_pose', 'D_all_visual'), ('F_raw_motion', 'E_plus_pose'), ('G_compensated_motion', 'F_raw_motion'),
             ('H_plus_metadata', 'G_compensated_motion'), ('production', 'H_plus_metadata')]
    for pair in pairs:
        if pair not in evaluated:
            rows.append(dict(comparison=f'{pair[0]} vs {pair[1]}', status='not evaluated',
                             note='No retrained matched-fold comparison exists; this is not a zero effect.'))
    for key, note in [('tracking_quality', 'tracking-quality associations'), ('role_click_audit', 'click-audit proxies')]:
        if bundle.get(key) is None:
            rows.append(dict(comparison=note, status='not evaluated', note=f'{key} report missing.'))
    return rows


# ----------------------------------------------------------------------------- ranking and rendering
def rank_priorities(findings, rules):
    strength = rules['evidence_strength']

    def key(f):
        impact = f.get('weighted_impact_pp')
        support = f.get('fold_support_fraction'); support = 1.0 if support is None else support
        if impact is not None and np.isfinite(impact):
            score = abs(impact) * strength[f['evidence_type']] * support * (0.5 if f['small_sample'] else 1.0)
            return (0, -score, -strength[f['evidence_type']])
        return (1, -strength[f['evidence_type']], -(f.get('sample_count') or 0))
    ranked = sorted(findings, key=key)[:rules['max_priorities']]
    for i, f in enumerate(ranked, 1): f['priority'] = i
    return ranked


def insufficient_evidence_row(bundle):
    if bundle['status'] == 'incomplete':
        missing = bundle.get('missing_folds') or []
        next_step = (f'Finish outer fold(s) {missing} (rerun with RUN_MODE="resume"), then regenerate this summary.'
                     if missing else 'Rerun the pooled report export, then regenerate this summary.')
    elif bundle['status'] == 'empty':
        next_step = ('This folder has no nested-diagnostics reports (for example a QUICK_SMOKE run). Train with '
                     'QUICK_SMOKE=False and EVALUATION_MODE="nested_cv", then regenerate this summary.')
    elif bundle['status'] == 'development':
        next_step = ('Development results screen ideas only. Promote a promising change to a new release and run the '
                     'full nested_cv evaluation (HPC) before drawing performance conclusions.')
    elif bundle['status'] == 'holdout':
        next_step = 'Run EVALUATION_MODE="nested_cv" to obtain component, tracking and ensemble reports.'
    else:
        next_step = ('Run the optional retrained ablations (RUN_ABLATIONS=True) and inspect visual audits; '
                     'no loaded report currently supports a ranked weakness.')
    return dict(priority=1, suspected_component='undetermined',
                finding='The weakest component cannot yet be determined from the loaded reports.',
                evidence_files='; '.join(c['file'] for c in bundle['coverage'] if c['status'] == 'present') or 'none',
                component_or_comparison='n/a', sample_count=None, fold_support='n/a', effect_value=None, effect_units='n/a',
                evidence_type='insufficient_evidence', suggested_change='Collect the missing evidence before changing the pipeline.',
                next_controlled_test=next_step, limitations='No evidence-backed ranking is possible yet.')


def build_pipeline_summary(run_dir, rules=None):
    bundle = load_run_reports(run_dir)
    analysis = analyze_reports(bundle, rules)
    ranked = rank_priorities(analysis['findings'], analysis['rules'])
    if not ranked and bundle['status'] != 'inference_only':
        ranked = [insufficient_evidence_row(bundle)]
    manifest = bundle.get('experiment_manifest') or {}
    config = manifest.get('config', {})
    identity = dict(run_id=Path(run_dir).name, run_dir=str(run_dir), dataset_id=config.get('dataset'),
                    split_id=bundle.get('split_id'), evaluation_mode=config.get('evaluation_mode'),
                    backbone=config.get('backbone', {'name': 'videomae', 'checkpoint': config.get('video'), 'note': 'legacy default'}),
                    training_policy=config.get('training_policy', 'finetune'),
                    feature_variant=config.get('ablation', 'production'),
                    seeds=manifest.get('seeds'), expected_folds=bundle.get('expected_folds'),
                    completed_folds=bundle.get('completed_folds'), missing_folds=bundle.get('missing_folds'))
    if bundle['status'] == 'inference_only':
        info = bundle['inference_run'] or {}
        identity.update(run_id=Path(run_dir).name, backbone=info.get('backbone'), inference_input=info.get('input'),
                        fitted_run=info.get('trained_run_dir'))
    summary = dict(schema_version=SUMMARY_SCHEMA_VERSION, identity=identity, status=bundle['status'],
                   overview=analysis['overview'], priorities=[{k: f.get(k) for k in PRIORITY_COLUMNS} for f in ranked],
                   priority_details=ranked, not_evaluated=analysis['not_evaluated'], coverage=bundle['coverage'],
                   warnings=bundle['warnings'], rules=analysis['rules'])
    return summary


def _fmt(value, digits=2):
    if value is None or (isinstance(value, float) and not math.isfinite(value)): return 'n/a'
    return f'{value:.{digits}f}' if isinstance(value, float) else str(value)


def render_summary_markdown(summary):
    ident = summary['identity']; lines = ['# Pipeline diagnosis and next improvements', '']
    backbone = ident.get('backbone')
    backbone_text = backbone.get('name', backbone) if isinstance(backbone, dict) else backbone
    lines += [f'**Run** `{ident["run_id"]}` | status **{summary["status"]}** | dataset `{ident.get("dataset_id")}` | '
              f'split `{ident.get("split_id")}` | backbone `{backbone_text}` ({ident.get("training_policy")}) | '
              f'feature variant `{ident.get("feature_variant")}` | folds {ident.get("completed_folds")} of {ident.get("expected_folds")}', '']
    if summary['status'] == 'inference_only':
        lines += ['This directory holds label-free predictions. No accuracy, loss or weakness ranking can be computed '
                  'without labels; summarize the fitted training run instead.', '']
    ov = summary['overview']
    if ov.get('components'):
        lines += ['## Overview (pooled outer-fold, exploratory)', '',
                  '| component | weighted acc. % | kick acc. % | keeper acc. % | weighted CE |', '|---|---|---|---|---|']
        for name, r in ov['components'].items():
            lines.append(f'| {name} | {_fmt(r["weighted_accuracy_pct"])} | {_fmt(r["kick_accuracy_pct"])} | '
                         f'{_fmt(r["keeper_accuracy_pct"])} | {_fmt(r["weighted_cross_entropy"], 4)} |')
        lines.append('')
    if ov.get('target_contributions'):
        parts = [f'{t}: {100 * c["weighted_error_contribution"]:.2f} pp ({c["wrong"]}/{c["clips"]} wrong)'
                 for t, c in ov['target_contributions'].items()]
        lines += ['Weighted score lost by target (0.70/0.30 x wrong / all evaluated clips): ' + '; '.join(parts), '']
    lines += ['## Investigation priorities', '', '_' + summary['rules']['ranking_rule'] + '_', '']
    if summary['priorities']:
        cols = PRIORITY_COLUMNS
        lines += ['| ' + ' | '.join(cols) + ' |', '|' + '---|' * len(cols)]
        for row in summary['priorities']:
            lines.append('| ' + ' | '.join(_fmt(row.get(c), 3).replace('|', '/') for c in cols) + ' |')
        lines.append('')
        if summary['priorities'][0]['evidence_type'] == 'insufficient_evidence':
            lines += ['**The weakest component cannot yet be determined.** Next diagnostic: '
                      + summary['priorities'][0]['next_controlled_test'], '']
    if summary['not_evaluated']:
        lines += ['## Not evaluated (not zero effects)', ''] + [f'- {r["comparison"]}: {r["note"]}' for r in summary['not_evaluated']] + ['']
    if summary['warnings']:
        lines += ['## Warnings', ''] + [f'- {w}' for w in summary['warnings']] + ['']
    lines += ['## Report coverage', ''] + [f'- `{c["file"]}`: {c["status"]}' for c in summary['coverage']] + ['']
    lines += ['## Interpretation', '', summary['rules']['interpretation'],
              'Image streams share one head and are not independent classifiers; overlapping slices are not additive; '
              'click containment is not identity correctness; possible tracking switches are visual observations. '
              'Fold counts are transparent heuristics, not significance tests. No competition-score improvement is promised.', '']
    return '\n'.join(lines)


def write_pipeline_summary(run_dir, rules=None, output_dir=None):
    """output_dir defaults to the run folder; reports on someone else's run/bundle should write elsewhere."""
    run = Path(run_dir); out = Path(output_dir) if output_dir is not None else run
    out.mkdir(parents=True, exist_ok=True)
    summary = build_pipeline_summary(run, rules)
    text = render_summary_markdown(summary)
    (out / 'pipeline_summary.md').write_text(text, encoding='utf-8')
    (out / 'pipeline_summary.json').write_text(json.dumps(summary, indent=2, default=str), encoding='utf-8')
    pd.DataFrame(summary['priorities'], columns=PRIORITY_COLUMNS).to_csv(out / 'improvement_priorities.csv', index=False)
    return summary, text
