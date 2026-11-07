# CELL: refinement
"""RUN_MODE='refine': improve a FINISHED nested_cv run without retraining any video model.

Reuses every saved fitted artifact (inner/outer video weights, scalers, CatBoost trees, priors, crops):
  1. Regenerate each fold's inner-validation predictions with the saved INNER models (inference only).
  2. Refit a class-weighted pose/motion CatBoost per stage on that stage's training partition (CPU, minutes);
     the original unweighted trees stay as the control.
  3. Reselect each task's ensemble from a predeclared candidate set on INNER predictions only
     (max inner accuracy, then lower inner cross-entropy, then fewer components), and optionally a
     center-class log-probability offset (adopted only if it strictly raises inner accuracy).
  4. Apply the frozen choices to the saved OUTER predictions (exploratory evaluation) and, with the saved
     outer models, to the private clips (submission). Persist an inference manifest that replays it exactly.
Outputs go to a NEW run folder next to the parent; the parent run is never modified.
"""
REFINE_VERSION = 'refine-v1'
REFINE_COMPONENTS = ['video_ensemble', 'pose_motion_tree', 'pose_motion_tree_weighted', 'class_prior']


def capped_inverse_sqrt_weights(y, n_classes, cap):
    """Per-sample weights ~ count(class)^-1/2 from THIS stage's training labels, mean 1, capped at `cap`."""
    counts = np.bincount(y, minlength=n_classes).astype(float)
    per_class = np.where(counts > 0, 1 / np.sqrt(np.maximum(counts, 1)), 0.)
    w = per_class[y]; w = w / w.mean()
    w = np.minimum(w, cap); return w / w.mean()


def fit_weighted_trees(features, labels, cap):
    trees = []
    for j, c in enumerate(CLASSES):
        tree = CatBoostClassifier(iterations=600, depth=4, learning_rate=.035, l2_leaf_reg=8, loss_function='MultiClass',
                                  verbose=False, random_seed=SEED, thread_count=4)
        tree.fit(features, labels[:, j], sample_weight=capped_inverse_sqrt_weights(labels[:, j], len(c), cap))
        trees.append(tree)
    return trees


def _task_metrics(p, y):
    p = np.asarray(p); acc = float((p.argmax(1) == y).mean())
    ce = float(-np.log(np.clip(p[np.arange(len(y)), y], 1e-12, 1)).mean())
    return acc, ce


def refine_candidates(names):
    """Predeclared candidates: each component alone, equal mixtures, and the original quarter-step grid over
    (video, tree variant, prior) for each tree variant. Returned as (label, {component: weight})."""
    out = [(f'only:{n}', {n: 1.0}) for n in names]
    trees = [n for n in ('pose_motion_tree', 'pose_motion_tree_weighted') if n in names]
    for t in trees:
        if 'video_ensemble' in names:
            out.append((f'equal:video+{t}', {'video_ensemble': .5, t: .5}))
            if 'class_prior' in names:
                out.append((f'equal:video+{t}+prior', {'video_ensemble': 1 / 3, t: 1 / 3, 'class_prior': 1 / 3}))
        base = [n for n in ('video_ensemble', t, 'class_prior') if n in names]
        for x in itertools.product(range(5), repeat=len(base)):
            if sum(x) == 4 and sum(v > 0 for v in x) > 1:
                out.append((f'grid:{t}:' + ','.join(map(str, x)), {n: v / 4 for n, v in zip(base, x) if v}))
    return out


def select_refined_task(components, y, j, offsets_grid):
    """Inner-only selection for task j. Returns (choice dict, candidate table rows)."""
    rows = []; best = None
    for order, (label, weights) in enumerate(refine_candidates([n for n in REFINE_COMPONENTS if n in components])):
        p = sum(w * np.asarray(components[n][j]) for n, w in weights.items())
        acc, ce = _task_metrics(p, y)
        key = (-acc, ce, len(weights), order)
        rows.append(dict(task=TARGETS[j], candidate=label, weights=json.dumps(weights), inner_accuracy=acc, inner_cross_entropy=ce))
        if best is None or key < best[0]: best = (key, label, weights, p, acc, ce)
    _, label, weights, p, acc, ce = best
    offset = 0.0
    if offsets_grid:
        center = 1
        trials = []
        for value in offsets_grid:
            off = [0.0] * len(CLASSES[j]); off[center] = float(value)
            a, c = _task_metrics(apply_class_offsets(p, off), y)
            trials.append((value, a, c))
        top = max(a for _, a, _ in trials)
        if top > acc:   # adopt only a strict inner-accuracy gain; smallest such offset, then lower CE
            offset = min((abs(v), c, v) for v, a, c in trials if a == top)[2]
    off = [0.0] * len(CLASSES[j]); off[1] = float(offset)
    for r in rows: r['selected'] = r['candidate'] == label
    return dict(candidate=label, weights=weights, inner_accuracy=acc, inner_cross_entropy=ce, center_offset=float(offset),
                offsets=off), rows


def _views(folder, indices):
    return {int(i): Path(folder) / f'{int(i)}.pkl' for i in indices}


def _numeric(paths, indices):
    return np.stack([ablation_numeric_features(load_view({0: paths[int(i)]}, 0), metadata[int(i)]) for i in indices])


def _stage_model(folder):
    serial = Path(folder) / 'backbone_serialization.json'
    architecture = json.loads(serial.read_text())['architecture'] if serial.exists() else None
    return MultiViewVideo(len(numeric_feature_manifest()), architecture=architecture)


def predict_saved_stage(folder, seeds, indices, paths):
    """Saved weights in eval mode on saved/label-free crops. No optimizer, no fitting, no selection."""
    folder = Path(folder); indices = [int(i) for i in indices]
    scaler = joblib.load(folder / 'numeric_scaler.joblib')
    numeric = scaler.transform(_numeric(paths, indices)).astype(np.float32)
    out = {}
    for seed in seeds:
        model = _stage_model(folder)
        model.load_state_dict(torch.load(folder / f'video_seed{seed}.pt', map_location='cpu', weights_only=True)['model'], strict=True)
        model.to(DEVICE).eval(); rows = [[] for _ in CLASSES]
        with torch.inference_mode():
            for k, i in enumerate(tqdm(indices, desc=f'{folder.name} seed {seed}', leave=False)):
                video = torch.from_numpy(prepare_video_tensor(load_view({0: paths[i]}, 0, decode_video=True)['video'],
                                                              IMAGE_MEAN, IMAGE_STD))[None]
                with torch.autocast(device_type='cuda' if DEVICE == 'cuda' else 'cpu', dtype=torch.float16,
                                    enabled=DEVICE == 'cuda'):
                    logits = model(video.to(DEVICE), torch.from_numpy(numeric[k:k + 1]).to(DEVICE))
                for j, p in enumerate(logits): rows[j].append(p.float().softmax(1).cpu().numpy())
        out[int(seed)] = [np.concatenate(r) for r in rows]
        del model; gc.collect()
        if torch.cuda.is_available(): torch.cuda.empty_cache()
    return out


def _cached(path, compute):
    path = Path(path)
    if path.exists():
        with np.load(path) as z: return {k: z[k] for k in z.files}
    arrays = compute(); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f'.{path.name}.{os.getpid()}.npz'); np.savez_compressed(tmp, **arrays); tmp.replace(path)
    return arrays


def _ensure_private_views(folder, cache_dir):
    """Private crops from the outer stage; missing ones are made label-free with the stage's SAVED selector."""
    paths = {}; selector = None
    for i in range(len(train), len(rows)):
        p = Path(folder) / f'{i}.pkl'
        if not p.exists():
            p = Path(cache_dir) / f'{i}.pkl'
            if not p.exists():
                if selector is None: selector = joblib.load(Path(folder) / 'roles_all_training.joblib')
                atomic_pickle(make_views(extract_raw(rows['clip'].iloc[i]), selector), p)
        paths[i] = p
    return paths


def refine_settings():
    return dict(version=REFINE_VERSION, weighted_tree=bool(REFINE_TREE_CLASS_WEIGHTS), weight_cap=float(REFINE_WEIGHT_CAP),
                class_offsets=bool(REFINE_CLASS_OFFSETS), center_offsets=list(REFINE_CENTER_OFFSETS) if REFINE_CLASS_OFFSETS else [],
                predict_private=bool(PREDICT_PRIVATE), candidates='single, equal mixtures, quarter grid over (video, tree, prior)',
                selection_rule='per task on inner validation: max accuracy, then lower cross-entropy, then fewer components')


def run_refinement(parent=None):
    parent = Path(parent or TRAINED_RUN_DIR)
    experiment = json.loads((parent / 'experiment_manifest.json').read_text())
    results = json.loads((parent / 'fold_results.json').read_text())
    if experiment['config'].get('evaluation_mode') != 'nested_cv' or len(results) != int(experiment['config']['outer_folds']):
        raise ValueError(f'{parent} is not a finished nested_cv run; refinement needs every fold.')
    plan = experiment['split_plan']; seeds = list(experiment['seeds']); settings = refine_settings()
    if list(plan['identity']['clips']) != train['clip'].tolist():
        raise ValueError('The loaded dataset does not match the run\'s saved split identity.')
    set_active_ablation(experiment['config'].get('ablation', 'production'))
    derived = parent.parent / f'{parent.name}__refine_{stable_hash(dict(parent=parent.name, settings=settings), 10)}'
    derived.mkdir(parents=True, exist_ok=True)
    print('Refinement run (parent untouched):', derived)
    with RunWriterLock(derived, 'refine'):
        write_attempt_record(derived, dict(parent=str(parent), refinement=settings), ENVIRONMENT_INFO)
        if REFINE_AUDITS: run_priority0_audits(parent, derived, plan)
        registry_parts = []; references = {}; fold_records = []; candidate_rows = []
        private_sum = [np.zeros((len(test), len(c))) for c in CLASSES] if PREDICT_PRIVATE else None
        manifest_members = []
        for item, record in zip(plan['folds'], results):
            fold = int(item['fold']); fdir = derived / f'fold{fold}'; fdir.mkdir(exist_ok=True)
            inner_dir = parent / 'stages' / Path(record['inner_stage_folder']).name
            outer_dir = parent / 'stages' / Path(record['stage_folder']).name
            inner, hold = np.asarray(item['inner_training'], int), np.asarray(item['inner_validation'], int)
            tr, va = np.asarray(item['training'], int), np.asarray(item['validation'], int)
            print(f'Fold {fold}: regenerating inner predictions with the saved inner models (no training)')
            hold_paths = _views(inner_dir, hold)
            iv = _cached(fdir / 'inner_video.npz', lambda: {f'seed{s}_task{j}': p[j] for s, p in
                         predict_saved_stage(inner_dir, seeds, hold, hold_paths).items() for j in range(2)})
            comps = {'video_ensemble': [sum(iv[f'seed{s}_task{j}'] for s in seeds) / len(seeds) for j in range(2)]}
            prior = json.loads((inner_dir / 'class_prior.json').read_text())['prior']
            comps['class_prior'] = [np.tile(np.asarray(prior[j]), (len(hold), 1)) for j in range(2)]
            hold_x = _numeric(hold_paths, hold)
            if (inner_dir / 'pose_camera_trees.joblib').exists():
                comps['pose_motion_tree'] = tree_probabilities(joblib.load(inner_dir / 'pose_camera_trees.joblib'), hold_x, CLASSES)
            if REFINE_TREE_CLASS_WEIGHTS and hold_x.shape[1]:
                wt_path = fdir / 'inner_trees_weighted.joblib'
                if not wt_path.exists():
                    joblib.dump(fit_weighted_trees(_numeric(_views(inner_dir, inner), inner), Y[inner], REFINE_WEIGHT_CAP), wt_path)
                comps['pose_motion_tree_weighted'] = tree_probabilities(joblib.load(wt_path), hold_x, CLASSES)
            original = select_diagnostic_mixture(comps, Y[hold])
            reproduced = (list(original[0]) == list(record['ensemble_components']) and
                          all(np.allclose(w, rw) for w, rw in zip(original[1], record['weights'])))
            if not reproduced:   # e.g. GPU fp16 nondeterminism flipping a near-tie; reported, never hidden
                print(f'WARNING fold {fold}: regenerated inner predictions do not reproduce the original ensemble choice '
                      f'({original[0]} {[w.tolist() for w in original[1]]} vs {record["ensemble_components"]} {record["weights"]}).')
            choices = []
            for j in range(2):
                choice, rows_j = select_refined_task(comps, Y[hold, j], j, REFINE_CENTER_OFFSETS if REFINE_CLASS_OFFSETS else [])
                choices.append(choice); candidate_rows += [dict(fold=fold, **r) for r in rows_j]
            references[fold] = select_reference_components(comps, Y[hold], TARGETS, CLASSES, WEIGHTS)
            # Outer: saved components, plus the weighted tree refit on the outer training partition only.
            with np.load(parent / 'folds' / f'fold{fold}' / 'component_predictions.npz') as z:
                assert np.array_equal(z['indices'], va)
                outer = {n: [z[f'{n}__task{j}'] for j in range(2)] for n in
                         {k.rsplit('__task', 1)[0] for k in z.files if '__task' in k and not k.startswith('private_')}}
            outer['baseline_ensemble'] = outer.pop('ensemble')
            outer_paths = _views(outer_dir, va)
            wt_outer = fdir / 'outer_trees_weighted.joblib'
            if REFINE_TREE_CLASS_WEIGHTS and hold_x.shape[1]:
                if not wt_outer.exists():
                    joblib.dump(fit_weighted_trees(_numeric(_views(outer_dir, tr), tr), Y[tr], REFINE_WEIGHT_CAP), wt_outer)
                outer['pose_motion_tree_weighted'] = tree_probabilities(joblib.load(wt_outer), _numeric(outer_paths, va), CLASSES)
            outer['ensemble'] = [apply_class_offsets(sum(w * outer[n][j] for n, w in choices[j]['weights'].items()),
                                                     choices[j]['offsets']) for j in range(2)]
            registry_parts.append((va, outer))
            fold_records.append(dict(fold=fold, stage_folder=record['stage_folder'], inner_stage_folder=record['inner_stage_folder'],
                seeds=seeds, selected_epochs=record['selected_epochs'],
                ensemble_components=sorted({n for c in choices for n in c['weights']}),
                weights=[[c['weights'].get(n, 0.0) for n in sorted({n for c in choices for n in c['weights']})] for c in choices],
                reference_components=references[fold], refined_choices=choices,
                original_inner_selection=dict(names=original[0], weights=[w.tolist() for w in original[1]], reproduced=bool(reproduced)),
                accuracy=[float((outer['ensemble'][j].argmax(1) == Y[va, j]).mean()) for j in range(2)],
                experiment_identity=stable_hash(dict(parent=experiment.get('experiment_identity'), settings=settings)),
                private_inference=bool(PREDICT_PRIVATE)))
            fold_records[-1]['score'] = float(WEIGHTS @ fold_records[-1]['accuracy'])
            if PREDICT_PRIVATE:
                ppaths = _ensure_private_views(outer_dir, derived / 'private_crops' / f'fold{fold}')
                pidx = list(range(len(train), len(rows)))
                pv = _cached(fdir / 'private_video.npz', lambda: {f'seed{s}_task{j}': p[j] for s, p in
                             predict_saved_stage(outer_dir, seeds, pidx, ppaths).items() for j in range(2)})
                px = _numeric(ppaths, pidx); pc = {'video_ensemble': [sum(pv[f'seed{s}_task{j}'] for s in seeds) / len(seeds) for j in range(2)]}
                oprior = json.loads((outer_dir / 'class_prior.json').read_text())['prior']
                pc['class_prior'] = [np.tile(np.asarray(oprior[j]), (len(pidx), 1)) for j in range(2)]
                if (outer_dir / 'pose_camera_trees.joblib').exists():
                    pc['pose_motion_tree'] = tree_probabilities(joblib.load(outer_dir / 'pose_camera_trees.joblib'), px, CLASSES)
                if wt_outer.exists(): pc['pose_motion_tree_weighted'] = tree_probabilities(joblib.load(wt_outer), px, CLASSES)
                for j in range(2):
                    private_sum[j] += apply_class_offsets(sum(w * pc[n][j] for n, w in choices[j]['weights'].items()),
                                                          choices[j]['offsets']) / len(plan['folds'])
            manifest_members.append((fold, choices, wt_outer if wt_outer.exists() else None))
            print(f'Fold {fold}: kick -> {choices[0]["candidate"]} (offset {choices[0]["center_offset"]}), '
                  f'keeper -> {choices[1]["candidate"]} (offset {choices[1]["center_offset"]})')
        return _write_refinement_outputs(parent, derived, experiment, plan, registry_parts, references, fold_records,
                                         candidate_rows, private_sum, manifest_members, settings)


def _write_refinement_outputs(parent, derived, experiment, plan, parts, references, fold_records, candidate_rows,
                              private_sum, members, settings):
    fold_ids = np.full(len(train), -1, dtype=int)
    for item in plan['folds']: fold_ids[item['validation']] = item['fold']
    registry = ComponentOOF(train, Y, fold_ids, TARGETS, CLASSES, WEIGHTS)
    for va, outer in parts:
        for name, p in outer.items():
            registry.add(name, va, p, train['clip'].iloc[va].tolist(), CLASSES)
    config = dict(experiment['config'], refinement=settings, refined_from=parent.name)
    atomic_write_text(derived / 'experiment_manifest.json', json.dumps(dict(experiment, config=config,
        interpretation='Refinement of a finished run: choices made on inner predictions only; outer results are exploratory.'), indent=2))
    shutil.copy2(parent / 'folds.csv', derived / 'folds.csv')
    atomic_write_text(derived / 'fold_results.json', json.dumps(fold_records, indent=2, default=str))
    pd.DataFrame(candidate_rows).to_csv(derived / 'refine_candidates.csv', index=False)
    quality = pd.read_csv(parent / 'tracking_quality.csv') if (parent / 'tracking_quality.csv').exists() else None
    for name in ['tracking_quality.csv', 'role_click_audit.csv', 'role_click_summary.csv', 'tracking_frames.csv']:
        if (parent / name).exists(): shutil.copy2(parent / name, derived / name)
    exported = registry.export(derived / 'component_reports',
                               quality=None if quality is None else quality.drop(columns=['source_id', 'fold'], errors='ignore'),
                               references=references, ensemble='ensemble')
    comparison = refinement_comparison(exported['metrics'], exported['errors'])
    comparison.to_csv(derived / 'refine_comparison.csv', index=False)
    oof = [np.full((len(train), len(c)), np.nan) for c in CLASSES]
    for va, outer in parts:
        for j in range(2): oof[j][va] = outer['ensemble'][j]
    np.savez_compressed(derived / 'predictions.npz', fold_ids=fold_ids, **{f'oof_{j}': p for j, p in enumerate(oof)},
                        **({f'private_{j}': p for j, p in enumerate(private_sum)} if private_sum is not None else {}))
    submission_path = None
    if private_sum is not None:
        validate_probability_arrays(private_sum, len(test), CLASSES)
        pred = pd.DataFrame({'clip': test['clip'].to_numpy(), **{t: decode_labels(private_sum[j], j) for j, t in enumerate(TARGETS)}})
        submission = template[['clip']].merge(pred, on='clip', how='left', validate='one_to_one')
        assert len(submission) == len(test) and not submission.isna().any().any()
        submission_path = derived / 'submission.csv'; submission.to_csv(submission_path, index=False)
    _write_refined_inference_manifest(parent, derived, members, settings)
    atomic_write_text(derived / 'refinement_manifest.json', json.dumps(dict(
        hypothesis='Reweighting the numeric specialist toward rare classes, reselecting task-specific ensembles on inner '
                   'validation and an inner-fitted center offset improve center-class recall without retraining video models.',
        changes=settings, controls=['baseline_ensemble (parent run)', 'pose_motion_tree (unweighted control)',
                                    'video_ensemble', 'class_prior'],
        selection_sources='inner validation of each fold (source-grouped)', evaluation='saved outer folds (exploratory; '
        'these outer labels were already inspected)', seeds=experiment['seeds'], parent=parent.name,
        adoption_rule='Choices are fixed by the inner rule above. Choosing between this submission and the baseline by '
                      'outer results turns those results into a tuning metric; report both.',
        retraining='None for video models. CatBoost specialists refit on each stage training partition (CPU).'), indent=2, default=str))
    display(comparison[comparison.fold == 'pooled'])
    print('Refinement outputs:', derived, '| submission:', submission_path)
    return dict(run=derived, comparison=comparison, submission=submission_path, exported=exported)


def refinement_comparison(metrics, errors):
    """Pooled and per-fold metrics per component, plus macro recall, center recall/precision and predicted counts,
    and paired per-fold differences against the baseline ensemble and the unweighted tree."""
    health = class_health_table(errors)
    rows = []
    for (component, fold), m in metrics.assign(fold=metrics.fold.astype(str)).groupby(['component', 'fold']):
        r = m.iloc[0]; row = dict(component=component, fold=fold, clips=int(r.clips), weighted_accuracy=float(r.weighted_accuracy),
                                  weighted_cross_entropy=float(r.weighted_cross_entropy))
        for j, t in enumerate(TARGETS):
            h = health[(health.component == component) & (health.fold == fold) & (health.target == t)].set_index('class_label')
            row[f'{t}_accuracy'] = float(r[f'{t}_accuracy']); row[f'{t}_cross_entropy'] = float(r[f'{t}_cross_entropy'])
            row[f'{t}_macro_recall'] = float(h.loc['__macro__', 'recall'])
            center = CLASSES[j][1]
            row[f'{t}_center_recall'] = float(h.loc[center, 'recall']); row[f'{t}_center_precision'] = float(h.loc[center, 'precision'])
            for label in CLASSES[j]: row[f'{t}_predicted__{label}'] = int(h.loc[label, 'predicted_support'])
        rows.append(row)
    table = pd.DataFrame(rows)
    for control in ('baseline_ensemble', 'pose_motion_tree'):
        if control not in set(table.component): continue
        base = table[table.component == control].set_index('fold')
        for metric in ('weighted_accuracy', 'weighted_cross_entropy'):
            table[f'{metric}_minus_{control}'] = [r[metric] - base.loc[r['fold'], metric] for _, r in table.iterrows()]
    return table


def _write_refined_inference_manifest(parent, derived, members, settings):
    src = parent / 'inference_manifest.json'
    if not src.exists():
        print('NOTE: parent has no inference_manifest.json; the refined ensemble cannot be replayed by inference mode.'); return
    manifest = json.loads(src.read_text()); rel = os.path.relpath(parent, derived).replace('\\', '/')
    files = {f'{rel}/{k}': v for k, v in manifest['files'].items()}
    by_fold = {fold: (choices, wt) for fold, choices, wt in members}
    for m in manifest['members']:
        choices, wt = by_fold[int(m['fold'])]
        for key in ('role_selector', 'numeric_scaler', 'trees', 'stage_dir'):
            if m.get(key): m[key] = f'{rel}/{m[key]}'
        for s in m['seeds']: s['weights'] = f'{rel}/{s["weights"]}'
        m['evaluation_crops'] = {c: f'{rel}/{p}' for c, p in m['evaluation_crops'].items()}
        names = sorted({n for c in choices for n in c['weights']})
        m['ensemble'] = dict(components=names, weights=[[c['weights'].get(n, 0.0) for n in names] for c in choices], seed_average='mean')
        m['class_offsets'] = [c['offsets'] for c in choices]
        m['extra_trees'] = {}
        if wt is not None:
            key = str(wt.relative_to(derived)).replace('\\', '/')
            m['extra_trees']['pose_motion_tree_weighted'] = key
            files[key] = dict(sha256=sha256_file(wt), bytes=wt.stat().st_size)
        m['fitted_artifact_id'] = fitted_artifact_id(dict(parent=m['fitted_artifact_id'], refinement=settings,
                                                          ensemble=m['ensemble'], offsets=m['class_offsets'],
                                                          extra={k: files[v]['sha256'] for k, v in m['extra_trees'].items()}))
    manifest.update(files=files, run_id=derived.name, refinement=dict(settings, parent=parent.name),
                    created=time.strftime('%Y-%m-%dT%H:%M:%S'))
    atomic_write_text(derived / 'inference_manifest.json', json.dumps(manifest, indent=2))


# ------------------------------------------------------------------------------------------- Priority 0 audits
def _raw_entry(clip):
    path = RAW / (clip + '.pkl')
    if not path.exists(): return None
    with path.open('rb') as f: return pickle.load(f)


def run_priority0_audits(parent, out, plan):
    """No retraining: label-mapping audit, temporal input audit, crop geometry, training-only label samples."""
    label_mapping_audit(out / 'label_mapping_audit.json')
    temporal, dims = [], {}
    for clip in tqdm(rows['clip'], desc='Temporal input audit (cached detections)', leave=False):
        value = _raw_entry(clip)
        if value is None:
            temporal.append(dict(clip=clip, cached=False)); continue
        t = np.asarray(value['times'], float)
        frame = decode_jpeg(value['frames_jpeg'][0]); dims[clip] = frame.shape[:2]
        temporal.append(dict(clip=clip, cached=True, frames=len(t), unique_frames=int(len(np.unique(t))),
                             first_time=float(t[0]), last_time=float(t[-1]), span_seconds=float(t[-1] - t[0]),
                             monotonic=bool(np.all(np.diff(t) >= 0)), within_window=bool(t[-1] - t[0] <= WINDOW + .1),
                             frame_height=frame.shape[0], frame_width=frame.shape[1]))
    pd.DataFrame(temporal).to_csv(out / 'temporal_input_audit.csv', index=False)
    frames_path = parent / 'tracking_frames.csv'
    if frames_path.exists():
        crop_geometry_report(pd.read_csv(frames_path), dims).to_csv(out / 'crop_geometry.csv', index=False)
    label_convention_samples(plan, out / 'label_convention_samples')
    print('Priority-0 audits written to', out)


def crop_geometry_report(frames, dims):
    """Per role: player height (fraction of frame and pixels in the model input), padding fraction and box jitter,
    for selected frames. Uses actual frame sizes from the detector cache."""
    needed = {'clip', 'role', 'frame', 'selected', 'box_x1', 'box_y1', 'box_x2', 'box_y2', 'crop_x1', 'crop_y1', 'crop_x2', 'crop_y2'}
    if not needed <= set(frames): return pd.DataFrame()
    f = frames[frames.selected.astype(bool)].copy()
    if f.empty: return pd.DataFrame()
    H = f['clip'].map(lambda c: dims.get(c, (np.nan, np.nan))[0]); W = f['clip'].map(lambda c: dims.get(c, (np.nan, np.nan))[1])
    bh = (f.box_y2 - f.box_y1) * H; bw = (f.box_x2 - f.box_x1) * W
    ch = (f.crop_y2 - f.crop_y1) * H; cw = (f.crop_x2 - f.crop_x1) * W
    f['player_height_fraction'] = f.box_y2 - f.box_y1
    f['player_height_model_pixels'] = bh * SIZE / np.maximum(np.maximum(ch, cw), 1)
    f['padding_fraction'] = 1 - (bh * bw) / np.maximum(ch * cw, 1)
    f = f.sort_values(['clip', 'role', 'frame'])
    cx = (f.box_x1 + f.box_x2) / 2 * W; cy = (f.box_y1 + f.box_y2) / 2 * H
    f['jitter_player_heights'] = np.hypot(cx.groupby([f['clip'], f['role']]).diff(), cy.groupby([f['clip'], f['role']]).diff()) / np.maximum(bh, 1)
    rows = []
    for role, g in f.groupby('role'):
        for col in ['player_height_fraction', 'player_height_model_pixels', 'padding_fraction', 'jitter_player_heights']:
            v = g[col].dropna()
            rows.append(dict(role=role, measure=col, frames=int(len(v)), median=float(v.median()) if len(v) else np.nan,
                             p10=float(v.quantile(.1)) if len(v) else np.nan, p90=float(v.quantile(.9)) if len(v) else np.nan))
    return pd.DataFrame(rows)


def label_convention_samples(plan, out, per_class=4):
    """Final cached frame of a few TRAINING clips (fold 0 outer-training partition) per kick class, with their labels,
    to check against the dataset documentation whether left/right are camera-relative. Never used for training."""
    import matplotlib.pyplot as plt
    out = Path(out); out.mkdir(parents=True, exist_ok=True)
    training = np.asarray(plan['folds'][0]['training'], int); rng = np.random.default_rng(SEED)
    for k, label in enumerate(CLASSES[0]):
        pool = training[Y[training, 0] == k]
        chosen = rng.choice(pool, size=min(per_class, len(pool)), replace=False) if len(pool) else []
        if not len(chosen): continue
        fig, axes = plt.subplots(1, len(chosen), figsize=(5 * len(chosen), 3.4), squeeze=False)
        for ax, i in zip(axes[0], chosen):
            value = _raw_entry(rows['clip'].iloc[i])
            if value is not None: ax.imshow(decode_jpeg(value['frames_jpeg'][-1]))
            ax.set_title(f'{rows["clip"].iloc[i]}\nkick={CLASSES[0][Y[i, 0]]}, keeper={CLASSES[1][Y[i, 1]]}', fontsize=9); ax.axis('off')
        fig.suptitle(f'Training clips labeled kick="{label}" (last sampled frame). Compare with the dataset definition of left/right.')
        fig.tight_layout(); fig.savefig(out / f'kick_{label}.png', dpi=110); plt.close(fig)
