# CELL: inference_helpers
"""Inference-only workflow: fitted-artifact manifest, validation, prediction caches and label-free prediction.

Nothing in this cell fits anything: no fit_video/fit_roles/diagnostic_stage/run_nested_diagnostics,
no scaler fitting, no CatBoost .fit, no checkpoint or ensemble selection, no optimizer. It only loads
saved artifacts and performs label-free decoding/detection/cropping through the injected `build_view` hook.
"""
import copy, hashlib, json, os, pickle, time
from pathlib import Path
import numpy as np
import pandas as pd

INFERENCE_SCHEMA_VERSION = 1
INFERENCE_CODE_VERSION = 'inference-v1'
LEGACY_BACKBONE_IDENTITY = dict(name='videomae', training_policy='finetune')


class InferenceArtifactError(RuntimeError):
    """A required fitted artifact is missing, modified or incompatible."""


def _sha256_bytes(data):
    return hashlib.sha256(data).hexdigest()


def sha256_file(path, chunk=8 * 1024 * 1024):
    digest = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(chunk), b''): digest.update(block)
    return digest.hexdigest()


def stable_hash(value, length=16):
    return hashlib.sha256(json.dumps(value, sort_keys=True, default=str).encode()).hexdigest()[:length]


def preprocessing_record(config, image_mean, image_std, camera_cache_version='similarity-jpeg-v1'):
    """Everything that determines evaluation crops/tensors for a fitted model (not the clip list)."""
    record = dict(frames=int(config['frames']), window=float(config['window']), size=int(config['size']),
                  detector_conf=config['detector_conf'], detector_imgsz=config['detector_imgsz'], pose=config['pose'],
                  tracking=config['tracking'], role_min_prob=config['role_min_prob'], crop_version=config['version'],
                  camera_cache_version=camera_cache_version, frame_sampling='last WINDOW seconds, quadratic spacing toward kick',
                  crop_policy='letterbox context + role crops (1.7 x 1.35 box) with full-frame fallback',
                  view_cache='JPEG quality 95', image_mean=[float(x) for x in image_mean],
                  image_std=[float(x) for x in image_std], input_layout='views x frames x channels x H x W, uint8 -> /255 -> (x-mean)/std')
    record['preprocessing_id'] = stable_hash({k: v for k, v in record.items() if k != 'preprocessing_id'})
    return record


def fitted_artifact_id(identity):
    """Stable ID of a fitted member. Must not include the requested prediction clip list."""
    forbidden = {'evaluation_indices', 'evaluation_clips', 'prediction_clips'}
    if forbidden & set(identity): raise ValueError('Fitted identity must exclude the prediction clip list')
    return stable_hash(identity, 24)


def _config_without_evaluation(config):
    return {k: v for k, v in config.items() if k not in ('smoke', 'smoke_max_batches')}


# ------------------------------------------------------------------------------------------- export
def export_inference_manifest(run_dir, *, preprocessing, backbone=None, training_labels=None, dataset_id=None,
                              extra_clip_names=None, holdout_config=None, overwrite=True, verbose=True):
    """Write inference_manifest.json for a nested_cv or holdout run whose trained artifacts exist.

    Used at the end of training and as the one-time exporter for existing runs. training_labels (a frame
    with clip + target columns) is used ONLY when a stage lacks class_prior.json, restricted to that stage's
    own training clips, and the recovered prior is persisted. Inference never needs labels.
    """
    run = Path(run_dir)
    experiment = run / 'experiment_manifest.json'
    if experiment.exists():
        manifest_in = json.loads(experiment.read_text())
        config = manifest_in['config']; plan = manifest_in['split_plan']; seeds = list(manifest_in['seeds'])
        clips = list(plan['identity']['clips']); mode = 'nested_cv'
        results_path = run / 'fold_results.json'
        if not results_path.exists(): raise InferenceArtifactError(f'Missing {results_path}')
        results = json.loads(results_path.read_text())
        expected = int(config['outer_folds'])
        done = sorted(int(r['fold']) for r in results)
        if done != list(range(expected)):
            raise InferenceArtifactError(f'Run is incomplete: folds {done} of {expected}. Resume training first; '
                                         'a partial fold ensemble is never exported.')
        folds = {int(f['fold']): f for f in plan['folds']}
        stages = [(int(r['fold']), r) for r in results]
    elif (run / 'holdout_manifest.json').exists() or holdout_config is not None:
        info = json.loads((run / 'holdout_manifest.json').read_text()) if (run / 'holdout_manifest.json').exists() else {}
        config = info.get('config', holdout_config); seeds = list(info.get('seeds', config['seeds'])); mode = 'holdout'
        split = pd.read_csv(run / 'dataset_splits.csv'); clips = split['clip'].tolist()
        candidates = sorted((run / 'stages').glob('train_validation_test_*'))
        if len(candidates) != 1: raise InferenceArtifactError(f'Expected one holdout stage folder, found {candidates}')
        stages = [(0, dict(stage_folder=str(candidates[0]), ensemble_components=['video_ensemble'],
                           weights=[[1.0], [1.0]], selected_epochs=None))]
        folds = {0: dict(training=np.flatnonzero(split.split == 'training').tolist(),
                         validation=np.flatnonzero(split.split != 'training').tolist())}
    else:
        raise InferenceArtifactError(f'{run} is neither a nested_cv run (experiment_manifest.json) nor a holdout run.')
    targets = ['kick_direction', 'goalkeeper_direction']
    classes = [['left', 'middle', 'right'], ['left', 'stands in the middle', 'right']]
    members = []; files = {}
    for fold, record in stages:
        stage = run / 'stages' / Path(record['stage_folder']).name
        if not stage.is_dir(): raise InferenceArtifactError(f'Missing stage folder {stage}')
        rel = lambda p: str(Path(p).relative_to(run)).replace('\\', '/')
        provenance_path = stage / 'component_provenance.json'
        component_prov = json.loads(provenance_path.read_text()) if provenance_path.exists() else None
        training_idx = (component_prov['training_indices'] if component_prov else folds[fold]['training'])
        if any(i >= len(clips) for i in training_idx): raise InferenceArtifactError('Training index outside public clips')
        training_clips = [clips[i] for i in training_idx]
        if mode == 'nested_cv' and sorted(training_idx) != sorted(folds[fold]['training']):
            raise InferenceArtifactError(f'Fold {fold} stage training indices differ from the saved split plan')
        heldout = [clips[i] for i in folds[fold]['validation']]
        required = dict(role_selector=stage / 'roles_all_training.joblib', numeric_scaler=stage / 'numeric_scaler.joblib')
        names = list(record['ensemble_components'])
        if 'pose_motion_tree' in names: required['trees'] = stage / 'pose_camera_trees.joblib'
        seed_entries = []
        for seed in seeds:
            path = stage / f'video_seed{seed}.pt'
            required[f'video_seed{seed}'] = path
            selected = (record.get('selected_epochs') or {}).get(str(seed))
            seed_entries.append(dict(seed=int(seed), weights=rel(path),
                weight_kind='best_validation' if mode == 'holdout' else 'selected_epoch_final_outer_refit',
                epochs=selected))
        missing = [str(p) for p in required.values() if not Path(p).exists()]
        if missing: raise InferenceArtifactError('Missing trained artifacts (no partial ensembles are exported):\n' + '\n'.join(missing))
        prior_path = stage / 'class_prior.json'
        if prior_path.exists():
            prior_info = json.loads(prior_path.read_text())
        elif 'class_prior' in names or mode == 'nested_cv':
            if training_labels is None:
                raise InferenceArtifactError(f'{prior_path} is missing. Re-run export with training_labels (public '
                    'training labels; only this stage\'s training clips are read) to recover and persist it.')
            labels = training_labels.set_index('clip').loc[training_clips]
            prior = [np.bincount(labels[t].map({c: i for i, c in enumerate(cs)}).to_numpy().astype(int),
                                 minlength=len(cs)) / len(labels) for t, cs in zip(targets, classes)]
            prior_info = dict(prior=[p.tolist() for p in prior], class_order=classes, training_clip_count=len(labels),
                              source='migrated_from_training_labels', training_clips_sha=stable_hash(training_clips))
            prior_path.write_text(json.dumps(prior_info, indent=2))
        else:
            prior_info = None
        if prior_info is not None and prior_info.get('training_clips_sha', stable_hash(training_clips)) != stable_hash(training_clips):
            raise InferenceArtifactError(f'{prior_path} was computed on different training clips')
        serial_path = stage / 'backbone_serialization.json'
        if serial_path.exists(): backbone_info = json.loads(serial_path.read_text())
        elif backbone is not None: backbone_info = backbone
        else: raise InferenceArtifactError(f'{serial_path} missing; pass the saved backbone architecture for migration.')
        evaluation_crops = {}
        crop_prov = stage / 'provenance.json'
        if crop_prov.exists():
            names_all = clips + list(extra_clip_names or [])
            for entry in json.loads(crop_prov.read_text()):
                if entry['kind'] != 'evaluation': continue
                for i in entry['prediction_indices']:
                    if i < len(names_all) and (stage / f'{i}.pkl').exists():
                        evaluation_crops[names_all[i]] = rel(stage / f'{i}.pkl')
        identity = dict(dataset=config.get('dataset'), split_id=config.get('split_id'), mode=mode, fold=fold,
                        training_clips_sha=stable_hash(training_clips), backbone=backbone_info['identity'],
                        feature_signature=stable_hash(config.get('feature_manifest')), seeds=seeds,
                        selected_epochs=record.get('selected_epochs'), preprocessing_id=preprocessing['preprocessing_id'],
                        config=_config_without_evaluation(config))
        member = dict(member_id=f'fold{fold}' if mode == 'nested_cv' else 'holdout', fold=fold,
                      kind='outer_refit' if mode == 'nested_cv' else 'holdout_best_validation',
                      stage_dir=rel(stage), fitted_artifact_id=fitted_artifact_id(identity),
                      role_selector=rel(required['role_selector']), numeric_scaler=rel(required['numeric_scaler']),
                      trees=rel(required['trees']) if 'trees' in required else None,
                      class_prior=None if prior_info is None else prior_info['prior'],
                      class_prior_source=None if prior_info is None else prior_info.get('source', 'training_labels_at_fit'),
                      seeds=seed_entries, ensemble=dict(components=names, weights=[list(map(float, w)) for w in record['weights']],
                                                        seed_average='mean'),
                      training_clips=training_clips, training_sources=(component_prov or {}).get('training_sources'),
                      heldout_public_clips=heldout, evaluation_crops=evaluation_crops,
                      backbone=backbone_info)
        for key, path in required.items(): files[rel(path)] = None
        members.append(member)
    for path in files: files[path] = dict(sha256=sha256_file(run / path), bytes=(run / path).stat().st_size)
    feature = dict(variant=config.get('ablation', 'production'), numeric_features=config.get('feature_manifest'),
                   numeric_dimension=len(config.get('feature_manifest') or []),
                   streams=members[0]['backbone'].get('streams'))
    backbones = {json.dumps(m['backbone']['identity'], sort_keys=True) for m in members}
    if len(backbones) != 1: raise InferenceArtifactError('Members disagree about the backbone identity')
    manifest = dict(schema_version=INFERENCE_SCHEMA_VERSION, code_version=INFERENCE_CODE_VERSION,
                    created=time.strftime('%Y-%m-%dT%H:%M:%S'), run_id=run.name, evaluation_mode=mode,
                    dataset_id=dataset_id or config.get('dataset'), split_id=config.get('split_id'),
                    targets=targets, classes=classes, task_weights=[0.7, 0.3], config=config,
                    preprocessing=preprocessing, feature=feature, backbone=members[0]['backbone'],
                    combination=dict(across_members='mean', default_routing='ensemble',
                                     note='Private/new clips: mean of member ensembles. Public OOF replay: each clip uses only '
                                          'the member whose outer fold held it out.'),
                    members=members, files=files)
    path = run / 'inference_manifest.json'
    if path.exists() and not overwrite: raise InferenceArtifactError(f'{path} exists')
    tmp = path.with_name(f'.{path.name}.{os.getpid()}.tmp'); tmp.write_text(json.dumps(manifest, indent=2)); tmp.replace(path)
    if verbose: print(f'Inference manifest: {path} ({len(members)} member(s), {len(files)} checksummed files)')
    return manifest


# ------------------------------------------------------------------------------------------- load/validate
def load_inference_manifest(run_dir, verify_checksums=True):
    run = Path(run_dir)
    path = run / 'inference_manifest.json'
    if not path.exists():
        raise InferenceArtifactError(f'{path} not found. For a run trained before this update, set '
                                     'MIGRATE_LEGACY_RUN=True once to export it from the saved artifacts.')
    manifest = json.loads(path.read_text())
    if manifest.get('schema_version') != INFERENCE_SCHEMA_VERSION:
        raise InferenceArtifactError(f'Unsupported manifest schema {manifest.get("schema_version")}')
    if manifest['targets'] != ['kick_direction', 'goalkeeper_direction'] or manifest['classes'] != [
            ['left', 'middle', 'right'], ['left', 'stands in the middle', 'right']] or not np.allclose(manifest['task_weights'], [.7, .3]):
        raise InferenceArtifactError('Manifest target/class order or task weights are incompatible')
    problems = []
    for rel, info in manifest['files'].items():
        p = run / rel
        if not p.exists(): problems.append(f'missing: {p}'); continue
        if p.stat().st_size != info['bytes']: problems.append(f'size changed: {p}'); continue
        if verify_checksums and sha256_file(p) != info['sha256']: problems.append(f'checksum changed: {p}')
    for m in manifest['members']:
        referenced = [m['role_selector'], m['numeric_scaler']] + [s['weights'] for s in m['seeds']] + ([m['trees']] if m['trees'] else [])
        for rel in referenced:
            if rel not in manifest['files']: problems.append(f'unlisted artifact: {run / rel}')
        if not m['seeds']: problems.append(f'{m["member_id"]}: no seed weights')
        names = m['ensemble']['components']
        if len(m['ensemble']['weights']) != 2 or any(len(w) != len(names) for w in m['ensemble']['weights']):
            problems.append(f'{m["member_id"]}: ensemble weights do not match components')
        if 'class_prior' in names and m['class_prior'] is None: problems.append(f'{m["member_id"]}: class prior missing')
        if 'pose_motion_tree' in names and not m['trees']: problems.append(f'{m["member_id"]}: trees missing')
    if problems: raise InferenceArtifactError('Inference artifacts are incomplete or modified:\n' + '\n'.join(problems))
    return manifest


# ------------------------------------------------------------------------------------------- routing
def route_clips(manifest, clip_names, routing):
    """Return {clip: [member_ids]} and an exposure label per clip."""
    members = manifest['members']
    routes = {}; exposure = {}
    for clip in clip_names:
        trained = [m['member_id'] for m in members if clip in set(m['training_clips'])]
        held = [m['member_id'] for m in members if clip in set(m['heldout_public_clips'])]
        if routing == 'oof_replay':
            if len(held) != 1:
                raise InferenceArtifactError(f'{clip}: OOF replay needs exactly one held-out member, found {held}')
            routes[clip] = held; exposure[clip] = 'out_of_fold'
        elif routing == 'ensemble':
            routes[clip] = [m['member_id'] for m in members]
            exposure[clip] = 'in_sample' if trained else 'not_in_training'
        else:
            raise ValueError(f'Unknown routing {routing}')
    return routes, exposure


def validate_probability_arrays(arrays, count, classes):
    if len(arrays) != 2: raise InferenceArtifactError('Expected two target probability arrays')
    for p, c in zip(arrays, classes):
        p = np.asarray(p)
        if p.shape != (count, len(c)) or not np.isfinite(p).all() or (p < -1e-7).any() or (p > 1 + 1e-6).any():
            raise InferenceArtifactError(f'Invalid probability array shape/value: {p.shape}')
        if not np.allclose(p.sum(1), 1, atol=1e-5): raise InferenceArtifactError('Probability rows must sum to one')
    return arrays


def combine_member(member, components, count):
    """Apply the member's frozen per-task ensemble weights to its saved components."""
    names = member['ensemble']['components']
    missing = [n for n in names if n not in components]
    if missing: raise InferenceArtifactError(f'{member["member_id"]}: missing components {missing}')
    return [sum(w * np.asarray(components[n][j]) for w, n in zip(member['ensemble']['weights'][j], names))
            for j in range(2)]


# ------------------------------------------------------------------------------------------- caches
def clip_cache_key(clip, content_id, preprocessing_id):
    """Geometry/crop cache key: decoded frames, detections and crops do not depend on metadata."""
    return stable_hash(dict(clip=clip, content=content_id, preprocessing=preprocessing_id, code=INFERENCE_CODE_VERSION), 20)


def prediction_cache_key(crop_key, metadata_row, fitted_artifact_id, feature_signature):
    """Prediction key: every effective model input. Metadata (preferred foot, gender) enters the numeric
    branch and the trees, so changing it must recompute; new weights (fitted id) invalidate too."""
    return stable_hash(dict(crop=crop_key, metadata=[float(x) for x in metadata_row], fitted=fitted_artifact_id,
                            features=feature_signature, code=INFERENCE_CODE_VERSION), 24)


def content_id_for(path, cache):
    """sha256 of the video file, memoized by (path, size, mtime)."""
    p = Path(path); stat = p.stat(); key = f'{p}|{stat.st_size}|{int(stat.st_mtime)}'
    if key not in cache: cache[key] = sha256_file(p)
    return cache[key]


def prepare_video_tensor(video_uint8, image_mean, image_std):
    """Identical arithmetic to ClipDataset without augmentation: views x T x C x H x W."""
    x = np.asarray(video_uint8).astype(np.float32) / 255.
    mean = np.asarray(image_mean, dtype=np.float32)[None, None, :, None, None]
    std = np.asarray(image_std, dtype=np.float32)[None, None, :, None, None]
    return (x - mean) / std


def tree_probabilities(trees, features, classes):
    out = []
    for j, c in enumerate(classes):
        prob = np.zeros((len(features), len(c)))
        prob[:, np.asarray(trees[j].classes_, dtype=int)] = trees[j].predict_proba(features)
        out.append(prob)
    return out


# ------------------------------------------------------------------------------------------- runner
def run_saved_inference(run_dir, clips, *, hooks, routing='ensemble', verify_checksums=True, device='cpu',
                        dataset_id=None, output_name=None, compare_oof=True, output_root=None):
    """Predict saved models on `clips` (DataFrame: clip, path, preferred_foot, gender[, source_id]).

    hooks: build_view(record, selector, member) -> view dict (label-free decode/detect/crop),
           save_view(view, path), load_view(path, decode_video) -> cached view (decoded 'video' when requested),
           model_factory(manifest, member) -> torch module with saved architecture (no pretrained download),
           numeric_features(view, metadata_row, feature_config) -> 1-D float array,
           load_joblib(path).
    """
    import torch
    run = Path(run_dir)
    manifest = load_inference_manifest(run, verify_checksums)
    before = {rel: info['sha256'] for rel, info in manifest['files'].items()}
    classes = manifest['classes']; pre = manifest['preprocessing']
    clips = clips.reset_index(drop=True)
    if not clips['clip'].is_unique: raise InferenceArtifactError('Requested clip IDs must be unique')
    names = clips['clip'].tolist()
    routes, exposure = route_clips(manifest, names, routing)
    content_cache = {}
    content = {r.clip: content_id_for(r.path, content_cache) for r in clips.itertuples()}
    metadata = np.column_stack([(clips.preferred_foot == 'left'), (clips.preferred_foot == 'right'),
                                (clips.gender == 'F'), (clips.gender == 'M')]).astype(np.float32)
    # Caches/outputs live under output_root (default: <run>/inference). Notebooks pass a folder outside the
    # trained run, so an imported or still-running run is never written to.
    base = Path(output_root) if output_root is not None else run / 'inference'
    stats = dict(crops_reused_training=0, crops_reused_inference=0, crops_built=0, predictions_reused=0, predictions_computed=0)
    member_outputs = {}
    feature_config = manifest['feature']
    for member in manifest['members']:
        needed = [i for i, c in enumerate(names) if member['member_id'] in routes[c]]
        if not needed: continue
        member_dir = base / 'predictions' / member['fitted_artifact_id']
        crop_dir = base / 'crops' / f'{member["member_id"]}__{member["fitted_artifact_id"][:12]}__{pre["preprocessing_id"][:12]}'
        member_dir.mkdir(parents=True, exist_ok=True); crop_dir.mkdir(parents=True, exist_ok=True)
        crop_keys = {i: clip_cache_key(names[i], content[names[i]], pre['preprocessing_id']) for i in needed}
        feature_signature = stable_hash(manifest['feature'])
        keys = {i: prediction_cache_key(crop_keys[i], metadata[i], member['fitted_artifact_id'], feature_signature)
                for i in needed}
        pending = [i for i in needed if not (member_dir / f'{keys[i]}.npz').exists()]
        components = {}
        if pending:
            selector = None; views = {}
            for i in pending:
                crop_path = crop_dir / f'{crop_keys[i]}.pkl'
                training_crop = member['evaluation_crops'].get(names[i])
                if crop_path.exists():
                    stats['crops_reused_inference'] += 1; views[i] = crop_path
                elif (training_crop and dataset_id is not None and dataset_id == manifest['dataset_id']
                      and (run / training_crop).exists()):   # bundles may omit training-time crops
                    stats['crops_reused_training'] += 1; views[i] = run / training_crop
                else:
                    if selector is None: selector = hooks['load_joblib'](run / member['role_selector'])
                    hooks['save_view'](hooks['build_view'](clips.iloc[i], selector, member), crop_path)
                    stats['crops_built'] += 1; views[i] = crop_path
            scaler = hooks['load_joblib'](run / member['numeric_scaler'])
            raw = np.stack([hooks['numeric_features'](hooks['load_view'](views[i], decode_video=False), metadata[i],
                                                      feature_config) for i in pending]).reshape(len(pending), -1)
            numeric = scaler.transform(raw).astype(np.float32)
            seed_probs = {}
            for entry in member['seeds']:
                model = hooks['model_factory'](manifest, member)
                state = torch.load(run / entry['weights'], map_location='cpu', weights_only=True)
                if 'optimizer' in state: raise InferenceArtifactError(f'{entry["weights"]} is a resumable checkpoint, not inference weights')
                model.load_state_dict(state['model'], strict=True); model.to(device).eval()
                rows = [[] for _ in classes]
                with torch.inference_mode():
                    for k, i in enumerate(pending):  # stream one decoded clip at a time (bounded memory)
                        view = hooks['load_view'](views[i], decode_video=True)
                        video = torch.from_numpy(prepare_video_tensor(view['video'], pre['image_mean'], pre['image_std']))[None]
                        n = torch.from_numpy(numeric[k:k + 1])
                        with torch.autocast(device_type='cuda' if str(device).startswith('cuda') else 'cpu',
                                            dtype=torch.float16, enabled=str(device).startswith('cuda')):
                            logits = model(video.to(device), n.to(device))
                        for j, p in enumerate(logits): rows[j].append(p.float().softmax(1).cpu().numpy())
                seed_probs[entry['seed']] = [np.concatenate(r) for r in rows]
                del model, state
            tree_probs = None
            if member['trees']:
                tree_probs = tree_probabilities(hooks['load_joblib'](run / member['trees']), raw, classes)
            for k, i in enumerate(pending):
                arrays = {f'seed{s}_task{j}': p[j][k] for s, p in seed_probs.items() for j in range(2)}
                if tree_probs is not None: arrays.update({f'tree_task{j}': tree_probs[j][k] for j in range(2)})
                tmp = member_dir / f'.{keys[i]}.{os.getpid()}.tmp.npz'
                np.savez(tmp, **arrays, clip=np.array(names[i]), content=np.array(content[names[i]]),
                         metadata=np.asarray(metadata[i]))
                tmp.replace(member_dir / f'{keys[i]}.npz')
            stats['predictions_computed'] += len(pending)
        stats['predictions_reused'] += len(needed) - len(pending)
        cached = [np.load(member_dir / f'{keys[i]}.npz') for i in needed]
        for c, i in zip(cached, needed):
            if (str(c['clip']) != names[i] or str(c['content']) != content[names[i]]
                    or not np.array_equal(c['metadata'], metadata[i])):
                raise InferenceArtifactError(f'Prediction cache mismatch for {names[i]}')
        seeds = [e['seed'] for e in member['seeds']]
        for s in seeds:
            components[f'video_seed{s}'] = [np.stack([c[f'seed{s}_task{j}'] for c in cached]) for j in range(2)]
        components['video_ensemble'] = [sum(components[f'video_seed{s}'][j] for s in seeds) / len(seeds) for j in range(2)]
        if member['trees']:
            components['pose_motion_tree'] = [np.stack([c[f'tree_task{j}'] for c in cached]) for j in range(2)]
        if member['class_prior'] is not None:
            components['class_prior'] = [np.tile(np.asarray(member['class_prior'][j]), (len(needed), 1)) for j in range(2)]
        components['ensemble'] = combine_member(member, components, len(needed))
        for name, arrays in components.items(): validate_probability_arrays(arrays, len(needed), classes)
        member_outputs[member['member_id']] = dict(indices=needed, components=components)
    final = [np.zeros((len(names), len(c))) for c in classes]
    counts = np.zeros(len(names))
    for member_id, out in member_outputs.items():
        for j in range(2): final[j][out['indices']] += out['components']['ensemble'][j]
        counts[out['indices']] += 1
    if (counts == 0).any(): raise InferenceArtifactError('Some clips received no member prediction')
    final = [p / counts[:, None] for p in final]
    validate_probability_arrays(final, len(names), classes)
    after = {rel: sha256_file(run / rel) for rel in before} if verify_checksums else before
    if after != before: raise InferenceArtifactError('A fitted artifact changed during inference')
    input_id = stable_hash(dict(clips=names, content=[content[c] for c in names], routing=routing,
                                metadata=metadata.tolist(),
                                preprocessing=pre['preprocessing_id'], members=[m['fitted_artifact_id'] for m in manifest['members']]))
    out_dir = base / 'runs' / (output_name or f'{routing}_{input_id}'); out_dir.mkdir(parents=True, exist_ok=True)
    table = pd.DataFrame({'clip': names, 'exposure': [exposure[c] for c in names],
                          'members': [';'.join(routes[c]) for c in names]})
    for j, (target, cs) in enumerate(zip(manifest['targets'], classes)):
        table[target] = np.asarray(cs)[final[j].argmax(1)]
        for k, label in enumerate(cs): table[f'{target}_probability_{k}__{label}'] = final[j][:, k]
    table.to_csv(out_dir / 'final_probabilities.csv', index=False)
    np.savez_compressed(out_dir / 'probabilities.npz', clip=np.asarray(names, dtype=str),
                        **{f'final_task{j}': final[j] for j in range(2)},
                        **{f'{mid}__{name}__task{j}': arr[j] for mid, out in member_outputs.items()
                           for name, arr in out['components'].items() for j in range(2)},
                        **{f'{mid}__indices': np.asarray(out['indices']) for mid, out in member_outputs.items()})
    comparison = None
    if compare_oof and routing == 'oof_replay' and (run / 'predictions.npz').exists() and manifest['evaluation_mode'] == 'nested_cv':
        saved = np.load(run / 'predictions.npz')
        order = {c: i for i, c in enumerate(json.loads((run / 'experiment_manifest.json').read_text())['split_plan']['identity']['clips'])}
        idx = [order[c] for c in names]
        comparison = {manifest['targets'][j]: float(np.abs(saved[f'oof_{j}'][idx] - final[j]).max()) for j in range(2)}
    record = dict(trained_run_dir=str(run), input=dict(count=len(names), routing=routing, input_id=input_id),
                  backbone=manifest['backbone']['identity'], members=[m['member_id'] for m in manifest['members']],
                  fitted_artifact_ids={m['member_id']: m['fitted_artifact_id'] for m in manifest['members']},
                  preprocessing_id=pre['preprocessing_id'], cache_statistics=stats,
                  exposure_counts=table.exposure.value_counts().to_dict(), max_abs_difference_vs_saved_oof=comparison,
                  note=('OOF replay: each public clip used only its held-out fold model.' if routing == 'oof_replay' else
                        'Fold-ensemble predictions; rows marked in_sample were seen by a training member and are not OOF.'),
                  labels_used=False, fitting_performed=False)
    (out_dir / 'inference_run.json').write_text(json.dumps(record, indent=2, default=str))
    return dict(final=final, table=table, members=member_outputs, record=record, output_dir=out_dir, manifest=manifest)


def write_submission(result, template, output_path):
    table = result['table']; manifest = result['manifest']
    submission = template[['clip']].merge(table[['clip'] + manifest['targets']], on='clip', how='left', validate='one_to_one')
    if submission.isna().any().any() or len(submission) != len(template):
        raise InferenceArtifactError('Submission does not cover every template clip')
    submission.to_csv(output_path, index=False)
    return submission
