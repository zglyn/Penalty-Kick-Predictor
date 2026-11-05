# CELL: nested_orchestration
"""Nested source-grouped experiment orchestration, embedded into the notebook."""

def saved_diagnostic_splits():
    identity=dict(dataset=DATA_ID,seed=SEED,outer_folds=OUTER_FOLDS,
                  clips=train['clip'].tolist(),sources=train.source_id.tolist())
    split_id=hashlib.sha256(json.dumps(identity,sort_keys=True).encode()).hexdigest()[:16]
    folder=OUTPUT/'diagnostic_splits'; folder.mkdir(exist_ok=True)
    path=folder/f'{split_id}.json'
    if path.exists():
        plan=json.loads(path.read_text())
        assert plan['identity']==identity, 'Saved split identity does not match the dataset.'
    else:
        folds=[]
        splitter=StratifiedGroupKFold(n_splits=OUTER_FOLDS,shuffle=True,random_state=SEED)
        for fold,(tr,va) in enumerate(splitter.split(np.zeros(len(train)),Y[:,0],train.source_id)):
            # Only outer-training labels may be consulted to ensure inner training has all classes.
            for offset in range(100):
                a,b=next(GroupShuffleSplit(n_splits=1,test_size=.2,random_state=SEED+offset).split(
                    tr,groups=train.source_id.iloc[tr]))
                inner,hold=tr[a],tr[b]
                if (train.source_id.iloc[inner].nunique()>=ROLE_FOLDS and
                    all(len(np.unique(Y[inner,j]))==len(c) for j,c in enumerate(CLASSES))): break
            else: raise ValueError('Cannot create inner training partition with all classes and enough sources.')
            folds.append(dict(fold=fold,training=tr.tolist(),validation=va.tolist(),
                              inner_training=inner.tolist(),inner_validation=hold.tolist()))
        plan=dict(split_id=split_id,identity=identity,folds=folds)
        atomic_write_text(path,json.dumps(plan,indent=2))
    coverage=np.zeros(len(train),dtype=int)
    for item in plan['folds']:
        tr,va,inner,hold=[np.asarray(item[k],dtype=int) for k in
                          ['training','validation','inner_training','inner_validation']]
        assert set(tr)|set(va)==set(range(len(train))) and not set(tr)&set(va)
        assert set(inner)|set(hold)==set(tr) and not set(inner)&set(hold)
        assert not set(train.source_id.iloc[tr]) & set(train.source_id.iloc[va])
        assert not set(train.source_id.iloc[inner]) & set(train.source_id.iloc[hold])
        coverage[va]+=1
    assert (coverage==1).all(), 'Saved outer validation partitions must cover each clip exactly once.'
    return plan


def diagnostic_stage(training,evaluation,name,validation=None,selected_epochs=None):
    # build_stage fits cross-fitted training crops and held-out evaluation crops using training sources only.
    folder,mapping,selector=build_stage(training,evaluation,name,selection=validation)
    component={}; selected={}
    for seed in VIDEO_SEEDS:
        epochs=VIDEO_EPOCHS if selected_epochs is None else int(selected_epochs[str(seed)])
        model,scaler=fit_video(training,mapping,folder,seed,validation_indices=validation,epochs_override=epochs)
        component[f'video_seed{seed}']=predict_video(model,scaler,evaluation,mapping)
        state=torch.load(folder/f'video_seed{seed}.pt',map_location='cpu',weights_only=True)
        selected[str(seed)]=int(state.get('best_epoch',epochs))
        del model,state; gc.collect(); torch.cuda.empty_cache()
    component['video_ensemble']=[sum(component[f'video_seed{s}'][j] for s in VIDEO_SEEDS)/len(VIDEO_SEEDS)
                                 for j in range(len(TARGETS))]
    x=np.stack([ablation_numeric_features(load_view(mapping,i),metadata[i]) for i in training])
    z=np.stack([ablation_numeric_features(load_view(mapping,i),metadata[i]) for i in evaluation])
    component['class_prior']=[]; trees=[]; priors=[]
    if x.shape[1]: component['pose_motion_tree']=[]
    for j,c in enumerate(CLASSES):
        prior=np.bincount(Y[training,j],minlength=len(c))/len(training)
        priors.append(prior.tolist())
        component['class_prior'].append(np.tile(prior,(len(evaluation),1)))
        if x.shape[1]:
            tree=CatBoostClassifier(iterations=30 if QUICK_SMOKE else 600,depth=4,learning_rate=.035,
                l2_leaf_reg=8,loss_function='MultiClass',verbose=False,random_seed=SEED,thread_count=4)
            tree.fit(x,Y[training,j]); prob=np.zeros((len(evaluation),len(c)))
            prob[:,np.asarray(tree.classes_,dtype=int)]=tree.predict_proba(z)
            component['pose_motion_tree'].append(prob); trees.append(tree)
    if trees: joblib.dump(trees,folder/'pose_camera_trees.joblib')
    # Training-only class prior persisted for inference; inference never needs labels to rebuild it.
    (folder/'class_prior.json').write_text(json.dumps(dict(prior=priors,class_order=CLASSES,
        training_clip_count=len(training),source='training_labels_at_fit',
        training_clips_sha=stable_hash(train['clip'].iloc[np.asarray(training)].tolist())),indent=2))
    provenance=dict(training_indices=list(map(int,training)),training_sources=sorted(set(rows.source_id.iloc[training])),
        evaluation_indices=list(map(int,evaluation)),checkpoint_selection_sources=[] if validation is None else
        sorted(set(rows.source_id.iloc[validation])),selected_epochs=selected,
        class_order=dict(zip(TARGETS,CLASSES)),features=numeric_feature_manifest(),
        components={name:dict(fit_sources=sorted(set(rows.source_id.iloc[training]))) for name in component},
        scaler_fit_sources=sorted(set(rows.source_id.iloc[training])),
        role_provenance_file=str(folder/'provenance.json'))
    public=np.asarray(evaluation)[np.asarray(evaluation)<len(train)]
    assert not set(rows.source_id.iloc[training]) & set(rows.source_id.iloc[public])
    (folder/'component_provenance.json').write_text(json.dumps(provenance,indent=2))
    return component,folder,mapping,selected


def select_diagnostic_mixture(components,truth):
    names=['video_ensemble']+(['pose_motion_tree'] if 'pose_motion_tree' in components else [])+['class_prior']
    grid=[np.asarray(x,dtype=float)/4 for x in itertools.product(range(5),repeat=len(names)) if sum(x)==4]
    weights=[]
    for j in range(len(TARGETS)):
        weights.append(max(grid,key=lambda w:(float((sum(v*components[n][j] for v,n in zip(w,names)).argmax(1)==truth[:,j]).mean()),
                                               -float(np.square(w).sum()))))
    return names,weights


def _nested_context(variant):
    """Switch RUN/STAGES/CFG to the variant's nested experiment; returns the restore tuple and split plan."""
    global RUN,STAGES,CFG
    saved=(RUN,STAGES,CFG)
    set_active_ablation(variant)
    plan=saved_diagnostic_splits()
    CFG=dict(CFG,diagnostics_version='nested-components-v1',split_id=plan['split_id'],
             ablation=variant,feature_manifest=numeric_feature_manifest(),gradient_accumulation=GRAD_ACCUM)
    if variant!='production':
        RUN=saved[0]/'ablations'/variant; RUN.mkdir(parents=True,exist_ok=True)
    STAGES=RUN/'stages'; STAGES.mkdir(exist_ok=True)
    fold_ids=np.full(len(train),-1,dtype=int)
    for item in plan['folds']: fold_ids[item['validation']]=item['fold']
    # Identical content from every fold job; atomic unique-temp writes never interleave.
    atomic_write_text(RUN/'folds.csv',train[['clip','source_id']].assign(fold=fold_ids).to_csv(index=False))
    atomic_write_text(RUN/'experiment_manifest.json',json.dumps(dict(config=CFG,split_plan=plan,
        classes=dict(zip(TARGETS,CLASSES)),seeds=VIDEO_SEEDS,experiment_identity=stable_hash(CFG),
        interpretation='Exploratory outer-fold diagnostics; repeated inspection is development feedback, not a final untouched estimate.'),indent=2))
    return saved,plan,fold_ids


def _restore_context(saved):
    global RUN,STAGES,CFG
    RUN,STAGES,CFG=saved
    set_active_ablation('production')


def fold_output_dir(fold):
    return RUN/'folds'/f'fold{int(fold)}'


def run_nested_fold(item,private_inference=False):
    """Inner selection, then the outer refit, for ONE saved fold. Writes only folds/fold<k>/ and its stages."""
    fold=int(item['fold']); out=fold_output_dir(fold)
    if (out/'COMPLETE').exists():
        result=json.loads((out/'result.json').read_text())
        if result['experiment_identity']!=stable_hash(CFG) or result['private_inference']!=bool(private_inference):
            raise ValueError(f'{out} was produced by a different configuration; it will not be mixed into this run.')
        print(f'Fold {fold}: already complete ({out}).'); return result
    out.mkdir(parents=True,exist_ok=True)
    with RunWriterLock(RUN,f'fold{fold}'):
        tr,va,inner,hold=[np.asarray(item[k],dtype=int) for k in
                          ['training','validation','inner_training','inner_validation']]
        ip,inner_folder,_,selected=diagnostic_stage(inner,hold,f'fold{fold}_inner',validation=hold)
        references=select_reference_components(ip,Y[hold],TARGETS,CLASSES,WEIGHTS)
        names,weights=select_diagnostic_mixture(ip,Y[hold])
        evaluation=np.r_[va,np.arange(len(train),len(rows))] if private_inference else va
        # Epoch counts are fixed using INNER validation before training on all outer-training sources.
        policy=hashlib.sha256(json.dumps(selected,sort_keys=True).encode()).hexdigest()[:8]
        op,folder,mapping,_=diagnostic_stage(tr,evaluation,f'fold{fold}_outer_{policy}',selected_epochs=selected)
        op['ensemble']=[sum(w*op[n][j] for w,n in zip(weights[j],names)) for j in range(len(TARGETS))]
        collect_tracking_diagnostics(va,mapping,fold,output_dir=RUN/f'tracking_fold{fold}')
        acc=[float((op['ensemble'][j][:len(va)].argmax(1)==Y[va,j]).mean()) for j in range(len(TARGETS))]
        result=dict(fold=fold,stage_folder=str(folder),inner_stage_folder=str(inner_folder),
            seeds=list(VIDEO_SEEDS),selected_epochs=selected,ensemble_components=names,
            weights=[w.tolist() for w in weights],reference_components=references,accuracy=acc,
            score=float(WEIGHTS@acc),selection_sources=sorted(set(train.source_id.iloc[hold])),
            fit_sources=sorted(set(train.source_id.iloc[tr])),outer_sources=sorted(set(train.source_id.iloc[va])),
            experiment_identity=stable_hash(CFG),private_inference=bool(private_inference),attempt=ATTEMPT_ID)
        assert not set(result['selection_sources']) & set(result['outer_sources'])
        arrays={f'{n}__task{j}':p[j][:len(va)] for n,p in op.items() for j in range(len(TARGETS))}
        if private_inference:
            arrays.update({f'private_ensemble__task{j}':op['ensemble'][j][len(va):] for j in range(len(TARGETS))})
        tmp=out/f'.component_predictions.{os.getpid()}.npz'
        np.savez_compressed(tmp,indices=va,clips=train['clip'].iloc[va].to_numpy(dtype=str),**arrays)
        tmp.replace(out/'component_predictions.npz')
        atomic_write_text(out/'result.json',json.dumps(result,indent=2))
        atomic_write_text(out/'COMPLETE',stable_hash(CFG))   # published last
        print(f'Fold {fold}: weighted accuracy={WEIGHTS@acc:.4f}; selected epochs={selected}')
        return result


def aggregate_nested(plan,fold_ids,private_inference=False):
    """Pool completed folds. Refuses while any fold is missing or was produced by another configuration."""
    missing=[f['fold'] for f in plan['folds'] if not (fold_output_dir(f['fold'])/'COMPLETE').exists()]
    if missing:
        print(f'Aggregation waits for folds {missing}; pooled reports are not written yet.'); return None
    reports=[json.loads((fold_output_dir(f['fold'])/'result.json').read_text()) for f in plan['folds']]
    for r in reports:
        if r['experiment_identity']!=stable_hash(CFG):
            raise ValueError(f'Fold {r["fold"]} has identity {r["experiment_identity"]}, expected {stable_hash(CFG)}.')
        if r['private_inference']!=bool(private_inference):
            raise ValueError(f'Fold {r["fold"]} private_inference={r["private_inference"]}; rerun it or change PREDICT_PRIVATE.')
    with RunWriterLock(RUN,'aggregate'):
        registry=ComponentOOF(train,Y,fold_ids,TARGETS,CLASSES,WEIGHTS)
        references={}; outer_mapping={}; quality=[]; frames=[]; click_rows=[]
        private=[np.zeros((len(test),len(c))) for c in CLASSES] if private_inference else None
        for item,r in zip(plan['folds'],reports):
            fold=int(item['fold']); va=np.asarray(item['validation'],dtype=int)
            references[fold]=r['reference_components']
            with np.load(fold_output_dir(fold)/'component_predictions.npz') as saved:
                if not np.array_equal(saved['indices'],va): raise ValueError(f'Fold {fold} predictions are misaligned')
                names=sorted({k.rsplit('__task',1)[0] for k in saved.files if '__task' in k and not k.startswith('private_')})
                for name in names:
                    registry.add(name,va,[saved[f'{name}__task{j}'] for j in range(len(TARGETS))],train['clip'].iloc[va].tolist(),CLASSES)
                if private_inference:
                    for j in range(len(TARGETS)): private[j]+=saved[f'private_ensemble__task{j}']/len(plan['folds'])
            folder=RUN/'stages'/Path(r['stage_folder']).name
            outer_mapping.update({int(i):folder/f'{int(i)}.pkl' for i in va})
            tdir=RUN/f'tracking_fold{fold}'
            quality.append(pd.read_csv(tdir/'tracking_quality.csv'))
            frames.append(pd.read_csv(tdir/'tracking_frames.csv'))
            try: click_rows.append(pd.read_csv(tdir/'role_click_audit.csv'))
            except pd.errors.EmptyDataError: pass
        atomic_write_text(RUN/'fold_results.json',json.dumps(reports,indent=2))
        q=pd.concat(quality,ignore_index=True)
        q.to_csv(RUN/'tracking_quality.csv',index=False)
        pd.concat(frames,ignore_index=True).to_csv(RUN/'tracking_frames.csv',index=False)
        combined_click_audits=pd.concat(click_rows,ignore_index=True) if click_rows else pd.DataFrame()
        combined_click_audits.to_csv(RUN/'role_click_audit.csv',index=False)
        summarize_role_click_audits(combined_click_audits).to_csv(RUN/'role_click_summary.csv',index=False)
        # Tracking exports carry source/fold keys too. Verify them against the saved
        # OOF assignment before removing duplicated columns for the clip-keyed join.
        expected=train[['clip','source_id']].assign(fold=fold_ids)
        if not q['clip'].is_unique or set(q['clip'])!=set(expected['clip']):
            raise ValueError('Tracking quality must cover every OOF clip exactly once.')
        aligned=expected.merge(q[['clip','source_id','fold']],on='clip',how='left',
                               suffixes=('_expected','_audit'),validate='one_to_one')
        if not ((aligned.source_id_expected==aligned.source_id_audit).all()
                and (aligned.fold_expected==aligned.fold_audit).all()):
            raise ValueError('Tracking quality source/fold provenance differs from the saved OOF plan.')
        exported=registry.export(RUN/'component_reports',
            quality=q.drop(columns=['source_id','fold']),references=references,ensemble='ensemble')
        # Standard artifact names retain compatibility with existing OOF diagnostics and heatmaps.
        oof=[np.full((len(train),len(c)),np.nan) for c in CLASSES]
        for item in plan['folds']:
            va=np.asarray(item['validation'])
            with np.load(fold_output_dir(item['fold'])/'component_predictions.npz') as saved:
                for j in range(len(TARGETS)): oof[j][va]=saved[f'ensemble__task{j}']
        tmp=RUN/f'.predictions.{os.getpid()}.npz'
        np.savez_compressed(tmp,fold_ids=fold_ids,
            **{f'oof_{j}':p for j,p in enumerate(oof)},
            **({f'private_{j}':p for j,p in enumerate(private)} if private is not None else {}))
        tmp.replace(RUN/'predictions.npz')
        export_run_inference_manifest(RUN)
        if GENERATE_TRACKING_AUDITS:
            # Export helper returns its errors table; use explicit ensemble rows only.
            errors=exported['errors']; errors=errors[errors.component=='ensemble']
            render_tracking_audits(errors,outer_mapping,RUN/'visual_audits',
                high_loss_count=AUDIT_HIGH_LOSS,correct_count=AUDIT_CORRECT,seed=SEED)
        return dict(oof=oof,private=private,fold_ids=fold_ids,reports=reports,exported=exported,run=str(RUN))


def run_nested_diagnostics(variant='production',private_inference=False,fold_selection=None):
    """fold_selection=None: every fold in this process, then aggregate. A list of fold numbers runs only
    those folds (one Slurm job per fold) and aggregates only once every fold is complete; [] aggregates only."""
    saved,plan,fold_ids=_nested_context(variant)
    try:
        selected=[f['fold'] for f in plan['folds']] if fold_selection is None else [int(f) for f in fold_selection]
        unknown=set(selected)-{f['fold'] for f in plan['folds']}
        if unknown: raise ValueError(f'FOLD_SELECTION has unknown folds {sorted(unknown)}')
        for item in plan['folds']:
            if item['fold'] in selected: run_nested_fold(item,private_inference)
        return aggregate_nested(plan,fold_ids,private_inference)
    finally:
        _restore_context(saved)


def run_optional_ablations():
    variants=list(ABLATION_VARIANTS)
    if len(set(variants))!=len(variants): raise ValueError('Duplicate ablation variants.')
    if any(name not in ABLATION_CONFIGS for name in variants): raise ValueError('Unknown ablation variant.')
    print(f'Ablation plan: {len(variants)} variants x {OUTER_FOLDS} folds x 2 stages x '
          f'{len(VIDEO_SEEDS)} seeds = {len(variants)*OUTER_FOLDS*2*len(VIDEO_SEEDS)} video fits (before cache reuse).')
    print('Numeric variants also fit two tabular specialists per stage; each stage fits cross-fitted role selectors.')
    if not RUN_ABLATIONS:
        print('Disabled. Set RUN_ABLATIONS=True only after inspecting diagnostics and the compute plan.'); return
    if QUICK_SMOKE: raise ValueError('Use the diagnostics smoke checks for cheap ablation checks; full ablations require QUICK_SMOKE=False.')
    if EVALUATION_MODE!='nested_cv': raise ValueError('Controlled ablations require the saved nested outer folds.')
    reference_path=RUN/'component_reports'/'component_metrics.csv'
    if not reference_path.exists(): raise ValueError('Finish the production nested diagnostics before ablations.')
    reference_manifest=json.loads((RUN/'experiment_manifest.json').read_text())
    split_id=saved_diagnostic_splits()['split_id']
    if reference_manifest['split_plan']['split_id']!=split_id:
        raise ValueError('Production report uses different saved folds; rerun production diagnostics first.')
    results={'production':dict(metrics=pd.read_csv(reference_path))}
    for variant in variants:
        if variant=='production': continue  # Main diagnostic run already supplies the reference.
        result=run_nested_diagnostics(variant,private_inference=False)
        candidate_manifest=json.loads((Path(result['run'])/'experiment_manifest.json').read_text())
        if candidate_manifest['split_plan']['split_id']!=split_id:
            raise ValueError('Ablation and production runs must use identical saved folds.')
        results[variant]=result['exported']
    # Comparison uses identical folds and reports both fold and pooled results.
    # This helper explicitly compares adjacent feature additions (and production
    # versus H), rather than treating every variant as a change from production.
    comparison=compare_ablation_metrics(results,RUN,component='ensemble')
    print('Saved matched-fold and pooled ablation deltas:',RUN/'ablation_comparisons.csv')
    if comparison.empty:
        print('No configured reference pairs were run; include both variants in a documented comparison pair.')
    return comparison
