# CELL: smoke_holdout_dispatch

def audit_localization(indices,mapping):
    report=[]
    for i in indices:
        clip=rows['clip'].iloc[i]
        if clip not in clicks.index: continue
        d=load_view(mapping,i); raw=extract_raw(clip)
        for r,name in enumerate(['kicker','keeper']):
            row=clicks.loc[clip]; t=float(row[f'{name}_click_time_seconds'])
            k=np.abs(raw['times']-t).argmin(); predicted=d['traces'][k,r]
            truth=row[[f'{name}_x_norm',f'{name}_y_norm']].to_numpy(dtype=float)
            report.append(dict(clip=clip,role=name,detected=bool(predicted[2]>0),
                distance=float(np.linalg.norm(predicted[:2]-truth)) if predicted[2]>0 else np.nan,
                sample_offset_seconds=float(abs(raw['times'][k]-t))))
    return report

def run_smoke():
    import time
    started=time.monotonic()
    folder,mapping,selector=build_stage(smoke_train,smoke_hold,'end_to_end_smoke')
    # Verify saved cross-fitting provenance against the actual prediction-source IDs.
    provenance=json.loads((folder/'provenance.json').read_text())
    for entry in provenance:
        prediction_sources=set(rows.source_id.iloc[entry['prediction_indices']])
        assert not prediction_sources & set(entry['fit_sources'])
    model,scaler=fit_video(smoke_train,mapping,folder,VIDEO_SEEDS[0])
    probabilities=predict_video(model,scaler,smoke_hold,mapping)
    for j,prob in enumerate(probabilities):
        assert prob.shape==(len(smoke_hold),len(CLASSES[j]))
        assert np.isfinite(prob).all() and np.allclose(prob.sum(1),1,atol=1e-5)
    # Reload the serialized model and scaler, and check inference reproducibility.
    checkpoint=folder/f'video_seed{VIDEO_SEEDS[0]}.pt'
    model.load_state_dict(torch.load(checkpoint,map_location=DEVICE,weights_only=True)['model'])
    reloaded_scaler=joblib.load(folder/'numeric_scaler.joblib')
    repeated=predict_video(model,reloaded_scaler,smoke_hold[:1],mapping)
    for j in range(len(TARGETS)): np.testing.assert_allclose(probabilities[j][:1],repeated[j],rtol=1e-3,atol=1e-4)
    timing_estimate=benchmark_video(model,scaler,smoke_train,mapping) if RUN_TIMING_BENCHMARK else None
    del model; gc.collect(); torch.cuda.empty_cache()
    # Exercise the numeric specialist and probability-ensemble interface too.
    x=np.stack([np.r_[numeric_summary(load_view(mapping,i)),metadata[i]] for i in smoke_train])
    z=np.stack([np.r_[numeric_summary(load_view(mapping,i)),metadata[i]] for i in smoke_hold])
    ensemble=[]
    for j,classes in enumerate(CLASSES):
        if len(np.unique(Y[smoke_train,j]))>1:
            tree=CatBoostClassifier(iterations=20,depth=2,verbose=False,random_seed=42,thread_count=4,loss_function='MultiClass')
        else:
            tree=DummyClassifier(strategy='prior')
        tree.fit(x,Y[smoke_train,j])
        p=np.zeros((len(smoke_hold),len(classes)))
        p[:,np.asarray(tree.classes_,dtype=int)]=tree.predict_proba(z)
        mixture=.5*probabilities[j]+.5*p
        assert np.isfinite(mixture).all() and np.allclose(mixture.sum(1),1,atol=1e-5)
        ensemble.append(mixture)
    preview=pd.DataFrame({'clip':rows['clip'].iloc[smoke_hold].to_numpy()})
    for j,(target,classes) in enumerate(zip(TARGETS,CLASSES)):
        preview[target]=np.array(classes)[ensemble[j].argmax(1)]
    assert not preview.isna().any().any()
    preview.to_csv(folder/'smoke_public_predictions.csv',index=False)
    training_checks=json.loads((folder/'video_training_checks.json').read_text())
    coverage=np.stack([load_view(mapping,i)['traces'][:,:,2]>0 for i in smoke_hold]).mean(axis=(0,1))
    report=dict(status='PASS',train_clips=len(smoke_train),heldout_clips=len(smoke_hold),
                train_sources=4,heldout_sources=2,role_counts=selector['counts'],
                heldout_role_detection_fraction=dict(zip(['kicker','keeper'],coverage.tolist())),
                elapsed_seconds=time.monotonic()-started,training=training_checks,
                timing=timing_estimate,
                checked=['role fit','source-cross-fitted crops','backbone gradients and updates',
                         'two-target inference','checkpoint reload','numeric specialist','ensemble'],
                note='Execution check only. Tiny-subset accuracy is not a performance estimate.')
    (RUN/'smoke_report.json').write_text(json.dumps(report,indent=2))
    print('END-TO-END SMOKE TEST: PASS'); print(json.dumps(report,indent=2)); display(preview)
    print('No competition CSV generated. Set QUICK_SMOKE=False, then restart and run all for full training.')

def probability_metrics(probabilities,indices):
    result={}; weighted=0.; score=0.
    for j,target in enumerate(TARGETS):
        p=probabilities[j]; truth=Y[indices,j]
        loss=float(-np.log(np.clip(p[np.arange(len(indices)),truth],1e-12,1)).mean())
        accuracy=float((p.argmax(1)==truth).mean())
        result[target+'_loss']=loss; result[target+'_accuracy']=accuracy
        weighted+=float(WEIGHTS[j])*loss; score+=float(WEIGHTS[j])*accuracy
    return dict(loss=weighted,weighted_accuracy=score,**result)

def run_holdout():
    import matplotlib.pyplot as plt
    training=split_indices['training']; validation=split_indices['validation']; testing=split_indices['test']
    (RUN/'holdout_manifest.json').write_text(json.dumps(dict(config=dict(CFG,ablation=ACTIVE_ABLATION_NAME,
        feature_manifest=numeric_feature_manifest()),seeds=VIDEO_SEEDS,
        interpretation='Holdout validation selects checkpoints: a tuning metric, not an untouched estimate.'),indent=2))
    folder,mapping,_=build_stage(training,np.r_[validation,testing],'train_validation_test')
    predictions={name:[np.zeros((len(ids),len(c))) for c in CLASSES]
                 for name,ids in [('validation',validation),('test',testing)]}
    histories=[]
    for seed in VIDEO_SEEDS:
        model,scaler=fit_video(training,mapping,folder,seed,validation_indices=validation)
        history=pd.read_csv(folder/f'loss_history_seed{seed}.csv'); history['seed']=seed; histories.append(history)
        # Test labels never select epochs, weights, scalers, crops, or hyperparameters.
        for name,ids in [('validation',validation),('test',testing)]:
            for j,p in enumerate(predict_video(model,scaler,ids,mapping)):
                predictions[name][j]+=p/len(VIDEO_SEEDS)
        del model; gc.collect(); torch.cuda.empty_cache()
    history=pd.concat(histories,ignore_index=True); history.to_csv(RUN/'loss_history.csv',index=False)
    fig,ax=plt.subplots(figsize=(9,5))
    for seed,h in history.groupby('seed'):
        ax.plot(h.epoch,h.train_loss,marker='o',label=f'Train / seed {seed}')
        ax.plot(h.epoch,h.validation_loss,marker='o',linestyle='--',label=f'Validation / seed {seed}')
    ax.set(xlabel='Epoch',ylabel='Weighted cross-entropy (70% kick / 30% keeper)',title='Training and validation loss')
    ax.legend(); ax.grid(alpha=.25); fig.tight_layout(); fig.savefig(RUN/'loss_curves.png',dpi=150)
    display(fig); plt.close(fig)
    metrics=pd.DataFrame([dict(split=name,clips=len(ids),**probability_metrics(predictions[name],ids))
        for name,ids in [('validation',validation),('test',testing)]])
    metrics.to_csv(RUN/'holdout_metrics.csv',index=False); display(metrics)
    np.savez_compressed(RUN/'holdout_predictions.npz',validation_indices=validation,test_indices=testing,
        **{f'{name}_{j}':p for name,ps in predictions.items() for j,p in enumerate(ps)})
    export_run_inference_manifest(RUN)
    print('Best validation checkpoint per seed restored. Test loss is a final assessment, not an epoch tuning curve.')
    print('Saved split manifest, loss history, curves, and final metrics to:',RUN)

def development_split():
    """Fixed, label-free, source-disjoint development subset (seeded source permutation). Saved once and reused;
    never re-drawn after looking at scores."""
    sources=np.array(sorted(train.source_id.unique()))
    order=np.random.default_rng(SEED).permutation(sources)
    n_train=max(ROLE_FOLDS,int(round(DEV_TRAIN_SOURCE_FRACTION*len(sources))))
    n_val=max(1,int(round(DEV_VALIDATION_SOURCE_FRACTION*len(sources))))
    if n_train+n_val>len(sources): raise ValueError('Development fractions leave no disjoint validation sources.')
    training=np.flatnonzero(train.source_id.isin(order[:n_train]).to_numpy())
    validation=np.flatnonzero(train.source_id.isin(order[n_train:n_train+n_val]).to_numpy())
    for j,c in enumerate(CLASSES):
        if len(np.unique(Y[training,j]))!=len(c):
            raise ValueError(f'Development training subset lacks a {TARGETS[j]} class; change DEV_TRAIN_SOURCE_FRACTION '
                             '(a new, recorded split), not the seed after seeing results.')
    plan=dict(dataset=DATA_ID,seed=SEED,train_source_fraction=DEV_TRAIN_SOURCE_FRACTION,
              validation_source_fraction=DEV_VALIDATION_SOURCE_FRACTION,training=training.tolist(),validation=validation.tolist(),
              training_sources=sorted(order[:n_train].tolist()),validation_sources=sorted(order[n_train:n_train+n_val].tolist()))
    split_id='dev_'+stable_hash(plan)
    path=OUTPUT/'splits'/f'{split_id}.json'
    if path.exists():
        assert json.loads(path.read_text())['plan']==plan, 'Saved development split differs'
    else:
        atomic_write_text(path,json.dumps(dict(split_id=split_id,plan=plan),indent=2))
    return split_id,training,validation


def run_development():
    """Bounded preset: one seed, few epochs, fixed subset. DEVELOPMENT ONLY: the validation subset also selects
    the checkpoint, so its score is a selection-biased screening metric, never complete OOF performance."""
    with RunWriterLock(RUN,'run'):
        split_id,training,validation=development_split()
        split=train[['clip','source_id']].copy(); split['split']='unused'
        split.loc[training,'split']='development_training'; split.loc[validation,'split']='development_validation'
        split.to_csv(RUN/'development_split.csv',index=False)
        atomic_write_text(RUN/'development_manifest.json',json.dumps(dict(config=CFG,split_id=split_id,seeds=VIDEO_SEEDS,
            epochs=VIDEO_EPOCHS,training_clips=len(training),validation_clips=len(validation),
            training_sources=int(train.source_id.iloc[training].nunique()),validation_sources=int(train.source_id.iloc[validation].nunique()),
            evaluation_scope='DEVELOPMENT ONLY: bounded subset, inner-validation screening; not complete OOF performance',
            experiment_identity=EXPERIMENT_ID,attempt=ATTEMPT_ID),indent=2))
        folder,mapping,_=build_stage(training,validation,'development',selection=validation)
        averaged=[np.zeros((len(validation),len(c))) for c in CLASSES]
        for seed in VIDEO_SEEDS:
            model,scaler=fit_video(training,mapping,folder,seed,validation_indices=validation)
            for j,p in enumerate(predict_video(model,scaler,validation,mapping)): averaged[j]+=p/len(VIDEO_SEEDS)
            del model; gc.collect(); torch.cuda.empty_cache()
        metrics=pd.DataFrame([dict(split='development_validation',clips=len(validation),
            evaluation_scope='development_only_selection_biased',**probability_metrics(averaged,validation))])
        metrics.to_csv(RUN/'development_metrics.csv',index=False)
        np.savez_compressed(RUN/'development_predictions.npz',validation_indices=validation,
                            **{f'validation_{j}':p for j,p in enumerate(averaged)})
        display(metrics)
        print('DEVELOPMENT RESULT ONLY (subset, one seed, checkpoint chosen on this validation subset). '
              'Compare only with other development runs on split', split_id)
        return metrics


# ------------------------------ Execution routing (RUN_MODE from Settings) ------------------------------
inference_result=None; diagnostic_result=None
if RUN_MODE == 'reports':
    pass                                           # reports mode stops before this cell; definitions only
elif RUN_MODE == 'inference':
    inference_result=run_inference_mode()          # loads saved artifacts only; no training or selection
elif QUICK_SMOKE:
    with RunWriterLock(RUN,'run'): run_smoke()
elif EVALUATION_MODE=='holdout':
    with RunWriterLock(RUN,'run'): run_holdout()
elif EVALUATION_MODE=='development':
    development_result=run_development()
else:
    diagnostic_result=run_nested_diagnostics('production',private_inference=PREDICT_PRIVATE,fold_selection=FOLD_SELECTION)
    if diagnostic_result is None:
        print('Selected fold job(s) finished; the pooled reports are written by the job that completes the last fold.')
    else:
        oof=diagnostic_result['oof']; private=diagnostic_result['private']
        fold_ids=diagnostic_result['fold_ids']; reports=diagnostic_result['reports']
        display(diagnostic_result['exported']['metrics'])
        if FOLD_SELECTION is None: run_optional_ablations()
