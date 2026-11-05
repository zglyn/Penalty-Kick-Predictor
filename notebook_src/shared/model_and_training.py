# CELL: model_and_training
metadata=(np.column_stack([(rows.preferred_foot=='left'),(rows.preferred_foot=='right'),
                          (rows.gender=='F'),(rows.gender=='M')]).astype(np.float32) if RUN_MODE!='reports' else None)

def load_view(mapping,i,decode_video=False):
    with Path(mapping[int(i)]).open('rb') as f: value=pickle.load(f)
    if decode_video:
        value['video']=np.stack([[decode_jpeg(frame).transpose(2,0,1) for frame in stream]
                                for stream in value['video_jpeg']])
    return value

def numeric_summary(value):
    x=value['numeric']
    return np.r_[x.mean(0),x.std(0),x[-1],x[-1]-x[0]]

class ClipDataset(torch.utils.data.Dataset):
    def __init__(self,indices,mapping,scaler,augment=False):
        self.indices=list(indices); self.mapping=mapping; self.scaler=scaler; self.augment=augment
    def __len__(self): return len(self.indices)
    def __getitem__(self,index):
        i=self.indices[index]; d=load_view(self.mapping,i,decode_video=True)
        x=d['video'].astype(np.float32)/255.
        if self.augment:
            gain=np.random.uniform(.85,1.15); shift=np.random.uniform(-.04,.04)
            x=np.clip(x*gain+shift,0,1)
        x=(x-IMAGE_MEAN[None,None,:,None,None])/IMAGE_STD[None,None,:,None,None]
        n=ablation_numeric_features(d,metadata[i])[None]
        n=self.scaler.transform(n)[0].astype(np.float32)
        y=Y[i] if i<len(train) else np.zeros(len(TARGETS),dtype=np.int64)
        return torch.from_numpy(x),torch.from_numpy(n),torch.tensor(y,dtype=torch.long)

# MultiViewVideo is created at the end of this cell from make_ablation_model_class(); the
# backbone (VideoMAE / V-JEPA 2.1 / V-JEPA 2) comes from the adapter selected in Run settings.

def direction_loss(logits, labels):
    assert len(logits) == len(TARGETS) == 2
    assert labels.ndim == 2 and labels.shape[1] == 2
    return (
        float(WEIGHTS[0]) * torch.nn.functional.cross_entropy(logits[0], labels[:, 0])
        + float(WEIGHTS[1]) * torch.nn.functional.cross_entropy(logits[1], labels[:, 1])
    )

@torch.inference_mode()
def evaluate_video_loss(model,scaler,indices,mapping):
    model.eval(); total=np.zeros(len(TARGETS)); count=0
    loader=torch.utils.data.DataLoader(ClipDataset(indices,mapping,scaler),batch_size=1,num_workers=0)
    for v,n,y in loader:
        with torch.autocast('cuda',dtype=torch.float16): logits=model(v.to(DEVICE),n.to(DEVICE))
        for j,p in enumerate(logits):
            total[j]+=torch.nn.functional.cross_entropy(p.float(),y[:,j].to(DEVICE),reduction='sum').item()
        count+=len(y)
    if not count: raise ValueError('Cannot evaluate an empty split')
    losses=total/count
    if not np.isfinite(losses).all(): raise FloatingPointError('Non-finite evaluation loss')
    return dict(loss=float(WEIGHTS@losses),**{t+'_loss':float(x) for t,x in zip(TARGETS,losses)})

def update_loss_plot(history,folder,seed,handle=None):
    # Update just this figure, preserving progress bars and earlier cell output.
    import matplotlib.pyplot as plt
    from IPython.display import display
    from matplotlib.ticker import MaxNLocator
    data=pd.DataFrame(history)
    with plt.ioff():
        fig,ax=plt.subplots(figsize=(9,5))
        ax.plot(data.epoch,data.train_loss,marker='o',label='Training')
        ax.plot(data.epoch,data.validation_loss,marker='o',linestyle='--',label='Validation')
        best=data.loc[data.validation_loss.idxmin()]
        ax.scatter([best.epoch],[best.validation_loss],s=110,marker='*',color='green',
                   zorder=5,label=f'Best validation: epoch {int(best.epoch)}')
        ax.set(xlabel='Epoch',ylabel='Weighted cross-entropy (70% kick / 30% keeper)',
               title=f'Seed {seed} | epoch {int(data.epoch.iloc[-1])}/{VIDEO_EPOCHS} | '
                     f'train {data.train_loss.iloc[-1]:.4f} | validation {data.validation_loss.iloc[-1]:.4f}')
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.legend(); ax.grid(alpha=.25); fig.tight_layout()
        try:
            fig.savefig(folder/f'loss_curves_seed{seed}.png',dpi=150)
            if handle is None: handle=display(fig,display_id=True)
            else: handle.update(fig)
        finally:
            plt.close(fig)
    return handle

def atomic_torch_save(state,path):
    # A failed write leaves the last complete checkpoint untouched.
    path=Path(path); temporary=path.with_name(f'.{path.name}.{os.getpid()}.tmp')  # unique per writer process
    with temporary.open('wb') as stream:
        torch.save(state,stream)
        stream.flush(); os.fsync(stream.fileno())
    temporary.replace(path)

def capture_training_rng():
    numpy_state=np.random.get_state()
    return dict(python=random.getstate(),numpy=dict(name=numpy_state[0],
        keys=numpy_state[1].tolist(),position=numpy_state[2],has_gauss=numpy_state[3],
        cached_gaussian=numpy_state[4]),torch=torch.get_rng_state(),
        cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [])

def restore_training_rng(state):
    random.setstate(state['python'])
    n=state['numpy']
    np.random.set_state((n['name'],np.asarray(n['keys'],dtype=np.uint32),
                         n['position'],n['has_gauss'],n['cached_gaussian']))
    torch.set_rng_state(state['torch'])
    if state['cuda'] and torch.cuda.is_available(): torch.cuda.set_rng_state_all(state['cuda'])

def fit_video(indices,mapping,folder,seed,validation_indices=None,epochs_override=None):
    planned_epochs=VIDEO_EPOCHS if epochs_override is None else int(epochs_override)
    assert planned_epochs>0
    checkpoint=folder/f'video_seed{seed}.pt'; scaler_path=folder/'numeric_scaler.joblib'
    resume_path=folder/f'video_seed{seed}_latest.pt'
    training_manifest=dict(seed=int(seed),training_indices=list(map(int,indices)),
        training_sources=sorted(set(rows.source_id.iloc[indices])),
        validation_indices=[] if validation_indices is None else list(map(int,validation_indices)),
        feature_manifest=numeric_feature_manifest(),streams=list(ablation_stream_ids()),
        planned_epochs=planned_epochs,gradient_accumulation=GRAD_ACCUM,
        config=CFG)
    manifest_path=folder/f'video_seed{seed}_fit_manifest.json'
    if manifest_path.exists():
        assert json.loads(manifest_path.read_text())==training_manifest, 'Checkpoint training provenance changed.'
    manifest_path.write_text(json.dumps(training_manifest,indent=2))
    torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)
    raw=np.stack([ablation_numeric_features(load_view(mapping,i),metadata[i]) for i in indices])
    scaler=fit_feature_scaler(raw); joblib.dump(scaler,scaler_path)
    model=MultiViewVideo(raw.shape[1]).to(DEVICE)
    serialization=dict(model.adapter.serialization(model.architecture),streams=list(ablation_stream_ids()),
                       parameters=trainable_parameter_report(model))
    (folder/'backbone_serialization.json').write_text(json.dumps(serialization,indent=2,default=str))
    optimizer=torch.optim.AdamW(apply_training_policy(model,TRAINING_POLICY),weight_decay=.05)
    amp=torch.amp.GradScaler('cuda')
    # Fine-tuning: a backbone parameter probe proves real encoder updates. Frozen probe: the encoder must
    # stay bit-identical while a head parameter changes.
    probe_name,probe=next((name,p) for name,p in model.named_parameters() if p.requires_grad and p.ndim>1
                          and (name.startswith('backbone.') or TRAINING_POLICY=='frozen_probe'))
    frozen_before=(next(p for p in model.backbone.parameters() if p.ndim>1).detach().float().cpu().clone()
                   if QUICK_SMOKE and TRAINING_POLICY=='frozen_probe' else None)
    probe_before=probe.detach().float().cpu().clone() if QUICK_SMOKE else None
    history=[]; best_validation=float('inf'); best_state=None; best_epoch=None; loss_plot=None
    start_epoch=0
    updates=0; gradient_observed=False; losses=[]
    loader=torch.utils.data.DataLoader(ClipDataset(indices,mapping,scaler,True),batch_size=1,shuffle=True,num_workers=0)
    batch_limit=min(len(loader),SMOKE_MAX_BATCHES) if QUICK_SMOKE else len(loader)
    if resume_path.exists() and not QUICK_SMOKE:
        saved=torch.load(resume_path,map_location='cpu',weights_only=True)
        assert saved['format_version']==1 and saved['seed']==seed
        assert saved['training_indices']==list(map(int,indices))
        assert saved['validation_indices']==([] if validation_indices is None else list(map(int,validation_indices)))
        assert saved['planned_epochs']==planned_epochs, 'Keep the same epoch setting when resuming.'
        saved_identity=saved.get('model_identity',LEGACY_IDENTITY)  # format 1 checkpoints predate the field
        if saved_identity!=BACKBONE_IDENTITY:
            raise ValueError(f'Checkpoint {resume_path} was trained with {saved_identity}; current settings are '
                             f'{BACKBONE_IDENTITY}. Backbone/policy changes must start a new run.')
        model.load_state_dict(saved['model']); optimizer.load_state_dict(saved['optimizer'])
        amp.load_state_dict(saved['amp'])
        start_epoch=int(saved['completed_epochs']); history=saved['history']
        best_validation=saved['best_validation']; best_state=saved['best_model']; best_epoch=saved['best_epoch']
        updates=saved['optimizer_updates']
        assert len(history)==start_epoch and 0<=start_epoch<=planned_epochs
        pd.DataFrame(history).to_csv(folder/f'loss_history_seed{seed}.csv',index=False)
        if validation_indices is not None and history:
            loss_plot=update_loss_plot(history,folder,seed,loss_plot)
        restore_training_rng(saved['rng'])
        del saved
        print(f'Restored seed {seed}: {start_epoch}/{planned_epochs} epochs complete. '
              + (f'Resuming at epoch {start_epoch+1}.' if start_epoch<planned_epochs else 'Training already complete.'))
    elif checkpoint.exists() and not QUICK_SMOKE:
        print(f'Seed {seed}: old weights-only checkpoint has no optimizer state; starting a fresh training run.')
    torch.cuda.synchronize()
    training_started=time.perf_counter()
    for epoch in range(start_epoch,planned_epochs):
        model.train(); optimizer.zero_grad(); total=0.
        for step,(v,n,y) in enumerate(tqdm(loader,desc=f'Video seed {seed} epoch {epoch+1}',leave=False)):
            v,n,y=v.to(DEVICE),n.to(DEVICE),y.to(DEVICE)
            with torch.autocast('cuda',dtype=torch.float16):
                logits=model(v,n)
                raw_loss=direction_loss(logits,y)
                # Correct normalization for the final short accumulation group.
                group_size=min(GRAD_ACCUM,batch_limit-(step//GRAD_ACCUM)*GRAD_ACCUM)
                loss=raw_loss/group_size
            if not torch.isfinite(raw_loss): raise FloatingPointError('Non-finite video training loss')
            amp.scale(loss).backward(); total+=raw_loss.detach().item(); losses.append(raw_loss.detach().item())
            if (step+1)%GRAD_ACCUM==0 or step+1==batch_limit:
                amp.unscale_(optimizer)
                if QUICK_SMOKE:
                    if probe.grad is not None:
                        gradient_observed |= bool(torch.isfinite(probe.grad).all() and probe.grad.abs().sum()>0)
                torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
                scale_before=amp.get_scale()
                amp.step(optimizer); amp.update()
                if amp.get_scale()>=scale_before: updates+=1
                optimizer.zero_grad()
            if step+1>=batch_limit: break
        record=dict(epoch=epoch+1,optimization_loss=total/batch_limit)
        if validation_indices is not None:
            # Eval-mode, unaugmented training loss is comparable to validation loss.
            for label,ids in [('train',indices),('validation',validation_indices)]:
                record.update({label+'_'+k:v for k,v in evaluate_video_loss(model,scaler,ids,mapping).items()})
            if record['validation_loss'] < best_validation:
                best_validation=record['validation_loss']
                best_state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
                best_epoch=epoch+1
        history.append(record)
        # The latest checkpoint is authoritative for history and best-model selection.
        # Store the best weights here too so an interrupted save cannot mismatch epochs.
        atomic_torch_save(dict(format_version=1,seed=seed,planned_epochs=planned_epochs,
            completed_epochs=epoch+1,training_indices=list(map(int,indices)),
            validation_indices=[] if validation_indices is None else list(map(int,validation_indices)),
            model={k:v.detach().cpu() for k,v in model.state_dict().items()},
            optimizer=optimizer.state_dict(),amp=amp.state_dict(),rng=capture_training_rng(),
            history=history,best_validation=best_validation,best_epoch=best_epoch,
            best_model=best_state,optimizer_updates=updates,model_identity=BACKBONE_IDENTITY),resume_path)
        print(f'Checkpoint saved: seed {seed}, epoch {epoch+1}/{planned_epochs} -> {resume_path}')
        pd.DataFrame(history).to_csv(folder/f'loss_history_seed{seed}.csv',index=False)
        print(record)
        if validation_indices is not None:
            loss_plot=update_loss_plot(history,folder,seed,loss_plot)
    torch.cuda.synchronize()
    training_seconds=time.perf_counter()-training_started
    if QUICK_SMOKE:
        delta=float((probe.detach().float().cpu()-probe_before).abs().max())
        frozen_delta=(float((next(p for p in model.backbone.parameters() if p.ndim>1).detach().float().cpu()
                             -frozen_before).abs().max()) if frozen_before is not None else None)
        checks=dict(training_policy=TRAINING_POLICY,backbone=BACKBONE_SPEC['name'],probe_parameter=probe_name,
                    probe_max_change=delta,frozen_backbone_max_change=frozen_delta,
                    finite_nonzero_probe_gradient=gradient_observed,optimizer_updates=updates,
                    trainable_parameters=trainable_parameter_report(model),
                    batches=len(losses),losses=losses,training_seconds=training_seconds,
                    seconds_per_training_batch=training_seconds/len(losses))
        (folder/'video_training_checks.json').write_text(json.dumps(checks,indent=2))
        assert gradient_observed and updates>0 and delta>0, f'Training update check failed: {checks}'
        assert frozen_delta in (None,0.0), f'Frozen encoder changed: {checks}'
    if validation_indices is None:
        atomic_torch_save(dict(model={k:v.detach().cpu() for k,v in model.state_dict().items()}),checkpoint)
    else:
        assert best_state is not None
        atomic_torch_save(dict(model=best_state,best_epoch=best_epoch,validation_loss=best_validation),checkpoint)
        model.load_state_dict(best_state)
    return model,scaler

@torch.inference_mode()
def predict_video(model,scaler,indices,mapping):
    model.eval(); out=[[] for _ in TARGETS]
    loader=torch.utils.data.DataLoader(ClipDataset(indices,mapping,scaler),batch_size=1,num_workers=0)
    torch.cuda.synchronize(); started=time.perf_counter()
    for v,n,_ in tqdm(loader,desc='Video inference',leave=False):
        with torch.autocast('cuda',dtype=torch.float16): pred=model(v.to(DEVICE),n.to(DEVICE))
        for j,p in enumerate(pred): out[j].append(p.float().softmax(1).cpu().numpy())
    torch.cuda.synchronize()
    model.last_inference_seconds_per_clip=(time.perf_counter()-started)/len(indices)
    return [np.concatenate(p) for p in out]

def fit_predict_stage(training,evaluation,name):
    folder,mapping,selector=build_stage(training,evaluation,name)
    neural=[np.zeros((len(evaluation),len(c))) for c in CLASSES]
    for seed in VIDEO_SEEDS:
        model,scaler=fit_video(training,mapping,folder,seed)
        for j,p in enumerate(predict_video(model,scaler,evaluation,mapping)): neural[j]+=p/len(VIDEO_SEEDS)
        del model; gc.collect(); torch.cuda.empty_cache()
    x=np.stack([ablation_numeric_features(load_view(mapping,i),metadata[i]) for i in training])
    z=np.stack([ablation_numeric_features(load_view(mapping,i),metadata[i]) for i in evaluation])
    predictions=[]; trees=[]
    for j,c in enumerate(CLASSES):
        tree=CatBoostClassifier(iterations=30 if QUICK_SMOKE else 600,depth=4,learning_rate=.035,l2_leaf_reg=8,
             loss_function='MultiClass',verbose=False,random_seed=42,thread_count=4)
        tree.fit(x,Y[training,j]); prob=np.zeros((len(evaluation),len(c)))
        prob[:,np.asarray(tree.classes_,dtype=int)]=tree.predict_proba(z)
        prior=np.bincount(Y[training,j],minlength=len(c))/len(training)
        predictions.append([neural[j],prob,np.tile(prior,(len(evaluation),1))]); trees.append(tree)
    joblib.dump(trees,folder/'pose_camera_trees.joblib')
    return predictions,folder,mapping,selector

def select_mixture(predictions,truth):
    grid=[np.array([a,b,4-a-b])/4 for a in range(5) for b in range(5-a)]
    return max(grid,key=lambda w:(accuracy_score(truth,sum(x*p for x,p in zip(w,predictions)).argmax(1)),-np.square(w).sum()))


# Stream-specific heads are retrained for each ablation; production keeps the original architecture.
MultiViewVideo=make_ablation_model_class()
