# CELL: timing_benchmark

def full_video_workload():
    if EVALUATION_MODE=='holdout':
        ntrain=len(split_indices['training']); nval=len(split_indices['validation'])
        seeds=len(FULL_TRAIN_SETTINGS['seeds']); epochs=FULL_TRAIN_SETTINGS['epochs']
        return dict(training_batches=seeds*epochs*ntrain,
            inference_clips=seeds*(epochs*(ntrain+nval)+nval+len(split_indices['test'])),video_fits=seeds)
    # Maximum-budget estimate: inner validation can choose fewer outer epochs.
    batches=0; inference=0; fits=0
    for item in saved_diagnostic_splits()['folds']:
        ninner=len(item['inner_training']); nhold=len(item['inner_validation'])
        ntrain=len(item['training']); nval=len(item['validation'])
        seeds=len(FULL_TRAIN_SETTINGS['seeds']); epochs=FULL_TRAIN_SETTINGS['epochs']
        batches+=seeds*epochs*(ninner+ntrain)
        inference+=seeds*(epochs*(ninner+nhold)+nhold+nval+(len(test) if PREDICT_PRIVATE else 0))
        fits+=2*seeds
    return dict(training_batches=int(batches),inference_clips=int(inference),video_fits=int(fits))


def summarize_benchmark(training_seconds,data_seconds,inference_seconds,workload):
    training_seconds=np.asarray(training_seconds,dtype=float)
    data_seconds=np.asarray(data_seconds,dtype=float)
    inference_seconds=np.asarray(inference_seconds,dtype=float)
    assert len(training_seconds)>0 and len(inference_seconds)>0
    assert len(training_seconds)==len(data_seconds)
    assert np.isfinite(training_seconds).all() and np.isfinite(inference_seconds).all()
    assert np.isfinite(data_seconds).all() and (data_seconds>=0).all()
    assert (training_seconds>0).all() and (inference_seconds>0).all()
    assert (data_seconds<=training_seconds).all()
    train_hours=float(training_seconds.mean()*workload['training_batches']/3600)
    infer_hours=float(inference_seconds.mean()*workload['inference_clips']/3600)
    subtotal=train_hours+infer_hours
    return dict(workload=workload,measured_training_batches=len(training_seconds),
        measured_inference_clips=len(inference_seconds),
        seconds_per_training_batch=float(training_seconds.mean()),
        training_batch_p50_seconds=float(np.median(training_seconds)),
        training_batch_p90_seconds=float(np.quantile(training_seconds,.9)),
        seconds_per_inference_clip=float(inference_seconds.mean()),
        training_data_loading_fraction=float(data_seconds.sum()/training_seconds.sum()),
        training_hours=train_hours,inference_hours=infer_hours,video_subtotal_hours=subtotal,
        planning_with_25_percent_allowance_hours=subtotal*1.25,
        planning_with_50_percent_allowance_hours=subtotal*1.5)

def benchmark_video(model,scaler,training_indices,mapping):
    accumulation=FULL_TRAIN_SETTINGS['accumulation']
    assert TIMING_WARMUP_BATCHES>=accumulation and TIMING_WARMUP_BATCHES%accumulation==0
    assert TIMING_MEASURED_BATCHES>=accumulation and TIMING_MEASURED_BATCHES%accumulation==0
    assert TIMING_INFERENCE_CLIPS>0
    dataset=ClipDataset(training_indices,mapping,scaler,True)
    loader=torch.utils.data.DataLoader(dataset,batch_size=1,shuffle=True,num_workers=0)
    iterator=iter(loader)
    optimizer=torch.optim.AdamW(apply_training_policy(model,TRAINING_POLICY),weight_decay=.05)
    amp=torch.amp.GradScaler('cuda')
    model.train(); optimizer.zero_grad(set_to_none=True)
    torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
    measured=[]; loading=[]; updates=0
    total=TIMING_WARMUP_BATCHES+TIMING_MEASURED_BATCHES
    for step in tqdm(range(total),desc='Timing benchmark: warm-up + measured training'):
        torch.cuda.synchronize(); started=time.perf_counter()
        try: v,n,y=next(iterator)
        except StopIteration:
            iterator=iter(loader); v,n,y=next(iterator)
        data_elapsed=time.perf_counter()-started
        v,n,y=v.to(DEVICE),n.to(DEVICE),y.to(DEVICE)
        with torch.autocast('cuda',dtype=torch.float16):
            logits=model(v,n)
            loss=direction_loss(logits,y)
            loss=loss/accumulation
        if not torch.isfinite(loss): raise FloatingPointError('Non-finite timing benchmark loss')
        amp.scale(loss).backward()
        if (step+1)%accumulation==0:
            amp.unscale_(optimizer); torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
            previous_scale=amp.get_scale(); amp.step(optimizer); amp.update()
            if step>=TIMING_WARMUP_BATCHES and amp.get_scale()>=previous_scale: updates+=1
            optimizer.zero_grad(set_to_none=True)
        torch.cuda.synchronize(); elapsed=time.perf_counter()-started
        if step>=TIMING_WARMUP_BATCHES:
            measured.append(elapsed); loading.append(data_elapsed)
    assert updates==TIMING_MEASURED_BATCHES//accumulation, 'Skipped optimizer updates; benchmark is not representative.'
    peak_gib=torch.cuda.max_memory_allocated()/1024**3
    del optimizer,amp; model.zero_grad(set_to_none=True)
    inference=[]; infer_dataset=ClipDataset(training_indices,mapping,scaler,False)
    model.eval()
    with torch.inference_mode():
        for step in range(2+TIMING_INFERENCE_CLIPS):
            torch.cuda.synchronize(); started=time.perf_counter()
            v,n,_=infer_dataset[step%len(infer_dataset)]
            with torch.autocast('cuda',dtype=torch.float16):
                outputs=model(v[None].to(DEVICE),n[None].to(DEVICE))
            outputs=[p.float().softmax(1).cpu().numpy() for p in outputs]
            torch.cuda.synchronize(); elapsed=time.perf_counter()-started
            assert all(np.isfinite(p).all() for p in outputs)
            if step>=2: inference.append(elapsed)
    result=summarize_benchmark(measured,loading,inference,full_video_workload())
    result.update(gpu=torch.cuda.get_device_name(),peak_training_gpu_allocated_gib=peak_gib,
        backbone=BACKBONE_IDENTITY,frames=N_FRAMES,resolution=SIZE,trainable_parameters=trainable_parameter_report(model),
        warmup_training_batches=TIMING_WARMUP_BATCHES,gradient_accumulation=accumulation,
        full_settings=FULL_TRAIN_SETTINGS,measured_batch_seconds=measured,
        caveat='Warmed small-subset maximum-epoch projection, not a full-run benchmark. Inner-selected outer epochs may reduce runtime. Includes inner loss evaluation; excludes checkpoint transfer overhead. Base estimate excludes preprocessing, role/tree fitting, model loading and checkpoint writes. Drive caching and GPU load can change throughput. Allowances are scenarios, not confidence bounds.')
    (RUN/'timing_estimate.json').write_text(json.dumps(result,indent=2))
    table=pd.DataFrame([
        ('Video training',result['training_hours']),('Video inference',result['inference_hours']),
        ('Video subtotal (excludes other stages)',result['video_subtotal_hours']),
        ('Subtotal + 25% planning allowance',result['planning_with_25_percent_allowance_hours']),
        ('Subtotal + 50% planning allowance',result['planning_with_50_percent_allowance_hours'])],columns=['Component','Estimated hours'])
    table.to_csv(RUN/'timing_estimate.csv',index=False)
    print(f"TIMING ESTIMATE: {result['gpu']}; {result['workload']['video_fits']} video fits")
    display(table.round(2))
    print(f"Data loading: {100*result['training_data_loading_fraction']:.1f}% of measured training time; peak allocated GPU memory: {peak_gib:.1f} GiB")
    print(result['caveat']); print('Saved:',RUN/'timing_estimate.json')
    return result

