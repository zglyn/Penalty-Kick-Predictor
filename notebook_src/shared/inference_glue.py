# CELL: inference_glue
# Notebook bindings for the inference-only workflow. These call only label-free decoding/detection/cropping
# (extract_raw, make_views with a SAVED selector) and load saved weights; no fitting function is referenced.

def resolve_inference_clips(spec):
    """Return (clip frame with clip/path/preferred_foot/gender/in_dataset, routing)."""
    columns=['clip','source_id','preferred_foot','gender']
    if spec=='private':
        frame=test[columns].copy(); routing='ensemble'
    elif spec=='public_oof':
        frame=train[columns].copy(); routing='oof_replay'   # metadata columns only; labels are not read
    else:
        frame=pd.read_csv(spec); routing='ensemble'
        missing={'clip','path','preferred_foot','gender'}-set(frame)
        if missing: raise ValueError(f'Inference CSV {spec} lacks columns {sorted(missing)}')
        bad=frame[~frame.preferred_foot.isin(['left','right'])|~frame.gender.isin(['F','M'])]
        if len(bad): raise ValueError(f'preferred_foot must be left/right and gender F/M (as in training): {bad.clip.tolist()}')
        frame['path']=[p if Path(p).is_absolute() else str(Path(spec).parent/p) for p in frame['path'].astype(str)]
        if 'source_id' not in frame: frame['source_id']='custom'
    if 'path' not in frame:
        frame['path']=[str(video_paths[c]) for c in frame['clip']]
    frame['path']=frame['path'].astype(str)
    for p in frame['path']:
        if not Path(p).is_file(): raise FileNotFoundError(f'Inference clip not found: {p}')
    frame['in_dataset']=[c in video_paths and Path(p).resolve()==Path(video_paths[c]).resolve()
                         for c,p in zip(frame['clip'],frame['path'])]
    return frame.reset_index(drop=True),routing

_INFERENCE_CONTENT={}
def inference_build_view(record,selector,member):
    if record['in_dataset']:
        raw=extract_raw(record['clip'])     # shared label-free detector cache for dataset clips
    else:
        key=content_id_for(record['path'],_INFERENCE_CONTENT)
        raw=extract_raw(record['clip'],path=record['path'],cache_path=INFERENCE_RAW/f'{key}.pkl')
    return make_views(raw,selector)

def inference_model_factory(manifest,member):
    set_active_ablation(manifest['feature']['variant'])
    if numeric_feature_manifest()!=manifest['feature']['numeric_features']:
        raise InferenceArtifactError('Saved numeric feature schema/order differs from this notebook version')
    if manifest['feature'].get('streams') is not None and list(ablation_stream_ids())!=list(manifest['feature']['streams']):
        raise InferenceArtifactError('Saved visual streams differ from this notebook version')
    # architecture= rebuilds the saved encoder configuration without downloading pretrained weights.
    return MultiViewVideo(manifest['feature']['numeric_dimension'],architecture=member['backbone']['architecture'])

def inference_numeric_features(view,metadata_row,feature):
    return ablation_numeric_features(view,metadata_row,ABLATION_CONFIGS[feature['variant']])

INFERENCE_HOOKS=dict(build_view=inference_build_view,save_view=atomic_pickle,
    load_view=lambda path,decode_video=True: load_view({0:path},0,decode_video=decode_video),
    model_factory=inference_model_factory,numeric_features=inference_numeric_features,load_joblib=joblib.load)

def migrate_legacy_run(run_dir):
    """One-time export for a run trained before inference manifests. Old folders are preserved: this only
    adds stages/*/class_prior.json (when missing) and inference_manifest.json."""
    run=Path(run_dir); experiment=json.loads((run/'experiment_manifest.json').read_text())
    config=experiment['config']
    mean,std=IMAGE_MEAN,IMAGE_STD  # cell above derived these from the SAVED backbone (processor or ImageNet)
    if 'backbone' in config:
        backbone=None  # new-style runs saved backbone_serialization.json in every stage
    else:
        from transformers import VideoMAEConfig
        spec=BACKBONE_SPEC  # derived from the saved config by the configuration cell (VideoMAE, saved checkpoint id)
        assert spec['family']=='videomae' and spec['checkpoint']==config['video'], 'Legacy runs are VideoMAE runs'
        architecture=dict(kind='videomae',config=VideoMAEConfig.from_pretrained(config['video']).to_dict())  # config JSON only
        backbone=dict(identity=backbone_identity(spec,'finetune'),spec=spec,architecture=architecture,
                      provenance=dict(note='migrated legacy VideoMAE run; architecture config fetched, weights from run'),
                      training_policy='finetune',policy_description=TRAINING_POLICIES['finetune'],
                      streams=list(ablation_stream_ids(ABLATION_CONFIGS[config.get('ablation','production')])))
    print('Migration may read public TRAINING labels to recover a missing class prior; each prior uses only '
          'its own stage training clips and is saved. Inference itself never reads labels.')
    return export_inference_manifest(run,preprocessing=preprocessing_record(config,mean,std,CAMERA_CACHE_VERSION),
        backbone=backbone,training_labels=train[['clip']+TARGETS],dataset_id=DATA_ID,
        extra_clip_names=test['clip'].tolist())

def export_run_inference_manifest(run_dir):
    """Called after training: same exporter, current-session preprocessing values."""
    try:
        return export_inference_manifest(run_dir,preprocessing=preprocessing_record(CFG,IMAGE_MEAN,IMAGE_STD,CAMERA_CACHE_VERSION),
            dataset_id=DATA_ID,extra_clip_names=test['clip'].tolist())
    except InferenceArtifactError as error:
        print('WARNING: inference manifest not exported; training outputs are intact.\n',error)

def inference_output_root(run_dir):
    return OUTPUT/'inference'/PLATFORM/Path(run_dir).name


def run_inference_mode():
    global INFERENCE_RAW
    if not (TRAINED_RUN_DIR/'inference_manifest.json').exists():
        if is_bundle(TRAINED_RUN_DIR):
            raise InferenceArtifactError('This bundle has no inference_manifest.json; export bundles from finished runs.')
        if not MIGRATE_LEGACY_RUN:
            raise InferenceArtifactError(f'{TRAINED_RUN_DIR/"inference_manifest.json"} not found. If this run finished '
                'before this notebook version, set MIGRATE_LEGACY_RUN=True once (exports from the saved artifacts).')
        migrate_legacy_run(TRAINED_RUN_DIR)
    manifest=load_inference_manifest(TRAINED_RUN_DIR,VERIFY_CHECKSUMS)
    pre=manifest['preprocessing']
    assert (N_FRAMES,SIZE,float(WINDOW))==(pre['frames'],pre['size'],pre['window']), 'Saved preprocessing not applied'
    assert np.allclose(IMAGE_MEAN,pre['image_mean']) and np.allclose(IMAGE_STD,pre['image_std'])
    raw_settings={k:v for k,v in RAW_CFG.items() if k not in ('dataset','click_timestamps')}
    INFERENCE_RAW=OUTPUT/('inference_detector_cache_'+stable_hash(raw_settings)); INFERENCE_RAW.mkdir(parents=True,exist_ok=True)
    frame,routing=resolve_inference_clips(INFERENCE_INPUT)
    print(f'Inference: {len(frame)} clip(s), routing={routing}, {len(manifest["members"])} fitted member(s), '
          f'backbone={manifest["backbone"]["identity"]["name"]} ({manifest["backbone"]["training_policy"]})')
    # Outputs/caches go to this platform's OUTPUT_ROOT, never into the trained run or imported bundle.
    out_root=inference_output_root(TRAINED_RUN_DIR)
    result=run_saved_inference(TRAINED_RUN_DIR,frame,hooks=INFERENCE_HOOKS,routing=routing,
        verify_checksums=VERIFY_CHECKSUMS,device=DEVICE,dataset_id=DATA_ID,output_root=out_root)
    release_pose_model()
    if INFERENCE_INPUT=='private':
        path=result['output_dir']/'submission.csv'
        submission=write_submission(result,template,path); display(submission.head())
        print('Saved submission:',path)
    print(json.dumps(result['record'],indent=2,default=str))
    if result['record']['max_abs_difference_vs_saved_oof'] is not None:
        print('OOF replay reproduces saved outer predictions within:',result['record']['max_abs_difference_vs_saved_oof'])
    return result
