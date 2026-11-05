# CELL: roles_and_crops

def role_examples(indices,role):
    features=[]; labels=[]; groups=[]; sample_weights=[]; audit=[]
    name=['kicker','keeper'][role]
    for i in indices:
        clip=rows['clip'].iloc[i]
        if clip not in clicks.index: continue
        raw=extract_raw(clip); d=raw['clicked'][role]
        point=clicks.loc[clip,[f'{name}_x_norm',f'{name}_y_norm']].to_numpy(dtype=float)
        selected,method=match_click_candidate(d,point)
        audit.append(dict(clip=clip,source_id=rows.source_id.iloc[i],role=name,
                          match=method,candidates=len(d['boxes'])))
        if selected is None: continue
        y=np.zeros(len(d['boxes']),dtype=int); y[selected]=1
        features.extend(d['features']); labels.extend(y); groups.extend([rows.source_id.iloc[i]]*len(y))
        # Crowded clips should not dominate training solely by having more negatives.
        sample_weights.extend([1/len(y)]*len(y))
    return np.asarray(features,dtype=np.float32),np.asarray(labels),groups,np.asarray(sample_weights),audit


def fit_roles(indices,path):
    source_ids=set(rows.source_id.iloc[indices])
    if path.exists() and not QUICK_SMOKE:
        saved=joblib.load(path)
        assert set(saved['sources'])==source_ids
        return saved
    models=[]; counts=[]
    for role in range(2):
        x,y,_,sample_weights,match_audit=role_examples(indices,role)
        if len(y)<(8 if QUICK_SMOKE else 30) or len(np.unique(y))<2:
            raise ValueError(f'Insufficient matched role examples for role {role}: {len(y)} candidates, {int(y.sum())} positives. Inspect detector matches or enlarge the smoke subset.')
        model=CatBoostClassifier(iterations=30 if QUICK_SMOKE else 350,depth=4,learning_rate=.04,l2_leaf_reg=8,
             loss_function='Logloss',verbose=False,random_seed=42,thread_count=4)
        model.fit(x,y,sample_weight=sample_weights); models.append(model)
        pd.DataFrame(match_audit).to_csv(path.with_name(path.stem+f'_role{role}_matches.csv'),index=False)
        counts.append(dict(candidates=len(y),positives=int(y.sum()),annotated_clips=len(match_audit)))
    result=dict(models=models,sources=sorted(source_ids),counts=counts)
    joblib.dump(result,path); return result

def transform_points(points,matrix):
    return np.asarray(points)@matrix[:2,:2].T+matrix[:2,2]

def assign_roles(detection,selector,previous,matrix,quality,shape):
    n=len(detection['boxes'])
    if not n: return [None,None],np.zeros(2)
    probs=np.stack([m.predict_proba(detection['features'])[:,1] for m in selector['models']])
    centers=(detection['boxes'][:,:2]+detection['boxes'][:,2:])/2
    h,w=shape[:2]; score=np.log(np.maximum(probs,1e-8))
    for role in range(2):
        if previous[role] is not None and quality>0:
            expected=transform_points(previous[role]*[w,h],matrix)/[w,h]
            score[role]-=2*np.linalg.norm(centers-expected,axis=1)
    choices=[None]+list(range(n)); best=None; best_score=-float('inf')
    for a,b in itertools.product(choices,repeat=2):
        if a is not None and a==b: continue
        pair=(a,b)
        if any(j is not None and probs[r,j]<ROLE_MIN_PROB for r,j in enumerate(pair)): continue
        total=sum(np.log(ROLE_MIN_PROB) if j is None else score[r,j] for r,j in enumerate(pair))
        if total>best_score: best=pair; best_score=total
    return list(best),np.array([0. if j is None else probs[r,j] for r,j in enumerate(best)])

def match_click_candidate(detection, point, near_limit=0.35):
    boxes=detection['boxes']
    if not len(boxes): return None,'no_detection'
    point=np.asarray(point,dtype=float)
    centers=(boxes[:,:2]+boxes[:,2:])/2
    distances=np.linalg.norm(centers-point,axis=1)/np.maximum(boxes[:,3]-boxes[:,1],.02)
    inside=np.all(point>=boxes[:,:2]-.01,axis=1)&np.all(point<=boxes[:,2:]+.01,axis=1)
    candidates=np.flatnonzero(inside)
    method='inside'
    if not len(candidates):
        candidates=np.flatnonzero(distances<=near_limit); method='near'
    if not len(candidates): return None,'unmatched'
    order=candidates[np.argsort(distances[candidates])]
    if len(order)>1 and distances[order[1]]-distances[order[0]]<.10:
        return None,'ambiguous'
    return int(order[0]),method


def bidirectional_role_tracks(raw,selector):
    count=len(raw['detections']); tracks=np.full((count,2),-1,dtype=int)
    confidence=np.zeros((count,2),dtype=np.float32)
    nominal=int(np.abs(raw['times']-(raw['times'][-1]-.23)).argmin())
    probs=[]
    for d in raw['detections']:
        probs.append(np.stack([m.predict_proba(d['features'])[:,1] for m in selector['models']])
                     if len(d['boxes']) else np.zeros((2,0)))
    # Prefer an automatic anchor near the public annotation timing; no click is read here.
    anchors=sorted(range(count),key=lambda t:abs(raw['times'][t]-raw['times'][nominal]))
    best=None
    for t in anchors:
        selected,p=assign_roles(raw['detections'][t],selector,[None,None],np.eye(3),0,raw['frames'][t].shape)
        found=sum(j is not None for j in selected)
        key=(found,float(p.sum())-.25*abs(raw['times'][t]-raw['times'][nominal]))
        if found and (best is None or key>best[0]): best=(key,t,selected,p)
        if found==2 and abs(raw['times'][t]-raw['times'][nominal])<.12 and p.min()>=.6: break
    if best is None: return tracks,confidence,nominal
    _,anchor,seeds,seed_probs=best
    for r,j in enumerate(seeds):
        if j is not None: tracks[anchor,r]=j; confidence[anchor,r]=seed_probs[r]
    for step in [1,-1]:
        states=[]
        for r,j in enumerate(seeds):
            if j is None: states.append(None); continue
            d=raw['detections'][anchor]; b=d['boxes'][j]; h,w=raw['frames'][anchor].shape[:2]
            states.append(dict(center=(b[:2]+b[2:])/2*[w,h],height=max((b[3]-b[1])*h,2.),
                appearance=d['features'][j,-24:].copy(),last_time=raw['times'][anchor]))
        for t in range(anchor+step,count if step==1 else -1,step):
            previous=t-step; h,w=raw['frames'][t].shape[:2]
            edge=t if step==1 else previous
            good=raw['camera_quality'][edge]>0
            matrix=raw['camera'][edge] if step==1 else np.linalg.inv(raw['camera'][edge])
            d=raw['detections'][t]; boxes=d['boxes']; centers=(boxes[:,:2]+boxes[:,2:])/2*[w,h]
            scores=[]
            for r,state in enumerate(states):
                score=np.full(len(boxes),-np.inf)
                if state is not None:
                    if good:
                        state['center']=transform_points(state['center'],matrix)
                        state['height']*=float(np.hypot(matrix[0,0],matrix[1,0]))
                    gap=abs(raw['times'][t]-state['last_time'])
                    if gap<=TRACK_MAX_GAP_SECONDS:
                        distances=np.linalg.norm(centers-state['center'],axis=1)/max(state['height'],2.)
                        allowed=min(TRACK_MAX_JUMP_HEIGHTS,.15+TRACK_SPEED_HEIGHTS_PER_SECOND*gap)
                        appearance=np.abs(d['features'][:,-24:]-state['appearance']).sum(axis=1)/3 if len(boxes) else np.zeros(0)
                        valid=(distances<=allowed)&(probs[t][r]>=.15)
                        score[valid]=-distances[valid]-.4*appearance[valid]+.35*np.log(np.maximum(probs[t][r,valid],1e-8))
                scores.append(score)
            choices=[None]+list(range(len(boxes)))
            selected=(None,None); value=-3.2
            for pair in itertools.product(choices,repeat=2):
                if pair[0] is not None and pair[0]==pair[1]: continue
                candidate=sum(-1.6 if j is None else scores[r][j] for r,j in enumerate(pair))
                if candidate>value: value=candidate; selected=pair
            for r,j in enumerate(selected):
                if j is None: continue
                b=boxes[j]; states[r]=dict(center=centers[j],height=max((b[3]-b[1])*h,2.),
                    appearance=d['features'][j,-24:].copy(),last_time=raw['times'][t])
                tracks[t,r]=j; confidence[t,r]=probs[t][r,j]
    return tracks,confidence,anchor


def letterbox(rgb):
    image=Image.fromarray(rgb); image.thumbnail((SIZE,SIZE))
    canvas=Image.new('RGB',(SIZE,SIZE),(123,116,103))
    canvas.paste(image,((SIZE-image.width)//2,(SIZE-image.height)//2))
    return np.asarray(canvas)

def role_crop(rgb,box):
    h,w=rgb.shape[:2]; center=(box[:2]+box[2:])/2; size=(box[2:]-box[:2])*[1.7,1.35]
    lo=np.maximum(0,center-size/2); hi=np.minimum(1,center+size/2)
    x1,y1=(lo*[w,h]).astype(int); x2,y2=(hi*[w,h]).astype(int)
    return rgb[y1:max(y1+1,y2),x1:max(x1+1,x2)]

def make_views(raw,selector):
    videos=[]; numeric=[]; traces=[]; previous=[None,None]; prior_points=[None,None]
    track_ids,track_confidence,anchor=bidirectional_role_tracks(raw,selector)
    for t,(rgb,d) in enumerate(zip(raw['frames'],raw['detections'])):
        matrix=raw['camera'][t]; quality=raw['camera_quality'][t]; h,w=rgb.shape[:2]
        selected=[None if j<0 else int(j) for j in track_ids[t]]
        prob=track_confidence[t]
        dt=raw['times'][t]-raw['times'][t-1] if t else 0.
        frame_views=[letterbox(rgb)]; values=[]; trace=[]
        for role,j in enumerate(selected):
            if j is None:
                frame_views.append(letterbox(rgb)); values.append(np.zeros(146,dtype=np.float32))
                previous[role]=None; prior_points[role]=None; trace.append([0,0,0]); continue
            box=d['boxes'][j]; center=(box[:2]+box[2:])/2; size=np.maximum(box[2:]-box[:2],1e-5)
            xy=d['joints'][j]; conf=d['confidence'][j]; rel=(xy-center)/size; rel[conf<.25]=0
            points=np.vstack([center,xy]); valid=np.r_[True,conf>=.25]
            raw_velocity=np.zeros((18,2)); compensated=np.zeros((18,2)); motion_valid=np.zeros(18)
            if prior_points[role] is not None and dt>1e-6:
                before,before_valid=prior_points[role]; good=valid&before_valid
                raw_velocity[good]=(points-before)[good]/dt
                if quality>0:
                    expected=transform_points(before*[w,h],matrix)/[w,h]
                    compensated[good]=(points-expected)[good]/dt; motion_valid[good]=1
            # Raw and compensated velocities coexist; validity distinguishes failure from stillness.
            feature=np.r_[rel.ravel(),conf,center,size,prob[role],raw_velocity.ravel(),
                          compensated.ravel(),motion_valid]
            assert len(feature)==146
            values.append(feature.astype(np.float32)); frame_views.append(letterbox(role_crop(rgb,box)))
            previous[role]=center; prior_points[role]=(points,valid); trace.append([*center,prob[role]])
        scale=np.hypot(matrix[0,0],matrix[1,0]); angle=np.arctan2(matrix[1,0],matrix[0,0])
        camera=np.array([matrix[0,2]/w,matrix[1,2]/h,np.log(max(scale,1e-6)),angle,quality,
                         dt,raw['times'][t]-raw['times'][-1]],dtype=np.float32)
        videos.append(np.stack(frame_views)); numeric.append(np.r_[*values,camera]); traces.append(trace)
    result=dict(video=np.stack(videos).transpose(1,0,4,2,3),numeric=np.asarray(numeric,dtype=np.float32),
                traces=np.asarray(traces,dtype=np.float32),track_ids=track_ids,anchor_frame=anchor)
    assert result['video'].shape==(3,N_FRAMES,3,SIZE,SIZE)
    assert np.isfinite(result['numeric']).all()
    return result

def stage_identity(training,selection,name):
    """Fitted-stage identity: training clips, checkpoint-selection clips and configuration.
    The evaluation (prediction) list is deliberately excluded: new prediction clips never retrain."""
    return dict(name=name,training=sorted(map(int,training)),
                selection=None if selection is None else sorted(map(int,selection)),config=CFG)

def resolve_stage_folder(training,selection,name):
    identity=stage_identity(training,selection,name)
    tag='s'+hashlib.sha256(json.dumps(identity,sort_keys=True).encode()).hexdigest()[:11]
    folder=STAGES/(name+'_'+tag)
    if folder.exists(): return folder
    for candidate in sorted(STAGES.glob(name+'_*')):
        marker=candidate/'stage_identity.json'
        if marker.exists():
            if json.loads(marker.read_text()).get('identity_tag')==tag: return candidate
            continue
        # Legacy folders also hashed the evaluation list. Reuse one only when its saved fit manifests prove
        # the same training clips, selection clips and configuration (so the fitted models are identical).
        manifests=sorted(candidate.glob('video_seed*_fit_manifest.json')); provenance=candidate/'provenance.json'
        if not manifests or not provenance.exists(): continue
        fits=[json.loads(m.read_text()) for m in manifests]
        crossfit=sorted(i for e in json.loads(provenance.read_text()) if e['kind']=='training_crossfit'
                        for i in e['prediction_indices'])
        if (crossfit==identity['training'] and all(f['config']==CFG and sorted(f['training_indices'])==identity['training']
                and sorted(f['validation_indices'])==(identity['selection'] or []) for f in fits)):
            marker.write_text(json.dumps(dict(identity_tag=tag,legacy_folder=True,
                note='Legacy stage reused: identical fitted identity; only the evaluation list differed.'),indent=2))
            print(f'Reusing existing stage {candidate.name}: same fitted identity, new prediction clips only.')
            return candidate
    return folder

def build_stage(training,evaluation,name,selection=None):
    training=np.asarray(training); evaluation=np.asarray(evaluation)
    assert not set(training)&set(evaluation)
    heldout_public=evaluation[evaluation<len(train)]
    assert not set(rows.source_id.iloc[training])&set(rows.source_id.iloc[heldout_public])
    # Private shared sources are allowed at final inference; outer/inner validation must be disjoint.
    folder=resolve_stage_folder(training,selection,name); folder.mkdir(exist_ok=True)
    if not (folder/'stage_identity.json').exists():
        identity=stage_identity(training,selection,name)
        (folder/'stage_identity.json').write_text(json.dumps(dict(identity_tag=folder.name.rsplit('_',1)[1],
            training_clips=len(identity['training']),selection_clips=None if selection is None else len(identity['selection']),
            note='Prediction clips are not part of this identity; changing them reuses the fitted models.'),indent=2))
    mapping={}; manifests=[]
    folds=GroupKFold(n_splits=ROLE_FOLDS).split(training,groups=rows.source_id.iloc[training])
    for k,(a,b) in enumerate(folds):
        fit_idx,predict_idx=training[a],training[b]
        assert not set(rows.source_id.iloc[fit_idx])&set(rows.source_id.iloc[predict_idx])
        selector=fit_roles(fit_idx,folder/f'roles_crossfit_{k}.joblib')
        for i in tqdm(predict_idx,desc=f'{name}: cross-fit crops {k}',leave=False):
            dest=folder/f'{i}.pkl'
            if not dest.exists(): atomic_pickle(make_views(extract_raw(rows['clip'].iloc[i]),selector),dest)
            mapping[int(i)]=dest
        manifests.append(dict(kind='training_crossfit',fit_sources=selector['sources'],prediction_indices=predict_idx.tolist()))
    selector=fit_roles(training,folder/'roles_all_training.joblib')
    for i in tqdm(evaluation,desc=f'{name}: held-out crops',leave=False):
        dest=folder/f'{i}.pkl'
        if not dest.exists(): atomic_pickle(make_views(extract_raw(rows['clip'].iloc[i]),selector),dest)
        mapping[int(i)]=dest
    # One evaluation entry accumulates every clip ever predicted from this fitted stage.
    previous=[]
    if (folder/'provenance.json').exists():
        previous=[i for e in json.loads((folder/'provenance.json').read_text()) if e['kind']=='evaluation'
                  for i in e['prediction_indices']]
    evaluated=sorted(set(map(int,previous))|set(map(int,evaluation)))
    assert not set(evaluated)&set(map(int,training))
    manifests.append(dict(kind='evaluation',fit_sources=selector['sources'],prediction_indices=evaluated))
    (folder/'provenance.json').write_text(json.dumps(manifests,indent=2))
    return folder,mapping,selector
