# CELL: extraction

def atomic_pickle(value,path):
    path=Path(path); tmp=path.with_name(f'.{path.name}.{os.getpid()}.tmp')  # unique per writer process
    if isinstance(value,dict) and 'video' in value:
        value=dict(value)
        video=value.pop('video')
        value['video_jpeg']=[[encode_jpeg(frame.transpose(1,2,0)) for frame in stream] for stream in video]
    with tmp.open('wb') as f: pickle.dump(value,f,protocol=pickle.HIGHEST_PROTOCOL)
    tmp.replace(path)

def encode_jpeg(rgb):
    ok,buffer=cv2.imencode('.jpg',cv2.cvtColor(rgb,cv2.COLOR_RGB2BGR),[cv2.IMWRITE_JPEG_QUALITY,95])
    if not ok: raise ValueError('JPEG cache encoding failed')
    return buffer.tobytes()

def decode_jpeg(buffer):
    image=cv2.imdecode(np.frombuffer(buffer,dtype=np.uint8),cv2.IMREAD_COLOR)
    if image is None: raise ValueError('Corrupt JPEG cache')
    return cv2.cvtColor(image,cv2.COLOR_BGR2RGB)

def decode_frames(path,click_times=()):
    recent=deque(); nearest=[None]*len(click_times); distances=[float('inf')]*len(click_times)
    with av.open(str(path)) as container:
        stream=container.streams.video[0]; rate=float(stream.average_rate or 25)
        for i,frame in enumerate(container.decode(stream)):
            t=float(frame.time) if frame.time is not None else i/rate
            im=frame.to_image(); im.thumbnail((DETECTOR_IMGSZ,DETECTOR_IMGSZ))
            rgb=np.asarray(im).copy()
            recent.append((t,rgb))
            while len(recent)>2 and recent[1][0]<t-WINDOW-.1: recent.popleft()
            for j,ct in enumerate(click_times):
                if abs(t-ct)<distances[j]: nearest[j]=(t,rgb); distances[j]=abs(t-ct)
    if not recent: raise ValueError(f'No frames: {path}')
    ts=np.array([a[0] for a in recent]); wanted=ts[-1]-np.linspace(np.sqrt(WINDOW),0,N_FRAMES)**2
    ids=np.abs(ts[:,None]-wanted).argmin(0)
    return [recent[i] for i in ids],nearest

def appearance_features(rgb,box):
    h,w=rgb.shape[:2]; a=np.clip(box,0,1)
    x1,y1,x2,y2=(a*np.array([w,h,w,h])).astype(int)
    patch=rgb[max(0,y1):max(y1+1,y2),max(0,x1):max(x1+1,x2)]
    if not patch.size: return np.zeros(24,dtype=np.float32)
    hsv=cv2.cvtColor(patch,cv2.COLOR_RGB2HSV)
    return np.concatenate([np.histogram(hsv[:,:,i],bins=8,range=(0,180 if i==0 else 256))[0]/(patch.shape[0]*patch.shape[1]) for i in range(3)])

_POSE_MODEL=None
def get_pose_model():
    # Loaded lazily: cached clips (and reports mode) never load or download the detector.
    global _POSE_MODEL
    if _POSE_MODEL is None: _POSE_MODEL=YOLO(POSE_WEIGHTS)
    return _POSE_MODEL

def release_pose_model():
    global _POSE_MODEL
    _POSE_MODEL=None; gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()

def detect(rgb):
    # PIL input avoids RGB/BGR ambiguity in Ultralytics numpy inputs.
    result=get_pose_model().predict(Image.fromarray(rgb),imgsz=DETECTOR_IMGSZ,conf=DETECTOR_CONF,verbose=False,
                                    device=0 if DEVICE=='cuda' else 'cpu')[0]
    boxes=result.boxes.xyxyn.cpu().numpy().astype(np.float32)
    joints=result.keypoints.xyn.cpu().numpy().astype(np.float32)
    confidence=result.keypoints.conf.cpu().numpy().astype(np.float32)
    features=[]
    for i,box in enumerate(boxes):
        center=(box[:2]+box[2:])/2; size=np.maximum(box[2:]-box[:2],1e-5)
        rel=(joints[i]-center)/size; rel[confidence[i]<.25]=0
        features.append(np.r_[box,center,size,size.prod(),float(result.boxes.conf[i]),
            rel.ravel(),confidence[i],appearance_features(rgb,box)])
    return dict(boxes=boxes,joints=joints,confidence=confidence,
                features=np.asarray(features,dtype=np.float32).reshape(len(boxes),85))

def background_mask(shape,boxes):
    h,w=shape[:2]; mask=np.full((h,w),255,dtype=np.uint8)
    # Exclude screen edges, often occupied by stationary graphics.
    mask[:int(.07*h)]=0; mask[int(.93*h):]=0
    for b in boxes:
        x1,y1,x2,y2=b*np.array([w,h,w,h]); pad=.08*max(x2-x1,y2-y1)
        cv2.rectangle(mask,(max(0,int(x1-pad)),max(0,int(y1-pad))),
                      (min(w-1,int(x2+pad)),min(h-1,int(y2+pad))),0,-1)
    return mask

def fit_similarity(src,dst):
    identity=np.eye(3,dtype=np.float32)
    if len(src)<12: return identity,0.
    affine,inliers=cv2.estimateAffinePartial2D(np.asarray(src,dtype=np.float32),np.asarray(dst,dtype=np.float32),
         method=cv2.RANSAC,ransacReprojThreshold=3,maxIters=3000,confidence=.995)
    if affine is None or inliers is None: return identity,0.
    keep=inliers.ravel().astype(bool); ratio=keep.mean()
    scale=float(np.hypot(affine[0,0],affine[1,0])); angle=float(np.arctan2(affine[1,0],affine[0,0]))
    residual=np.linalg.norm(np.asarray(src)@affine[:,:2].T+affine[:,2]-dst,axis=1)
    if keep.sum()<10 or ratio<.5 or not .65<scale<1.5 or abs(angle)>.4 or np.median(residual[keep])>2.5:
        return identity,0.
    matrix=identity.copy(); matrix[:2]=affine
    return matrix,float(ratio)

def camera_motion(previous,current,prev_boxes,cur_boxes):
    identity=np.eye(3,dtype=np.float32)
    if previous.shape!=current.shape: return identity,0.
    a=cv2.cvtColor(previous,cv2.COLOR_RGB2GRAY); b=cv2.cvtColor(current,cv2.COLOR_RGB2GRAY)
    mask=background_mask(previous.shape,prev_boxes); nextmask=background_mask(current.shape,cur_boxes)
    p=cv2.goodFeaturesToTrack(a,maxCorners=1000,qualityLevel=.01,minDistance=8,mask=mask)
    if p is None or len(p)<12: return identity,0.
    q,status,_=cv2.calcOpticalFlowPyrLK(a,b,p,None,winSize=(31,31),maxLevel=4)
    if q is None: return identity,0.
    back,status2,_=cv2.calcOpticalFlowPyrLK(b,a,q,None,winSize=(31,31),maxLevel=4)
    if back is None: return identity,0.
    src=p.reshape(-1,2); dst=q.reshape(-1,2)
    good=(status.ravel()>0)&(status2.ravel()>0)&(np.linalg.norm(src-back.reshape(-1,2),axis=1)<1.5)
    h,w=a.shape; inside=(dst[:,0]>=0)&(dst[:,0]<w)&(dst[:,1]>=0)&(dst[:,1]<h)
    good &= inside
    ix=np.clip(dst[:,0].astype(int),0,w-1); iy=np.clip(dst[:,1].astype(int),0,h-1)
    good &= nextmask[iy,ix]>0
    src,dst=src[good],dst[good]
    # Require spatial coverage, not merely one small moving patch.
    if len(src)<12 or np.ptp(src[:,0])<.2*w or np.ptp(src[:,1])<.15*h: return identity,0.
    return fit_similarity(src,dst)

def extract_raw(clip,path=None,cache_path=None):
    # Default: dataset clip in the shared detector cache. Inference of other videos passes an explicit
    # path and content-keyed cache_path; those never receive annotation clicks.
    dest=RAW/(clip+'.pkl') if cache_path is None else Path(cache_path)
    if dest.exists():
        with dest.open('rb') as f: value=pickle.load(f)
        value['frames']=[decode_jpeg(frame) for frame in value.pop('frames_jpeg')]
        return value
    annotation=clicks.loc[clip] if cache_path is None and clip in clicks.index else None
    ctimes=[] if annotation is None else [float(annotation[f'{r}_click_time_seconds']) for r in ['kicker','keeper']]
    sampled,clicked=decode_frames(video_paths[clip] if path is None else path,ctimes)
    detections=[detect(rgb) for t,rgb in sampled]
    matrices=[np.eye(3,dtype=np.float32)]; quality=[0.]
    for i in range(1,N_FRAMES):
        if sampled[i][0]==sampled[i-1][0]: matrix,q=np.eye(3,dtype=np.float32),0.
        else: matrix,q=camera_motion(sampled[i-1][1],sampled[i][1],detections[i-1]['boxes'],detections[i]['boxes'])
        matrices.append(matrix); quality.append(q)
    value=dict(times=np.array([x[0] for x in sampled]),frames=[x[1] for x in sampled],
               detections=detections,camera=np.stack(matrices),camera_quality=np.array(quality),
               clicked=[dict(time=t,**detect(rgb)) for t,rgb in clicked])
    value['frames_jpeg']=[encode_jpeg(frame) for frame in value.pop('frames')]
    atomic_pickle(value,dest)
    # Identical first-run and cache-hit inputs for downstream crop processing.
    value['frames']=[decode_jpeg(frame) for frame in value.pop('frames_jpeg')]
    return value

if RUN_MODE in TRAINING_MODES:
    work_clips=rows['clip'].iloc[np.r_[smoke_train,smoke_hold]] if QUICK_SMOKE else rows['clip']
    for clip in tqdm(work_clips,desc='Label-free detections, exact click frames, camera motion'): extract_raw(clip)
    release_pose_model()
    if QUICK_SMOKE: print('Smoke extraction complete; continue through role training, fine-tuning, and inference below.')
else:
    print(f'{RUN_MODE} mode: no bulk extraction here (inference decodes only uncached requested clips).')

