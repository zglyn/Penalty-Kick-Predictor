# CELL: heatmaps
# Explain highest-loss OUTER validation clips, using their held-out-fold models.
# Tube occlusion obscures the same crop-relative region across tracked frames.
# Red loss regions: masking hurts the true class. Blue: masking helps it.
# These are sensitivity maps, not segmentation masks or proof of causation.

def occlusion_regions(size,grid):
    assert 1<=grid<=size
    edges=np.linspace(0,size,grid+1,dtype=int)
    return [(r,c,int(edges[r]),int(edges[r+1]),int(edges[c]),int(edges[c+1]))
            for r in range(grid) for c in range(grid)]


def occlude_player_video(video,region,valid,fill='blur'):
    # video: T,C,H,W uint8. Missing-role full-frame fallbacks remain untouched.
    _,_,y0,y1,x0,x1=region; result=video.copy()
    for t in np.flatnonzero(valid):
        if fill=='blur':
            rgb=video[t].transpose(1,2,0)
            replacement=cv2.GaussianBlur(rgb,(51,51),0).transpose(2,0,1)
            result[t,:,y0:y1,x0:x1]=replacement[:,y0:y1,x0:x1]
        elif fill=='mean':
            patch=video[t,:,y0:y1,x0:x1]
            result[t,:,y0:y1,x0:x1]=patch.mean(axis=(1,2),keepdims=True).astype(video.dtype)
        else: raise ValueError('HEATMAP_FILL must be blur or mean')
    return result


def occlusion_effects(base,perturbed,true_label,predicted_label):
    base=np.asarray(base,dtype=float); perturbed=np.asarray(perturbed,dtype=float)
    loss_delta=-np.log(np.clip(perturbed[...,true_label],1e-12,1))+np.log(np.clip(base[true_label],1e-12,1))
    predicted_drop=base[predicted_label]-perturbed[...,predicted_label]
    return loss_delta,predicted_drop


def combine_occluded_ensemble(oof_probability,video_baseline,video_occluded,video_weight):
    # Trees/priors are unchanged; only the visual branch is intervened on.
    changed=np.asarray(oof_probability)+float(video_weight)*(np.asarray(video_occluded)-np.asarray(video_baseline))
    if not np.isfinite(changed).all() or changed.min() < -2e-3 or changed.max()>1.002:
        raise ValueError('Reconstructed ensemble is inconsistent. Check fold checkpoints, seeds and crop cache.')
    changed=np.clip(changed,0,1)
    return changed/changed.sum(axis=-1,keepdims=True)


def assert_heldout_explanation(folder,index):
    provenance=json.loads((folder/'provenance.json').read_text())
    entries=[x for x in provenance if x['kind']=='evaluation' and int(index) in x['prediction_indices']]
    assert len(entries)==1, 'Heatmap clip must be an evaluation clip, not a training cross-fit clip.'
    assert train.source_id.iloc[index] not in entries[0]['fit_sources'], 'Training-source leakage in explanation.'


@torch.inference_mode()
def seed_occlusion_probabilities(model,scaler,index,mapping):
    model.eval()
    d=load_view(mapping,index,decode_video=True)
    video,numeric,_=ClipDataset([index],mapping,scaler)[0]
    video=video[None].to(DEVICE); numeric=numeric[None].to(DEVICE)
    if not hasattr(model,'encode_stream'):
        raise NotImplementedError(f'Occlusion heatmaps are not supported for {type(model).__name__}.')
    def encode(v):
        # Backbone adapter: VideoMAE, V-JEPA 2.1 and V-JEPA 2 share [B,T,C,H,W] -> [B,feature_dim].
        with torch.autocast('cuda',dtype=torch.float16):
            return model.encode_stream(v)
    def probabilities(features):
        with torch.autocast('cuda',dtype=torch.float16):
            fused=torch.cat(features+[model.numeric(numeric)],dim=1)
            logits=[head(model.dropout(fused)) for head in model.heads]
        return [x.float().softmax(1)[0].cpu().numpy() for x in logits]
    features=[encode(video[:,view]) for view in range(3)]
    baseline=probabilities(features)
    # Cached unchanged-stream embeddings must reproduce the normal forward method.
    with torch.autocast('cuda',dtype=torch.float16): direct=model(video,numeric)
    for a,b in zip(baseline,direct):
        np.testing.assert_allclose(a,b.float().softmax(1)[0].cpu().numpy(),atol=2e-4,rtol=2e-3)
    maps=[np.empty((2,HEATMAP_GRID,HEATMAP_GRID,len(c)),dtype=np.float32) for c in CLASSES]
    regions=occlusion_regions(SIZE,HEATMAP_GRID)
    for role in range(2):
        valid=d['traces'][:,role,2]>0
        for region in tqdm(regions,desc=f'Occlude {rows["clip"].iloc[index]} role {role}',leave=False):
            r,c,*_=region
            if not valid.any(): altered=baseline
            else:
                pixels=occlude_player_video(d['video'][role+1],region,valid,HEATMAP_FILL).astype(np.float32)/255
                pixels=(pixels-IMAGE_MEAN[None,:,None,None])/IMAGE_STD[None,:,None,None]
                changed=list(features)
                changed[role+1]=encode(torch.from_numpy(pixels)[None].to(DEVICE))
                altered=probabilities(changed)
            for j in range(len(TARGETS)): maps[j][role,r,c]=altered[j]
    return baseline,maps


def save_tracking_preview(index,folder,output):
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    d=load_view({index:folder/f'{index}.pkl'},index)
    raw=extract_raw(rows['clip'].iloc[index])
    anchor=int(d.get('anchor_frame',np.abs(raw['times']-(raw['times'][-1]-.23)).argmin()))
    selected=sorted(set([0,anchor,len(raw['frames'])-1]))
    fig,axes=plt.subplots(1,len(selected),figsize=(6*len(selected),4),squeeze=False)
    for ax,t in zip(axes[0],selected):
        rgb=raw['frames'][t]; h,w=rgb.shape[:2]; ax.imshow(rgb)
        for role,color in enumerate(['orange','cyan']):
            if 'track_ids' in d and d['track_ids'][t,role]>=0:
                box=raw['detections'][t]['boxes'][d['track_ids'][t,role]]*[w,h,w,h]
                ax.add_patch(Rectangle(box[:2],box[2]-box[0],box[3]-box[1],fill=False,edgecolor=color,linewidth=2))
            elif d['traces'][t,role,2]>0:
                ax.scatter(*list(d['traces'][t,role,:2]*[w,h]),c=color,s=70)
        clip=rows['clip'].iloc[index]
        if clip in clicks.index:
            for name,color in [('kicker','orange'),('keeper','cyan')]:
                row=clicks.loc[clip]
                if abs(raw['times'][t]-row[f'{name}_click_time_seconds'])<=.06:
                    ax.scatter(row[f'{name}_x_norm']*w,row[f'{name}_y_norm']*h,marker='x',c=color,s=100)
        ax.set_title(f'{raw["times"][t]-raw["times"][-1]:+.2f}s from final frame'); ax.axis('off')
    fig.suptitle(f'{rows["clip"].iloc[index]} | orange kicker, cyan keeper; crosses: public audit clicks only')
    fig.tight_layout(); fig.savefig(output,dpi=150,bbox_inches='tight'); display(fig); plt.close(fig)


def run_validation_heatmaps():
    import matplotlib.pyplot as plt
    from matplotlib.colors import TwoSlopeNorm
    assert not QUICK_SMOKE and (fold_ids>=0).all()
    output=RUN/'heatmaps'; output.mkdir(exist_ok=True)
    ranking=np.zeros(len(train))
    for j,w in enumerate(WEIGHTS):
        ranking+=float(w)*(-np.log(np.clip(oof[j][np.arange(len(train)),Y[:,j]],1e-12,1)))
    chosen=np.argsort(-ranking)[:min(HEATMAP_TOP_K,len(train))]
    fold_reports={int(r['fold']):r for r in json.loads((RUN/'fold_results.json').read_text())}
    rows_report=[]
    for fold in sorted(set(fold_ids[chosen])):
        report=fold_reports[int(fold)]
        folder=Path(report['stage_folder'])
        indices=[int(i) for i in chosen if fold_ids[i]==fold]
        mapping={i:folder/f'{i}.pkl' for i in indices}
        for i in indices: assert_heldout_explanation(folder,i)
        seeds=report.get('seeds',VIDEO_SEEDS)
        accumulated={i:dict(base=[np.zeros(len(c)) for c in CLASSES],
            maps=[np.zeros((2,HEATMAP_GRID,HEATMAP_GRID,len(c))) for c in CLASSES]) for i in indices}
        for seed in seeds:
            scaler=joblib.load(folder/'numeric_scaler.joblib')
            model=MultiViewVideo(scaler.n_features_in_).to(DEVICE)
            state=torch.load(folder/f'video_seed{seed}.pt',map_location='cpu',weights_only=True)
            model.load_state_dict(state['model'])
            for i in indices:
                base,maps=seed_occlusion_probabilities(model,scaler,i,mapping)
                for j in range(len(TARGETS)):
                    accumulated[i]['base'][j]+=base[j]/len(seeds)
                    accumulated[i]['maps'][j]+=maps[j]/len(seeds)
            del model,state; gc.collect(); torch.cuda.empty_cache()
        for i in indices:
            clip=rows['clip'].iloc[i]; stem=Path(clip).stem; d=load_view(mapping,i,decode_video=True)
            save_tracking_preview(i,folder,output/f'{stem}_tracking.png')
            # Verify the unmasked reconstructed ensemble against its stored OOF prediction.
            trees=joblib.load(folder/'pose_camera_trees.joblib')
            numeric=np.r_[numeric_summary(d),metadata[i]][None]
            eval_provenance=next(x for x in json.loads((folder/'provenance.json').read_text()) if x['kind']=='evaluation')
            fit_indices=np.flatnonzero(train.source_id.isin(eval_provenance['fit_sources']).to_numpy())
            for j,target in enumerate(TARGETS):
                video_base=accumulated[i]['base'][j]; video_maps=accumulated[i]['maps'][j]
                mixture=np.asarray(report['weights'][j]); tree_p=np.zeros(len(CLASSES[j]))
                tree_p[np.asarray(trees[j].classes_,dtype=int)]=trees[j].predict_proba(numeric)[0]
                prior=np.bincount(Y[fit_indices,j],minlength=len(CLASSES[j]))/len(fit_indices)
                reconstructed=mixture[0]*video_base+mixture[1]*tree_p+mixture[2]*prior
                np.testing.assert_allclose(reconstructed,oof[j][i],atol=2e-3,rtol=2e-3,
                    err_msg='Heatmap model/crop/ensemble does not reproduce OOF probabilities')
                perturbed=combine_occluded_ensemble(oof[j][i],video_base,video_maps,mixture[0])
                true_label=int(Y[i,j]); predicted_label=int(oof[j][i].argmax())
                loss_map,prediction_map=occlusion_effects(oof[j][i],perturbed,true_label,predicted_label)
                fig,axes=plt.subplots(2,3,figsize=(13,8))
                scales=[max(float(np.abs(loss_map).max()),1e-6),max(float(np.abs(prediction_map).max()),1e-6)]
                for role,name in enumerate(['kicker','keeper']):
                    valid=np.flatnonzero(d['traces'][:,role,2]>0)
                    t=int(valid[-1]) if len(valid) else N_FRAMES-1
                    image=d['video'][role+1,t].transpose(1,2,0)
                    axes[role,0].imshow(image); axes[role,0].set_title(f'{name}: tracked {len(valid)}/{N_FRAMES} frames')
                    for col,(heat,label) in enumerate([(loss_map[role],'Change in true-class loss'),
                                                       (prediction_map[role],'Drop in predicted-class probability')],start=1):
                        ax=axes[role,col]; ax.imshow(image)
                        resized=cv2.resize(heat.astype(np.float32),(SIZE,SIZE),interpolation=cv2.INTER_NEAREST)
                        overlay=ax.imshow(resized,cmap='coolwarm',alpha=.6,
                            norm=TwoSlopeNorm(vmin=-scales[col-1],vcenter=0,vmax=scales[col-1]))
                        ax.set_title(label); fig.colorbar(overlay,ax=ax,fraction=.046)
                    for ax in axes[role]: ax.axis('off')
                    rows_report.append(dict(clip=clip,fold=int(fold),target=target,role=name,
                        weighted_oof_loss=float(ranking[i]),true_class=CLASSES[j][true_label],
                        predicted_class=CLASSES[j][predicted_label],video_ensemble_weight=float(mixture[0]),
                        tracked_fraction=len(valid)/N_FRAMES,max_loss_increase=float(loss_map[role].max()),
                        largest_loss_decrease=float(loss_map[role].min())))
                fig.suptitle(f'{clip} | {target}: true={CLASSES[j][true_label]}, predicted={CLASSES[j][predicted_label]}\n'
                    f'OOF weighted loss={ranking[i]:.3f}; video weight={mixture[0]:.2f}; {HEATMAP_FILL} tube occlusion across tracked frames')
                fig.tight_layout(); fig.savefig(output/f'{stem}_{target}_heatmap.png',dpi=150,bbox_inches='tight')
                display(fig); plt.close(fig)
                np.savez_compressed(output/f'{stem}_{target}_maps.npz',loss_delta=loss_map,
                    predicted_probability_drop=prediction_map,base_probability=oof[j][i],perturbed_probability=perturbed)
    pd.DataFrame(rows_report).to_csv(output/'heatmap_summary.csv',index=False)
    pd.DataFrame({'clip':train['clip'].to_numpy(),'weighted_oof_loss':ranking}).sort_values(
        'weighted_oof_loss',ascending=False).to_csv(output/'validation_loss_ranking.csv',index=False)
    print('Saved held-out heatmaps and tracking previews to:',output)
    print('Context and numeric pose/motion inputs remain fixed. Duplicate evidence can make a region appear unimportant.')

if GENERATE_HEATMAPS and RUN_MODE in {'train', 'resume'} and not QUICK_SMOKE and EVALUATION_MODE=='nested_cv' and diagnostic_result is not None:
    run_validation_heatmaps()
else:
    print('Heatmaps run after full OOF validation; smoke mode does not estimate trustworthy high-loss examples.')

