# CELL: diagnostics_helpers
"""Strict two-target OOF exports. Embedded verbatim into the Colab notebook."""
import json
from pathlib import Path
import numpy as np
import pandas as pd


def validate_component_probabilities(probabilities, count, class_order, expected_classes):
    if [list(x) for x in class_order] != [list(x) for x in expected_classes]:
        raise ValueError('Probability class order does not match the saved label mapping')
    if len(probabilities) != 2 or len(expected_classes) != 2:
        raise ValueError('Exactly kick direction and goalkeeper direction are required')
    arrays=[]
    for p,classes in zip(probabilities,expected_classes):
        p=np.asarray(p,dtype=float)
        if p.shape != (count,len(classes)):
            raise ValueError(f'Probability shape {p.shape}; expected {(count,len(classes))}')
        if not np.isfinite(p).all() or (p<0).any() or (p>1).any():
            raise ValueError('Probabilities must be finite and in [0,1]')
        if not np.allclose(p.sum(axis=1),1,rtol=0,atol=1e-5):
            raise ValueError('Probability rows must sum to one')
        arrays.append(p)
    return arrays


def component_metrics(truth, probabilities, targets, classes, weights):
    truth=np.asarray(truth)
    if truth.shape!=(len(truth),2) or not len(truth):
        raise ValueError('Need a nonempty aligned two-target label matrix')
    if list(targets)!=['kick_direction','goalkeeper_direction'] or not np.allclose(weights,[.7,.3]):
        raise ValueError('Competition metric must be 70% kick / 30% goalkeeper')
    arrays=validate_component_probabilities(probabilities,len(truth),classes,classes)
    result={}; weighted_accuracy=0.; weighted_ce=0.
    for j,(target,p) in enumerate(zip(targets,arrays)):
        y=truth[:,j]
        if not np.isfinite(y).all() or not np.equal(y,y.astype(int)).all() or (y<0).any() or (y>=len(classes[j])).any():
            raise ValueError('Invalid class index in truth')
        y=y.astype(int)
        accuracy=float((p.argmax(1)==y).mean())
        ce=float(-np.log(np.clip(p[np.arange(len(y)),y],1e-12,1)).mean())
        result[target+'_accuracy']=accuracy; result[target+'_cross_entropy']=ce
        weighted_accuracy+=float(weights[j])*accuracy; weighted_ce+=float(weights[j])*ce
    return dict(clips=len(truth),weighted_accuracy=weighted_accuracy,
                weighted_cross_entropy=weighted_ce,**result)


def select_reference_components(inner_components,inner_truth,targets,classes,weights):
    """Call only on inner-validation predictions; independent reference per task."""
    if not inner_components: raise ValueError('No inner components to select from')
    metrics={name:component_metrics(inner_truth,p,targets,classes,weights)
             for name,p in inner_components.items()}
    return [min(metrics,key=lambda name:(-metrics[name][target+'_accuracy'],
            metrics[name][target+'_cross_entropy'],name)) for target in targets]


class ComponentOOF:
    def __init__(self,frame,truth,fold_ids,targets,classes,weights):
        self.frame=frame.reset_index(drop=True).copy(); self.truth=np.asarray(truth)
        self.folds=np.asarray(fold_ids); self.targets=list(targets)
        self.classes=[list(c) for c in classes]; self.weights=np.asarray(weights,dtype=float)
        n=len(self.frame)
        if not {'clip','source_id'}<=set(frame) or not frame['clip'].is_unique:
            raise ValueError('Unique clip IDs and source_id are required')
        if self.truth.shape!=(n,2) or self.folds.shape!=(n,) or not n:
            raise ValueError('Frame, labels, and saved fold assignments must align')
        if not np.issubdtype(self.folds.dtype,np.integer) or (self.folds<0).any():
            raise ValueError('Every evaluated clip must have a saved outer fold')
        if self.frame.source_id.isna().any(): raise ValueError('Missing source ID')
        groups=self.frame.assign(_fold=self.folds).groupby('source_id',dropna=False)._fold.nunique()
        if (groups!=1).any(): raise ValueError('One source appears in multiple outer folds')
        if len(targets)!=2 or not np.allclose(self.weights,[.7,.3]):
            raise ValueError('Expected two targets and 70/30 metric weights')
        # Validate labels even before any probability assignment.
        component_metrics(self.truth,[np.full((n,len(c)),1/len(c)) for c in self.classes],
                          self.targets,self.classes,self.weights)
        for j,target in enumerate(self.targets):
            if target in self.frame:
                expected=np.asarray(self.classes[j])[self.truth[:,j].astype(int)]
                if not np.array_equal(self.frame[target].to_numpy(),expected):
                    raise ValueError('Frame labels do not match the supplied label matrix')
        self.probabilities={}; self.coverage={}

    def add(self,component,indices,probabilities,clip_ids,class_order):
        ids=np.asarray(indices)
        if ids.ndim!=1 or not np.issubdtype(ids.dtype,np.integer) or not len(ids):
            raise ValueError('Prediction indices must be a nonempty integer vector')
        if len(np.unique(ids))!=len(ids) or (ids<0).any() or (ids>=len(self.frame)).any():
            raise ValueError('Duplicate or invalid prediction index')
        if not np.array_equal(np.asarray(clip_ids),self.frame['clip'].iloc[ids].to_numpy()):
            raise ValueError('Prediction clip order is misaligned')
        arrays=validate_component_probabilities(probabilities,len(ids),class_order,self.classes)
        if component not in self.probabilities:
            self.probabilities[component]=[np.full((len(self.frame),len(c)),np.nan) for c in self.classes]
            self.coverage[component]=np.zeros(len(self.frame),dtype=bool)
        if self.coverage[component][ids].any(): raise ValueError('Duplicate OOF predictions')
        for saved,p in zip(self.probabilities[component],arrays): saved[ids]=p
        self.coverage[component][ids]=True

    def require_complete(self):
        if not self.probabilities: raise ValueError('No component predictions')
        for name,p in self.probabilities.items():
            if not self.coverage[name].all():
                raise ValueError(f'Incomplete OOF coverage for {name}: {int(self.coverage[name].sum())}/{len(self.frame)}')
            validate_component_probabilities(p,len(self.frame),self.classes,self.classes)

    def error_rows(self,quality=None):
        self.require_complete(); chunks=[]
        for name,probs in self.probabilities.items():
            columns=[x for x in ['clip','source_id','preferred_foot','gender'] if x in self.frame]
            result=self.frame[columns].copy(); result['fold']=self.folds; result['component']=name
            result['weighted_loss']=0.; result['weighted_error']=0.
            for j,target in enumerate(self.targets):
                p=probs[j]; y=self.truth[:,j].astype(int); pred=p.argmax(1)
                result[target+'_true']=np.asarray(self.classes[j])[y]
                result[target+'_predicted']=np.asarray(self.classes[j])[pred]
                result[target+'_true_probability']=p[np.arange(len(y)),y]
                ce=-np.log(np.clip(p[np.arange(len(y)),y],1e-12,1))
                result[target+'_cross_entropy']=ce; result[target+'_correct']=pred==y
                for k,label in enumerate(self.classes[j]):
                    result[f'{target}_probability_{k}__{label}']=p[:,k]
                result['weighted_loss']+=float(self.weights[j])*ce
                result['weighted_error']+=float(self.weights[j])*(pred!=y)
            if quality is not None:
                if 'clip' not in quality or not quality['clip'].is_unique:
                    raise ValueError('Quality indicators require one row per clip')
                if set(quality['clip'])!=set(self.frame['clip']):
                    raise ValueError('Quality indicators must cover precisely the evaluated clips')
                overlap=(set(quality)-{'clip'})&set(result)
                if overlap: raise ValueError(f'Quality columns overlap report columns: {sorted(overlap)}')
                result=result.merge(quality,on='clip',how='left',validate='one_to_one',sort=False)
            chunks.append(result)
        return pd.concat(chunks,ignore_index=True)

    def metric_table(self):
        self.require_complete(); records=[]
        for name,p in self.probabilities.items():
            for fold in [*sorted(np.unique(self.folds)),'pooled']:
                ids=np.arange(len(self.frame)) if fold=='pooled' else np.flatnonzero(self.folds==fold)
                records.append(dict(component=name,fold=fold,**component_metrics(self.truth[ids],
                    [a[ids] for a in p],self.targets,self.classes,self.weights)))
        return pd.DataFrame(records)

    def ensemble_comparison(self,references,ensemble='ensemble'):
        self.require_complete()
        if ensemble not in self.probabilities: raise ValueError('Missing final ensemble')
        records=[]
        for fold in sorted(np.unique(self.folds)):
            reference=references.get(int(fold),references.get(str(int(fold))))
            if reference is None or len(reference)!=2: raise ValueError('Missing inner-selected references')
            ids=np.flatnonzero(self.folds==fold)
            for j,target in enumerate(self.targets):
                if reference[j] not in self.probabilities: raise ValueError('Reference component not exported')
                y=self.truth[ids,j]; a=self.probabilities[reference[j]][j][ids].argmax(1)
                b=self.probabilities[ensemble][j][ids].argmax(1)
                corrected=int(((a!=y)&(b==y)).sum()); harmed=int(((a==y)&(b!=y)).sum())
                records.append(dict(fold=int(fold),target=target,reference_component=reference[j],clips=len(ids),
                    mistakes_corrected=corrected,correct_changed_to_mistake=harmed,
                    disagreements=int((a!=b).sum()),disagreement_rate=float((a!=b).mean()),
                    accuracy_delta=(corrected-harmed)/len(ids),
                    weighted_score_delta=float(self.weights[j])*(corrected-harmed)/len(ids),
                    contribution_to_pooled_weighted_delta=float(self.weights[j])*(corrected-harmed)/len(self.frame)))
        for j,target in enumerate(self.targets):
            subset=[r for r in records if r['target']==target]
            corrected=sum(r['mistakes_corrected'] for r in subset); harmed=sum(r['correct_changed_to_mistake'] for r in subset)
            disagreements=sum(r['disagreements'] for r in subset); n=len(self.frame)
            records.append(dict(fold='pooled',target=target,reference_component='fold-specific inner-selected reference',
                clips=n,mistakes_corrected=corrected,correct_changed_to_mistake=harmed,disagreements=disagreements,
                disagreement_rate=disagreements/n,accuracy_delta=(corrected-harmed)/n,
                weighted_score_delta=float(self.weights[j])*(corrected-harmed)/n,
                contribution_to_pooled_weighted_delta=float(self.weights[j])*(corrected-harmed)/n))
        return pd.DataFrame(records)

    def export(self,output,quality=None,references=None,ensemble='ensemble'):
        self.require_complete(); output=Path(output); output.mkdir(parents=True,exist_ok=True)
        errors=self.error_rows(quality); metrics=self.metric_table()
        slices,confusions=component_slice_reports(errors,self.targets,self.classes,self.weights,len(self.frame))
        errors.to_csv(output/'component_clip_errors.csv',index=False)
        metrics.to_csv(output/'component_metrics.csv',index=False)
        errors.sort_values(['component','weighted_loss'],ascending=[True,False],kind='stable').to_csv(output/'highest_loss.csv',index=False)
        errors.sort_values(['component','weighted_error','weighted_loss'],ascending=[True,False,False],kind='stable').to_csv(output/'greatest_accuracy_error.csv',index=False)
        slices.to_csv(output/'component_slices.csv',index=False); confusions.to_csv(output/'component_confusions.csv',index=False)
        manifest=dict(targets=self.targets,classes=self.classes,weights=self.weights.tolist(),
            components=list(self.probabilities),clips=self.frame['clip'].tolist(),fold_ids=self.folds.tolist(),
            interpretation='Outer-fold diagnostics are exploratory development feedback, not an untouched final estimate. '
                'Loss is cross-entropy; score is accuracy. Slices overlap and are not additive. Quality associations are not causal.',
            probability_columns={target:[f'{target}_probability_{k}__{label}' for k,label in enumerate(classes)]
                                 for target,classes in zip(self.targets,self.classes)})
        if references is not None:
            manifest['inner_selected_reference_components']={str(k):list(v) for k,v in references.items()}
        (output/'component_manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
        arrays={'clip':self.frame['clip'].to_numpy(dtype=str),'fold_ids':self.folds,'truth':self.truth}
        for index,(name,probs) in enumerate(self.probabilities.items()):
            for j,p in enumerate(probs): arrays[f'component_{index}_target_{j}']=p
        np.savez_compressed(output/'component_oof_probabilities.npz',**arrays)
        result=dict(errors=errors,metrics=metrics,slices=slices,confusions=confusions)
        if references is not None:
            comparison=self.ensemble_comparison(references,ensemble)
            comparison.to_csv(output/'ensemble_comparison.csv',index=False); result['comparison']=comparison
        return result


def component_slice_reports(errors,targets,classes,weights,population):
    """Each component has the same complete evaluated population denominator."""
    data=errors.copy(); fields=[c for c in ['source_id','preferred_foot','gender'] if c in data]
    fields += [t+'_true' for t in targets]
    # Bin continuous quality indicators; retain categorical fields as reported.
    quality_fields=[c for c in data if any(k in c.lower() for k in
        ['tracking','tracked','coverage','pose','camera','fallback','jump','gap','confidence','clipping'])
        and not c.startswith(tuple(targets)) and c not in fields]
    for c in quality_fields:
        if pd.api.types.is_numeric_dtype(data[c]) and not pd.api.types.is_bool_dtype(data[c]):
            name=c+'_slice'
            if any(k in c for k in ['fraction','coverage','confidence','rate']):
                data[name]=pd.cut(data[c],[-np.inf,0,.5,.9,np.inf],labels=['zero_or_less','low_(0,.5]','medium_(.5,.9]','high_(.9,+inf)']).astype('object').fillna('missing')
            else:
                data[name]=pd.cut(data[c],[-np.inf,0,1,5,np.inf],labels=['zero_or_less','(0,1]','(1,5]','above_5']).astype('object').fillna('missing')
            fields.append(name)
        elif data[c].nunique(dropna=False)<=30: fields.append(c)
    records=[]; matrices=[]
    for component,part in data.groupby('component',sort=False):
        if len(part)!=population: raise ValueError('Slice denominator must be the full evaluated population')
        for field in fields:
            for value,g in part.groupby(field,dropna=False,observed=True):
                row=dict(component=component,slice_field=field,slice_value=str(value),clips=len(g),
                    small_slice=len(g)<20,weighted_accuracy=1-float(g.weighted_error.mean()),
                    mean_weighted_loss=float(g.weighted_loss.mean()),
                    contribution_to_total_weighted_errors=float(g.weighted_error.sum()/population),
                    population_denominator=population)
                for j,t in enumerate(targets):
                    row[t+'_accuracy']=float(g[t+'_correct'].mean())
                    row[t+'_mean_cross_entropy']=float(g[t+'_cross_entropy'].mean())
                    row[t+'_weighted_error_contribution']=float(weights[j])*(~g[t+'_correct']).sum()/population
                records.append(row)
        for j,t in enumerate(targets):
            for true in classes[j]:
                for predicted in classes[j]:
                    matrices.append(dict(component=component,target=t,true_label=true,predicted_label=predicted,
                        clips=int(((part[t+'_true']==true)&(part[t+'_predicted']==predicted)).sum())))
    return pd.DataFrame(records),pd.DataFrame(matrices)


def smoke_component_reports(output):
    """Synthetic smoke scores test arithmetic only; they are not model performance."""
    targets=['kick_direction','goalkeeper_direction']; classes=[['left','middle','right'],['left','stands in the middle','right']]
    weights=[.7,.3]; y=np.array([[0,1],[1,2],[2,0],[0,1]])
    frame=pd.DataFrame({'clip':['a','b','c','d'],'source_id':['s0','s0','s1','s1'],'gender':['F','M','F','M']})
    folds=np.array([0,0,1,1]); base=[np.eye(3)[y[:,j]]*.8+.2/3 for j in range(2)]
    worse=[p.copy() for p in base]; worse[0][0]=[.1,.8,.1]; worse[1][1]=[.8,.1,.1]
    report=ComponentOOF(frame,y,folds,targets,classes,weights)
    report.add('video_seed42',[0,1],[p[:2] for p in worse],frame['clip'][:2],classes)
    try: report.require_complete()
    except ValueError: pass
    else: raise AssertionError('Incomplete OOF predictions accepted')
    report.add('video_seed42',[2,3],[p[2:] for p in worse],frame['clip'][2:],classes)
    report.add('ensemble',np.arange(4),base,frame['clip'],classes)
    for bad_ids,bad_clips,bad_classes in [([0],['wrong'],classes),([0],['a'],[classes[0][::-1],classes[1]])]:
        try: report.add('bad',bad_ids,[p[:1] for p in base],bad_clips,bad_classes)
        except ValueError: pass
        else: raise AssertionError('Misalignment accepted')
    for bad_probabilities in [[np.full((1,3),np.nan),base[1][:1]],
                              [np.zeros((1,3)),base[1][:1]],
                              [np.ones((1,2))*.5,base[1][:1]]]:
        try: report.add('bad',[0],bad_probabilities,['a'],classes)
        except ValueError: pass
        else: raise AssertionError('Invalid probability array accepted')
    try: report.add('video_seed42',[0],[p[:1] for p in base],['a'],classes)
    except ValueError: pass
    else: raise AssertionError('Duplicate OOF prediction accepted')
    try: ComponentOOF(frame,y,np.array([0,1,1,1]),targets,classes,weights)
    except ValueError: pass
    else: raise AssertionError('Source leakage accepted')
    quality=pd.DataFrame({'clip':frame['clip'],'kicker_tracked_fraction':[0,.5,1,1],
                          'pose_valid_fraction':[0,.2,.8,1],'camera_valid_fraction':[0,1,1,1]})
    results=report.export(output,quality=quality,references={0:['video_seed42']*2,1:['video_seed42']*2})
    e=results['errors'].query("component=='video_seed42'")
    np.testing.assert_allclose(e.weighted_error,[.7,.3,0,0])
    np.testing.assert_allclose(e.weighted_loss,.7*e.kick_direction_cross_entropy+.3*e.goalkeeper_direction_cross_entropy)
    pooled=results['comparison'].query("fold=='pooled'")
    np.testing.assert_allclose(pooled.weighted_score_delta.sum(),.25)
    source_slices=results['slices'].query("component=='video_seed42' and slice_field=='source_id'")
    np.testing.assert_allclose(source_slices.contribution_to_total_weighted_errors.sum(),.25)
    assert results['confusions'].groupby(['component','target']).clips.sum().eq(4).all()
    selected=select_reference_components({'good':base,'bad':worse},y,targets,classes,weights)
    assert selected==['good','good']
    return dict(status='PASS',checks=['complete OOF coverage','clip/class alignment','70/30 accuracy and CE',
        'slice population denominator','ensemble corrections','CSV/NPZ export'],
        note='Synthetic execution check only; scores are not performance estimates.')


"""Post-prediction role/tracking diagnostics; embed this file as a notebook cell.

These functions never generate crops, fit selectors, or change model inputs.
Click agreement is a localization proxy, not an identity annotation.
"""
import json
from pathlib import Path
import numpy as np
import pandas as pd


def _inside_box(point, box):
    return bool(np.all(point >= box[:2]) and np.all(point <= box[2:]))


def _longest_missing_run(valid):
    longest = current = 0
    for present in valid:
        current = 0 if present else current + 1
        longest = max(longest, current)
    return longest


def tracking_clip_diagnostics(clip, source_id, fold, raw, view, annotation=None,
                              timestamp_tolerance=.10, jump_threshold_heights=1.25):
    """Return (one clip-quality row, per-frame rows, heldout click rows).

    Track indices refer to the cached detections used to generate the actual crops.
    Missing detections produce full-frame fallback crops in the current pipeline.
    Distances use pixels divided by selected player height, preserving aspect ratio.
    """
    times = np.asarray(raw['times'], dtype=float)
    ids = np.asarray(view.get('track_ids', np.full((len(times), 2), -1)), dtype=int)
    traces = np.asarray(view['traces'])
    if ids.shape != (len(times), 2) or traces.shape != (len(times), 2, 3):
        raise ValueError('Tracking/cache shape does not match sampled timestamps')
    if 'track_ids' not in view:
        raise ValueError('Rebuild old crop caches: selected detection indices are required for auditing')
    base = dict(clip=clip, source_id=source_id, fold=int(fold))
    summary = dict(base)
    frames = []
    role_records = {}
    for r, role in enumerate(['kicker', 'keeper']):
        records = []
        previous_box = None
        previous_t = None
        previous_center = None
        for t, (rgb, detection) in enumerate(zip(raw['frames'], raw['detections'])):
            boxes = np.asarray(detection['boxes']).reshape(-1, 4)
            h, w = rgb.shape[:2]
            j = int(ids[t, r])
            present = 0 <= j < len(boxes)
            if j >= len(boxes):
                raise ValueError('Saved track points outside cached detections')
            record = dict(base, role=role, frame=t, time_seconds=float(times[t]),
                          detection_count=len(boxes), selected_detection=j,
                          selected=present, fallback='none' if present else 'full_frame',
                          role_confidence=float(traces[t, r, 2]) if present else 0.,
                          camera_valid=bool(raw['camera_quality'][t] > 0),
                          camera_quality=float(raw['camera_quality'][t]),
                          pose_visible_fraction=0., crop_clipped=False,
                          raw_jump_player_heights=np.nan, gap_since_previous_selected_seconds=np.nan,
                          suspicious_jump=False)
            for name in ['box_x1','box_y1','box_x2','box_y2',
                         'crop_x1','crop_y1','crop_x2','crop_y2']:
                record[name] = np.nan
            if present:
                box = boxes[j].astype(float)
                center = (box[:2] + box[2:]) / 2
                crop_size = (box[2:] - box[:2]) * [1.7, 1.35]
                intended_lo, intended_hi = center - crop_size/2, center + crop_size/2
                lo, hi = np.clip(intended_lo, 0, 1), np.clip(intended_hi, 0, 1)
                record.update(dict(zip(['box_x1','box_y1','box_x2','box_y2'], box.tolist())))
                record.update(dict(zip(['crop_x1','crop_y1','crop_x2','crop_y2'], np.r_[lo,hi].tolist())))
                record['crop_clipped'] = bool(np.any(intended_lo < 0) or np.any(intended_hi > 1))
                confidence = np.asarray(detection['confidence'])[j]
                record['pose_visible_fraction'] = float((confidence >= .25).mean())
                if previous_box is not None:
                    height = max(float((previous_box[3]-previous_box[1])*h), 2.)
                    jump = float(np.linalg.norm(center*[w,h] - previous_center*[w,h])/height)
                    record['raw_jump_player_heights'] = jump
                    record['gap_since_previous_selected_seconds'] = float(times[t]-previous_t)
                    record['suspicious_jump'] = jump > jump_threshold_heights
                previous_box, previous_center, previous_t = box, center, times[t]
            records.append(record)
        role_records[role] = records
        frames.extend(records)
        selected = np.array([x['selected'] for x in records])
        summary[role+'_tracked_fraction'] = float(selected.mean())
        summary[role+'_fallback_fraction'] = float((~selected).mean())
        summary[role+'_missing_frame_count'] = int((~selected).sum())
        summary[role+'_longest_missing_run_frames'] = _longest_missing_run(selected)
        summary[role+'_tracking_quality'] = ('missing' if not selected.any() else
                                            'partial' if selected.mean() < .8 else 'mostly_tracked')
        summary[role+'_mean_confidence'] = float(np.mean([x['role_confidence'] for x in records]))
        summary[role+'_pose_visible_fraction'] = float(np.mean([x['pose_visible_fraction'] for x in records]))
        summary[role+'_pose_quality'] = ('low' if summary[role+'_pose_visible_fraction'] < .5 else 'high')
        summary[role+'_clipped_crop_fraction'] = float(np.mean([x['crop_clipped'] for x in records]))
        summary[role+'_suspicious_jump_count'] = int(sum(x['suspicious_jump'] for x in records))
        # Observed gaps are sample-grid statistics, not continuous-time identities.
        gaps = [x['gap_since_previous_selected_seconds'] for x in records
                if np.isfinite(x['gap_since_previous_selected_seconds'])]
        summary[role+'_max_selected_gap_seconds'] = float(max(gaps)) if gaps else np.nan
    summary['camera_valid_fraction'] = float((np.asarray(raw['camera_quality']) > 0).mean())
    summary['camera_quality'] = 'low' if summary['camera_valid_fraction'] < .5 else 'high'
    summary['detector_empty_fraction'] = float(np.mean([len(x['boxes']) == 0 for x in raw['detections']]))
    summary['pose_quality'] = 'low' if np.mean([summary[r+'_pose_visible_fraction'] for r in ['kicker','keeper']]) < .5 else 'high'
    summary['tracking_quality'] = 'low' if min(summary[r+'_tracked_fraction'] for r in ['kicker','keeper']) < .8 else 'high'
    click_rows = []
    if annotation is not None:
        for r, role in enumerate(['kicker','keeper']):
            fields = [role+'_click_time_seconds', role+'_x_norm', role+'_y_norm']
            if any(f not in annotation or pd.isna(annotation[f]) for f in fields):
                continue
            click_time = float(annotation[fields[0]])
            point = np.array([annotation[fields[1]], annotation[fields[2]]], dtype=float)
            t = int(np.argmin(np.abs(times-click_time)))
            record = role_records[role][t]
            boxes = np.asarray(raw['detections'][t]['boxes']).reshape(-1,4)
            candidates_inside = [i for i,b in enumerate(boxes) if _inside_box(point,b)]
            selected_inside = bool(record['selected'] and record['selected_detection'] in candidates_inside)
            mismatch = float(abs(times[t]-click_time))
            eligible = mismatch <= timestamp_tolerance
            # Exact click-nearest detector frame diagnoses detection independently,
            # without rerunning role selection or altering sampled crop inputs.
            exact = raw.get('clicked', [])
            exact_d = exact[r] if len(exact)>r else None
            exact_boxes = np.asarray(exact_d['boxes']).reshape(-1,4) if exact_d is not None else np.empty((0,4))
            exact_match = any(_inside_box(point,b) for b in exact_boxes) if exact_d is not None else None
            distance = np.nan
            if record['selected']:
                box = boxes[record['selected_detection']]
                h,w = raw['frames'][t].shape[:2]
                center = (box[:2]+box[2:])/2
                distance = float(np.linalg.norm((point-center)*[w,h])/max((box[3]-box[1])*h,2.))
            if not eligible: failure = 'timestamp_mismatch_unresolved'
            elif not candidates_inside: failure = 'no_click_containing_detection_proxy'
            elif not record['selected']: failure = 'role_missing_with_candidate'
            elif not selected_inside: failure = 'selected_other_box_proxy'
            else: failure = 'click_inside_selected_box_proxy'
            click_rows.append(dict(base,role=role,annotation_time_seconds=click_time,
                sampled_time_seconds=float(times[t]),sample_offset_seconds=mismatch,
                timestamp_eligible=eligible,timestamp_tolerance_seconds=timestamp_tolerance,
                selected=record['selected'],missing_role=not record['selected'],
                click_inside_selected_box=selected_inside,
                click_distance_player_heights=distance,
                detection_count=len(boxes),click_containing_detection_count=len(candidates_inside),
                unmatched_detection_proxy=not bool(candidates_inside),
                failure_category=failure,exact_click_detector_available=exact_d is not None,
                exact_click_frame_offset_seconds=float(abs(exact_d['time']-click_time)) if exact_d is not None else np.nan,
                exact_click_unmatched_detection_proxy=(not exact_match) if exact_d is not None else np.nan))
    return summary, frames, click_rows


def assert_tracking_audit_provenance(indices, mapping, fit_sources=None):
    """Require evidence that audited clips were held out from the role selector."""
    held = set(rows.source_id.iloc[list(indices)].tolist())
    if fit_sources is not None:
        assert not held & set(fit_sources), 'Audit sources overlap learned role fit sources'
    for i in indices:
        provenance = Path(mapping[int(i)]).parent/'provenance.json'
        if provenance.exists():
            entries = json.loads(provenance.read_text())
            matching = [e for e in entries if int(i) in e['prediction_indices']]
            assert len(matching)==1 and matching[0]['kind']=='evaluation', 'Audit requires heldout evaluation crops'
            assert rows.source_id.iloc[int(i)] not in matching[0]['fit_sources']
        elif fit_sources is None:
            raise ValueError('Tracking audit requires saved provenance or explicit role fit sources')


def summarize_role_click_audits(audits):
    """Aggregate rates with explicit timestamp eligibility; empty annotations are valid."""
    summaries=[]
    if not len(audits): return pd.DataFrame(summaries)
    for role,all_rows in audits.groupby('role'):
        aligned=all_rows[all_rows.timestamp_eligible]
        summaries.append(dict(role=role,annotated_clips=len(all_rows),timestamp_eligible_clips=len(aligned),
            timestamp_mismatch_rate=float((~all_rows.timestamp_eligible).mean()),
            mean_timestamp_offset_seconds=float(all_rows.sample_offset_seconds.mean()),
            missing_role_rate=float(aligned.missing_role.mean()) if len(aligned) else np.nan,
            unmatched_detection_proxy_rate=float(aligned.unmatched_detection_proxy.mean()) if len(aligned) else np.nan,
            click_inside_selected_box_rate=float(aligned.click_inside_selected_box.mean()) if len(aligned) else np.nan,
            mean_distance_player_heights=float(aligned.click_distance_player_heights.mean()),
            note='Nearest sampled prediction, not inference rerun at annotation time. '
                 'Aligned sample frames only for role rates; localization proxies are not identity accuracy.'))
    return pd.DataFrame(summaries)


def collect_tracking_diagnostics(indices,mapping,fold,output_dir=None,fit_sources=None):
    """Audit only already-predicted public clips; returns quality/frame/click tables."""
    indices=list(map(int,indices))
    assert_tracking_audit_provenance(indices,mapping,fit_sources)
    quality=[]; frame_rows=[]; click_rows=[]
    for i in indices:
        row=rows.iloc[i]; clip=row['clip']
        annotation=clicks.loc[clip] if clip in clicks.index else None
        summary,frames,audits=tracking_clip_diagnostics(clip,row.source_id,fold,
            extract_raw(clip),load_view(mapping,i),annotation)
        quality.append(summary); frame_rows.extend(frames); click_rows.extend(audits)
    quality=pd.DataFrame(quality); frames=pd.DataFrame(frame_rows); audits=pd.DataFrame(click_rows)
    if output_dir is not None:
        output_dir=Path(output_dir); output_dir.mkdir(parents=True,exist_ok=True)
        quality.to_csv(output_dir/'tracking_quality.csv',index=False)
        frames.to_csv(output_dir/'tracking_frames.csv',index=False)
        audits.to_csv(output_dir/'role_click_audit.csv',index=False)
        summarize_role_click_audits(audits).to_csv(output_dir/'role_click_summary.csv',index=False)
        (output_dir/'tracking_limitations.txt').write_text(
            'All audits are post-prediction. Click-containing boxes are localization proxies, not identity truth. '
            'No identity-switch rate is measurable from the available isolated clicks. Raw jumps may reflect camera motion. '
            'Role predictions are taken from the nearest sampled classification frame; inference is not rerun at the annotated time. '
            'Exact click-nearest detector frames are used only for the separate detector-containment proxy. '
            'Timestamp-ineligible clicks are excluded from role rates. Missing roles use full-frame crops. '
            'Tracking-quality associations with prediction errors are not causal component effects.\n')
    return quality,frames,audits


def render_tracking_audits(error_df,mapping,output_dir,high_loss_count=3,correct_count=2,seed=42):
    """Save context/pose/crop panels for high-loss and reproducibly sampled correct clips."""
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    output_dir=Path(output_dir); output_dir.mkdir(parents=True,exist_ok=True)
    errors=error_df.copy()
    loss_column='weighted_loss' if 'weighted_loss' in errors else 'weighted_log_loss'
    errors=errors.drop_duplicates('clip').sort_values('clip')
    high=errors.sort_values([loss_column,'clip'],ascending=[False,True]).head(high_loss_count)
    correct=errors[(errors.weighted_error==0)&~errors['clip'].isin(high['clip'])]
    if len(correct): correct=correct.sample(n=min(correct_count,len(correct)),random_state=seed)
    selected=pd.concat([high.assign(audit_selection='highest_loss'),correct.assign(audit_selection='sampled_correct')])
    selected[['clip','audit_selection']].assign(selection_seed=seed).to_csv(output_dir/'visual_audit_selection.csv',index=False)
    clip_indices={rows['clip'].iloc[int(i)]:int(i) for i in mapping}
    for _,entry in selected.iterrows():
        i=clip_indices[entry['clip']]; assert_tracking_audit_provenance([i],mapping)
        raw=extract_raw(entry['clip']); view=load_view(mapping,i,decode_video=True)
        times=np.asarray(raw['times']); anchor=int(view.get('anchor_frame',len(times)//2))
        time_ids=sorted(set([0,anchor,len(times)//2,len(times)-1]))
        fig,axes=plt.subplots(len(time_ids),3,figsize=(12,3.2*len(time_ids)),squeeze=False)
        for row_number,t in enumerate(time_ids):
            rgb=raw['frames'][t]; h,w=rgb.shape[:2]; ax=axes[row_number,0]; ax.imshow(rgb)
            detection=raw['detections'][t]
            for r,(role,color) in enumerate([('kicker','orange'),('keeper','cyan')]):
                j=int(view['track_ids'][t,r]); present=0<=j<len(detection['boxes'])
                if present:
                    box=np.asarray(detection['boxes'][j])*[w,h,w,h]
                    ax.add_patch(Rectangle(box[:2],box[2]-box[0],box[3]-box[1],fill=False,edgecolor=color,linewidth=2))
                    joints=np.asarray(detection['joints'][j])*[w,h]
                    visible=np.asarray(detection['confidence'][j])>=.25
                    ax.scatter(joints[visible,0],joints[visible,1],s=8,c=color)
                if entry['clip'] in clicks.index:
                    annotation=clicks.loc[entry['clip']]
                    offset=abs(float(times[t])-float(annotation[f'{role}_click_time_seconds']))
                    if offset<=.10:
                        ax.scatter(annotation[f'{role}_x_norm']*w,annotation[f'{role}_y_norm']*h,
                                   s=70,c=color,marker='x',linewidths=2)
                axes[row_number,r+1].imshow(view['video'][r+1,t].transpose(1,2,0))
                axes[row_number,r+1].set_title(f'{role}: '+('selected crop' if present else 'FULL-FRAME FALLBACK'))
            ax.set_title(f'Context + selected boxes/pose | t={times[t]:.3f}s')
            for a in axes[row_number]: a.axis('off')
        fig.suptitle(f"{entry['clip']} | {entry['audit_selection']} | loss={entry[loss_column]:.3f}\n"
                     'Orange kicker / cyan keeper; crosses: post-prediction public clicks within 0.10 s. Not identity ground truth.')
        fig.tight_layout(); fig.savefig(output_dir/(Path(entry['clip']).stem+'_audit.png'),dpi=130,bbox_inches='tight')
        plt.close(fig)
    return selected[['clip','audit_selection']]


def tracking_diagnostics_smoke(output_dir):
    """CPU-only synthetic audit checks; no videos, labels, detector, or model needed."""
    output_dir=Path(output_dir); output_dir.mkdir(parents=True,exist_ok=True)
    box=np.array([[.4,.3,.6,.7]],dtype=float)
    detection=dict(boxes=box,joints=np.full((1,17,2),.5),confidence=np.ones((1,17)))
    empty=dict(boxes=np.empty((0,4)),joints=np.empty((0,17,2)),confidence=np.empty((0,17)))
    raw=dict(times=np.array([0.,.5,1.]),frames=[np.zeros((100,200,3),dtype=np.uint8) for _ in range(3)],
             detections=[detection,empty,detection],camera_quality=np.array([0.,0.,1.]),
             clicked=[dict(time=0.,**detection),dict(time=.5,**empty)])
    view=dict(track_ids=np.array([[0,-1],[-1,-1],[0,-1]]),
              traces=np.array([[[.5,.5,.8],[0,0,0]],[[0,0,0],[0,0,0]],[[.5,.5,.7],[0,0,0]]]))
    annotation={'kicker_click_time_seconds':0.,'kicker_x_norm':.6,'kicker_y_norm':.5,
                'keeper_click_time_seconds':.5,'keeper_x_norm':.2,'keeper_y_norm':.4}
    quality,frames,audits=tracking_clip_diagnostics('synthetic.mp4','heldout_source',0,raw,view,annotation)
    assert quality['kicker_tracked_fraction']==2/3 and quality['keeper_fallback_fraction']==1.
    assert quality['kicker_longest_missing_run_frames']==1
    assert quality['keeper_longest_missing_run_frames']==3
    np.testing.assert_allclose(audits[0]['click_distance_player_heights'],.5)
    assert audits[0]['click_inside_selected_box']
    assert audits[1]['missing_role'] and audits[1]['unmatched_detection_proxy']
    assert frames[1]['fallback']=='full_frame'
    # A temporally remote click must not be described as an identity/role failure.
    off=dict(annotation,kicker_click_time_seconds=3.)
    _,_,mismatch=tracking_clip_diagnostics('synthetic.mp4','heldout_source',0,raw,view,off)
    assert mismatch[0]['failure_category']=='timestamp_mismatch_unresolved'
    # A selected wrong box can be distinguished from no click-containing detection.
    two=dict(detection,boxes=np.r_[box,[[.1,.2,.3,.6]]],
             joints=np.full((2,17,2),.5),confidence=np.ones((2,17)))
    raw_other=dict(raw,detections=[two,empty,detection])
    view_other=dict(view,track_ids=np.array([[1,-1],[-1,-1],[0,-1]]))
    _,_,wrong=tracking_clip_diagnostics('synthetic.mp4','heldout_source',0,raw_other,view_other,annotation)
    assert wrong[0]['failure_category']=='selected_other_box_proxy'
    pd.DataFrame([quality]).to_csv(output_dir/'tracking_quality.csv',index=False)
    pd.DataFrame(frames).to_csv(output_dir/'tracking_frames.csv',index=False)
    pd.DataFrame(audits).to_csv(output_dir/'role_click_audit.csv',index=False)
    return dict(status='PASS',checks=['height-normalized aspect-correct distances','missing role/full-frame fallback',
        'tracking coverage and gaps','timestamp mismatch exclusion','detector/selector failure proxies'],
        limitation='Synthetic execution check only; no performance or identity-switch estimate.')


"""Self-contained, retraining-based stream and numeric-feature ablations.

The notebook embeds this source. No companion file is needed in Colab.
"""
import copy
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


ABLATION_CONFIGS = {
    'A_context': dict(streams=[0], pose=False, motion='none', metadata=False),
    'B_context_kicker': dict(streams=[0, 1], pose=False, motion='none', metadata=False),
    'C_context_keeper': dict(streams=[0, 2], pose=False, motion='none', metadata=False),
    'D_all_visual': dict(streams=[0, 1, 2], pose=False, motion='none', metadata=False),
    'E_plus_pose': dict(streams=[0, 1, 2], pose=True, motion='none', metadata=False),
    'F_raw_motion': dict(streams=[0, 1, 2], pose=True, motion='raw', metadata=False),
    'G_compensated_motion': dict(streams=[0, 1, 2], pose=True, motion='compensated', metadata=False),
    'H_plus_metadata': dict(streams=[0, 1, 2], pose=True, motion='compensated', metadata=True),
    # This exact feature set preserves the pre-diagnostics production representation.
    'production': dict(streams=[0, 1, 2], pose=True, motion='both', metadata=True,
                       legacy_full_summary=True),
}
ACTIVE_ABLATION_NAME = 'production'
ACTIVE_ABLATION = copy.deepcopy(ABLATION_CONFIGS[ACTIVE_ABLATION_NAME])

# Per-player make_views layout: 34 normalized joint coordinates, 17 confidences,
# center x/y, width/height, role confidence, 36 raw velocities, 36 compensated
# velocities, 18 validity flags. Two players (146 each), then seven camera/time fields.
ROLE_FEATURE_WIDTH = 146
FRAME_FEATURE_WIDTH = 299
STATIC_POSE = tuple(range(0, 56))
RAW_MOTION = tuple(range(56, 92))
COMPENSATED_MOTION = tuple(range(92, 128))
COMPENSATED_VALIDITY = tuple(range(128, 146))
CAMERA_FEATURES = tuple(range(292, 297))
TIME_FEATURES = (297, 298)
SUMMARY_OPERATIONS = ('mean', 'std', 'last', 'last_minus_first')
METADATA_NAMES = ('preferred_foot_left', 'preferred_foot_right', 'gender_F', 'gender_M')


def set_active_ablation(name):
    global ACTIVE_ABLATION_NAME, ACTIVE_ABLATION
    if name not in ABLATION_CONFIGS:
        raise ValueError(f'Unknown ablation: {name}')
    ACTIVE_ABLATION_NAME = name
    ACTIVE_ABLATION = copy.deepcopy(ABLATION_CONFIGS[name])
    return copy.deepcopy(ACTIVE_ABLATION)


def ablation_stream_ids(config=None):
    config = ACTIVE_ABLATION if config is None else config
    streams = tuple(config['streams'])
    if not streams or len(set(streams)) != len(streams) or any(s not in (0, 1, 2) for s in streams):
        raise ValueError('Streams must be a nonempty unique subset of context/kicker/keeper [0,1,2].')
    return streams


def _ablation_numeric_blocks(config):
    """Return (columns, reductions, group); exclusions happen before every reduction."""
    if config.get('legacy_full_summary', False):
        return [(tuple(range(FRAME_FEATURE_WIDTH)), SUMMARY_OPERATIONS, 'production_all')]
    blocks = []
    if config['pose']:
        # No pose delta/std/temporal pooling in the pose-only condition: these can
        # themselves encode movement. Pose and geometry are measured at the final frame.
        blocks.append((tuple(r * ROLE_FEATURE_WIDTH + j for r in range(2) for j in STATIC_POSE),
                       ('last',), 'static_pose_geometry_confidence'))
    if config['motion'] in ('raw', 'both'):
        blocks.append((tuple(r * ROLE_FEATURE_WIDTH + j for r in range(2) for j in RAW_MOTION),
                       SUMMARY_OPERATIONS, 'raw_motion'))
    if config['motion'] in ('compensated', 'both'):
        blocks.append((tuple(r * ROLE_FEATURE_WIDTH + j for r in range(2) for j in COMPENSATED_MOTION),
                       SUMMARY_OPERATIONS, 'compensated_motion'))
        blocks.append((tuple(r * ROLE_FEATURE_WIDTH + j for r in range(2) for j in COMPENSATED_VALIDITY),
                       ('mean', 'last'), 'compensated_validity'))
    if config['motion'] != 'none':
        # Keep exactly the same time features in raw/compensated comparisons.
        blocks.append((TIME_FEATURES, ('mean', 'last'), 'sampling_time'))
    return blocks


def numeric_feature_manifest(config=None):
    config = ACTIVE_ABLATION if config is None else config
    result = []
    for columns, reductions, group in _ablation_numeric_blocks(config):
        for operation in reductions:
            result.extend(dict(group=group, raw_column=int(i), reduction=operation,
                               name=f'{group}__column_{i}__{operation}') for i in columns)
    if config['metadata']:
        result.extend(dict(group='metadata', raw_column=None, reduction='none', name=name)
                      for name in METADATA_NAMES)
    return result


def ablation_numeric_features(value, metadata_row, config=None):
    config = ACTIVE_ABLATION if config is None else config
    ablation_stream_ids(config)
    x = np.asarray(value['numeric'], dtype=np.float32)
    if x.ndim != 2 or x.shape[1] != FRAME_FEATURE_WIDTH or len(x) == 0:
        raise ValueError(f'Expected nonempty T x {FRAME_FEATURE_WIDTH} numeric cache, got {x.shape}')
    features = []
    for columns, reductions, _ in _ablation_numeric_blocks(config):
        selected = x[:, columns]
        operations = dict(mean=lambda: selected.mean(0), std=lambda: selected.std(0),
                          last=lambda: selected[-1], last_minus_first=lambda: selected[-1]-selected[0])
        features.extend(operations[name]() for name in reductions)
    if config['metadata']:
        meta = np.asarray(metadata_row, dtype=np.float32)
        if meta.shape != (4,):
            raise ValueError('Expected the four existing preferred-foot/gender metadata indicators')
        features.append(meta)
    result = np.concatenate(features).astype(np.float32) if features else np.empty(0, dtype=np.float32)
    if len(result) != len(numeric_feature_manifest(config)) or not np.isfinite(result).all():
        raise ValueError('Non-finite or mismatched ablation features')
    return result


class EmptyNumericScaler:
    """A visual-only model has no numeric branch and no learned preprocessing."""
    n_features_in_ = 0

    def transform(self, values):
        values = np.asarray(values)
        if values.ndim != 2 or values.shape[1] != 0:
            raise ValueError('Visual-only scaler requires an empty numeric feature matrix')
        return values.astype(np.float32)


def fit_feature_scaler(training_features):
    values = np.asarray(training_features)
    if values.ndim != 2 or not len(values):
        raise ValueError('Scaler input must be a nonempty training matrix')
    if values.shape[1] == 0:
        return EmptyNumericScaler()
    from sklearn.preprocessing import StandardScaler
    return StandardScaler().fit(values)


def make_ablation_model_class():
    """Call after the notebook imports torch/transformers; use as MultiViewVideo.

    Keeping this a factory lets feature smoke checks run without importing torch
    or downloading any weights. Notebook globals are read at runtime: BACKBONE_SPEC,
    TRAINING_POLICY, N_FRAMES, SIZE, CLASSES. The encoder comes from BackboneAdapter.
    """
    import torch

    class ConfigurableMultiViewVideo(torch.nn.Module):
        def __init__(self, numeric_dim, architecture=None):
            """architecture=None loads pretrained encoder weights; a saved architecture dict rebuilds the
            same modules without a pretrained download (inference then loads the run's own weights)."""
            super().__init__()
            self.stream_ids = ablation_stream_ids()
            self.has_numeric = numeric_dim > 0
            self.feature_signature = ablation_configuration_manifest()['feature_signature']
            if numeric_dim != len(numeric_feature_manifest()):
                raise ValueError('Numeric feature dimension differs from active ablation manifest')
            spec = globals().get('BACKBONE_SPEC') or backbone_spec('videomae')
            self.training_policy = globals().get('TRAINING_POLICY', 'finetune')
            if (spec['frames'], spec['size']) != (N_FRAMES, SIZE):
                raise ValueError(f'Backbone expects {spec["frames"]} frames at {spec["size"]}px; crops use {N_FRAMES} at {SIZE}px')
            self.adapter = BackboneAdapter(spec, self.training_policy, repo_dir=globals().get('VJEPA2_REPO_DIR'))
            self.backbone, self.norm, self.architecture = self.adapter.build(architecture)
            hidden = self.adapter.feature_dim
            self.numeric = (torch.nn.Sequential(torch.nn.Linear(numeric_dim, 128),
                torch.nn.GELU(), torch.nn.Dropout(.3)) if self.has_numeric else torch.nn.Identity())
            fused_dim = len(self.stream_ids) * hidden + (128 if self.has_numeric else 0)
            self.heads = torch.nn.ModuleList([torch.nn.Linear(fused_dim, n) for n in map(len, CLASSES)])
            self.dropout = torch.nn.Dropout(.3)
            apply_training_policy(self, self.training_policy)  # sets requires_grad explicitly

        def train(self, mode=True):
            super().train(mode)
            if self.training_policy == 'frozen_probe':
                self.backbone.eval()  # frozen encoder: no dropout/drop-path, no gradient
            return self

        def encode_stream(self, video):
            """[B, T, C, H, W] normalized -> [B, hidden] via the backbone adapter."""
            if self.training_policy == 'frozen_probe':
                with torch.no_grad():
                    pooled = self.adapter.pool_tokens(self.backbone, video)
                return self.norm(pooled)
            return self.adapter.encode_video(self.backbone, self.norm, video)

        def forward(self, video, numeric):
            # Unselected streams never enter the encoder and have no head columns.
            streams = [self.encode_stream(video[:, view]) for view in self.stream_ids]
            if self.has_numeric:
                streams.append(self.numeric(numeric))
            elif numeric.shape[-1] != 0:
                raise ValueError('Visual-only ablation unexpectedly received numeric features')
            feature = torch.cat(streams, dim=1)
            return [head(self.dropout(feature)) for head in self.heads]

    return ConfigurableMultiViewVideo


def ablation_configuration_manifest(name=None):
    name = ACTIVE_ABLATION_NAME if name is None else name
    config = copy.deepcopy(ABLATION_CONFIGS[name])
    features = numeric_feature_manifest(config)
    manifest = dict(name=name, configuration=config, streams=list(ablation_stream_ids(config)),
                    numeric_dimension=len(features), numeric_features=features,
                    architecture_note='Shared pretrained video backbone; concatenate only selected streams. '
                    'Classification heads are resized for stream count. Numeric MLP is omitted for visual-only variants. '
                    'Fewer streams reduce encoder calls and head parameters; these compute differences are intentional.',
                    scope='Retrained ablation, not inference masking. Role detection/tracking and crop policy stay fixed. '
                    'Pose-only means final-frame geometry/confidence; motion groups have explicit temporal summaries. '
                    'Image streams inherently retain visible pose, motion, and camera information.')
    manifest['feature_signature'] = hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest()[:16]
    return manifest


def estimate_ablation_fits(names, outer_folds, seeds, inner_splits=1):
    """One inner selection fit + one outer refit per seed and fold by default."""
    if outer_folds < 2 or inner_splits < 1 or not seeds:
        raise ValueError('Provide outer fold count, at least one inner split, and seeds')
    rows = []
    for name in names:
        manifest = ablation_configuration_manifest(name)
        fits = int(outer_folds * len(seeds) * (inner_splits + 1))
        rows.append(dict(variant=name, video_model_fits=fits, outer_folds=outer_folds,
                         seeds=len(seeds), visual_streams=len(manifest['streams']),
                         numeric_features=manifest['numeric_dimension'],
                         encoder_stream_fits=fits * len(manifest['streams'])))
    return pd.DataFrame(rows)


def run_controlled_ablations(run_experiment, names, outer_folds, seeds, output_dir, enabled=False):
    """Callback must use the same persisted outer/inner assignments for every variant.

    run_experiment(name) is the notebook's full nested grouped experiment, including
    inner checkpoint/ensemble selection and outer assessment. No prediction masking.
    """
    names = list(names)
    if len(set(names)) != len(names):
        raise ValueError('Duplicate ablation names')
    planned = estimate_ablation_fits(names, outer_folds, seeds)
    print(planned.to_string(index=False))
    print('Total planned video model fits:', int(planned.video_model_fits.sum()))
    print('Separate role-selector and tabular fits are additional; decoded detections may be reused.')
    if not enabled:
        print('Ablations disabled. Set RUN_ABLATIONS=True only after inspecting the fit budget.')
        return {}
    output_dir = Path(output_dir); output_dir.mkdir(parents=True, exist_ok=True)
    planned.to_csv(output_dir/'ablation_fit_budget.csv', index=False)
    previous = ACTIVE_ABLATION_NAME
    results = {}
    try:
        for name in names:
            set_active_ablation(name)
            (output_dir/f'{name}_feature_manifest.json').write_text(
                json.dumps(ablation_configuration_manifest(), indent=2), encoding='utf-8')
            results[name] = run_experiment(name)
    finally:
        set_active_ablation(previous)
    return results


ABLATION_REFERENCES = {
    'B_context_kicker': 'A_context', 'C_context_keeper': 'A_context',
    'D_all_visual': 'A_context', 'E_plus_pose': 'D_all_visual',
    'F_raw_motion': 'E_plus_pose', 'G_compensated_motion': 'F_raw_motion',
    'H_plus_metadata': 'G_compensated_motion', 'production': 'H_plus_metadata',
}


def compare_ablation_metrics(results, output_dir, component='ensemble'):
    """Compare matched folds and pooled rows using ComponentOOF export results."""
    records = []
    for variant, reference in ABLATION_REFERENCES.items():
        if variant not in results or reference not in results:
            continue
        variant_table = results[variant]['metrics']
        reference_table = results[reference]['metrics']
        variant_table = variant_table[variant_table.component == component].copy()
        reference_table = reference_table[reference_table.component == component].copy()
        for table in (variant_table, reference_table): table['fold'] = table['fold'].astype(str)
        if variant_table.empty or reference_table.empty:
            raise ValueError(f'No {component} metrics for ablation comparison')
        if set(variant_table.fold) != set(reference_table.fold):
            raise ValueError('Cannot compare experiments with different evaluated folds')
        merged = variant_table.merge(reference_table, on='fold', suffixes=('_variant', '_reference'),
                                     validate='one_to_one')
        if not (merged.clips_variant == merged.clips_reference).all():
            raise ValueError('Cannot compare experiments with different evaluated sample counts')
        for row in merged.to_dict('records'):
            record = dict(variant=variant, reference=reference, component=component,
                          fold=row['fold'], clips=int(row['clips_variant']))
            for metric in ['weighted_accuracy', 'weighted_cross_entropy',
                           'kick_direction_accuracy', 'goalkeeper_direction_accuracy']:
                record[metric + '_reference'] = row[metric + '_reference']
                record[metric + '_variant'] = row[metric + '_variant']
                record[metric + '_delta'] = row[metric + '_variant'] - row[metric + '_reference']
            records.append(record)
    table = pd.DataFrame(records)
    output_dir = Path(output_dir); output_dir.mkdir(parents=True, exist_ok=True)
    table.to_csv(output_dir / 'ablation_comparisons.csv', index=False)
    return table


def ablation_feature_smoke_test():
    """CPU checks: excluded columns cannot affect features, including derived summaries."""
    rng = np.random.default_rng(19)
    cache = dict(numeric=rng.normal(size=(5, FRAME_FEATURE_WIDTH)).astype(np.float32))
    meta = np.array([1, 0, 0, 1], dtype=np.float32)
    for name, config in ABLATION_CONFIGS.items():
        baseline = ablation_numeric_features(cache, meta, config)
        kept = {entry['raw_column'] for entry in numeric_feature_manifest(config)
                if entry['raw_column'] is not None}
        excluded = sorted(set(range(FRAME_FEATURE_WIDTH)) - kept)
        changed = cache['numeric'].copy()
        changed[:, excluded] = 10000 + rng.normal(size=(len(changed), len(excluded)))
        changed_meta = meta if config['metadata'] else 1-meta
        np.testing.assert_array_equal(baseline, ablation_numeric_features(dict(numeric=changed), changed_meta, config))
        assert len(baseline) == ablation_configuration_manifest(name)['numeric_dimension']
    assert len(ablation_numeric_features(cache, meta, ABLATION_CONFIGS['A_context'])) == 0
    assert ablation_stream_ids(ABLATION_CONFIGS['B_context_kicker']) == (0, 1)
    assert not set(RAW_MOTION) & {e['raw_column'] for e in numeric_feature_manifest(ABLATION_CONFIGS['G_compensated_motion'])}
    production = ablation_numeric_features(cache, meta, ABLATION_CONFIGS['production'])
    x = cache['numeric']
    np.testing.assert_allclose(production, np.r_[x.mean(0), x.std(0), x[-1], x[-1]-x[0], meta])
    # Empty features remain genuinely absent; do not insert a learnable constant branch.
    assert fit_feature_scaler(np.empty((3, 0))).transform(np.empty((2, 0))).shape == (2, 0)
    return dict(status='PASS', variants_checked=len(ABLATION_CONFIGS),
                checks=['stream selection', 'excluded raw/derived features', 'metadata exclusion',
                        'visual-only empty branch', 'production feature compatibility'],
                limitation='Feature smoke only; video retraining and GPU forward/backward require Colab.')


def ablation_architecture_smoke_test():
    """Train tiny synthetic CPU variants through the actual configurable class and backbone adapter.

    Does not download VideoMAE or estimate video prediction performance. Covers both training policies.
    """
    import torch
    from types import SimpleNamespace

    class TinyBackbone(torch.nn.Module):
        def __init__(self):
            super().__init__(); self.projection = torch.nn.Linear(3, 4)

        def gradient_checkpointing_enable(self): pass

        def forward(self, pixel_values):
            pooled = pixel_values.mean(dim=(1, 3, 4))
            return SimpleNamespace(last_hidden_state=self.projection(pooled)[:, None, :])

    class TinyConfig(SimpleNamespace):
        def to_dict(self): return dict(vars(self))

    class TinyPretrained:
        def __init__(self, config=None):
            self.videomae = TinyBackbone(); self.fc_norm = torch.nn.LayerNorm(4)
            self.config = TinyConfig(hidden_size=4, num_frames=2, image_size=4)

        @classmethod
        def from_pretrained(cls, name, **kwargs):
            return cls()

    tiny_spec = dict(backbone_spec('videomae'), checkpoint='synthetic-cpu-smoke', frames=2, size=4, feature_dim=4)
    overrides = dict(VideoMAEForVideoClassification=TinyPretrained, VIDEO_ID='synthetic-cpu-smoke',
                     BACKBONE_SPEC=tiny_spec, TRAINING_POLICY='finetune',
                     N_FRAMES=2, SIZE=4, CLASSES=[['left', 'middle', 'right'],
                                                ['left', 'stands in the middle', 'right']])
    previous = {key: globals().get(key) for key in overrides}
    previous_name = ACTIVE_ABLATION_NAME
    results = []
    try:
        globals().update(overrides)
        with torch.random.fork_rng(devices=[]):
            # CPU generator only: do not alter a Colab CUDA RNG stream.
            torch.random.default_generator.manual_seed(42)
            for name, policy in [('A_context', 'finetune'), ('B_context_kicker', 'finetune'), ('B_context_kicker', 'frozen_probe')]:
                globals()['TRAINING_POLICY'] = policy
                set_active_ablation(name)
                model = make_ablation_model_class()(0)
                video = torch.randn(2, 3, 2, 3, 4, 4); numeric = torch.empty(2, 0)
                model.eval(); baseline = model(video, numeric)
                changed = video.clone()
                excluded = sorted(set(range(3)) - set(ablation_stream_ids()))
                changed[:, excluded] += 10000
                for p, q in zip(baseline, model(changed, numeric)):
                    torch.testing.assert_close(p, q, rtol=0, atol=0)
                assert not list(model.numeric.parameters())
                assert model.heads[0].in_features == len(ablation_stream_ids()) * 4
                optimizer = torch.optim.AdamW(apply_training_policy(model, policy), lr=.01)
                before = model.backbone.projection.weight.detach().clone()
                head_before = model.heads[0].weight.detach().clone()
                model.train(); logits = model(video, numeric); y = torch.tensor([0, 2])
                loss = .7 * torch.nn.functional.cross_entropy(logits[0], y) + .3 * torch.nn.functional.cross_entropy(logits[1], y)
                loss.backward(); optimizer.step()
                assert not torch.equal(head_before, model.heads[0].weight)
                if policy == 'finetune':
                    assert not torch.equal(before, model.backbone.projection.weight)
                else:
                    assert torch.equal(before, model.backbone.projection.weight) and not model.backbone.training
                results.append(dict(variant=name, policy=policy, status='PASS', numeric_dimension=0,
                                    streams=list(ablation_stream_ids())))
    finally:
        set_active_ablation(previous_name)
        for key, value in previous.items():
            if value is None: globals().pop(key, None)
            else: globals()[key] = value
    return dict(status='PASS', variants=results,
                limitation='Tiny synthetic CPU architectures only; no pretrained-video performance estimate.')


# Cheap synthetic checks: no videos, labels from private clips, pretrained weights, or GPU needed.
RUN_DIAGNOSTICS_SMOKE = RUN_MODE in {'smoke', 'train', 'resume'}  # pure definitions in inference/reports modes
DIAGNOSTICS_SMOKE_OUTPUT = Path(SCRATCH_ROOT) / 'diagnostics_smoke'
if RUN_DIAGNOSTICS_SMOKE:
    smoke_output=DIAGNOSTICS_SMOKE_OUTPUT
    smoke_component_reports(smoke_output/'components')
    tracking_diagnostics_smoke(smoke_output/'tracking')
    print(ablation_feature_smoke_test())
    print(ablation_architecture_smoke_test())
    print('DIAGNOSTICS CPU SMOKE: PASS. Synthetic scores are not performance estimates.',smoke_output)
