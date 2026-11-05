# CELL: validation_report
# Run after all outer validation folds finish.
def validation_loss_report(train, Y, oof, targets, classes, weights, fold_ids):
    assert len(targets) == len(classes) == len(weights) == len(oof) == 2
    assert Y.shape == (len(train), 2)
    assert np.asarray(fold_ids).shape == (len(train),) and (np.asarray(fold_ids) >= 0).all()
    assert np.isclose(np.sum(weights), 1)
    columns = [c for c in ['clip', 'source_id', 'preferred_foot', 'gender'] if c in train]
    errors = train[columns].copy()
    errors['weighted_log_loss'] = 0.0
    errors['weighted_error'] = 0.0
    summary = []
    class_summary = []
    matrices = {}
    for j, target in enumerate(targets):
        p = np.asarray(oof[j])
        assert p.shape == (len(train), len(classes[j]))
        assert np.isfinite(p).all() and (p >= 0).all()
        assert np.allclose(p.sum(axis=1), 1, atol=1e-5)
        truth = Y[:, j]
        pred = p.argmax(axis=1)
        loss = -np.log(np.clip(p[np.arange(len(train)), truth], 1e-12, 1.0))
        wrong = (pred != truth).astype(float)
        errors[target + '_true'] = np.asarray(classes[j])[truth]
        errors[target + '_pred'] = np.asarray(classes[j])[pred]
        errors[target + '_true_probability'] = p[np.arange(len(train)), truth]
        errors[target + '_log_loss'] = loss
        errors[target + '_wrong'] = wrong
        errors['weighted_log_loss'] += weights[j] * loss
        errors['weighted_error'] += weights[j] * wrong
        summary.append(dict(target=target, accuracy=1-wrong.mean(), mean_log_loss=loss.mean(),
                            weighted_score_lost=weights[j]*wrong.mean()))
        for k, label in enumerate(classes[j]):
            mask = truth == k
            class_summary.append(dict(target=target, true_class=label, clips=int(mask.sum()),
                accuracy=float(1-wrong[mask].mean()) if mask.any() else np.nan,
                mean_log_loss=float(loss[mask].mean()) if mask.any() else np.nan,
                weighted_score_lost=float(weights[j]*wrong[mask].sum()/len(train))))
        matrices[target] = pd.crosstab(errors[target+'_true'], errors[target+'_pred']).reindex(
            index=classes[j], columns=classes[j], fill_value=0)
    return errors, pd.DataFrame(summary), pd.DataFrame(class_summary), matrices

if RUN_MODE in {'train', 'resume'} and not QUICK_SMOKE and EVALUATION_MODE=='nested_cv' and diagnostic_result is not None:
    errors, target_report, class_report, confusion_tables = validation_loss_report(
        train, Y, oof, TARGETS, CLASSES, WEIGHTS, fold_ids
    )
    # Same-fold majority baseline distinguishes learned signal from class balance.
    majority_error=np.zeros(len(train))
    for fold in np.unique(fold_ids):
        fit=np.flatnonzero(fold_ids!=fold); held=np.flatnonzero(fold_ids==fold)
        for j,w in enumerate(WEIGHTS):
            majority=np.bincount(Y[fit,j],minlength=len(CLASSES[j])).argmax()
            majority_error[held]+=float(w)*(Y[held,j]!=majority)
    print('Fold-majority baseline:',1-majority_error.mean(),
          '| Model lift:',majority_error.mean()-errors['weighted_error'].mean())
    # Correlate prediction errors with role/camera coverage, using held-out caches only.
    for role in ['kicker','keeper']: errors[role+'_tracked_fraction']=np.nan
    errors['camera_valid_fraction']=np.nan
    for record in reports:
        folder=Path(record['stage_folder'])
        for i in np.flatnonzero(fold_ids==record['fold']):
            cached=load_view({int(i):folder/f'{i}.pkl'},int(i))
            for r,role in enumerate(['kicker','keeper']):
                errors.loc[i,role+'_tracked_fraction']=float((cached['traces'][:,r,2]>0).mean())
            errors.loc[i,'camera_valid_fraction']=float((cached['numeric'][:,-3]>0).mean())
    for role in ['kicker','keeper']:
        errors[role+'_tracking']=pd.cut(errors[role+'_tracked_fraction'],[-.01,0,.5,1.01],
                                       labels=['missing','partial','mostly tracked'])
    print('New OOF score:', 1-errors['weighted_error'].mean())
    print('Per-target performance:'); display(target_report)
    print('True classes costing the most leaderboard points:')
    display(class_report.sort_values('weighted_score_lost', ascending=False))
    print('Most confidently wrong / highest-loss validation clips:')
    display(errors.sort_values('weighted_log_loss', ascending=False).head(20))
    for field in ['source_id', 'preferred_foot', 'gender', 'kicker_tracking', 'keeper_tracking']:
        if field not in errors: continue
        grouped = errors.groupby(field).agg(
            clips=('clip', 'size'), mean_log_loss=('weighted_log_loss', 'mean'),
            error_rate=('weighted_error', 'mean'), error_sum=('weighted_error', 'sum'))
        grouped['weighted_score_lost'] = grouped.pop('error_sum') / len(errors)
        grouped.to_csv(RUN / f'validation_by_{field}.csv')
        print(f'Error breakdown by {field} (inspect sample counts):')
        display(grouped.sort_values('weighted_score_lost', ascending=False).head(20))
    for target, matrix in confusion_tables.items():
        print(target, '(rows: true, columns: predicted)'); display(matrix)
        matrix.to_csv(RUN / f'confusion_{target}.csv')
    errors.to_csv(RUN / 'validation_errors.csv', index=False)
    target_report.to_csv(RUN / 'validation_targets.csv', index=False)
    class_report.to_csv(RUN / 'validation_classes.csv', index=False)
else:
    print('OOF diagnostics apply to nested_cv training runs. Holdout mode saves loss_history.csv and holdout_metrics.csv.')

