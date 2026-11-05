# CELL: submission
if RUN_MODE == 'inference':
    if inference_result is not None and (inference_result['output_dir']/'submission.csv').exists():
        print('Inference submission:', inference_result['output_dir']/'submission.csv')
        offer_download(inference_result['output_dir']/'submission.csv')
    else:
        print('Inference output (no submission for this input):', inference_result['output_dir'])
elif QUICK_SMOKE:
    print("Smoke mode: no competition submission is generated.")
elif EVALUATION_MODE=='development':
    if PREDICT_PRIVATE and development_result is not None:
        print('Development submissions (one per backbone; subset-trained, labeled development):')
        display(development_result['comparison'][['backbone','status','best_validation_loss','submission']])
        for path in development_result['comparison'].submission.dropna(): offer_download(path)
    else:
        print('Development run complete; set PREDICT_PRIVATE=True for per-backbone development submissions.')
elif EVALUATION_MODE=='holdout':
    print('Holdout run complete. Submissions come from nested_cv or development runs (or inference mode).')
elif diagnostic_result is None:
    print('Nested folds are still incomplete; no submission until every fold is aggregated.')
elif not PREDICT_PRIVATE:
    print('Private inference was disabled. Set PREDICT_PRIVATE=True and rerun nested diagnostics to export a submission.')
else:
    submission_targets = ["kick_direction", "goalkeeper_direction"]
    submission_weights = np.array([0.70, 0.30])
    indices = [TARGETS.index(target) for target in submission_targets]

    pred = pd.DataFrame({"clip": test["clip"].to_numpy()})
    for target, j in zip(submission_targets, indices):
        probabilities = np.asarray(private[j])
        classes = np.asarray(CLASSES[j])
        assert probabilities.shape == (len(test), len(classes))
        assert np.isfinite(probabilities).all()
        assert (probabilities >= 0).all()
        assert np.allclose(probabilities.sum(axis=1), 1, atol=1e-5)
        pred[target] = classes[probabilities.argmax(axis=1)]

    submission = template[["clip"]].merge(
        pred, on="clip", how="left", validate="one_to_one"
    )
    assert len(submission) == len(test) == 900
    assert submission["clip"].is_unique
    assert not submission.isna().any().any()
    assert set(submission["clip"]) == set(test["clip"])
    assert list(submission.columns) == ["clip"] + submission_targets

    # Recalculate validation accuracy using the new competition metric.
    assert (fold_ids >= 0).all(), "Finish all folds before exporting."
    new_accuracies = np.array([
        accuracy_score(Y[:, j], oof[j].argmax(axis=1)) for j in indices
    ])
    print("OOF accuracies:", dict(zip(submission_targets, new_accuracies)))
    print(f"New weighted OOF score: {submission_weights @ new_accuracies:.6f}")

    output_path = RUN / "submission.csv"
    submission.to_csv(output_path, index=False)
    export_config = dict(CFG, submission_targets=submission_targets,
                         submission_weights=submission_weights.tolist())
    (RUN / "config.json").write_text(json.dumps(export_config, indent=2))
    import sys
    (RUN / "environment.txt").write_text(
        subprocess.check_output([sys.executable, "-m", "pip", "freeze"], text=True)
    )
    display(submission.head())
    print("Saved:", output_path)
    offer_download(output_path)
