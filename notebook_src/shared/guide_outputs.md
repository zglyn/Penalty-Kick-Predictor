## Outputs and how to read them

Run folders live at `OUTPUT_ROOT/runs/<platform>/<EXPERIMENT_NAME>__learned_<experiment id>/`. Each execution writes `attempts/<attempt id>/attempt.json`; per-fold jobs write `folds/fold<k>/` and pooled files appear only after every fold is complete.

| Output | Meaning |
|---|---|
| `experiment_manifest.json`, `folds.csv` | Configuration, class mappings, seeds, fixed outer/inner assignments and interpretation |
| `component_reports/component_metrics.csv` | Per-fold and pooled kick/keeper accuracy and cross-entropy, plus weighted 70/30 versions |
| `component_reports/component_oof_probabilities.npz`, `component_manifest.json` | Aligned complete OOF probabilities, labels, clip order, component and class mapping |
| `component_reports/component_clip_errors.csv` | One row per component/clip with both targets, all probabilities, correctness, loss, metadata and quality |
| `component_reports/highest_loss.csv`, `greatest_accuracy_error.csv` | Separate rankings for confident mistakes versus weighted accuracy errors |
| `component_reports/ensemble_comparison.csv` | Corrections, newly introduced mistakes, disagreement and score change versus an inner-selected reference |
| `component_reports/component_slices.csv`, `component_confusions.csv` | Counts, accuracy/loss, full-population error contributions and confusion matrices |
| `tracking_quality.csv`, `tracking_frames.csv` | Coverage, gaps, jumps, confidence, crop clipping, explicit full-frame fallback, pose and camera quality |
| `role_click_audit.csv`, `role_click_summary.csv` | Timestamp offsets, normalized distance, containment and detector/selector failure proxies |
| `visual_audits/` | Several timestamps per selected clip, context/boxes/pose/crops and public post-prediction click overlays |
| `stages/*/component_provenance.json`, `provenance.json`, `video_seed*_fit_manifest.json` | Sources permitted for each component, crop cross-fitting, scaler and checkpoint selection |
| `ablations/<variant>/`, `ablation_comparisons.csv` | Separately retrained experiments and matched-fold/pooled differences for available reference pairs |
| `stages/*/class_prior.json`, `backbone_serialization.json`, `stage_identity.json` | Training-only priors, backbone/architecture/pooling/trainable parameters, fitted-stage identity (excludes prediction clips) |
| `inference_manifest.json` | Everything inference needs: relative paths + sha256 of selectors, scalers, trees, seed weights; priors; frozen ensemble weights; preprocessing; feature schema; class order |
| `OUTPUT_ROOT/inference/<platform>/<run>/` | Inference-only crop caches, per-clip prediction caches keyed by fitted-artifact ID + clip content + preprocessing + metadata, and `runs/<input>/` outputs (`final_probabilities.csv`, `probabilities.npz`, `inference_run.json`, `submission.csv`) |
| `pipeline_summary.md/.json`, `improvement_priorities.csv` | Final *Pipeline diagnosis and next improvements* (report-only) |
| `folds/fold<k>/result.json`, `component_predictions.npz`, `COMPLETE` | One outer fold's selection, weights and predictions (written only by that fold's job) |
| `development_manifest.json`, `development_metrics.csv`, `development_split.csv` | Development preset: subset, one seed, few epochs. DEVELOPMENT ONLY, never a full OOF result |
| `locks/`, `attempts/` | Writer locks (heartbeat) and per-attempt records (host, Slurm IDs, settings, versions, code fingerprint) |

Weighted accuracy is `0.70*kick_accuracy + 0.30*keeper_accuracy`; higher is better. Weighted cross-entropy is a separate loss; lower is better. High cross-entropy can identify confident wrong predictions even when two clips have the same weighted accuracy error. Slice error contributions use the entire evaluated population as denominator. Slices overlap and must not be added together; inspect counts before interpreting small slices. Quality/performance associations do not prove causation.

Role audits compare predictions at the nearest sampled classification timestamp, record mismatch, and exclude large mismatches from summary rates. Separate annotation-nearest detections help distinguish detector failures from role-selection failures. Click-inside-box is a localization proxy, not identity proof. These data cannot establish an automatic identity-switch error rate. Visual audits help inspect possible switches without changing predictions.
