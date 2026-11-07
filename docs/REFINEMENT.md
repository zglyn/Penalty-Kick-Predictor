# Refinement without video retraining (`RUN_MODE='refine'`)

Implements the low-cost parts of `DIAGNOSTIC_IMPROVEMENT_SPEC_2026-11-06.md`. The spec measured these failures on the finished VideoMAE nested run:
- the center class is never predicted (0/181 middle kicks, 0/63 stationary keepers);
- kick direction is much weaker than keeper direction;
- role detection and tracking are often unavailable;
- the video models add little over the numeric pose/motion specialist.

Refinement reuses every saved weight. No video model is fine-tuned again.

## What it does, per outer fold

1. **Regenerate inner-validation predictions** from the saved inner video checkpoints. This is inference only, cached in `fold<k>/inner_video.npz`.
2. **Reproduce the original inner selection** (`select_diagnostic_mixture`). The result is recorded in `fold_results.json` as `original_inner_selection.reproduced`, and a warning is printed if it differs.
3. **Refit a class-weighted pose/motion CatBoost** (`pose_motion_tree_weighted`) on each stage's training partition. It uses capped inverse-sqrt class weights with mean 1 and cap `REFINE_WEIGHT_CAP`, runs on the CPU and takes minutes. The unweighted tree stays as a control.
4. **Reselect each task's ensemble on inner predictions only.** The candidates are declared in advance:
   - single components;
   - equal mixtures;
   - the quarter grid over video, tree and prior.

   The rule is: max inner accuracy, then lower cross-entropy, then fewer components.
5. **Optionally fit a center-class log-probability offset** from `REFINE_CENTER_OFFSETS`, on inner predictions. It is adopted only when inner accuracy strictly rises.
6. **Apply the frozen choices** to the saved outer predictions (an exploratory comparison) and to the private clips (a new `submission.csv`, made with the saved outer models).

## Outputs: `<parent>__refine_<id>/` next to the parent run (the parent is never modified)

| File | Content |
|---|---|
| `submission.csv` | Refined private predictions |
| `refine_comparison.csv` | Per fold and pooled: weighted accuracy, macro and center recall/precision, predicted-class counts, paired difference vs `baseline_ensemble` and `pose_motion_tree` |
| `refine_candidates.csv` | Every candidate's inner accuracy and cross-entropy; the selected one is marked |
| `component_reports/` | Standard component metrics, with the parent ensemble as `baseline_ensemble` |
| `class_health.csv`, `pipeline_summary.md` | Per-class support, predictions, precision, recall and zero-prediction alerts; the summary leads with these measurements |
| `label_mapping_audit.json`, `temporal_input_audit.csv`, `crop_geometry.csv`, `label_convention_samples/` | Priority-0 audits (label encode/decode round trip, no flips or reversals, frame sampling, crop geometry, training-only label examples) |
| `inference_manifest.json` | Replays the refined ensemble with `RUN_MODE='inference'` without reselecting; refers to the parent's weights via `../<parent>` |
| `refinement_manifest.json` | Hypothesis, changes, controls, selection source and adoption rule |

Bundles cannot be exported from a refinement folder, because it references the parent's weights. Export the parent and keep the two folders next to each other.

**Reading the result.** Outer folds were already inspected when the spec was written, so the outer comparison is exploratory. The submission's choices were fixed on inner data. Picking between the baseline and the refined submission by outer score would make that score a tuning metric. Report both.

## Training diagnostics added for future runs

`fit_video` now records seen class counts per epoch, and per-class validation predictions and recall. `training_history_report.csv` flags these patterns:
- train loss not falling;
- best epoch is the first epoch;
- validation loss rising after the best epoch.

## Deferred: spec items that need retraining

These are not implemented. Each needs new crops or new video training, so run them only if the refine results and the Priority-0 audits point to them:
- detector or tiled-candidate changes, and track continuation;
- missing-role masks;
- a crop-policy experiment;
- frozen-probe and backbone comparisons;
- a tiny overfit sanity check.
