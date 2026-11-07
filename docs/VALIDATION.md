# Validation status (2026-11-05)

This report uses four distinct terms:

- **Implemented**: the code exists.
- **Synthetically tested**: tested on CPU with synthetic data and tiny randomly initialized models.
- **Tested with a real checkpoint/GPU**: not done for any backbone.
- **Full performance evaluated**: not done.

No HPC access, Slurm job, real video, real checkpoint download or GPU benchmark happened while these notebooks were written.

## Environment of the actual tests

- Python 3.13, PyTorch 2.15 (CPU build), Transformers 5.3.0, CatBoost and scikit-learn.
- `python -m pytest tests` → **68 passed, 0 failed** in 159 s.
- `python tools/check_consistency.py` → PASS.
- `python tools/build_notebooks.py --check` → generated files are current.

## Validation ladder

| Level | Check | Status |
|---|---|---|
| **A: static/unit** | Notebook schema and compilation, metric arithmetic, source separation, cache invalidation, routing, consistency | **Passed** (synthetic, CPU) |
| **B: real-model smoke** | Real checkpoint and GPU, real clips through decoding, roles, crops, forward/backward, save/load and inference | **Not run.** Run `RUN_MODE='smoke'` on Colab and on the target HPC GPU. |
| **C: recovery/artifacts** | Interrupt after a saved epoch and resume; inference with fitting disabled; bundle copy and reload | **Passed synthetically** (CPU, tiny models). Still needed on real hardware: a real preemption and requeue on Cloud Bursting. |
| **D: bounded pilot** | Representative source-disjoint subset, several epochs; memory, time and storage; finite losses; crop audits | **Not run.** Use `hpc/jobs/pilot.json` (development preset, saved split). |
| **E: full evaluation** | All saved folds with coverage checks | **Not run.** No performance claims are made. |

The spec says to complete A–D before freezing a long-run baseline. Only A, and C synthetically, are done. Before launching the frozen HPC baseline, do the following:

1. Run the smoke job on the chosen partition and record peak memory and throughput.
2. Run the pilot job, then inspect `visual_audits/`, the loss curves and storage use.
3. Resolve the `UNVERIFIED` sbatch fields (see `hpc/ENVIRONMENT.md`).

## Passed (actual CPU tests)

| Requirement | Tests |
|---|---|
| Two-target metrics, splits, role provenance, summaries | `test_pipeline_summary.py` (16), `test_notebook_integrity.py::test_existing_cpu_diagnostics_smoke_checks_still_pass` |
| Epoch recovery (bit-identical after interruption) and identity mismatch rejection | `test_resume_and_stage_identity.py` (VideoMAE and V-JEPA 2, both policies) |
| Changed metadata changes prediction identity and recomputes | `test_changed_metadata_changes_prediction_identity_and_recomputes` |
| HPC notebook has no Colab dependency | `test_hpc_notebook_has_no_colab_dependency`, `check_consistency.py`, HPC reports run through the real batch runner |
| Custom inference without competition files or labels | `test_custom_inference_needs_no_competition_files_or_labels` (real dataset, configuration and inference cells; no ZIP on disk) |
| Reports without model/video imports, fresh processes, both notebooks | `test_reports_mode_hpc_batch_runner_is_lightweight`, `test_reports_mode_colab_notebook_is_lightweight_without_drive` (also asserts no `google` import), `test_reports_only_in_fresh_process_without_torch` |
| Equivalent outputs across notebook wrappers | `test_hpc_and_colab_wrappers_produce_identical_inference` (max abs diff ≤ 1e-6; observed 0), `test_identity_ignores_platform_and_paths_but_runs_are_isolated` |
| Independent run directories, concurrent writer isolation | `test_writer_locks_isolate_concurrent_writers` (a live second writer is refused; stale and requeue takeover), `test_parallel_fold_jobs_aggregate_once_and_match_the_monolithic_run` (two concurrent fold processes match the single-process run) |
| Bundle relocation and checksums; missing members fail | `test_bundle_relocates_verifies_and_rejects_incomplete_transfers` |
| Inference never fits, selects or builds an optimizer | `forbid_fitting` in every inference test, plus `test_fresh_process_inference_with_every_fit_entry_point_disabled` |
| Development preset is labeled and never exported as full evaluation | `test_development_preset_is_labeled_and_never_exported_as_full_evaluation` |
| Batch runner settings and release integrity | `test_batch_runner_rejects_unknown_settings_and_modified_releases` |
| Settings fail before expensive work | `test_incompatible_settings_fail_before_anything_expensive[colab/hpc]` |
| Refinement reuses weights, leaves the parent untouched, selects on inner data only, and replays through inference | `test_refinement.py` (9: parent bytes identical, no crops or role fits, original inner choices reproduced, fresh inference matches saved predictions ≤1e-6, label-audit catches a swapped decoder, class-health arithmetic) |
| Backbone adapters | VideoMAE and V-JEPA 2: real Transformers classes with tiny configs. V-JEPA 2.1: a **stand-in** of the official interface (running Meta's repository code was not permitted in the authoring sandbox). |

## Failed

None in the final run. Earlier failures during development (test-harness quoting, a missing stub module, an over-broad module check, and a dispatch-cell guard) were fixed and re-run.

## Not run (needs real resources)

- Real-backbone smoke tests (level B) on Colab and on HPC, for VideoMAE, V-JEPA 2 and V-JEPA 2.1 (including the strict load of the real 2.1 checkpoint).
- Bounded pilot (level D) and full nested evaluation (level E).
- Slurm submission, `--requeue` after a real spot preemption, MPS sharing, Singularity overlay execution, and Open OnDemand Jupyter kernel setup.
- Throughput and peak-memory measurements on L4 or A100.
