# CELL: batch_files_and_release
# Interactive helper (skipped by the batch runner); runs before anything heavy and then stops. WRITE_BATCH_FILES writes the runner, the Slurm script and a
# job JSON with the CURRENT settings beside this notebook. FREEZE_RELEASE copies notebook + runner + sbatch +
# settings into OUTPUT_ROOT/releases/<id>/ read-only; run long jobs from that folder so edits never touch them.
RUNNER_SOURCE = {{RUNNER_SOURCE}}
SBATCH_SOURCE = {{SBATCH_SOURCE}}
JOB_SETTINGS = dict(EXPERIMENT_SETTINGS, RUN_MODE=RUN_MODE, FOLD_SELECTION=FOLD_SELECTION,
                    DATASET_PATH=str(DATASET_PATH), OUTPUT_ROOT=str(OUTPUT_ROOT), SCRATCH_ROOT=str(SCRATCH_ROOT),
                    GENERATE_TRACKING_AUDITS=GENERATE_TRACKING_AUDITS, GENERATE_HEATMAPS=GENERATE_HEATMAPS)
if WRITE_BATCH_FILES or FREEZE_RELEASE:
    if NOTEBOOK_PATH is None: raise ValueError('Set NOTEBOOK_PATH to this notebook file (e.g. the path shown in Jupyter).')
    _folder = Path(NOTEBOOK_PATH).resolve().parent
    atomic_write_text(_folder/'run_penalty_kick.py', RUNNER_SOURCE)
    atomic_write_text(_folder/'penalty_kick.sbatch', SBATCH_SOURCE)
    _job = _folder/f'{EXPERIMENT_NAME}__{RUN_MODE}.json'
    atomic_write_text(_job, json.dumps(JOB_SETTINGS, indent=2))
    print('Wrote', _folder/'run_penalty_kick.py', _folder/'penalty_kick.sbatch', _job)
    if FREEZE_RELEASE:
        release = create_release(OUTPUT_ROOT, [NOTEBOOK_PATH, _folder/'run_penalty_kick.py', _folder/'penalty_kick.sbatch', _job],
                                 JOB_SETTINGS, extra=dict(environment=ENVIRONMENT_INFO, backbone=BACKBONE,
                                 training_policy=TRAINING_POLICY))
        print(f'Launch the frozen baseline:\n  cd {release}\n  sbatch penalty_kick.sbatch {_job.name}')
    stop_notebook('Batch files written' + (' and release frozen' if FREEZE_RELEASE else '') + '; nothing was trained here.')
