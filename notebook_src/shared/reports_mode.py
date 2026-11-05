# CELL: reports_mode
# RUN_MODE='reports' ends here: only json/pandas/numpy are used. No torch, video decoder, detector,
# dataset ZIP or model download. A bundle is verified first; output never goes into a foreign run/bundle.
def report_output_dir(run_dir):
    return report_root_for(run_dir) / 'reports' / f'{Path(run_dir).name}__{ATTEMPT_ID}'


ATTEMPT_ID = make_attempt_id(PLATFORM)
if RUN_MODE == 'reports':
    if not TRAINED_RUN_DIR.is_dir(): raise FileNotFoundError(f'TRAINED_RUN_DIR not found: {TRAINED_RUN_DIR}')
    if is_bundle(TRAINED_RUN_DIR):
        verify_bundle(TRAINED_RUN_DIR, VERIFY_CHECKSUMS); print('Bundle verified:', TRAINED_RUN_DIR)
    REPORT_DIR = report_output_dir(TRAINED_RUN_DIR)
    pipeline_summary, pipeline_summary_markdown = write_pipeline_summary(TRAINED_RUN_DIR, output_dir=REPORT_DIR)
    show_markdown(pipeline_summary_markdown)
    if EXPORT_BUNDLE_TO:   # a reports job can also package the finished run for transfer
        exported_bundle = export_bundle(TRAINED_RUN_DIR, Path(EXPORT_BUNDLE_TO)); verify_bundle(exported_bundle)
    stop_notebook(f'Reports written to {REPORT_DIR}. Reports mode stops here (nothing heavy was imported).')
