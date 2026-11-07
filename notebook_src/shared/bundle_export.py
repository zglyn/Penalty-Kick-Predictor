# CELL: bundle_export
# Optional: copy a finished run into a relocatable, checksummed bundle (e.g. HPC -> Drive). The copy is staged,
# verified and only then marked complete; an interrupted transfer has no marker and is rejected on import.
if EXPORT_BUNDLE_TO and RUN_MODE == 'refine':
    print('Refinement runs reference their parent run; copy the parent and refinement folders together instead of a bundle.')
elif EXPORT_BUNDLE_TO:
    _source = RUN if RUN_MODE in TRAINING_MODES else TRAINED_RUN_DIR
    exported_bundle = export_bundle(_source, Path(EXPORT_BUNDLE_TO))
    verify_bundle(exported_bundle)
    print('Bundle ready. Load it elsewhere with RUN_MODE="inference" or "reports" and TRAINED_RUN_DIR =', exported_bundle)
