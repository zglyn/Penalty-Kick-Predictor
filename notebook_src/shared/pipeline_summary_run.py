# CELL: pipeline_summary_run
# Pipeline diagnosis for the run this session trained, or for the fitted run used by inference
# (inference outputs carry no labels). Summaries of foreign runs/bundles are written outside them.
SUMMARY_RUN_DIR = TRAINED_RUN_DIR if RUN_MODE == 'inference' else RUN
SUMMARY_OUTPUT = report_output_dir(SUMMARY_RUN_DIR) if RUN_MODE == 'inference' else RUN
if QUICK_SMOKE:
    print('Smoke runs produce no outer-fold reports; the summary below will state that evidence is insufficient.')
pipeline_summary, pipeline_summary_markdown = write_pipeline_summary(SUMMARY_RUN_DIR, output_dir=SUMMARY_OUTPUT)
print('Saved:', Path(SUMMARY_OUTPUT)/'pipeline_summary.md', '|', Path(SUMMARY_OUTPUT)/'pipeline_summary.json',
      '|', Path(SUMMARY_OUTPUT)/'improvement_priorities.csv')
show_markdown(pipeline_summary_markdown)
