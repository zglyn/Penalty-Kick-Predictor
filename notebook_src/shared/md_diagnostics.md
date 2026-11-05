## Diagnostics helpers and CPU-only smoke checks
These checks run automatically in smoke/train/resume modes before data loading. It embeds all reporting, tracking, and ablation code and produces small synthetic reports under `SCRATCH_ROOT/diagnostics_smoke`. It does not train a real video model or estimate competition performance. The GPU pipeline smoke is `RUN_MODE='smoke'`.
