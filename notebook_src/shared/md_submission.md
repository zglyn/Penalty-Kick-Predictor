Submission and interpretation
This compensates motion features, not video pixels. Original frames preserve fine visual detail;
the learned video model still sees camera movement. Compensation is approximate and invalid
estimates are masked, never presented as measured zero motion. It does not reconstruct 3D camera
rotation, correct strong perspective/parallax, or guarantee robustness across cuts. Role selection
still depends on person detection; missing or ambiguous training matches are skipped.
All backbone parameters are fine-tuned; this is not a frozen feature-only implementation.
Run-all is expensive (inner and outer training, multiple seeds, three video streams). Keep
checkpoints/caches on Drive. The default smoke mode does not enter full CV or write a submission.
Verify available Drive space; cross-fitted RGB caches can use
tens of GB. Cache IDs include data and pipeline config to avoid reuse across different runs.
Outer scores are assessment, not another tuning set. No winning score is guaranteed.
