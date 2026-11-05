Timing benchmark: run this on the GPU intended for full training
The smoke run benchmarks 8 warm-up plus 24 measured training minibatches by default,
then warmed inference. It uses the full run's gradient accumulation and optimizer settings.
Only training-source clips are used for these extra updates. No benchmark weights are saved.
Each timing includes loading a cache from OUTPUT_ROOT, JPEG decoding, preprocessing and device transfers.
The output shows training/inference hours, observed data-loading share and peak GPU memory.
The 25% and 50% allowances are planning scenarios, not confidence bounds. Preprocessing,
role/tree fitting, model loading and checkpoint writes are not included in the base estimate.
Repeated access to a small sample can make storage caching look better than a full run.
Set RUN_TIMING_BENCHMARK=False if you only want the short execution smoke test.
