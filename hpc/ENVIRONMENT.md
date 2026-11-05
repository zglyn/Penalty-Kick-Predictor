# HPC setup: NYU HPC Cloud Bursting

## What the course codelab states (verified source material)

| Topic | Codelab fact |
|---|---|
| Access | Open OnDemand only: https://ood-burst.hpc.nyu.edu/ (NYU VPN off campus). From OOD: launch compute nodes, Jupyter notebooks, terminals, small transfers, and batch submission from Jupyter terminals. |
| Scheduler and account | Slurm, account `cs_gy_6643-2026fa`; 300 GPU hours (18,000 minutes) per student; ample CPU time. |
| Partitions | `interactive`; `n2c48m24` (CPU only); `g2-standard-12` (1 L4), `g2-standard-24` (2 L4), `g2-standard-48` (4 L4); `c12m85-a100-1` (1 A100 40GB), `c24m170-a100-2` (2 A100 40GB). |
| Preemption | Google Cloud spot instances can be preempted at any time. Use checkpoint/restart, save checkpoints to `/scratch/$NETID`, and add `#SBATCH --requeue`. Jobs are requeued automatically when instances are shut down. |
| GPU sharing | `#SBATCH --comment="gpu_mps=yes"` lets several jobs share one GPU (MPS). The codelab recommends running jobs in parallel this way. |
| Software | Conda environments inside Singularity with overlay files. Templates are in `/share/apps/overlay-fs-ext3` and images in `/share/apps/images`. Guide: https://services.rt.nyu.edu/docs/hpc/containers/singularity_with_conda/ (it includes Slurm batch usage). |
| Advice | Checkpoint all jobs and add "the preempt script". Ask the TAs early, because queue waits grow late in the semester. |

## Not stated in the codelab (placeholders in the files)

These are marked `UNVERIFIED` in `penalty_kick.sbatch`. Confirm each one before the first batch job:

- the **wall-time limit** of each partition (`--time`)
- whether **`--gres=gpu:N`** is required in addition to choosing a GPU partition
- **CPU and memory** requests (`--cpus-per-task`, `--mem`)
- the exact **overlay template** and **Singularity image** file names (pick from the two folders above), and the overlay setup commands. These come from the linked NYU guide, which was not attached.
- whether **`/scratch` is purged** after inactivity, and whether a separate persistent home or archive is intended for results
- the **"preempt script"** mentioned in the codelab. It was not attached. The runner logs SIGTERM; recovery relies on `--requeue` plus epoch checkpoints.
- whether the OOD Jupyter app can use a **Singularity-backed kernel**, or whether interactive work should run in a terminal on a compute node

## One-time environment setup

Do this on a compute node, never on a login node. Start an OOD terminal or Jupyter session on `n2c48m24` or a GPU partition, then follow the NYU *Singularity with conda* guide:

1. **Make the overlay.** Copy an overlay template from `/share/apps/overlay-fs-ext3` to `/scratch/$USER/penalty_kick/env/` and unpack it. Name it `overlay-penalty.ext3`, or set `PENALTY_OVERLAY` to your name.
2. **Install conda.** Choose a CUDA-capable image from `/share/apps/images` and set `PENALTY_IMAGE`. Start the image read-write with the overlay, and install Miniforge/Miniconda into `/ext3` with an `/ext3/env.sh` activation script, as the guide describes.
3. **Install the pinned packages.** Inside that environment, install the versions that the HPC notebook's environment cell checks: `numpy==2.2.6`, `scipy>=1.15,<1.17`, `pandas>=2.2.3,<3`, `scikit-learn==1.7.2`, `av==16.1.0`, `catboost==1.2.8`, `transformers==5.3.0`, `huggingface-hub>=1.32,<2`, `ultralytics==8.4.0`, `opencv-python-headless`, `tqdm`, `timm>=1.0.15,<2`, `einops>=0.8,<1`, a CUDA build of `torch>=2.6` with `torchvision`, plus `jupyter`/`ipykernel` and `packaging`.
4. **Freeze the overlay.** Exit and use the overlay read-only (`:ro`) from then on. Jobs never install packages. The notebook refuses to run if a version differs.
5. **Prepare data and logs.**
   ```bash
   mkdir -p /scratch/$USER/penalty_kick/{data,outputs,staging,logs}
   ```
   Then upload `penalty-kick-prediction-cs-gy-6643.zip` to `/scratch/$USER/penalty_kick/data/` with the OOD file browser.
6. **Pre-download weights (optional).** Run one `RUN_MODE='smoke'` job, so the VideoMAE, YOLO and (if used) V-JEPA weights are cached before long jobs start.

## Storage policy used by the notebook

- **`OUTPUT_ROOT=/scratch/$NETID/penalty_kick/outputs`** holds everything that must survive preemption: checkpoints, reports, bundles, releases and splits. This follows the codelab.
- Checkpoints are written atomically: a temporary file, then fsync, then rename. A preempted save leaves the previous complete checkpoint in place.
- `inference_manifest.json` (with checksums) is the completion marker for a trained run. Bundles and releases are staged in hidden `.partial` folders and renamed only after verification.
- **`SCRATCH_ROOT=/scratch/$NETID/penalty_kick/staging`** holds the extracted videos. These can be regenerated from the ZIP.
- **Node-local storage** (`$TMPDIR`, `/tmp`) is lost with a spot node. The notebook refuses to use it as `OUTPUT_ROOT` unless `ALLOW_EPHEMERAL_OUTPUT=True`.
- **Recovery is at epoch boundaries.** After a preemption, the unfinished epoch repeats. Mid-epoch recovery is not implemented.

## Running jobs

From an OOD terminal, in the release folder (or the folder containing the notebook and runner):

```bash
sbatch penalty_kick.sbatch jobs/smoke.json
```

```bash
squeue -u $USER
```

```bash
scancel <jobid>
```

Logs go to `/scratch/$USER/penalty_kick/logs/penalty-kick_<jobid>_<task>.out`.

- **Smoke and pilot first.** Each GPU hour comes out of the 300-hour budget. Use the smoke and pilot timings to choose between `g2-standard-12` (L4) and `c12m85-a100-1` (A100).
- **One fold per task.** `sbatch --array=0-4 penalty_kick.sbatch jobs/baseline_nested.json --fold-from-array` runs one outer fold per task, each with its own lock and its own `folds/fold<k>/` output. The job that finishes the last fold pools the reports.
- **Sharing a GPU.** With `gpu_mps=yes`, several folds may share one GPU. Check peak memory from the smoke/timing report first: three video streams fine-tuned on a 24 GB L4 may not leave room for a second job.
