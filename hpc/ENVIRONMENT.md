# HPC setup: NYU HPC Cloud Bursting

Sources:
- the course codelab
- NYU's official HPC documentation, read from its source repository [NYU-RTS/rts-docs](https://github.com/NYU-RTS/rts-docs) (commit `ca4980d`, 2026-10-29). This is the site published at services.rt.nyu.edu.

## The codelab's broken link

The codelab links to `services.rt.nyu.edu/docs/hpc/containers/singularity_with_conda/`. Two things are going on:

- **HTTP 431** ("Request Header Fields Too Large") comes from your browser, not the page. It is sending too many or too-large `nyu.edu` cookies. Clear the cookies and site data for `services.rt.nyu.edu` and `nyu.edu`, or open the link in a private window.
- **The page was renamed.** NYU switched from Singularity to Apptainer. The guide is now *Apptainer with Conda* (`docs/hpc/07_containers/02_apptainer_with_conda.md` in the repo). The OOD Jupyter version is *Jupyter Notebook with Conda/Singularity in OOD* (`docs/hpc/09_ood/02_jupyter_with_conda_singularity.mdx`). Both can be read on GitHub without the 431 problem.

## Facts

| Topic | Fact | Source |
|---|---|---|
| Access for this course | Open OnDemand at https://ood-burst.hpc.nyu.edu/; NYU VPN when off campus | codelab |
| Account, budget | Slurm account `cs_gy_6643-2026fa`; 300 GPU hours | codelab |
| Partitions | `interactive`, `n2c48m24` (CPU), `g2-standard-12/24/48` (1/2/4 L4), `c12m85-a100-1` / `c24m170-a100-2` (1/2 A100 40GB) | codelab |
| What bursting is | A separate Slurm cluster on Google Cloud, next to NYU's main cluster *Torch*. Access is by approval: you can log in to the burst login node without it, but jobs are refused (`Invalid account or account/partition combination specified`). | NYU docs, *HPC Bursting* |
| **Storage** | **Burst compute nodes have their own persistent `/home` and `/scratch`, separate from Torch's.** The burst login node shows Torch's `/home` and `/scratch`, but jobs on compute nodes do not, so data must be copied to the burst file system. NYU's example: inside an interactive burst job, run `scp <NetID>@dtn.torch.hpc.nyu.edu:/path/to/files /home/<NetID>/`. | NYU docs |
| Preemption | Spot instances can be preempted; use `#SBATCH --requeue` and checkpoint under `/scratch/$NETID` | codelab |
| GPU sharing | `#SBATCH --comment="gpu_mps=yes"` | codelab (NYU docs use the same comment syntax) |
| GPU request | `--gres=gpu:N`. The burst visualization example combines a partition with `--gres=gpu:p100:1`. | NYU docs |
| Containers | Apptainer is the default; `singularity` commands are the older name. Writable setup: `apptainer exec --fakeroot --overlay <file>.ext3:rw <image>.sif /bin/bash`. Jobs use `apptainer exec --nv --overlay <file>.ext3:ro <image>.sif ...`. `--nv` passes the GPU into the container. | NYU docs |
| Overlay | `/share/apps/overlay-fs-ext3/overlay-15GB-500K.ext3.gz` (15 GB, 500K files), copied and gunzipped | NYU docs |
| Images | `ls /share/apps/images/`. Examples in the docs: `cuda-13.3.1-ubuntu-26.04.sif` and `cuda12.3.2-cudnn9.0.0-ubuntu-22.04.4.sif` (these are Torch paths; list the folder on the burst cluster) | NYU docs |
| Concurrent jobs | An overlay opened read-write is locked. Many jobs can share it only while every job mounts it `:ro`. | NYU docs |
| Install packages on a compute node | Login nodes allow 2 GB of memory per user and run a different OS image from the compute nodes | NYU docs |
| Torch scratch | 5 TB / 5M files, no backups, files not accessed for 60 days are purged (the burst file system is separate; its policy is not stated) | NYU docs |
| Wall times (Torch) | `show_slurm_qos`: interactive 6 h; GPU 48 h or 7 days. Burst limits are not documented; check with `sinfo -o "%P %l"` on the burst cluster. | NYU docs |
| SSH (Torch) | `ssh <NetID>@login.torch.hpc.nyu.edu` from PowerShell, with Microsoft device-login MFA; SSH keys are not supported. Then `ssh burst` for the burst login node. Your course says to use OOD; try SSH only if the TAs allow it. | NYU docs |

## One-time setup

All commands run in an Open OnDemand terminal on the burst cluster (or `ssh burst`).

1. **Check `~/.bashrc`.** If it contains a `>>> conda initialize >>>` block, comment it out, then log out and back in. NYU warns that this block breaks installs inside Apptainer.
2. **Start a compute-node session.** Never install on the login node. For example:
   ```bash
   srun --account=cs_gy_6643-2026fa --partition=interactive --pty /bin/bash
   ```
3. **Create the folders and the overlay:**
   ```bash
   mkdir -p /scratch/$USER/penalty_kick/{data,outputs,staging,logs,env,code}
   cd /scratch/$USER/penalty_kick/env
   cp -rp /share/apps/overlay-fs-ext3/overlay-15GB-500K.ext3.gz . && gunzip overlay-15GB-500K.ext3.gz
   mv overlay-15GB-500K.ext3 overlay-penalty.ext3
   ls /share/apps/images/          # pick a CUDA 12.x image
   ```
4. **Open the overlay read-write and install Miniforge into it:**
   ```bash
   apptainer exec --fakeroot --overlay overlay-penalty.ext3:rw /share/apps/images/<cuda-image>.sif /bin/bash
   ```
   Then, at the `Apptainer>` prompt:
   ```bash
   wget --no-check-certificate https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh
   bash Miniforge3-Linux-x86_64.sh -b -p /ext3/miniforge3
   printf '#!/bin/bash\nunset -f which\nsource /ext3/miniforge3/etc/profile.d/conda.sh\nexport PATH=/ext3/miniforge3/bin:$PATH\nexport PYTHONPATH=/ext3/miniforge3/bin:$PATH\n' > /ext3/env.sh
   source /ext3/env.sh
   conda config --remove channels defaults; conda install -y pip ipykernel
   ```
5. **Install the pinned packages** that the HPC notebook checks (still at the `Apptainer>` prompt):
   ```bash
   pip install torch torchvision          # a CUDA build matching the image
   pip install numpy==2.2.6 "scipy>=1.15,<1.17" "pandas>=2.2.3,<3" scikit-learn==1.7.2 av==16.1.0 catboost==1.2.8 \
       transformers==5.3.0 "huggingface-hub>=1.32,<2" ultralytics==8.4.0 opencv-python-headless tqdm \
       "timm>=1.0.15,<2" "einops>=0.8,<1" jupyter packaging matplotlib
   exit
   ```
   From then on, use the overlay read-only (`:ro`).
6. **Test the GPU.** Inside a GPU job, for example `srun --account=cs_gy_6643-2026fa --partition=g2-standard-12 --gres=gpu:1 --pty /bin/bash`:
   ```bash
   apptainer exec --nv --overlay /scratch/$USER/penalty_kick/env/overlay-penalty.ext3:ro /share/apps/images/<cuda-image>.sif \
     /bin/bash -c 'source /ext3/env.sh; python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"'
   ```
7. **Get the data onto the burst file system.**
   - Small files: upload them with the OOD file browser.
   - Larger files: put them on Torch (Globus endpoint `nyu#torch`, WinSCP, or `scp` to `dtn.torch.hpc.nyu.edu`). Then, from an interactive burst job, run:
     ```bash
     scp <NetID>@dtn.torch.hpc.nyu.edu:/scratch/<NetID>/penalty-kick-prediction-cs-gy-6643.zip /scratch/$USER/penalty_kick/data/
     ```
   - Before submitting, check the copy from a compute node with `ls /scratch/$USER/penalty_kick/data`. A file visible only on the login node is not visible to jobs.
8. **Get the code:**
   ```bash
   cd /scratch/$USER && git clone https://github.com/zglyn/penalty-kick-predictor
   cp penalty-kick-predictor/notebooks/Penalty_Kick_HPC.ipynb penalty-kick-predictor/hpc/run_penalty_kick.py penalty-kick-predictor/hpc/penalty_kick.sbatch /scratch/$USER/penalty_kick/code/
   cp -r penalty-kick-predictor/hpc/jobs /scratch/$USER/penalty_kick/code/
   ```
   Then edit `penalty_kick.sbatch`: set `--time`, and set `IMAGE`/`PENALTY_IMAGE` to your `.sif`.
9. **Optional: a Jupyter kernel for OOD.** Following NYU's OOD guide:
   ```bash
   mkdir -p ~/.local/share/jupyter/kernels
   cp -R /share/apps/kernel_template ~/.local/share/jupyter/kernels/penalty
   ```
   - In `kernels/penalty/python`, change the `singularity exec` line to your overlay (`:ro`) and image.
   - In `kernel.json`, set the python path to `/home/<NetID>/.local/share/jupyter/kernels/penalty/python` and the display name to `penalty`.
   - In OOD, start Jupyter on a GPU partition and pick the `penalty` kernel.

## Running jobs

From `/scratch/$USER/penalty_kick/code` on the burst cluster:

```bash
sbatch penalty_kick.sbatch jobs/smoke.json
```

```bash
squeue -u $USER
```

```bash
scancel <jobid>
```

Logs go to `/scratch/$USER/penalty_kick/logs/`.

Run the smoke job and the pilot first, then `jobs/baseline_nested.json`, or one fold per task with `--array=0-4 ... --fold-from-array`. See `docs/WORKFLOW.md`.

## Storage policy used by the notebook

- **`OUTPUT_ROOT=/scratch/$NETID/penalty_kick/outputs`** on the burst file system, as the codelab says. It holds checkpoints, reports, bundles and releases. Checkpoints are written atomically, so a preempted save leaves the previous one intact.
- **Copy finished results back.** Copy them to Torch, Google Drive (rclone or Globus) or your laptop. NYU documents no purge policy for the burst file system, and Torch's scratch is purged after 60 days without access.
- **Node-local temporary storage** is refused as `OUTPUT_ROOT`.
- **Recovery is at epoch boundaries.** After a preemption, the unfinished epoch repeats.

## Still to confirm on the cluster

- **Wall-time limit per partition.** Check with `sinfo -o "%P %l"`.
- **Whether your account can use each partition.** Test with an `srun` on that partition.
- **Which CUDA image to use.** Check `ls /share/apps/images/` on burst.
- **The "preempt script"** the codelab mentions.
