#!/bin/bash
# Penalty-kick batch job for NYU HPC Cloud Bursting (Slurm). Submit from an Open OnDemand terminal:
#   mkdir -p /scratch/$USER/penalty_kick/logs
#   sbatch penalty_kick.sbatch job.json                    # one job (smoke, pilot, holdout, or all folds)
#   sbatch --array=0-4 penalty_kick.sbatch job.json --fold-from-array   # one outer fold per task
# Status: squeue -u $USER     Cancel: scancel <jobid>     Logs: /scratch/$USER/penalty_kick/logs/
#
# Verified from the course codelab: account, partition names, --requeue, --comment="gpu_mps=yes",
# checkpoints under /scratch/$NETID, conda overlays (/share/apps/overlay-fs-ext3, /share/apps/images).
# From NYU's docs (github.com/NYU-RTS/rts-docs, 2026-10): Apptainer replaces Singularity; jobs mount the
# overlay read-only with --nv; GPUs are requested with --gres=gpu:N (the burst example also names a partition).
# Still to confirm on the burst login node: wall-time limit (sinfo -o "%P %l"), CPU/memory, image name.
#SBATCH --job-name=penalty-kick
#SBATCH --account=cs_gy_6643-2026fa
#SBATCH --partition=g2-standard-12          # 1x L4 (codelab). 1x A100 40GB: c12m85-a100-1
#SBATCH --requeue                           # codelab: spot instances may be preempted; jobs are requeued
#SBATCH --comment="gpu_mps=yes"             # codelab: lets several jobs share one GPU (MPS)
#SBATCH --output=/scratch/%u/penalty_kick/logs/%x_%A_%a.out
#SBATCH --time=UNVERIFIED_SET_ME            # e.g. 12:00:00 -- check: sinfo -o "%P %l" on the burst login node
#SBATCH --gres=gpu:1                        # NYU docs: GPUs are requested with --gres
##SBATCH --cpus-per-task=8                  # UNVERIFIED (g2-standard-12 has 12 vCPUs per its name)
##SBATCH --mem=40G                          # UNVERIFIED
set -euo pipefail
SETTINGS=${1:?usage: sbatch penalty_kick.sbatch job.json [--fold-from-array]}
shift
JOB_DIR=${PENALTY_JOB_DIR:-$SLURM_SUBMIT_DIR}   # folder holding the notebook + runner (or a frozen release): submit from it
# From NYU's "Apptainer with Conda" guide (the codelab's singularity_with_conda link moved). Your files:
OVERLAY=${PENALTY_OVERLAY:-/scratch/$USER/penalty_kick/env/overlay-penalty.ext3}
IMAGE=${PENALTY_IMAGE:-UNVERIFIED_SET_ME.sif}   # a CUDA image listed by: ls /share/apps/images  (on the burst cluster)
echo "job=$SLURM_JOB_ID array=${SLURM_ARRAY_TASK_ID:-none} restarts=${SLURM_RESTART_COUNT:-0} host=$(hostname)"
apptainer exec --nv --overlay "$OVERLAY":ro "$IMAGE" /bin/bash -c "
  source /ext3/env.sh
  python '$JOB_DIR/run_penalty_kick.py' --notebook '$JOB_DIR/Penalty_Kick_HPC.ipynb' --settings '$SETTINGS' $*
"
