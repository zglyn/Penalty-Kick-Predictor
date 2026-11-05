# CELL: environment
# HPC environment checks. No Colab imports, no Drive mount, no package installation, no Colab paths.
import json, os, sys
from pathlib import Path
# Pinned to the same versions as the Colab installation cell (see hpc/ENVIRONMENT.md for the overlay setup).
REQUIREMENTS = ['numpy==2.2.6', 'scipy>=1.15,<1.17', 'pandas>=2.2.3,<3', 'scikit-learn==1.7.2', 'av==16.1.0',
                'catboost==1.2.8', 'transformers==5.3.0', 'huggingface-hub>=1.32,<2', 'ultralytics==8.4.0',
                'tqdm', 'timm>=1.0.15,<2', 'einops>=0.8,<1', 'torch>=2.6', 'torchvision']

_job = os.environ.get('SLURM_JOB_ID')
ENVIRONMENT_INFO = dict(platform='hpc', python=sys.version.split()[0], host=os.uname().nodename, slurm_job_id=_job,
                        partition=os.environ.get('SLURM_JOB_PARTITION'), account=os.environ.get('SLURM_JOB_ACCOUNT'),
                        restart_count=os.environ.get('SLURM_RESTART_COUNT'),
                        container=os.environ.get('SINGULARITY_CONTAINER') or os.environ.get('APPTAINER_CONTAINER'),
                        cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'), batch=IS_BATCH)
print(json.dumps(ENVIRONMENT_INFO))
if RUN_MODE in {'smoke', 'train', 'resume'} and not _job and not ALLOW_OUTSIDE_SLURM:
    raise RuntimeError('No Slurm allocation (SLURM_JOB_ID unset): this looks like a login node. Start an Open OnDemand '
                       'Jupyter session on a GPU partition or submit penalty_kick.sbatch.')
if RUN_MODE != 'reports':
    from importlib.metadata import version, PackageNotFoundError
    from packaging.requirements import Requirement
    _bad = []
    for _text in REQUIREMENTS + ['opencv-python-headless>=4.10|opencv-python>=4.10']:
        _options = _text.split('|'); _ok = False; _found = []
        for _option in _options:
            _req = Requirement(_option)
            try:
                _v = version(_req.name); _found.append(f'{_req.name} {_v}')
                _ok = _ok or _req.specifier.contains(_v, prereleases=True)
            except PackageNotFoundError:
                pass
        if not _ok: _bad.append(f'{_text} (found: {", ".join(_found) or "missing"})')
    if _bad:
        raise RuntimeError('The Singularity/conda environment does not match the pinned requirements (this notebook '
                           'and its jobs never install packages):\n- ' + '\n- '.join(_bad) + '\nSee hpc/ENVIRONMENT.md.')
    if not ENVIRONMENT_INFO['container']:
        print('NOTE: not inside a Singularity container; the codelab recommends Singularity + conda overlays.')
