# CELL: install
# Run this cell once, then use Runtime > Restart session before proceeding.
import sys, os, json, hashlib, subprocess
from pathlib import Path
from importlib.metadata import version, PackageNotFoundError
from packaging.requirements import Requirement

INSTALL_REQUIREMENTS = [
    'numpy==2.2.6', 'scipy>=1.15,<1.17', 'pandas>=2.2.3,<3',
    'scikit-learn==1.7.2', 'av==16.1.0', 'catboost==1.2.8',
    'transformers==5.3.0', 'huggingface-hub>=1.32,<2',
    'ultralytics==8.4.0', 'opencv-python==4.12.0.88', 'tqdm',
    # Official V-JEPA 2.1 repository code needs timm and einops (V-JEPA 2 uses transformers).
    'timm>=1.0.15,<2', 'einops>=0.8,<1',
]
SETUP_MARKER = Path('/content/penalty_dependencies.json')
SETUP_SIGNATURE = hashlib.sha256(json.dumps(INSTALL_REQUIREMENTS).encode()).hexdigest()

def kernel_identity():
    # Distinguish kernels even if an operating-system PID is reused.
    stat=Path('/proc/self/stat')
    start=stat.read_text().rsplit(')',1)[1].split()[19] if stat.exists() else ''
    return f'{os.getpid()}:{start}'

def requirements_match():
    for text in INSTALL_REQUIREMENTS:
        req=Requirement(text)
        try: installed=version(req.name)
        except PackageNotFoundError: return False
        if not req.specifier.contains(installed,prereleases=True): return False
    return True

if sys.version_info < (3,11):
    raise RuntimeError('This dependency set requires Python 3.11 or newer; use a current Colab runtime.')
state=json.loads(SETUP_MARKER.read_text()) if SETUP_MARKER.exists() else {}
if state.get('signature')==SETUP_SIGNATURE and requirements_match():
    if state.get('installed_kernel')==kernel_identity():
        print('Dependencies installed. REQUIRED: Runtime > Restart session, then run from the next code cell.')
    else:
        print('Dependencies already installed and session restarted. Continue with the next code cell.')
else:
    print('Python:',sys.version.split()[0])
    # Preserve Colab CUDA builds; let pip resolve the remaining packages together.
    constraints=Path('/content/penalty_torch_constraints.txt')
    constraints.write_text('\n'.join(f'{name}=={version(name)}' for name in ['torch','torchvision'])+'\n')
    subprocess.check_call([sys.executable,'-m','pip','install','--only-binary=:all:',
                           '--constraint',str(constraints),*INSTALL_REQUIREMENTS])
    SETUP_MARKER.write_text(json.dumps(dict(signature=SETUP_SIGNATURE,
        installed_kernel=kernel_identity(),requirements=INSTALL_REQUIREMENTS),indent=2))
    print('\nINSTALLATION COMPLETE. REQUIRED: Runtime > Restart session.')
    print('Then run from the next code cell. Do not import NumPy/sklearn in this session.')
