# CELL: environment
# Colab environment: restart guard, Drive only when a path this mode uses needs it. No NumPy/torch imports.
import json, os, sys
from pathlib import Path

_setup_path = Path('/content/penalty_dependencies.json')
if _setup_path.exists():
    _setup = json.loads(_setup_path.read_text())
    _stat = Path('/proc/self/stat')
    _start = _stat.read_text().rsplit(')', 1)[1].split()[19] if _stat.exists() else ''
    if _setup['installed_kernel'] == f'{os.getpid()}:{_start}':
        raise RuntimeError('Restart required: Runtime > Restart session, then run from Settings again.')
    from importlib.metadata import version
    from packaging.requirements import Requirement
    for _text in _setup['requirements']:
        if not Requirement(_text).specifier.contains(version(Requirement(_text).name), prereleases=True):
            raise RuntimeError(f'Dependency changed: {_text}. Rerun the installation cell and restart.')
elif RUN_MODE != 'reports':
    raise RuntimeError('Run the dependency installation cell first, then restart the session. '
                       '(Reports mode needs only pandas/numpy and skips this requirement.)')

# Mount Drive only if a path this mode actually uses lives on Drive.
_used = [OUTPUT_ROOT] if RUN_MODE != 'reports' else []
if NEEDS_DATASET: _used.append(DATASET_PATH)
if TRAINED_RUN_DIR is not None: _used.append(TRAINED_RUN_DIR)
if CUSTOM_INFERENCE: _used.append(INFERENCE_INPUT)
if EXPORT_BUNDLE_TO: _used.append(EXPORT_BUNDLE_TO)
if any(needs_drive(p) for p in _used) and not os.path.ismount('/content/drive'):
    from google.colab import drive
    drive.mount('/content/drive')
else:
    print('Drive not mounted (no path used by this mode needs it, or it is already mounted).')
ENVIRONMENT_INFO = dict(platform='colab', python=sys.version.split()[0])
print('WARNING: SCRATCH_ROOT (/content) is ephemeral and Colab Pro+ runtimes can still be disconnected or replaced. '
      'Completed checkpoints go to OUTPUT_ROOT on Drive; training resumes from the last completed epoch.')
