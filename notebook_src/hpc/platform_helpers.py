# CELL: platform_helpers
# HPC-specific behavior used by the shared cells. Pure Python; no Colab or Drive code exists in this notebook.
import os
from pathlib import Path
IS_BATCH = os.environ.get('PENALTY_BATCH') == '1'   # set by run_penalty_kick.py


class StopNotebook(Exception):
    def _render_traceback_(self): return []


def stop_notebook(message):
    print(message)
    if IS_BATCH: raise SystemExit(0)
    raise StopNotebook(message)


def show_markdown(text):
    if IS_BATCH: print(text); return
    from IPython.display import Markdown, display
    display(Markdown(text))


def offer_download(path):
    print('Saved (copy it with the Open OnDemand file browser or scp):', path)


def is_ephemeral_path(path):
    """Node-local or temporary storage disappears with the (spot) node; /scratch/$NETID is the codelab's
    checkpoint location."""
    text = str(Path(path).expanduser())
    roots = [os.environ.get(k) for k in ('TMPDIR', 'SLURM_TMPDIR')] + ['/tmp', '/dev/shm']
    return any(r and (text == r or text.startswith(r.rstrip('/') + '/')) for r in roots)


def report_root_for(run_dir):
    return Path(OUTPUT_ROOT)
