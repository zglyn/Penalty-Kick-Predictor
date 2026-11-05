# CELL: platform_helpers
# Colab-specific behavior used by the shared cells. Pure Python.
import os
from pathlib import Path
IS_BATCH = False


class StopNotebook(Exception):
    """Ends Run-all quietly after a mode finished (e.g. reports) without a traceback."""
    def _render_traceback_(self): return []


def stop_notebook(message):
    print(message)
    raise StopNotebook(message)


def show_markdown(text):
    from IPython.display import Markdown, display
    display(Markdown(text))


def offer_download(path):
    from google.colab import files
    files.download(str(path))


def is_ephemeral_path(path):
    """/content is local runtime disk (lost when the runtime is replaced); /content/drive is Google Drive."""
    text = str(Path(path).expanduser())
    return text.startswith('/content') and not text.startswith('/content/drive')


def needs_drive(path):
    return str(path).startswith('/content/drive')


def report_root_for(run_dir):
    # Reports on local files stay local so they never force Drive authentication.
    return Path(OUTPUT_ROOT) if needs_drive(run_dir) else Path(SCRATCH_ROOT)
