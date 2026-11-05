#!/usr/bin/env python
"""Run Penalty_Kick_HPC.ipynb's code cells as a batch job (no browser, installs, Drive or prompts).

    python run_penalty_kick.py --notebook Penalty_Kick_HPC.ipynb --settings job.json [--set KEY=VALUE ...]
                               [--fold-from-array]

The notebook is the single source of computational code: this runner executes its code cells in order in one
namespace, applying settings overrides immediately after the Settings cell. Unknown setting names are rejected.
If the notebook sits in a frozen release (RELEASE.json beside it), every release file is checksum-verified first.
--fold-from-array sets FOLD_SELECTION=[$SLURM_ARRAY_TASK_ID] (one outer fold per array task).
Preemption: Slurm requeues the job (#SBATCH --requeue); training resumes from the last completed epoch
checkpoint and repeats the unfinished epoch.
"""
import argparse, ast, hashlib, json, os, signal, sys, time
from pathlib import Path


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    p.add_argument('--notebook', required=True)
    p.add_argument('--settings', help='JSON object of settings overrides')
    p.add_argument('--set', action='append', default=[], metavar='KEY=VALUE', help='Python literal override')
    p.add_argument('--fold-from-array', action='store_true')
    return p.parse_args(argv)


def verify_release(folder):
    info = json.loads((folder / 'RELEASE.json').read_text())
    for name, digest in info['files'].items():
        actual = hashlib.sha256((folder / name).read_bytes()).hexdigest()
        if actual != digest: raise SystemExit(f'Release file {name} was modified; refusing to run a changed release.')
    print('Release verified:', info['release_id'])
    return info


def load_overrides(args):
    overrides = {}
    if args.settings: overrides.update(json.loads(Path(args.settings).read_text()))
    for item in args.set:
        key, _, value = item.partition('=')
        try: overrides[key] = ast.literal_eval(value)
        except (ValueError, SyntaxError): overrides[key] = value
    if args.fold_from_array:
        task = os.environ.get('SLURM_ARRAY_TASK_ID')
        if task is None: raise SystemExit('--fold-from-array needs a Slurm array job (SLURM_ARRAY_TASK_ID).')
        overrides['FOLD_SELECTION'] = [int(task)]
    return overrides


def apply_overrides(ns, overrides):
    unknown = sorted(k for k in overrides if k not in ns)
    if unknown: raise SystemExit(f'Unknown settings {unknown}; names must match the notebook Settings cell.')
    for key, value in overrides.items():
        if isinstance(ns[key], Path) and value is not None: value = Path(value)
        ns[key] = value
    print('Settings overrides:', json.dumps({k: str(v) for k, v in overrides.items()}))


def main(argv=None):
    args = parse_args(argv)
    notebook = Path(args.notebook).resolve()
    if (notebook.parent / 'RELEASE.json').exists(): verify_release(notebook.parent)
    cells = [''.join(c['source']) for c in json.loads(notebook.read_text())['cells'] if c['cell_type'] == 'code']
    if not any("PLATFORM = 'hpc'" in c for c in cells): raise SystemExit('This runner executes the HPC notebook only.')
    os.environ['PENALTY_BATCH'] = '1'; os.environ.setdefault('MPLBACKEND', 'Agg')
    overrides = load_overrides(args)

    def on_term(signum, frame):
        print(f'Received signal {signum} (likely spot preemption) at {time.ctime()}; the requeued job resumes from the '
              'last completed epoch checkpoint.', flush=True)
        sys.exit(143)
    signal.signal(signal.SIGTERM, on_term)
    ns = dict(__name__='__main__', NOTEBOOK_PATH=notebook,
              display=lambda *objs, **kw: [print(getattr(o, 'to_string', lambda: o)()) for o in objs])
    for index, source in enumerate(cells):
        if source.startswith('# CELL: batch_files_and_release'): continue   # interactive helper only
        print(f'--- notebook code cell {index} ---', flush=True)
        try:
            exec(compile(source, f'{notebook.name}:cell{index}', 'exec'), ns)
        except SystemExit as stop:
            if stop.code in (0, None): print('Mode finished.'); return 0
            raise
        if source.startswith('# CELL: settings'): apply_overrides(ns, overrides)
    print('Notebook run complete.')
    return 0


if __name__ == '__main__':
    sys.exit(main())
