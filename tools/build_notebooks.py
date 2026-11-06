#!/usr/bin/env python
"""Generate notebooks/Penalty_Kick_Colab_VideoMAE_Full.ipynb and notebooks/Penalty_Kick_HPC.ipynb from notebook_src/.

notebook_src/shared/ is the single maintained source of computational code; notebook_src/colab and
notebook_src/hpc hold only platform cells (settings defaults, environment, install/batch helpers, text).
Every cell records its source file in metadata so tools/check_consistency.py can detect drift.
Run:  python tools/build_notebooks.py           (writes)      python tools/build_notebooks.py --check
"""
import argparse, hashlib, json, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / 'notebook_src'

COMMON = ['shared/settings_validation.py', '{p}/environment.py', 'shared/run_artifacts.py']
CORE = ['shared/constants_and_summary_helpers.py', 'shared/reports_mode.py', 'shared/heavy_imports.py',
        'shared/md_dataset.md', 'shared/dataset_loading.py',
        'shared/md_adapters.md', 'shared/backbone_adapters.py', 'shared/inference_helpers.py',
        'shared/md_diagnostics.md', 'shared/diagnostics_helpers.py',
        'shared/md_smoke.md', 'shared/experiment_configuration.py',
        'shared/md_extraction.md', 'shared/extraction.py', 'shared/md_roles.md', 'shared/roles_and_crops.py',
        'shared/md_model.md', 'shared/model_and_training.py', 'shared/md_timing.md', 'shared/timing_benchmark.py',
        'shared/md_nested.md', 'shared/nested_orchestration.py',
        'shared/md_inference_bindings.md', 'shared/inference_glue.py', 'shared/smoke_holdout_dispatch.py',
        'shared/md_validation_report.md', 'shared/validation_report.py', 'shared/md_heatmaps.md', 'shared/heatmaps.py',
        'shared/md_submission.md', 'shared/submission.py', 'shared/md_summary.md', 'shared/pipeline_summary_run.py',
        'shared/bundle_export.py', 'shared/md_references.md', 'shared/guide_outputs.md']
LAYOUT = {
    'colab': ['colab/title.md', 'colab/install.md', 'colab/install.py', 'colab/restart.md', 'colab/settings.py',
              'colab/platform_helpers.py', *COMMON, *CORE, 'colab/recipes.md'],
    'hpc': ['hpc/title.md', 'hpc/settings.md', 'hpc/settings.py', 'hpc/platform_helpers.py', *COMMON,
            'hpc/batch_files_and_release.py', *CORE, 'hpc/recipes.md'],
}
OUTPUT = {'colab': ROOT / 'notebooks' / 'Penalty_Kick_Colab_VideoMAE_Full.ipynb', 'hpc': ROOT / 'notebooks' / 'Penalty_Kick_HPC.ipynb'}
COPIES = {SRC / 'hpc' / 'run_penalty_kick.py': ROOT / 'hpc' / 'run_penalty_kick.py',
          SRC / 'hpc' / 'penalty_kick.sbatch': ROOT / 'hpc' / 'penalty_kick.sbatch'}
METADATA = {
    'colab': {'accelerator': 'GPU', 'colab': {'gpuType': 'A100', 'machine_shape': 'hm', 'provenance': []},
              'kernelspec': {'display_name': 'Python 3', 'name': 'python3'}, 'language_info': {'name': 'python'}},
    'hpc': {'kernelspec': {'display_name': 'Python 3 (penalty-kick overlay)', 'language': 'python', 'name': 'python3'},
            'language_info': {'name': 'python'}},
}


def shared_files():
    names = []
    for platform in LAYOUT:
        for item in LAYOUT[platform]:
            if item.startswith('shared/') and item not in names: names.append(item)
    return names


def shared_fingerprint():
    digest = hashlib.sha256()
    for name in shared_files():
        digest.update(name.encode() + b'\0' + (SRC / name).read_bytes() + b'\0')
    return digest.hexdigest()[:20]


def render(platform):
    fingerprint = shared_fingerprint(); cells = []
    for item in LAYOUT[platform]:
        rel = item.format(p=platform)
        text = (SRC / rel).read_text()
        text = text.replace('{{SHARED_CODE_FINGERPRINT}}', fingerprint)
        if '{{RUNNER_SOURCE}}' in text:
            text = text.replace('{{RUNNER_SOURCE}}', repr((SRC / 'hpc' / 'run_penalty_kick.py').read_text()))
            text = text.replace('{{SBATCH_SOURCE}}', repr((SRC / 'hpc' / 'penalty_kick.sbatch').read_text()))
        kind = 'markdown' if rel.endswith('.md') else 'code'
        cell = dict(cell_type=kind, id=rel.replace('/', '-').replace('.', '-').replace('_', '-'),
                    metadata={'penalty_source': rel, 'penalty_shared': rel.startswith('shared/')},
                    source=text.splitlines(keepends=True))
        if kind == 'code': cell.update(execution_count=None, outputs=[])
        cells.append(cell)
    return dict(cells=cells, metadata=METADATA[platform], nbformat=4, nbformat_minor=5)


def dumps(nb):
    return json.dumps(nb, indent=1, ensure_ascii=False) + '\n'


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--check', action='store_true')
    args = parser.parse_args(); stale = []
    for platform, path in OUTPUT.items():
        text = dumps(render(platform))
        if args.check:
            if not path.exists() or path.read_text() != text: stale.append(str(path))
        else:
            path.parent.mkdir(parents=True, exist_ok=True); path.write_text(text); print('wrote', path)
    for src, dst in COPIES.items():
        if args.check:
            if not dst.exists() or dst.read_text() != src.read_text(): stale.append(str(dst))
        else:
            dst.parent.mkdir(parents=True, exist_ok=True); dst.write_text(src.read_text()); print('wrote', dst)
    if stale:
        print('Generated files are stale; run python tools/build_notebooks.py:\n' + '\n'.join(stale)); return 1
    print('shared code fingerprint:', shared_fingerprint())
    return 0


if __name__ == '__main__':
    sys.exit(main())
