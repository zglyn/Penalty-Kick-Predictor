"""Both notebooks: schema, compilation, generation freshness, shared-code consistency, platform separation,
defaults, self-containment and the VideoMAE call-site audit."""
import ast, json, re, subprocess, sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))
from nb_harness import NOTEBOOKS, REPO, cell_source, notebook_cells, run_cell

ORDER = ['settings', 'platform_helpers', 'settings_validation', 'environment', 'run_artifacts',
         'constants_and_summary_helpers', 'reports_mode', 'heavy_imports', 'dataset_loading', 'backbone_adapters',
         'inference_helpers', 'diagnostics_helpers', 'experiment_configuration', 'extraction', 'roles_and_crops',
         'model_and_training', 'timing_benchmark', 'nested_orchestration', 'inference_glue', 'smoke_holdout_dispatch',
         'validation_report', 'heatmaps', 'submission', 'pipeline_summary_run', 'bundle_export']


def markers(platform):
    _, cells = notebook_cells(platform)
    return [c.split('\n', 1)[0].replace('# CELL: ', '') for c in cells if c.startswith('# CELL: ')]


@pytest.mark.parametrize('platform', ['colab', 'hpc'])
def test_schema_compilation_and_order(platform):
    import nbformat
    nb = nbformat.read(str(NOTEBOOKS[platform]), as_version=4)
    nbformat.validate(nb)
    for i, cell in enumerate(nb.cells):
        if cell.cell_type == 'code':
            compile(cell.source, f'{platform}:{i}', 'exec')
            assert not cell.outputs and cell.execution_count is None
    assert [m for m in markers(platform) if m in ORDER] == ORDER
    local = re.compile(r'^\s*(from|import)\s+(nb_harness|build_notebooks|check_consistency|test_\w+)\b', re.M)
    for c in notebook_cells(platform)[1]:
        assert not local.search(c) and 'TODO' not in c


def test_generated_files_are_fresh_and_consistent():
    assert subprocess.run([sys.executable, str(REPO / 'tools' / 'build_notebooks.py'), '--check'], capture_output=True).returncode == 0
    out = subprocess.run([sys.executable, str(REPO / 'tools' / 'check_consistency.py')], capture_output=True, text=True)
    assert out.returncode == 0, out.stdout


def test_consistency_checker_detects_drift(tmp_path):
    sys.path.insert(0, str(REPO / 'tools'))
    import check_consistency
    nb = json.loads(NOTEBOOKS['hpc'].read_text())
    target = next(c for c in nb['cells'] if c['metadata'].get('penalty_source') == 'shared/inference_helpers.py')
    target['source'].append('\n# drift\n')
    (tmp_path / 'hpc.ipynb').write_text(json.dumps(nb))
    assert any('differs' in p for p in check_consistency.check(NOTEBOOKS['colab'], tmp_path / 'hpc.ipynb'))
    nb = json.loads(NOTEBOOKS['hpc'].read_text())
    nb['cells'][-1]['source'] = ['from google.colab import drive\n']; nb['cells'][-1]['cell_type'] = 'code'
    nb['cells'][-1].update(outputs=[], execution_count=None)
    (tmp_path / 'hpc2.ipynb').write_text(json.dumps(nb))
    assert any('google' in p for p in check_consistency.check(NOTEBOOKS['colab'], tmp_path / 'hpc2.ipynb'))


def test_hpc_notebook_has_no_colab_dependency():
    nb, cells = notebook_cells('hpc')
    code = '\n'.join(t for t, c in zip(cells, nb['cells']) if c['cell_type'] == 'code')
    for pattern in [r'google\.colab', r'drive\.mount', r"['\"]/content", r'pip install', r'files\.download']:
        assert not re.search(pattern, code), pattern


def settings_values(platform):
    values = {}
    for node in ast.parse(cell_source('settings', platform)).body:
        if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name):
            try: values[node.targets[0].id] = ast.literal_eval(node.value)
            except ValueError: values[node.targets[0].id] = ast.unparse(node.value)
    return values


def test_settings_share_keys_and_inexpensive_defaults():
    colab, hpc = settings_values('colab'), settings_values('hpc')
    hpc_only = {'NETID', 'ALLOW_OUTSIDE_SLURM', 'WRITE_BATCH_FILES', 'FREEZE_RELEASE', 'NOTEBOOK_PATH'}
    assert set(colab) == set(hpc) - hpc_only
    for v in (colab, hpc):
        assert v['RUN_MODE'] == 'smoke' and v['PREDICT_PRIVATE'] is False and v['RUN_ABLATIONS'] is False
        assert v['TRAINING_POLICY'] == 'finetune' and v['BACKBONE'] == 'videomae'
    assert colab['DATASET_PATH'] == "Path('/content/drive/MyDrive/Fall2026/CV/penalty-kick-prediction-cs-gy-6643.zip')"
    assert colab['OUTPUT_ROOT'] == "Path('/content/drive/MyDrive/Fall2026/CV/penalty_kick_outputs')"
    assert colab['EVALUATION_MODE'] == 'development' and hpc['EVALUATION_MODE'] == 'nested_cv'
    assert '/scratch/' in hpc['OUTPUT_ROOT'] and '/content' not in str(hpc.values())
    for platform in ('colab', 'hpc'):
        assert not re.search(r'^\s*import (numpy|torch|pandas)', cell_source('settings', platform), re.M)


def test_videomae_specific_calls_only_inside_adapter_or_preflight():
    allowed = {'backbone_adapters', 'heavy_imports', 'diagnostics_helpers', 'experiment_configuration', 'inference_glue'}
    for platform in ('colab', 'hpc'):
        nb, cells = notebook_cells(platform)
        for c, cell in zip(cells, nb['cells']):
            if cell['cell_type'] != 'code': continue
            name = c.split('\n', 1)[0].replace('# CELL: ', '')
            for pattern in ['.videomae', 'fc_norm', 'pixel_values=', 'VideoMAEForVideoClassification.from_pretrained']:
                assert not (pattern in c and name not in allowed), f'{pattern} in {platform}:{name}'


def test_inference_path_references_no_fitting_entry_points():
    forbidden = ['fit_video', 'fit_roles', 'diagnostic_stage', 'run_nested_diagnostics', 'fit_feature_scaler',
                 '.fit(', 'select_diagnostic_mixture', 'select_reference_components', 'torch.optim', 'build_stage']
    for name in ['inference_helpers', 'inference_glue']:
        code = '\n'.join(l for l in cell_source(name).splitlines() if not l.strip().startswith('#'))
        code = re.sub(r'"""[\s\S]*?"""', '', code)
        for f in forbidden: assert f not in code, f'{name} references {f}'


def test_existing_cpu_diagnostics_smoke_checks_still_pass(tmp_path):
    ns = dict(__name__='__smoke__', RUN_MODE='reports', SCRATCH_ROOT=tmp_path)
    run_cell('backbone_adapters', ns); run_cell('diagnostics_helpers', ns)
    assert ns['smoke_component_reports'](tmp_path / 'components')['status'] == 'PASS'
    assert ns['tracking_diagnostics_smoke'](tmp_path / 'tracking')['status'] == 'PASS'
    assert ns['ablation_feature_smoke_test']()['status'] == 'PASS'
    arch = ns['ablation_architecture_smoke_test']()
    assert arch['status'] == 'PASS' and {v['policy'] for v in arch['variants']} == {'finetune', 'frozen_probe'}
