"""Spec-driven workflow tests: prediction-cache identity, ZIP-free custom inference, lightweight reports for both
notebooks (HPC through the real batch runner), bundles, writer isolation, parallel fold jobs, development preset,
releases and cross-wrapper equivalence. CPU only; synthetic data and tiny models (no performance meaning)."""
import json, os, shutil, subprocess, sys, textwrap, time
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent))
import nb_harness as h

PY = sys.executable


@pytest.fixture(scope='module')
def trained(tmp_path_factory):
    root = tmp_path_factory.mktemp('platform_run')
    ns = h.base_namespace(root)
    result = h.train_nested(ns)
    ns.update(TRAINED_RUN_DIR=Path(result['run']), RUN_MODE='inference', INFERENCE_RAW=root / 'raw')
    ns['INFERENCE_RAW'].mkdir()
    return dict(ns=ns, run=Path(result['run']), root=root)


def custom_csv(ns, path, gender_flip=None):
    rows = []
    for c, f, g in zip(ns['test']['clip'][:3], ns['test'].preferred_foot[:3], ns['test'].gender[:3]):
        if c == gender_flip: g = 'M' if g == 'F' else 'F'
        rows.append(dict(clip=c, path=str(ns['video_paths'][c]), preferred_foot=f, gender=g))
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def infer_csv(ns, csv, out):
    frame, routing = ns['resolve_inference_clips'](str(csv))
    with h.forbid_fitting(ns):
        return ns['run_saved_inference'](ns['TRAINED_RUN_DIR'], frame, hooks=ns['INFERENCE_HOOKS'], routing=routing,
                                         device='cpu', dataset_id=None, output_root=out)


def test_changed_metadata_changes_prediction_identity_and_recomputes(trained, tmp_path):
    ns = trained['ns']; out = tmp_path / 'inference_out'
    r1 = infer_csv(ns, custom_csv(ns, tmp_path / 'a.csv'), out)
    assert r1['record']['cache_statistics']['predictions_computed'] == 3 * 2
    r2 = infer_csv(ns, custom_csv(ns, tmp_path / 'a.csv'), out)
    assert r2['record']['cache_statistics']['predictions_computed'] == 0       # identical inputs reuse
    flipped = ns['test']['clip'].iloc[1]
    r3 = infer_csv(ns, custom_csv(ns, tmp_path / 'b.csv', gender_flip=flipped), out)
    stats = r3['record']['cache_statistics']
    assert stats['predictions_computed'] == 1 * 2 and stats['predictions_reused'] == 2 * 2   # only the changed clip
    assert stats['crops_built'] == 0                                            # crops are metadata-independent
    assert not any((trained['run'] / 'inference').glob('**/*'))                # nothing written into the run


def test_custom_inference_needs_no_competition_files_or_labels(trained, tmp_path, monkeypatch):
    """Real dataset_loading + experiment_configuration + run_inference_mode with no ZIP on disk."""
    import types
    monkeypatch.setitem(sys.modules, 'ultralytics', types.SimpleNamespace(YOLO=lambda *a, **k: None))
    ns = dict(trained['ns'])
    ns.update(RUN_MODE='inference', NEEDS_DATASET=False, CUSTOM_INFERENCE=True, DATASET_PATH=tmp_path / 'missing.zip',
              INFERENCE_INPUT=str(custom_csv(ns, tmp_path / 'custom.csv')), OUTPUT_ROOT=tmp_path / 'out',
              TRAINED_RUN_DIR=trained['run'], VERIFY_CHECKSUMS=True, MIGRATE_LEGACY_RUN=False, DEVICE='cpu',
              hashlib=__import__('hashlib'), zipfile=__import__('zipfile'), BACKBONE='vjepa2_1_vitb_384',
              BACKBONE_REVISION=None, TRAINING_POLICY='frozen_probe', EVALUATION_MODE='nested_cv', QUICK_SMOKE=False,
              PREDICT_PRIVATE=False)
    ns.pop('find_one', None)
    for name in ['dataset_loading', 'experiment_configuration']:
        h.run_cell(name, ns)
    assert ns['DATA_ID'] is None and len(ns['train']) == 0 and ns['Y'].shape == (0, 2) and ns['template'] is None
    assert (ns['N_FRAMES'], ns['SIZE']) == (2, 4)                                # saved settings, not defaults
    assert ns['BACKBONE_SPEC']['name'] == 'videomae' and ns['TRAINING_POLICY'] == 'finetune'
    h.install_label_free_stubs(ns)
    h.run_cell('inference_glue', ns); h.install_label_free_stubs(ns)
    with h.forbid_fitting(ns):
        result = ns['run_inference_mode']()
    assert len(result['table']) == 3 and set(result['table'].exposure) == {'not_in_training'}
    assert str(result['output_dir']).startswith(str(tmp_path / 'out'))
    assert not (tmp_path / 'missing.zip').exists()


def write_report_run(root):
    """A complete synthetic report run made by the notebook's own exporters (see test_pipeline_summary)."""
    sys.path.insert(0, str(Path(__file__).parent))
    from test_pipeline_summary import make_run
    ns = dict(__name__='__r__', RUN_MODE='reports', Path=Path, SCRATCH_ROOT=root / 's')
    h.run_cell('diagnostics_helpers', ns); h.run_cell('constants_and_summary_helpers', ns)
    run, _ = make_run(ns, root / 'reported_run', harm=True)
    return run


REPORT_PROBE = '''
import json, sys, runpy
from pathlib import Path
heavy = ('torch', 'cv2', 'av', 'transformers', 'ultralytics', 'catboost', 'sklearn')
'''


def test_reports_mode_hpc_batch_runner_is_lightweight(tmp_path):
    run = write_report_run(tmp_path)
    settings = tmp_path / 'job.json'
    settings.write_text(json.dumps(dict(RUN_MODE='reports', TRAINED_RUN_DIR=str(run), OUTPUT_ROOT=str(tmp_path / 'out'),
                                        SCRATCH_ROOT=str(tmp_path / 'scratch'))))
    script = REPORT_PROBE + textwrap.dedent(f'''
        sys.argv = ['run_penalty_kick.py', '--notebook', {str(h.NOTEBOOKS['hpc'])!r}, '--settings', {str(settings)!r}]
        sys.path.insert(0, {str(h.REPO / 'hpc')!r})
        import run_penalty_kick
        code = run_penalty_kick.main(sys.argv[1:])
        print(json.dumps(dict(code=code, heavy=[m for m in heavy if m in sys.modules])))
    ''')
    out = subprocess.run([PY, '-c', script], capture_output=True, text=True, env=dict(os.environ, USER='testnetid'))
    assert out.returncode == 0, out.stdout[-2000:] + out.stderr[-2000:]
    result = json.loads(out.stdout.strip().splitlines()[-1])
    assert result == dict(code=0, heavy=[])
    reports = list((tmp_path / 'out' / 'reports').glob('reported_run__*/pipeline_summary.md'))
    assert len(reports) == 1 and not (run / 'pipeline_summary.md').exists()     # never writes into the source run


def test_reports_mode_colab_notebook_is_lightweight_without_drive(tmp_path):
    run = write_report_run(tmp_path)
    replacements = [("RUN_MODE = 'train'", "RUN_MODE = 'reports'"),
                    ('TRAINED_RUN_DIR = None', f'TRAINED_RUN_DIR = Path({str(run)!r})'),
                    ("SCRATCH_ROOT = Path('/content/penalty_scratch')", f'SCRATCH_ROOT = Path({str(tmp_path / "scratch")!r})')]
    script = REPORT_PROBE + f'REPLACEMENTS = {replacements!r}\n' + textwrap.dedent(f'''
        nb = json.loads(Path({str(h.NOTEBOOKS['colab'])!r}).read_text())
        ns = dict(__name__='__main__')
        import builtins
        real_import = builtins.__import__
        def guard(name, *a, **k):
            if name.startswith('google'): raise AssertionError('Colab/Drive import attempted in local reports mode')
            return real_import(name, *a, **k)
        builtins.__import__ = guard
        stopped = False
        for c in nb['cells']:
            src = ''.join(c['source'])
            if c['cell_type'] != 'code' or src.startswith('# CELL: install'): continue
            if src.startswith('# CELL: settings'):
                for old, new in REPLACEMENTS: src = src.replace(old, new)
            try: exec(src, ns)
            except Exception as e:
                if type(e).__name__ == 'StopNotebook': stopped = True; break
                raise
        print(json.dumps(dict(stopped=stopped, heavy=[m for m in heavy if m in sys.modules])))
    ''')
    out = subprocess.run([PY, '-c', script], capture_output=True, text=True)
    assert out.returncode == 0, out.stdout[-2000:] + out.stderr[-2000:]
    assert json.loads(out.stdout.strip().splitlines()[-1]) == dict(stopped=True, heavy=[])
    assert len(list((tmp_path / 'scratch' / 'reports').glob('*/pipeline_summary.md'))) == 1


def test_bundle_relocates_verifies_and_rejects_incomplete_transfers(trained, tmp_path):
    ns = trained['ns']
    bundle = ns['export_bundle'](trained['run'], tmp_path / 'hpc_side')
    moved = tmp_path / 'drive_side' / bundle.name
    shutil.copytree(bundle, moved)                                              # different absolute root
    shutil.rmtree(bundle)
    ns['verify_bundle'](moved)
    assert not any('_latest.pt' in p.name for p in moved.rglob('*'))           # no optimizer/resume state
    frame, routing = ns['resolve_inference_clips']('private')
    with h.forbid_fitting(ns):
        r = ns['run_saved_inference'](moved, frame, hooks=ns['INFERENCE_HOOKS'], routing=routing, device='cpu',
                                      dataset_id=ns['DATA_ID'], output_root=tmp_path / 'out')
    saved = np.load(trained['run'] / 'predictions.npz')
    for j in range(2): np.testing.assert_allclose(r['final'][j], saved[f'private_{j}'], atol=1e-6)
    assert r['record']['cache_statistics']['crops_built'] > 0                    # bundle omits crops: rebuilt label-free
    summary = ns['write_pipeline_summary'](moved, output_dir=tmp_path / 'rep')[0]
    assert summary['status'] == 'complete'
    partial = tmp_path / 'partial'; shutil.copytree(moved, partial); (partial / ns['BUNDLE_MARKER']).unlink()
    with pytest.raises(FileNotFoundError, match='incomplete'): ns['verify_bundle'](partial)
    member = next(p for p in moved.rglob('*.joblib'))
    member.unlink()
    with pytest.raises(IOError, match='missing'): ns['verify_bundle'](moved)


def test_writer_locks_isolate_concurrent_writers(tmp_path):
    ns = dict(__name__='__lock__', PLATFORM='colab')
    h.run_cell('run_artifacts', ns); ns['ATTEMPT_ID'] = 'attempt-a'
    holder = subprocess.Popen([PY, '-c', textwrap.dedent(f'''
        import json, sys, time
        nb = json.loads(open({str(h.NOTEBOOKS['hpc'])!r}).read())
        src = [''.join(c['source']) for c in nb['cells'] if ''.join(c['source']).startswith('# CELL: run_artifacts')][0]
        ns = dict(__name__='x', PLATFORM='hpc', ATTEMPT_ID='attempt-hpc'); exec(src, ns)
        with ns['RunWriterLock']({str(tmp_path)!r}, 'fold0'):
            print('LOCKED', flush=True); time.sleep(6)
    ''')], stdout=subprocess.PIPE, text=True)
    assert holder.stdout.readline().strip() == 'LOCKED'
    with pytest.raises(ns['RunLockedError'], match='attempt-hpc'):
        with ns['RunWriterLock'](tmp_path, 'fold0'): pass
    with ns['RunWriterLock'](tmp_path, 'fold1'): pass                           # other scopes stay available
    holder.wait()
    lock = tmp_path / 'locks' / 'fold0.lock'
    lock.write_text(json.dumps(dict(attempt='dead', pid=1, host='gone', heartbeat=time.time() - 3600)))
    with ns['RunWriterLock'](tmp_path, 'fold0'): pass                           # stale heartbeat -> takeover
    os.environ['SLURM_JOB_ID'] = '777'
    try:
        lock.write_text(json.dumps(dict(attempt='prev', pid=1, host='other', slurm_job_id='777', heartbeat=time.time())))
        with ns['RunWriterLock'](tmp_path, 'fold0'): pass                       # requeued job -> takeover
    finally:
        del os.environ['SLURM_JOB_ID']
    assert not lock.exists()


PARALLEL_FOLD = '''
import sys, json
sys.path.insert(0, {tests!r})
import nb_harness as h
ns = h.base_namespace({root!r})
with h.cpu_training_patches():
    result = ns['run_nested_diagnostics']('production', private_inference=True, fold_selection=[{fold}])
print(json.dumps(dict(fold={fold}, aggregated=result is not None, run=str(ns['RUN']))))
'''


def test_parallel_fold_jobs_aggregate_once_and_match_the_monolithic_run(trained, tmp_path):
    procs = [subprocess.Popen([PY, '-c', PARALLEL_FOLD.format(tests=str(Path(__file__).parent), root=str(tmp_path), fold=k)],
                              stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True) for k in (0, 1)]
    outs = [p.communicate(timeout=1500) for p in procs]
    for p, (o, e) in zip(procs, outs): assert p.returncode == 0, e[-3000:]
    results = [json.loads([l for l in o.splitlines() if l.startswith('{"fold"')][-1]) for o, _ in outs]
    run = Path(results[0]['run'])
    if not any(r['aggregated'] for r in results):          # both finished together: either may aggregate later
        ns = h.base_namespace(tmp_path)
        with h.cpu_training_patches():
            assert ns['run_nested_diagnostics']('production', private_inference=True, fold_selection=[]) is not None
    parallel = np.load(run / 'predictions.npz'); mono = np.load(trained['run'] / 'predictions.npz')
    for key in ['oof_0', 'oof_1', 'private_0', 'private_1']:
        np.testing.assert_allclose(parallel[key], mono[key], atol=1e-6)
    assert {p.name for p in (run / 'folds').iterdir()} == {'fold0', 'fold1'} and not list((run / 'locks').iterdir())
    assert json.loads((run / 'inference_manifest.json').read_text())['members']


def test_fold_from_another_configuration_is_never_aggregated(trained, tmp_path):
    copy = tmp_path / trained['run'].name; shutil.copytree(trained['run'], copy)
    result = json.loads((copy / 'folds' / 'fold1' / 'result.json').read_text()); result['experiment_identity'] = 'other'
    (copy / 'folds' / 'fold1' / 'result.json').write_text(json.dumps(result))
    ns = h.base_namespace(tmp_path / 'ns')
    ns['RUN'] = copy; ns['STAGES'] = copy / 'stages'
    plan, fold_ids = ns['saved_diagnostic_splits'](), None
    saved, plan, fold_ids = ns['_nested_context']('production')
    try:
        with pytest.raises(ValueError, match='identity'):
            ns['aggregate_nested'](plan, fold_ids, private_inference=True)
    finally:
        ns['_restore_context'](saved)


def test_development_preset_is_labeled_and_never_exported_as_full_evaluation(tmp_path):
    ns = h.base_namespace(tmp_path)
    ns.update(EVALUATION_MODE='development', VIDEO_SEEDS=[42], VIDEO_EPOCHS=2)
    mode = ns['RUN_MODE']; ns['RUN_MODE'] = 'reports'; h.run_cell('smoke_holdout_dispatch', ns); ns['RUN_MODE'] = mode
    with h.cpu_training_patches():
        metrics = ns['run_development']()
    run = ns['RUN']
    assert metrics.evaluation_scope.iloc[0] == 'development_only_selection_biased'
    manifest = json.loads((run / 'development_manifest.json').read_text())
    assert manifest['evaluation_scope'].startswith('DEVELOPMENT ONLY')
    split_files = list((ns['OUTPUT'] / 'splits').glob('dev_*.json'))
    assert len(split_files) == 1
    tr, va = ns['development_split']()[1:]
    assert set(ns['train'].source_id.iloc[tr]).isdisjoint(ns['train'].source_id.iloc[va])
    assert ns['development_split']()[0] == json.loads(split_files[0].read_text())['split_id']   # fixed, reused
    summary = ns['write_pipeline_summary'](run, output_dir=tmp_path / 'rep')[0]
    assert summary['status'] == 'development' and summary['priorities'][0]['evidence_type'] == 'insufficient_evidence'
    pre = dict(preprocessing_id='x')
    with pytest.raises(ns['InferenceArtifactError'], match='neither a nested_cv run'):
        ns['export_inference_manifest'](run, preprocessing=pre)


def test_batch_runner_rejects_unknown_settings_and_modified_releases(tmp_path):
    sys.path.insert(0, str(h.REPO / 'hpc'))
    import run_penalty_kick
    with pytest.raises(SystemExit, match='Unknown settings'):
        run_penalty_kick.apply_overrides(dict(RUN_MODE='smoke'), dict(RUN_MDOE='train'))
    ns = dict(OUTPUT_ROOT=Path('/x'), RUN_MODE='smoke')
    run_penalty_kick.apply_overrides(ns, dict(OUTPUT_ROOT='/y', RUN_MODE='train'))
    assert ns == dict(OUTPUT_ROOT=Path('/y'), RUN_MODE='train')
    os.environ['SLURM_ARRAY_TASK_ID'] = '3'
    try:
        args = run_penalty_kick.parse_args(['--notebook', 'x', '--fold-from-array'])
        assert run_penalty_kick.load_overrides(args) == dict(FOLD_SELECTION=[3])
    finally:
        del os.environ['SLURM_ARRAY_TASK_ID']
    art = dict(__name__='__rel__', PLATFORM='hpc', SHARED_CODE_FINGERPRINT='fp')
    h.run_cell('run_artifacts', art, 'hpc')
    files = []
    for name in ['Penalty_Kick_HPC.ipynb', 'run_penalty_kick.py', 'penalty_kick.sbatch']:
        p = tmp_path / 'src' / name; p.parent.mkdir(exist_ok=True)
        shutil.copy(h.NOTEBOOKS['hpc'] if name.endswith('ipynb') else h.REPO / 'hpc' / name, p); files.append(p)
    release = art['create_release'](tmp_path / 'out', files, dict(EXPERIMENT_NAME='baseline', BACKBONE='videomae'))
    assert art['create_release'](tmp_path / 'out', files, dict(EXPERIMENT_NAME='baseline', BACKBONE='videomae')) == release
    info = run_penalty_kick.verify_release(release)
    assert set(info['files']) == {'Penalty_Kick_HPC.ipynb', 'run_penalty_kick.py', 'penalty_kick.sbatch', 'settings.json'}
    target = release / 'run_penalty_kick.py'; os.chmod(target, 0o644); target.write_text(target.read_text() + '\n# edit')
    with pytest.raises(SystemExit, match='modified'): run_penalty_kick.verify_release(release)
    other = art['create_release'](tmp_path / 'out', files, dict(EXPERIMENT_NAME='baseline', BACKBONE='vjepa2_vitl_256'))
    assert other != release                                                      # changed experiment -> new release


def test_hpc_and_colab_wrappers_produce_identical_inference(trained):
    script = textwrap.dedent(f'''
        import os, sys, json; os.environ['PENALTY_WRAPPER'] = 'hpc'
        sys.path.insert(0, {str(Path(__file__).parent)!r})
        import nb_harness as h, numpy as np
        from pathlib import Path
        ns = h.base_namespace({str(trained['root'])!r}, run_mode='inference')
        assert ns['PLATFORM'] == 'hpc' and 'google.colab' not in sys.modules
        run = Path({str(trained['run'])!r}); ns.update(TRAINED_RUN_DIR=run, INFERENCE_RAW=run.parent / 'raw_hpc')
        frame, routing = ns['resolve_inference_clips']('public_oof')
        with h.forbid_fitting(ns):
            r = ns['run_saved_inference'](run, frame, hooks=ns['INFERENCE_HOOKS'], routing=routing, device='cpu',
                                          dataset_id=ns['DATA_ID'], output_root=run.parent / 'hpc_out')
        print(json.dumps(r['record']['max_abs_difference_vs_saved_oof']))
    ''')
    out = subprocess.run([PY, '-c', script], capture_output=True, text=True)
    assert out.returncode == 0, out.stderr[-3000:]
    diff = json.loads(out.stdout.strip().splitlines()[-1])
    assert max(diff.values()) <= 1e-6
