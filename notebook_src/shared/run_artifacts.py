# CELL: run_artifacts
"""Run directories, writer isolation, attempt records, artifact bundles and frozen releases.

Pure Python (no NumPy/torch), so reports mode can use it. Layout under OUTPUT_ROOT:
    runs/<platform>/<experiment name>__<experiment id>/   one writer platform per run directory
        attempts/<attempt id>/attempt.json               one record per execution (retries, Slurm requeues)
        locks/<scope>.lock                               run, fold<k>, aggregate (heartbeat-refreshed)
    splits/  (shared, immutable saved split assignments)  bundles/<bundle id>/  releases/<release id>/
Experiment identity hashes only computation (data, split, model, features, preprocessing, optimizer);
mount points, platform, job IDs and attempt IDs never change it.
"""
import datetime, hashlib, json, os, shutil, socket, threading, time
from pathlib import Path

ARTIFACT_SCHEMA_VERSION = 2
LOCK_STALE_SECONDS = 600       # a lock whose heartbeat is older than this belongs to a dead process
LOCK_HEARTBEAT_SECONDS = 60


def stable_hash(value, length=16):
    return hashlib.sha256(json.dumps(value, sort_keys=True, default=str).encode()).hexdigest()[:length]


def sha256_file(path, chunk=8 * 1024 * 1024):
    digest = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(chunk), b''): digest.update(block)
    return digest.hexdigest()


def atomic_write_text(path, text):
    """Unique temp name per process, fsync, then rename: concurrent writers never interleave bytes and an
    interrupted write leaves the previous complete file in place."""
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f'.{path.name}.{os.getpid()}.{threading.get_ident()}.tmp')
    with open(tmp, 'w', encoding='utf-8') as f:
        f.write(text); f.flush(); os.fsync(f.fileno())
    os.replace(tmp, path)


def make_attempt_id(platform):
    job = os.environ.get('SLURM_JOB_ID')
    task = os.environ.get('SLURM_ARRAY_TASK_ID')
    stamp = datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    return f'{stamp}_{platform}_' + (f'job{job}' + (f'_{task}' if task else '') if job else f'pid{os.getpid()}')


class RunLockedError(RuntimeError):
    pass


class RunWriterLock:
    """Exclusive writer for one scope of a run directory (e.g. 'run', 'fold3', 'aggregate').

    A second live writer fails immediately. A Slurm requeue of the same job (same SLURM_JOB_ID) or a lock
    whose heartbeat is older than LOCK_STALE_SECONDS is taken over, so preempted spot jobs can resume.
    """

    def __init__(self, run_dir, scope):
        self.path = Path(run_dir) / 'locks' / f'{scope}.lock'
        self.owner = dict(scope=scope, platform=globals().get('PLATFORM', 'unknown'),
                          attempt=globals().get('ATTEMPT_ID', 'unknown'), host=socket.gethostname(),
                          pid=os.getpid(), slurm_job_id=os.environ.get('SLURM_JOB_ID'))
        self._stop = threading.Event(); self._thread = None

    def _write(self):
        atomic_write_text(self.path, json.dumps(dict(self.owner, heartbeat=time.time()), indent=2))

    def _beat(self):
        while not self._stop.wait(LOCK_HEARTBEAT_SECONDS):
            try: self._write()
            except OSError: pass

    def __enter__(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        try:
            fd = os.open(self.path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.close(fd)
        except FileExistsError:
            try: holder = json.loads(self.path.read_text())
            except (OSError, ValueError): holder = {}
            same_job = holder.get('slurm_job_id') and holder.get('slurm_job_id') == self.owner['slurm_job_id']
            same_process = holder.get('pid') == os.getpid() and holder.get('host') == self.owner['host']
            stale = time.time() - float(holder.get('heartbeat', 0)) > LOCK_STALE_SECONDS
            if not (same_job or same_process or stale):
                raise RunLockedError(f'{self.path} is held by {holder.get("platform")} attempt {holder.get("attempt")} '
                                     f'on {holder.get("host")} (heartbeat {time.time() - float(holder.get("heartbeat", 0)):.0f}s ago). '
                                     'Two writers must never share a run directory; use another EXPERIMENT_NAME/OUTPUT_ROOT.')
            print(f'Taking over lock {self.path.name} ({"requeued job" if same_job else "stale heartbeat"}).')
        self._write()
        self._thread = threading.Thread(target=self._beat, daemon=True); self._thread.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        try:
            holder = json.loads(self.path.read_text())
            if holder.get('attempt') == self.owner['attempt'] and holder.get('pid') == os.getpid():
                self.path.unlink()
        except (OSError, ValueError):
            pass
        return False


def write_attempt_record(run_dir, settings, environment):
    """One immutable record per execution attempt; never overwritten by retries."""
    path = Path(run_dir) / 'attempts' / globals()['ATTEMPT_ID'] / 'attempt.json'
    record = dict(attempt=globals()['ATTEMPT_ID'], platform=globals().get('PLATFORM'), started=time.strftime('%Y-%m-%dT%H:%M:%S%z'),
                  host=socket.gethostname(), slurm=dict((k, v) for k, v in os.environ.items() if k.startswith('SLURM_')),
                  settings=settings, environment=environment, code_fingerprint=globals().get('SHARED_CODE_FINGERPRINT'))
    atomic_write_text(path, json.dumps(record, indent=2, default=str))
    return path


# ------------------------------------------------------------------ bundles (relocatable, checksummed)
BUNDLE_MARKER = 'BUNDLE_COMPLETE.json'
BUNDLE_REPORT_FILES = ['experiment_manifest.json', 'folds.csv', 'fold_results.json', 'predictions.npz',
                       'tracking_quality.csv', 'tracking_frames.csv', 'role_click_audit.csv', 'role_click_summary.csv',
                       'ablation_comparisons.csv', 'holdout_manifest.json', 'holdout_metrics.csv', 'loss_history.csv',
                       'dataset_splits.csv', 'development_manifest.json', 'development_metrics.csv',
                       'pipeline_summary.md', 'pipeline_summary.json', 'improvement_priorities.csv']


def bundle_members(run_dir, include_reports=True):
    """Relative paths of everything inference and report review need (never resumable optimizer state)."""
    run = Path(run_dir); members = set()
    manifest = run / 'inference_manifest.json'
    if manifest.exists():
        members.add('inference_manifest.json')
        info = json.loads(manifest.read_text())
        members.update(info['files'])
        for m in info['members']:
            for name in ('class_prior.json', 'backbone_serialization.json', 'provenance.json', 'component_provenance.json'):
                if (run / m['stage_dir'] / name).exists(): members.add(f"{m['stage_dir']}/{name}")
    if include_reports:
        members.update(f for f in BUNDLE_REPORT_FILES if (run / f).exists())
        for folder in ('component_reports', 'ablations'):
            members.update(str(p.relative_to(run)).replace('\\', '/') for p in (run / folder).rglob('*')
                           if p.is_file() and not p.name.endswith('.pt') and (folder != 'ablations' or p.suffix in ('.json', '.csv')))
        members.update(str(p.relative_to(run)).replace('\\', '/') for p in run.glob('stages/*/loss_history_seed*.csv'))
    if not members: raise FileNotFoundError(f'{run} has neither an inference manifest nor report files')
    return sorted(m for m in members if '_latest.pt' not in m)


def export_bundle(run_dir, destination, include_reports=True):
    """Copy a completed run into destination/<bundle id>/ with relative paths and checksums.
    Files are staged in a hidden partial directory; the completion marker is written last, then renamed."""
    run = Path(run_dir); members = bundle_members(run, include_reports)
    checksums = {m: sha256_file(run / m) for m in members}
    bundle_id = f'{run.name}__{stable_hash(checksums, 12)}'
    final = Path(destination) / bundle_id
    if (final / BUNDLE_MARKER).exists():
        print('Bundle already exported:', final); return final
    partial = Path(destination) / f'.{bundle_id}.partial'
    for m in members:
        target = partial / m; target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(run / m, target)
        if sha256_file(target) != checksums[m]: raise IOError(f'Copy verification failed for {m}')
    atomic_write_text(partial / BUNDLE_MARKER, json.dumps(dict(schema=ARTIFACT_SCHEMA_VERSION, bundle_id=bundle_id,
        source_run=run.name, source_platform=globals().get('PLATFORM'), created=time.strftime('%Y-%m-%dT%H:%M:%S%z'),
        files={m: dict(sha256=c, bytes=(run / m).stat().st_size) for m, c in checksums.items()}), indent=2))
    if final.exists(): shutil.rmtree(final)
    os.replace(partial, final)
    print(f'Exported {len(members)} files to {final}')
    return final


def verify_bundle(bundle_dir, verify_checksums=True):
    """A copied bundle is usable only with its completion marker and every listed file intact."""
    bundle = Path(bundle_dir); marker = bundle / BUNDLE_MARKER
    if not marker.exists():
        raise FileNotFoundError(f'{bundle} has no {BUNDLE_MARKER}: the transfer is incomplete or this is not a bundle.')
    info = json.loads(marker.read_text()); problems = []
    for rel, meta in info['files'].items():
        p = bundle / rel
        if not p.exists(): problems.append(f'missing: {p}')
        elif p.stat().st_size != meta['bytes']: problems.append(f'size changed: {p}')
        elif verify_checksums and sha256_file(p) != meta['sha256']: problems.append(f'checksum changed: {p}')
    if problems: raise IOError('Bundle verification failed:\n' + '\n'.join(problems))
    return info


def is_bundle(path):
    return (Path(path) / BUNDLE_MARKER).exists()


# ------------------------------------------------------------------ frozen releases
def create_release(output_root, files, settings, extra=None):
    """Freeze code + configuration: copy the listed files (notebook, runner, submission script) and the
    settings into releases/<release id>/, record checksums, make files read-only, then write RELEASE.json.
    Any later edit produces a different release id; the running release is never overwritten."""
    files = {Path(f).name: Path(f) for f in files}
    digests = {name: sha256_file(path) for name, path in files.items()}
    identity = dict(files=digests, settings=settings, code_fingerprint=globals().get('SHARED_CODE_FINGERPRINT'))
    release_id = f"{settings.get('EXPERIMENT_NAME', 'release')}__{stable_hash(identity, 12)}"
    final = Path(output_root) / 'releases' / release_id
    if (final / 'RELEASE.json').exists():
        print('Release already exists (unchanged):', final); return final
    partial = final.with_name(f'.{release_id}.partial')
    if partial.exists(): shutil.rmtree(partial)
    partial.mkdir(parents=True)
    for name, path in files.items(): shutil.copy2(path, partial / name)
    atomic_write_text(partial / 'settings.json', json.dumps(settings, indent=2, default=str))
    digests['settings.json'] = sha256_file(partial / 'settings.json')
    atomic_write_text(partial / 'RELEASE.json', json.dumps(dict(release_id=release_id, schema=ARTIFACT_SCHEMA_VERSION,
        created=time.strftime('%Y-%m-%dT%H:%M:%S%z'), files=digests, **(extra or {}),
        code_fingerprint=globals().get('SHARED_CODE_FINGERPRINT'),
        rule='Immutable: bug fixes or experiment changes create a new release.'), indent=2, default=str))
    for p in partial.iterdir(): os.chmod(p, 0o444)
    os.replace(partial, final)
    print('Release frozen:', final)
    return final


def verify_release(release_dir):
    info = json.loads((Path(release_dir) / 'RELEASE.json').read_text())
    bad = [n for n, d in info['files'].items() if sha256_file(Path(release_dir) / n) != d]
    if bad: raise IOError(f'Release {release_dir} was modified: {bad}')
    return info
