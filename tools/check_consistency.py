#!/usr/bin/env python
"""Drift check between the two notebooks' common code (usable without the authoring sources).

    python tools/check_consistency.py notebooks/Penalty_Kick_Colab.ipynb notebooks/Penalty_Kick_HPC.ipynb

Checks: every shared cell exists in both notebooks with identical text, in the same relative order; both
notebooks embed the same SHARED_CODE_FINGERPRINT; the HPC notebook has no Colab import, Drive mount or Colab path.
"""
import json, re, sys
from pathlib import Path


def cells(path):
    nb = json.loads(Path(path).read_text())
    return [(c['metadata'].get('penalty_source'), ''.join(c['source']), c['cell_type']) for c in nb['cells']]


def check(colab_path, hpc_path):
    problems = []
    colab, hpc = cells(colab_path), cells(hpc_path)
    shared_c = [(s, t) for s, t, _ in colab if s and s.startswith('shared/')]
    shared_h = [(s, t) for s, t, _ in hpc if s and s.startswith('shared/')]
    if [s for s, _ in shared_c] != [s for s, _ in shared_h]:
        problems.append('Shared cell lists/order differ between notebooks')
    for (s1, t1), (s2, t2) in zip(shared_c, shared_h):
        if s1 == s2 and t1 != t2: problems.append(f'Shared cell {s1} differs between notebooks')
    prints = []
    for path, nb in [(colab_path, colab), (hpc_path, hpc)]:
        found = set(re.findall(r"SHARED_CODE_FINGERPRINT = '([0-9a-f]+)'", ''.join(t for _, t, _ in nb)))
        if len(found) != 1: problems.append(f'{path}: expected one SHARED_CODE_FINGERPRINT, found {found}')
        prints.append(found)
    if prints[0] != prints[1]: problems.append(f'Fingerprints differ: {prints}')
    hpc_code = '\n'.join(t for _, t, k in hpc if k == 'code')
    for pattern in [r'google\.colab', r'drive\.mount', r"['\"]/content"]:
        if re.search(pattern, hpc_code): problems.append(f'HPC notebook contains {pattern}')
    return problems


if __name__ == '__main__':
    root = Path(__file__).resolve().parents[1]
    args = sys.argv[1:] or [root / 'notebooks' / 'Penalty_Kick_Colab.ipynb', root / 'notebooks' / 'Penalty_Kick_HPC.ipynb']
    issues = check(*args)
    print('\n'.join(issues) if issues else 'Consistency check: PASS (shared cells identical, fingerprints match, HPC is Colab-free)')
    sys.exit(1 if issues else 0)
