# CELL: dataset_loading
# Opens the competition ZIP only when the mode needs it (training, private or public inference).
# Custom-video inference never touches it: no ZIP, labels, clip-count assertions or submission template.
EXPECTED_COUNTS = dict(public=1300, private=900)   # competition dataset audit (not applied to custom inference)

def extract_dataset(zip_path, scratch_root):
    signature = hashlib.sha256()
    with Path(zip_path).open('rb') as f:
        for chunk in iter(lambda: f.read(8*1024*1024), b''): signature.update(chunk)
    data_id = signature.hexdigest()[:16]
    root = Path(scratch_root) / 'penalty_data' / data_id
    if not (root / '.complete').exists():
        # Extract into a private partial folder, then rename: concurrent jobs never see half a dataset.
        partial = root.with_name(f'.{data_id}.{os.getpid()}.partial')
        with zipfile.ZipFile(zip_path) as z:
            for info in tqdm(z.infolist(), desc='Extract'):
                destination = (partial / info.filename).resolve()
                assert destination.is_relative_to(partial.resolve()), 'Unsafe archive path'
                z.extract(info, partial)
        (partial / '.complete').touch()
        try: os.replace(partial, root)
        except OSError:
            import shutil; shutil.rmtree(partial, ignore_errors=True)   # another job finished first
    return data_id, root

def find_one(names):
    matches = [p for p in ROOT.rglob('*.csv') if p.name in names]
    assert len(matches) == 1, matches
    return matches[0]

if NEEDS_DATASET:
    DATA_ID, ROOT = extract_dataset(DATASET_PATH, SCRATCH_ROOT)
    train = pd.read_csv(find_one({'train_ground_truth.csv','public_ground_truth.csv'}))
    test = pd.read_csv(find_one({'private_metadata.csv'}))
    template = pd.read_csv(find_one({'submission_template.csv'}))
    assert len(train)==EXPECTED_COUNTS['public'] and len(test)==EXPECTED_COUNTS['private']
    assert train['clip'].is_unique and test['clip'].is_unique and template['clip'].is_unique
    assert set(template['clip'])==set(test['clip'])
    assert not set(train['clip']) & set(test['clip'])
    video_paths = {}
    for p in ROOT.rglob('*.mp4'):
        assert p.name not in video_paths, f'Duplicate filename: {p.name}'
        video_paths[p.name] = p
    assert set(pd.concat([train['clip'],test['clip']])) <= video_paths.keys()
    Y = np.column_stack([train[t].map({c:i for i,c in enumerate(cs)}).to_numpy() for t,cs in zip(TARGETS,CLASSES)])
    assert not pd.isna(Y).any()
    Y=Y.astype(int)
    display(train[TARGETS].apply(pd.Series.value_counts))
    print('Public sources:', train.source_id.nunique(), 'Private sources:', test.source_id.nunique())
else:
    # Custom inference: no competition data. Empty tables keep later definitions valid.
    DATA_ID, ROOT, video_paths, template = None, None, {}, None
    _meta = ['clip','source_id','preferred_foot','gender']
    train = pd.DataFrame(columns=_meta+TARGETS); test = pd.DataFrame(columns=_meta)
    Y = np.zeros((0, len(TARGETS)), dtype=int)
    print('Custom-video inference: the competition ZIP is not opened.')
