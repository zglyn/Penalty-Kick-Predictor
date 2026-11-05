# CELL: backbone_adapters
"""Backbone adapters: VideoMAE (baseline), V-JEPA 2.1 (official repo) and V-JEPA 2 (Transformers).

Every adapter maps a normalized [batch, frames, channels, height, width] tensor to [batch, feature_dim].
The multi-view model keeps its attribute names `backbone` (encoder) and `norm` (pooled-feature norm), so
existing VideoMAE checkpoints keep their exact state-dict keys. Nothing here downloads weights at import.
"""
import copy, hashlib, json, sys, zipfile
from pathlib import Path

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
VJEPA2_REPO = 'facebookresearch/vjepa2'
# main @ 2026-03-23 ("Fix figure (#143)"), checked 2026-11-05. Its src/hub/backbones.py points
# VJEPA_BASE_URL at http://localhost:8300 "for testing", so the hub entry is built with
# pretrained=False and the encoder is loaded here from the README's official checkpoint URL.
VJEPA2_REVISION = '204698b45b3712590f06245fbfba32d3be539812'

BACKBONES = {
    'videomae': dict(
        family='videomae', loader='transformers.VideoMAEForVideoClassification',
        checkpoint='MCG-NJU/videomae-base-finetuned-kinetics', revision=None,
        frames=16, size=224, tubelet=2, feature_dim=768, input_layout='B,T,C,H,W',
        normalization='checkpoint image processor mean/std',
        pooling='mean over encoder tokens, then the checkpoint fc_norm (original notebook behavior)',
        gradient_checkpointing='VideoMAEModel.gradient_checkpointing_enable()',
        trainable_policies=('finetune', 'frozen_probe')),
    'vjepa2_1_vitb_384': dict(
        family='vjepa2_1', loader='official repo src.hub.backbones.vjepa2_1_vit_base_384(pretrained=False)',
        repo=VJEPA2_REPO, revision=VJEPA2_REVISION, entry_point='vjepa2_1_vit_base_384',
        checkpoint='https://dl.fbaipublicfiles.com/vjepa2/vjepa2_1_vitb_dist_vitG_384.pt',
        checkpoint_key='ema_encoder', frames=16, size=384, tubelet=2, feature_dim=768,
        input_layout='encoder expects B,C,T,H,W; adapter permutes from B,T,C,H,W',
        normalization='ImageNet mean/std (official evals make_transforms)',
        pooling='encoder forward(training=False) returns final-norm tokens; mean over tokens, then a new LayerNorm',
        gradient_checkpointing='VisionTransformer.use_activation_checkpointing attribute (official forward)',
        trainable_policies=('finetune', 'frozen_probe')),
    'vjepa2_vitl_256': dict(
        family='vjepa2', loader='transformers.VJEPA2Model (encoder only; predictor discarded)',
        checkpoint='facebook/vjepa2-vitl-fpc64-256', revision=None,
        frames=16, size=256, tubelet=2, feature_dim=1024, input_layout='B,T,C,H,W',
        normalization='ImageNet mean/std (VJEPA2VideoProcessor defaults)',
        pooling='encoder last_hidden_state (final LayerNorm applied); mean over tokens, then a new LayerNorm',
        gradient_checkpointing='VJEPA2Model.gradient_checkpointing_enable() before taking .encoder',
        trainable_policies=('finetune', 'frozen_probe')),
}
TRAINING_POLICIES = {
    'finetune': 'All encoder, norm, numeric-MLP and head parameters are trainable (original notebook policy).',
    'frozen_probe': 'Encoder frozen (eval mode, no gradients); pooled-feature norm, numeric MLP and heads trainable. '
                    'Plumbing/compute probe, not a substitute for fine-tuning.',
}


def backbone_spec(name, revision=None):
    if name not in BACKBONES:
        raise ValueError(f'Unknown BACKBONE {name!r}; choose one of {sorted(BACKBONES)}')
    spec = copy.deepcopy(BACKBONES[name]); spec['name'] = name
    if revision is not None:
        if spec['family'] == 'vjepa2_1' and revision != spec['revision']:
            raise ValueError('V-JEPA 2.1 code revision is pinned; edit VJEPA2_REVISION deliberately after re-validating.')
        spec['revision'] = revision
    spec['frames'] = int(spec['frames']); spec['size'] = int(spec['size'])
    if spec['frames'] % spec['tubelet']:
        raise ValueError('Frame count must be divisible by the tubelet size')
    return spec


def backbone_identity(spec, policy):
    """Short serializable identity used in configs, checkpoints and manifests."""
    if policy not in TRAINING_POLICIES: raise ValueError(f'Unknown TRAINING_POLICY {policy!r}')
    if policy not in spec['trainable_policies']: raise ValueError(f'{policy} is not supported for {spec["name"]}')
    keys = ['name', 'family', 'checkpoint', 'revision', 'frames', 'size', 'feature_dim', 'pooling']
    return dict({k: spec.get(k) for k in keys}, training_policy=policy)


def _clean_vjepa_keys(state):
    # Same cleaning as the official src/hub/backbones.py::_clean_backbone_key.
    return {k.replace('module.', '').replace('backbone.', ''): v for k, v in state.items()}


def _sha256_file(path, chunk=8 * 1024 * 1024):
    digest = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(chunk), b''): digest.update(block)
    return digest.hexdigest()


def vjepa2_repo_dir(revision, repo_dir=None):
    """Download (once) and return the official repo at a pinned commit. repo_dir overrides for tests."""
    import torch, urllib.request
    if repo_dir is not None:
        return Path(repo_dir)
    target = Path(torch.hub.get_dir()) / f'facebookresearch_vjepa2_{revision}'
    if not (target / 'src' / 'hub' / 'backbones.py').exists():
        target.parent.mkdir(parents=True, exist_ok=True)
        archive = target.with_suffix('.zip')
        urllib.request.urlretrieve(f'https://github.com/{VJEPA2_REPO}/archive/{revision}.zip', archive)
        with zipfile.ZipFile(archive) as z:
            root = z.namelist()[0].split('/')[0]
            z.extractall(target.parent)
        (target.parent / root).rename(target); archive.unlink()
    return target


def _import_vjepa2_hub(repo):
    repo = str(Path(repo).resolve())
    for package in ('src', 'app'):
        loaded = sys.modules.get(package)
        origin = getattr(loaded, '__path__', None)
        if loaded is not None and not any(str(p).startswith(repo) for p in (origin or [])):
            raise RuntimeError(f'A different top-level package {package!r} is already imported; restart the '
                               'session before loading the official V-JEPA repository code.')
    if repo not in sys.path: sys.path.insert(0, repo)
    import importlib
    return importlib.import_module('src.hub.backbones')


class BackboneAdapter:
    """Construction, preprocessing contract, encode_video, checkpointing and serialization for one backbone."""

    def __init__(self, spec, policy='finetune', repo_dir=None):
        self.spec = spec; self.policy = policy; self.repo_dir = repo_dir
        backbone_identity(spec, policy)
        self.feature_dim = int(spec['feature_dim'])
        self.frames = spec['frames']; self.size = spec['size']
        self.provenance = {}

    # ---------- construction ----------
    def build(self, architecture=None):
        """Return (encoder, norm, architecture). architecture=None loads pretrained weights; a saved
        architecture dict constructs the same modules without downloading pretrained weights."""
        import torch
        family = self.spec['family']
        if family == 'videomae':
            vm = globals()['VideoMAEForVideoClassification']
            if architecture is None:
                pretrained = vm.from_pretrained(self.spec['checkpoint'], **({'revision': self.spec['revision']}
                                                                            if self.spec.get('revision') else {}))
                config = pretrained.config
                self.provenance['resolved_revision'] = getattr(config, '_commit_hash', None)
            else:
                from transformers import VideoMAEConfig
                pretrained = vm(VideoMAEConfig.from_dict(architecture['config']))
                config = pretrained.config
            if config.num_frames != self.frames or config.image_size != self.size:
                raise ValueError(f'VideoMAE checkpoint expects {config.num_frames} frames at {config.image_size}px; '
                                 f'notebook uses {self.frames} at {self.size}px')
            encoder, norm = pretrained.videomae, copy.deepcopy(pretrained.fc_norm)
            hidden = config.hidden_size
            arch = dict(kind='videomae', config=_jsonable(config.to_dict()))
        elif family == 'vjepa2':
            from transformers import VJEPA2Config, VJEPA2Model
            if architecture is None:
                kwargs = {'revision': self.spec['revision']} if self.spec.get('revision') else {}
                full = VJEPA2Model.from_pretrained(self.spec['checkpoint'], **kwargs)
                self.provenance['resolved_revision'] = getattr(full.config, '_commit_hash', None)
            else:
                full = VJEPA2Model(VJEPA2Config.from_dict(architecture['config']))
            config = full.config
            if config.crop_size != self.size or config.tubelet_size != self.spec['tubelet']:
                raise ValueError(f'V-JEPA 2 checkpoint uses crop {config.crop_size}, tubelet {config.tubelet_size}')
            if self.policy == 'finetune': full.gradient_checkpointing_enable()
            encoder = full.encoder; del full.predictor
            hidden = config.hidden_size; norm = torch.nn.LayerNorm(hidden)
            arch = dict(kind='vjepa2', config=_jsonable(config.to_dict()))
        elif family == 'vjepa2_1':
            hub = _import_vjepa2_hub(vjepa2_repo_dir(self.spec['revision'], self.repo_dir))
            overrides = dict(architecture.get('entry_kwargs', {})) if architecture else {}
            encoder, predictor = getattr(hub, self.spec['entry_point'])(pretrained=False, **overrides)
            del predictor  # world-model predictor is not used for direction classification
            if architecture is None:
                path = self._download_vjepa2_1_checkpoint()
                try:
                    state = torch.load(path, map_location='cpu', weights_only=True)
                except Exception as error:  # official file may hold non-tensor training metadata
                    print(f'NOTE: weights_only load failed ({type(error).__name__}); loading the official Meta '
                          'checkpoint the way the official hub code does (weights_only=False).')
                    state = torch.load(path, map_location='cpu', weights_only=False)
                if self.spec['checkpoint_key'] not in state:
                    raise KeyError(f'{self.spec["checkpoint_key"]} missing from {path}; keys={sorted(state)}')
                weights = _clean_vjepa_keys(state[self.spec['checkpoint_key']])
                result = encoder.load_state_dict(weights, strict=True)  # exact key/shape validation
                self.provenance.update(checkpoint_sha256=_sha256_file(path), checkpoint_file=str(path),
                                       load_result=str(result), encoder_tensors=len(weights))
                del state, weights
            hidden = int(encoder.embed_dim); norm = torch.nn.LayerNorm(hidden)
            arch = dict(kind='vjepa2_1', entry_point=self.spec['entry_point'], revision=self.spec['revision'],
                        entry_kwargs=overrides, embed_dim=hidden, depth=len(encoder.blocks))
        else:
            raise ValueError(f'Unsupported backbone family {family}')
        if hidden != self.feature_dim:
            raise ValueError(f'{self.spec["name"]} hidden size {hidden} != declared feature_dim {self.feature_dim}')
        if self.policy == 'finetune': self.enable_gradient_checkpointing(encoder)
        return encoder, norm, arch

    def _download_vjepa2_1_checkpoint(self):
        import torch
        folder = Path(torch.hub.get_dir()) / 'checkpoints'; folder.mkdir(parents=True, exist_ok=True)
        path = folder / Path(self.spec['checkpoint']).name
        if not path.exists():
            torch.hub.download_url_to_file(self.spec['checkpoint'], str(path), progress=True)
        return path

    def enable_gradient_checkpointing(self, encoder):
        family = self.spec['family']
        if family == 'videomae':
            encoder.gradient_checkpointing_enable()
        elif family == 'vjepa2':
            # transformers 5.x GradientCheckpointingLayer flag, set via the parent model before extraction.
            layers = [m for m in encoder.modules() if hasattr(m, 'gradient_checkpointing')]
            if not any(getattr(m, 'gradient_checkpointing') for m in layers):
                raise RuntimeError('V-JEPA 2 gradient checkpointing was not enabled by the Transformers API')
        elif family == 'vjepa2_1':
            if not hasattr(encoder, 'use_activation_checkpointing'):
                raise RuntimeError('Official V-JEPA 2.1 encoder lacks use_activation_checkpointing')
            encoder.use_activation_checkpointing = True

    # ---------- forward ----------
    def encode_video(self, encoder, norm, video):
        """video: normalized float tensor [B, T, C, H, W] in chronological order -> [B, feature_dim]."""
        return norm(self.pool_tokens(encoder, video))

    def pool_tokens(self, encoder, video):
        """Encoder tokens mean-pooled to [B, feature_dim], before the pooled-feature norm."""
        if video.ndim != 5 or video.shape[1] != self.frames or video.shape[2] != 3:
            raise ValueError(f'Expected [B,{self.frames},3,H,W], got {tuple(video.shape)}')
        family = self.spec['family']
        if family == 'videomae':
            tokens = encoder(pixel_values=video).last_hidden_state
        elif family == 'vjepa2':
            tokens = encoder(pixel_values_videos=video).last_hidden_state
        else:
            tokens = encoder(video.permute(0, 2, 1, 3, 4).contiguous(), training=False)
        if tokens.ndim != 3 or tokens.shape[-1] != self.feature_dim:
            raise ValueError(f'Unexpected encoder token shape {tuple(tokens.shape)}')
        return tokens.mean(1)

    # ---------- serialization ----------
    def serialization(self, architecture):
        return dict(identity=backbone_identity(self.spec, self.policy), spec=_jsonable(self.spec),
                    architecture=architecture, provenance=_jsonable(self.provenance),
                    training_policy=self.policy, policy_description=TRAINING_POLICIES[self.policy])


def _jsonable(value):
    return json.loads(json.dumps(value, default=lambda x: x.tolist() if hasattr(x, 'tolist') else str(x)))


def apply_training_policy(model, policy):
    """Freeze/unfreeze according to policy; returns optimizer parameter groups with explicit learning rates."""
    import torch
    head = list(model.norm.parameters()) + list(model.numeric.parameters()) + list(model.heads.parameters())
    if policy == 'finetune':
        for p in model.parameters(): p.requires_grad_(True)
        return [dict(params=list(model.backbone.parameters()), lr=1e-5, name='backbone'),
                dict(params=head, lr=1e-4, name='norm_numeric_heads')]
    if policy == 'frozen_probe':
        for p in model.backbone.parameters(): p.requires_grad_(False)
        for p in head: p.requires_grad_(True)
        return [dict(params=head, lr=1e-3, name='norm_numeric_heads')]
    raise ValueError(f'Unknown training policy {policy}')


def trainable_parameter_report(model):
    groups = {'backbone': model.backbone, 'norm': model.norm, 'numeric': model.numeric, 'heads': model.heads}
    return {k: dict(trainable=int(sum(p.numel() for p in m.parameters() if p.requires_grad)),
                    total=int(sum(p.numel() for p in m.parameters()))) for k, m in groups.items()}
