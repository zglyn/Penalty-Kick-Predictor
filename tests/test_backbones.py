"""Backbone adapters through the notebook's ConfigurableMultiViewVideo.

VideoMAE and V-JEPA 2 use the real Transformers 5.3.0 classes with tiny random configs (no downloads).
V-JEPA 2.1: executing the official repository code was not permitted in the authoring sandbox, so the
adapter is exercised against a STAND-IN module that reproduces the official interface read from the pinned
commit (src.hub.backbones.<entry>(pretrained=False) -> (encoder, predictor); encoder.forward(x[B,C,T,H,W],
masks=None, training=False) -> tokens; use_activation_checkpointing; ema_encoder checkpoint key with
'module.'/'backbone.' prefixes). Real-checkpoint validation of 2.1 remains a Colab GPU task.
"""
import sys, textwrap
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).parent))
from nb_harness import CLASSES, run_cell


def namespace(spec_name, policy, frames=2, size=4, feature_dim=8, **spec_overrides):
    from transformers import VideoMAEConfig, VideoMAEForVideoClassification, VJEPA2Config, VJEPA2Model
    ns = dict(__name__='__bb__', RUN_MODE='reports', Path=Path, CLASSES=CLASSES, N_FRAMES=frames, SIZE=size, SCRATCH_ROOT=Path('/tmp/penalty_test_scratch'))
    run_cell('backbone_adapters', ns); run_cell('diagnostics_helpers', ns)
    vm = VideoMAEConfig(image_size=size, patch_size=2, num_frames=frames, tubelet_size=2, hidden_size=feature_dim,
                        intermediate_size=16, num_hidden_layers=1, num_attention_heads=2, num_labels=3)
    jc = VJEPA2Config(crop_size=size, patch_size=2, frames_per_clip=frames, tubelet_size=2, hidden_size=feature_dim,
                      num_attention_heads=2, num_hidden_layers=2, pred_hidden_size=8, pred_num_hidden_layers=1,
                      pred_num_attention_heads=2)

    class TinyVM(VideoMAEForVideoClassification):
        @classmethod
        def from_pretrained(cls, name, **k): torch.manual_seed(0); return cls(vm)

    class TinyJ(VJEPA2Model):
        @classmethod
        def from_pretrained(cls, name, **k): torch.manual_seed(0); return cls(jc)
    spec = ns['backbone_spec'](spec_name); spec.update(frames=frames, size=size, feature_dim=feature_dim, **spec_overrides)
    ns.update(VideoMAEForVideoClassification=TinyVM, BACKBONE_SPEC=spec, TRAINING_POLICY=policy, _TinyJ=TinyJ)
    return ns


@pytest.fixture
def patched_vjepa2(monkeypatch):
    import transformers
    def install(ns): monkeypatch.setattr(transformers, 'VJEPA2Model', ns['_TinyJ'])
    return install


def build(ns, dim=0, architecture=None):
    ns['set_active_ablation']('D_all_visual' if dim == 0 else 'production')
    return ns['make_ablation_model_class']()(dim, architecture=architecture)


@pytest.mark.parametrize('name', ['videomae', 'vjepa2_vitl_256'])
@pytest.mark.parametrize('policy', ['finetune', 'frozen_probe'])
def test_shape_gradients_policy_and_save_load(name, policy, patched_vjepa2):
    ns = namespace(name, policy); patched_vjepa2(ns)
    torch.manual_seed(1)
    model = build(ns)
    groups = ns['apply_training_policy'](model, policy)
    video = torch.randn(2, 3, 2, 3, 4, 4); numeric = torch.zeros(2, 0)
    pooled = model.encode_stream(video[:, 0])
    assert pooled.shape == (2, 8) and torch.isfinite(pooled).all()
    model.train(); out = model(video, numeric)
    assert [o.shape for o in out] == [(2, 3), (2, 3)]
    loss = .7 * torch.nn.functional.cross_entropy(out[0], torch.tensor([0, 1])) + .3 * torch.nn.functional.cross_entropy(out[1], torch.tensor([2, 0]))
    loss.backward()
    enc_grads = [p.grad for p in model.backbone.parameters()]
    if policy == 'finetune':
        assert any(g is not None and g.abs().sum() > 0 for g in enc_grads)
        assert {g['name'] for g in groups} == {'backbone', 'norm_numeric_heads'}
        flags = [m for m in model.backbone.modules() if getattr(m, 'gradient_checkpointing', False)]
        assert flags, 'gradient checkpointing should be active through the backbone API'
    else:
        assert all(g is None for g in enc_grads) and not model.backbone.training
        assert [g['name'] for g in groups] == ['norm_numeric_heads']
    assert model.heads[0].weight.grad.abs().sum() > 0
    report = ns['trainable_parameter_report'](model)
    assert (report['backbone']['trainable'] == 0) == (policy == 'frozen_probe')
    # save/load equivalence via saved architecture: no pretrained construction path is used
    model.eval(); expected = [o.detach() for o in model(video, numeric)]
    state = {k: v.clone() for k, v in model.state_dict().items()}
    ns['VideoMAEForVideoClassification'].from_pretrained = classmethod(lambda *a, **k: (_ for _ in ()).throw(AssertionError('download')))
    ns['_TinyJ'].from_pretrained = classmethod(lambda *a, **k: (_ for _ in ()).throw(AssertionError('download')))
    clone = build(ns, architecture=model.architecture)
    clone.load_state_dict(state, strict=True); clone.eval()
    for a, b in zip(expected, clone(video, numeric)): torch.testing.assert_close(a, b, rtol=0, atol=0)
    serial = model.adapter.serialization(model.architecture)
    assert serial['identity']['name'] == name and serial['training_policy'] == policy


def test_videomae_state_dict_keys_unchanged_for_old_checkpoints():
    ns = namespace('videomae', 'finetune')
    keys = set(build(ns, dim=len(ns['numeric_feature_manifest'](ns['ABLATION_CONFIGS']['production']))).state_dict())
    assert {k.split('.')[0] for k in keys} == {'backbone', 'norm', 'numeric', 'heads'}
    assert any(k.startswith('backbone.embeddings.patch_embeddings') for k in keys) and 'norm.weight' in keys


def test_unsupported_configurations_fail_clearly():
    ns = namespace('videomae', 'finetune')
    with pytest.raises(ValueError, match='Unknown BACKBONE'): ns['backbone_spec']('vjepa3')
    with pytest.raises(ValueError, match='Unknown TRAINING_POLICY'): ns['backbone_identity'](ns['backbone_spec']('videomae'), 'lora')
    with pytest.raises(ValueError, match='pinned'): ns['backbone_spec']('vjepa2_1_vitb_384', revision='main')
    ns['N_FRAMES'] = 4
    with pytest.raises(ValueError, match='frames'): build(ns)


STANDIN = textwrap.dedent('''
    import torch
    class VisionTransformer(torch.nn.Module):
        """Stand-in mirroring app/vjepa_2_1/models/vision_transformer.py's interface (not the official code)."""
        def __init__(self, img_size=(384, 384), patch_size=16, num_frames=64, tubelet_size=2, embed_dim=8, depth=2, **kw):
            super().__init__()
            self.embed_dim = embed_dim; self.patch_size = patch_size; self.tubelet_size = tubelet_size
            self.use_activation_checkpointing = False
            self.patch_embed = torch.nn.Conv3d(3, embed_dim, (tubelet_size, patch_size, patch_size), (tubelet_size, patch_size, patch_size))
            self.blocks = torch.nn.ModuleList([torch.nn.Linear(embed_dim, embed_dim) for _ in range(depth)])
            self.norms_block = torch.nn.ModuleList([torch.nn.LayerNorm(embed_dim)])
            self.layout_seen = None
        def forward(self, x, masks=None, training=False):
            assert x.ndim == 5 and x.shape[1] == 3, 'official encoder expects B,C,T,H,W'
            self.layout_seen = tuple(x.shape)
            x = self.patch_embed(x).flatten(2).transpose(1, 2)
            for blk in self.blocks:
                x = torch.utils.checkpoint.checkpoint(blk, x, use_reentrant=False) if self.use_activation_checkpointing and x.requires_grad else blk(x)
            return self.norms_block[-1](x)
    def vjepa2_1_vit_base_384(*, pretrained=True, **kwargs):
        assert pretrained is False, 'adapter must not use the hub weight URL (localhost at the pinned commit)'
        return VisionTransformer(patch_size=2), torch.nn.Linear(1, 1)
''')


@pytest.fixture
def standin_repo(tmp_path, monkeypatch):
    (tmp_path / 'src' / 'hub').mkdir(parents=True)
    (tmp_path / 'src' / '__init__.py').write_text(''); (tmp_path / 'src' / 'hub' / '__init__.py').write_text('')
    (tmp_path / 'src' / 'hub' / 'backbones.py').write_text(STANDIN)
    (tmp_path / 'app').mkdir(); (tmp_path / 'app' / '__init__.py').write_text('')
    for m in [m for m in sys.modules if m == 'src' or m.startswith('src.') or m == 'app']: monkeypatch.delitem(sys.modules, m)
    yield tmp_path
    for m in [m for m in sys.modules if m == 'src' or m.startswith('src.') or m == 'app']: sys.modules.pop(m, None)
    if str(tmp_path.resolve()) in sys.path: sys.path.remove(str(tmp_path.resolve()))


def test_vjepa2_1_adapter_layout_checkpointing_and_strict_checkpoint_load(standin_repo, tmp_path):
    ns = namespace('vjepa2_1_vitb_384', 'finetune')
    ns['VJEPA2_REPO_DIR'] = standin_repo
    reference = ns['_import_vjepa2_hub'](standin_repo).vjepa2_1_vit_base_384(pretrained=False)[0]
    ckpt = tmp_path / 'vjepa2_1_vitb_dist_vitG_384.pt'
    torch.save({'ema_encoder': {'module.backbone.' + k: v for k, v in reference.state_dict().items()},
                'predictor': {}}, ckpt)
    ns['BackboneAdapter']._download_vjepa2_1_checkpoint = lambda self: ckpt
    model = build(ns)
    for k, v in reference.state_dict().items():
        torch.testing.assert_close(model.backbone.state_dict()[k], v)        # cleaned keys loaded strictly
    assert model.backbone.use_activation_checkpointing is True
    assert model.adapter.provenance['checkpoint_sha256'] and model.adapter.provenance['encoder_tensors'] == len(reference.state_dict())
    video = torch.randn(1, 3, 2, 3, 4, 4); model.train()
    out = model(video, torch.zeros(1, 0)); (out[0].sum() + out[1].sum()).backward()
    assert model.backbone.layout_seen == (1, 3, 2, 4, 4)                     # B,C,T,H,W after permutation
    assert model.backbone.patch_embed.weight.grad.abs().sum() > 0
    clone = build(ns, architecture=model.architecture)                     # no checkpoint read on this path
    clone.load_state_dict(model.state_dict(), strict=True)
    torch.save({'ema_encoder': {k: v for k, v in list(reference.state_dict().items())[1:]}}, ckpt)
    with pytest.raises(RuntimeError, match='Missing key'): build(ns)
    torch.save({'target_encoder': reference.state_dict()}, ckpt)
    with pytest.raises(KeyError, match='ema_encoder'): build(ns)
