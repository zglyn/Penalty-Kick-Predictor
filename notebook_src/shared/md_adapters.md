## Backbone adapters and inference helpers
**Backbones.** `BACKBONE` selects one adapter; each has explicit construction, preprocessing contract, feature dimension, `encode_video`, gradient checkpointing and serialization:

| `BACKBONE` | Loader | Frames x resolution | Feature | Pooling (design choice to test on inner validation) |
|---|---|---|---|---|
| `videomae` (baseline) | `MCG-NJU/videomae-base-finetuned-kinetics` via Transformers | 16 x 224 | 768 | token mean + checkpoint `fc_norm` (original) |
| `vjepa2_1_vitb_384` | official `facebookresearch/vjepa2` code pinned at commit `204698b4…` (`vjepa2_1_vit_base_384`, `pretrained=False`), encoder weights from the README checkpoint `vjepa2_1_vitb_dist_vitG_384.pt` (`ema_encoder`, strict key/shape load) | 16 x 384 | 768 | token mean + new LayerNorm |
| `vjepa2_vitl_256` | `facebook/vjepa2-vitl-fpc64-256` via Transformers `VJEPA2Model` (encoder only) | 16 x 256 | 1024 | token mean + new LayerNorm |

The V-JEPA world-model predictor and the action-conditioned model are not used; the two direction heads are trained on the competition targets. Frames keep the notebook's chronological pre-kick window (16 frames over the final 1.6 s, quadratic spacing); crops are letterboxed rather than center-cropped, which differs from the official evaluation transform. V-JEPA uses the official ImageNet mean/std. The official hub file at that commit points weights at `localhost` ("for testing"), so the adapter builds the hub architecture with `pretrained=False` and downloads the encoder from `dl.fbaipublicfiles.com` itself; V-JEPA 2 is never silently substituted if 2.1 fails. Crop caches are keyed by frames/resolution/backbone, so a new backbone regenerates crops from the shared detector cache (decoded detections and role geometry are reused). Comparing backbones on the same saved folds and seeds is meaningful only as a non-compute-matched comparison: model sizes and resolutions differ.

**Inference helpers** below build and validate `inference_manifest.json` (relative artifact paths, sha256 checksums, backbone/architecture, preprocessing, feature schema, class order, per-fold scalers/selectors/trees/priors, frozen ensemble weights, seed weights) and run label-free prediction with separate, content-keyed prediction caches.
