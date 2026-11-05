# CELL: heavy_imports
# Training/inference dependencies (never reached in reports mode). Catches API conflicts before any
# decoding or large download, and checks GPU visibility before a model is loaded.
import os, json, hashlib, zipfile, random
from pathlib import Path
import numpy as np
import scipy, pandas as pd, sklearn, torch, torchvision, av, cv2
from PIL import Image
from tqdm.auto import tqdm
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
from catboost import CatBoostClassifier
from ultralytics import YOLO
from transformers import VideoMAEConfig, VideoMAEImageProcessor, VideoMAEForVideoClassification
from packaging.version import Version
import joblib
if Version(torch.__version__.split('+')[0]) < Version('2.6'):
    raise RuntimeError('VideoMAE checkpoint loading requires PyTorch >= 2.6.')
# Exercise the actual fine-tuning API with a tiny randomly initialized CPU model.
_tiny=VideoMAEForVideoClassification(VideoMAEConfig(image_size=32,patch_size=16,num_frames=2,
    tubelet_size=2,hidden_size=32,intermediate_size=64,num_hidden_layers=1,num_attention_heads=4,num_labels=3))
_out=_tiny(pixel_values=torch.zeros(1,2,3,32,32)).logits
assert _out.shape==(1,3) and hasattr(_tiny,'videomae') and hasattr(_tiny,'fc_norm')
_out.sum().backward()
assert any(p.grad is not None for p in _tiny.videomae.parameters())
del _tiny,_out
print('Dependency imports and VideoMAE forward/backward preflight: PASS')
if BACKBONE == 'vjepa2_vitl_256':
    from transformers import VJEPA2Config, VJEPA2Model
    _tiny=VJEPA2Model(VJEPA2Config(crop_size=32,frames_per_clip=4,hidden_size=32,num_attention_heads=4,
        num_hidden_layers=1,pred_hidden_size=32,pred_num_hidden_layers=1,pred_num_attention_heads=4))
    _tiny.gradient_checkpointing_enable(); _tiny.train()
    _out=_tiny.encoder(pixel_values_videos=torch.zeros(1,4,3,32,32)).last_hidden_state
    assert _out.shape==(1,2*2*2,32); _out.mean().backward()
    assert any(p.grad is not None for p in _tiny.encoder.parameters()); del _tiny,_out
    print('V-JEPA 2 (Transformers) encoder forward/backward preflight: PASS')
elif BACKBONE == 'vjepa2_1_vitb_384':
    import timm, einops  # official V-JEPA 2.1 repository dependencies
    print('V-JEPA 2.1 dependencies importable; exact encoder keys/shapes are validated (strict) when weights load.')
SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
ENVIRONMENT_INFO.update(torch=torch.__version__, cuda=torch.version.cuda, device=DEVICE,
                        gpus=[torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())])
print('Device:', DEVICE, ENVIRONMENT_INFO['gpus'])
if RUN_MODE in {'smoke', 'train', 'resume'} and DEVICE != 'cuda':
    raise RuntimeError('No GPU is visible to PyTorch. Colab: select a GPU runtime. HPC: request a GPU partition '
                       '(and the Singularity --nv flag).')
Path(OUTPUT_ROOT).mkdir(parents=True, exist_ok=True)
if NEEDS_DATASET and not Path(DATASET_PATH).is_file():
    raise FileNotFoundError(f'Dataset ZIP not found: {DATASET_PATH}')
