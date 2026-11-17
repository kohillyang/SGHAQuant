"""SAM and DINOv3 native image encoders."""
from pathlib import Path
import subprocess
import tempfile

import numpy as np

from .runtime import ROOT, engine_environment

ENCODERS = {'sam1': ('vit_b', 'vit_l', 'vit_h'), 'dinov3': ('vit_l',)}

def command(family, model, precision, weights=ROOT / 'weights', batch=1):
    if family not in ENCODERS or model not in ENCODERS[family]:
        raise ValueError(f'Unknown encoder: {family}/{model}')
    if precision not in ('w4a4', 'w8a8'):
        raise ValueError('precision must be w4a4 or w8a8')
    root = Path(weights).resolve() / family / model
    for name in ('model.bin', 'model.json', 'model.manifest'):
        if not (root / 'common' / name).is_file():
            raise FileNotFoundError(f'{root / "common" / name}; run download.py first')
    for name in ('weights.bin', 'fc2static.bin'):
        if not (root / precision / name).is_file():
            raise FileNotFoundError(f'{root / precision / name}; run download.py first')
    binary = 'dinov3' if family == 'dinov3' else ('sam1_dh80' if model == 'vit_h' else 'sam1_dh64')
    executable = ROOT / 'build' / binary
    if not executable.is_file():
        raise FileNotFoundError(f'{executable}; run python build.py --family {family}')
    args = [str(executable), '--model', str(root / 'common/model'), '--batch', str(batch),
            '--f16acc', '--no-tune', '--int4' if precision == 'w4a4' else '--int8-fc2',
            '--gptq', str(root / precision / 'weights.bin'),
            '--fc2-static', str(root / precision / 'fc2static.bin')]
    if family == 'dinov3':
        args += ['--fc2-fp16-b0', '--attn-acc', '1']
    return args

def preprocess(image, family):
    """Return the engine input and resize metadata for one RGB PIL image."""
    import torch
    from torchvision.transforms import InterpolationMode, v2
    from torchvision.transforms.functional import resize
    width, height = image.size
    if family == 'sam1':
        scale = 1024 / max(height, width)
        h, w = int(height * scale + 0.5), int(width * scale + 0.5)
        resized = np.array(resize(image, [h, w], interpolation=InterpolationMode.BILINEAR))
        x = torch.from_numpy(resized).permute(2, 0, 1).float()
        mean = torch.tensor([123.675, 116.28, 103.53]).view(3, 1, 1)
        std = torch.tensor([58.395, 57.12, 57.375]).view(3, 1, 1)
        x = torch.nn.functional.pad((x - mean) / std, (0, 1024 - w, 0, 1024 - h))
        return x.numpy()[None], {'original_size': [height, width], 'resized_size': [h, w]}
    if family == 'dinov3':
        transform = v2.Compose([v2.ToImage(), v2.Resize(256, interpolation=InterpolationMode.BICUBIC,
                                                      antialias=True), v2.CenterCrop(224)])
        x = transform(image).as_subclass(torch.Tensor).numpy()
        return x[None], {'original_size': [height, width], 'crop_size': [224, 224]}
    raise ValueError(f'Unknown family: {family}')

def encode_array(inputs, family, model, precision, weights=ROOT / 'weights', eager=False):
    """Encode preprocessed NCHW arrays: SAM float32/1024, DINOv3 uint8/224."""
    inputs = np.asarray(inputs)
    side = 1024 if family == 'sam1' else 224
    dtype = np.dtype(np.float32 if family == 'sam1' else np.uint8)
    if inputs.ndim != 4 or inputs.shape[1:] != (3, side, side) or inputs.dtype != dtype:
        raise ValueError(f'Expected NCHW [B,3,{side},{side}] {dtype}, got {inputs.shape} {inputs.dtype}')
    if inputs.shape[0] < 1:
        raise ValueError('Batch must contain at least one image')
    args = command(family, model, precision, weights, inputs.shape[0])
    with tempfile.TemporaryDirectory(prefix='sghaquant-encoder-') as temp:
        temp = Path(temp)
        inputs.tofile(temp / 'input.bin')
        args += ['--input', str(temp / 'input.bin'), '--out-dir', str(temp), '--warmup', '0', '--iters', '1']
        if eager:
            args += ['--no-graph']
        env = engine_environment()
        env.pop('NANO_LN_L1', None)
        result = subprocess.run(args, capture_output=True, text=True, env=env)
        if result.returncode:
            raise RuntimeError(result.stdout + result.stderr)
        if family == 'sam1':
            features = np.fromfile(temp / 'out.emb.bin', dtype=np.float32).reshape(inputs.shape[0], 256, 64, 64)
            output = {'embedding': features}
        else:
            tokens = np.fromfile(temp / 'tokens.bin', dtype=np.float16).reshape(inputs.shape[0], 201, 1024)
            output = {'tokens': tokens, 'cls': tokens[:, 0], 'registers': tokens[:, 1:5], 'patches': tokens[:, 5:]}
        if not all(np.isfinite(value).all() for value in output.values()):
            raise RuntimeError('Encoder produced non-finite features')
        return output
