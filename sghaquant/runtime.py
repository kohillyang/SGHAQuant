"""Native encoder and SAM2.1 decoder integration."""
import os
from pathlib import Path
import subprocess
import tempfile

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
MODELS = {
    'tiny': 'configs/sam2.1/sam2.1_hiera_t.yaml',
    'small': 'configs/sam2.1/sam2.1_hiera_s.yaml',
    'base_plus': 'configs/sam2.1/sam2.1_hiera_b+.yaml',
    'large': 'configs/sam2.1/sam2.1_hiera_l.yaml',
}
SEEDS = list(range(20261010, 20261015))
SHAPES = {'vf': (1, 256, 64, 64), 's4': (1, 32, 256, 256), 's8': (1, 64, 128, 128)}

def engine_command(model, precision, seed=20261010, weights=ROOT / 'weights', engine=ROOT / 'build/sam21'):
    folder = Path(weights).resolve() / model / (f'w4a4/{seed}' if precision == 'w4a4' else 'w8a8')
    required = ['model.bin', 'model.json', 'model.manifest', 'weights.bin']
    if precision == 'w4a4':
        required += ['factors.bin', 'fc2static.bin']
    for name in required:
        if not (folder / name).is_file():
            raise FileNotFoundError(f'{folder / name}; run download.py first')
    cmd = [str(Path(engine).resolve()), '--model', str(folder / 'model'), '--f16acc', '--no-tune',
           '--gptq', str(folder / 'weights.bin'), '--win8-pool', '--win8-glob']
    if precision == 'w4a4':
        cmd += ['--int4', '--dhd', str(folder / 'factors.bin'), '--fc2-static', str(folder / 'fc2static.bin')]
    else:
        cmd += ['--int8-fc2', '--rot']
    return cmd

def engine_environment():
    env = dict(os.environ)
    # Fix the LayerNorm implementation used by the paper's W8A8 row.
    for key in list(env):
        if key.startswith('NANO_'):
            del env[key]
    env['NANO_LN_L1'] = '1'
    return env

class Encoder:
    """Persistent native encoder; input is float32 NCHW, 1 x 3 x 1024 x 1024."""
    def __init__(self, model, precision, seed=20261010, weights=ROOT / 'weights', engine=ROOT / 'build/sam21'):
        self.work = tempfile.TemporaryDirectory(prefix='sghaquant-')
        self.directory = Path(self.work.name)
        if any(c.isspace() for c in str(self.directory)):
            self.work.cleanup()
            raise ValueError('Set TMPDIR to a path without spaces for the native encoder protocol')
        cmd = engine_command(model, precision, seed, weights, engine)
        self.proc = subprocess.Popen(cmd + ['--serve', '--warmup', '0', '--iters', '1'],
                                     stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                     stderr=subprocess.STDOUT, text=True, bufsize=1,
                                     env=engine_environment())
        try:
            self._wait('SERVE_READY')
        except BaseException:
            self.close()
            raise

    def _wait(self, marker):
        lines = []
        while True:
            line = self.proc.stdout.readline()
            if not line:
                raise RuntimeError('Native encoder stopped:\n' + ''.join(lines[-25:]))
            lines.append(line)
            if line.startswith(marker):
                return

    def __call__(self, image):
        if hasattr(image, 'detach'):
            image = image.detach().cpu().numpy()
        image = np.asarray(image, dtype=np.float32)
        if image.shape != (1, 3, 1024, 1024):
            raise ValueError(f'Expected (1, 3, 1024, 1024), received {image.shape}')
        image.tofile(self.directory / 'input.bin')
        self.proc.stdin.write(f'{self.directory / "input.bin"} {self.directory}\n')
        self.proc.stdin.flush()
        self._wait('SERVED')
        return {name: np.fromfile(self.directory / f'ours_{name}.bin', dtype=np.float32).reshape(shape)
                for name, shape in SHAPES.items()}

    def close(self):
        if self.proc.poll() is None:
            try:
                self.proc.stdin.write('quit\n')
                self.proc.stdin.flush()
                self.proc.wait(timeout=10)
            except (BrokenPipeError, subprocess.TimeoutExpired):
                self.proc.kill()
                self.proc.wait()
        self.proc.stdin.close()
        self.proc.stdout.close()
        self.work.cleanup()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()

def load_predictor(model, weights=ROOT / 'weights'):
    import torch
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    checkpoint = Path(weights) / model / 'reference.pt'
    sam = build_sam2(MODELS[model], str(checkpoint), device='cuda').eval()
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    return SAM2ImagePredictor(sam)

def native_forward(sam, encoder):
    import torch
    no_memory = sam.no_mem_embed.detach().float().view(1, -1, 1, 1)
    def forward(image):
        output = encoder(image)
        maps = [torch.from_numpy(output[k]).to(image.device) for k in ('s4', 's8', 'vf')]
        # The encoder export already adds no_mem_embed; SAM2ImagePredictor adds it once.
        maps[-1] = maps[-1] - no_memory
        positions = [sam.image_encoder.neck.position_encoding(x).to(x.dtype) for x in maps]
        return {'backbone_fpn': maps, 'vision_features': maps[-1], 'vision_pos_enc': positions}
    return forward
