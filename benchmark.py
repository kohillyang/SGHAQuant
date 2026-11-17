# Copyright 2026 SGHAQuant contributors
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Measure encoder and neck latency with CUDA graphs."""
import argparse
import json
import os
from pathlib import Path
import re
import statistics
import subprocess
import tempfile
import time

from sghaquant.runtime import MODELS, ROOT, SEEDS, engine_command, engine_environment
from sghaquant.encoders import ENCODERS, command as encoder_command, preprocess

def cool_gpu(limit):
    device = os.environ.get('CUDA_VISIBLE_DEVICES', '0').split(',')[0]
    deadline = time.monotonic() + 300
    while True:
        result = subprocess.run(['nvidia-smi', '-i', device, '--query-gpu=temperature.gpu',
                                 '--format=csv,noheader,nounits'], check=True, capture_output=True, text=True)
        temperature = int(result.stdout.strip())
        if temperature <= limit:
            return
        if time.monotonic() >= deadline:
            raise RuntimeError(f'GPU remains above {limit} C; run the benchmark on an idle GPU')
        time.sleep(2)

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--family', choices=['sam21', *ENCODERS], default='sam21')
    p.add_argument('--model', choices=[*MODELS, 'vit_b', 'vit_l', 'vit_h', 'all'], default='all')
    p.add_argument('--precision', choices=['w4a4', 'w8a8', 'both'], default='both')
    p.add_argument('--seed', type=int, choices=SEEDS, default=SEEDS[0])
    p.add_argument('--batch', nargs='+', type=int, default=[1, 8])
    p.add_argument('--warmup', type=int, default=200)
    p.add_argument('--iters', type=int, default=200)
    p.add_argument('--rounds', type=int, default=6)
    inputs = p.add_mutually_exclusive_group()
    inputs.add_argument('--input', type=Path, help='native NCHW input tensor; omitted: zeros')
    inputs.add_argument('--image', type=Path, help='RGB image, repeated to fill the batch')
    p.add_argument('--max-start-temperature', type=int, default=50)
    p.add_argument('--weights', type=Path, default=ROOT / 'weights')
    p.add_argument('--engine', type=Path)
    p.add_argument('--output', type=Path, default=ROOT / 'outputs/benchmark.json')
    args = p.parse_args()
    if min(args.batch + [args.iters, args.rounds]) < 1 or args.warmup < 0:
        p.error('batch, iters and rounds must be positive; warmup must be nonnegative')
    available = MODELS if args.family == 'sam21' else ENCODERS[args.family]
    if args.model != 'all' and args.model not in available:
        p.error(f'{args.family} models: {list(available)}')
    models = list(available) if args.model == 'all' else [args.model]
    precisions = ['w8a8', 'w4a4'] if args.precision == 'both' else [args.precision]
    measurements = {}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temp = tempfile.TemporaryDirectory(prefix='sghaquant-bench-')
    prepared = None
    if args.image:
        from PIL import Image
        import numpy as np
        image = Image.open(args.image).convert('RGB')
        if args.family == 'sam21':
            from sam2.utils.transforms import SAM2Transforms
            tensor = SAM2Transforms(resolution=1024, mask_threshold=0.0)(image)
            args.input = Path(temp.name) / 'input.bin'
            tensor.numpy().astype(np.float32).tofile(args.input)
        else:
            prepared, _ = preprocess(image, args.family)
    for batch in args.batch:
        current_input = args.input
        if prepared is not None:
            current_input = Path(temp.name) / f'input_b{batch}.bin'
            np.repeat(prepared, batch, axis=0).tofile(current_input)
        cells = [(model, precision) for model in models for precision in precisions]
        for rep in range(args.rounds):
            for model, precision in cells if rep % 2 == 0 else reversed(cells):
                cool_gpu(args.max_start_temperature)
                if args.family == 'sam21':
                    cmd = engine_command(model, precision, args.seed, args.weights, args.engine or ROOT / 'build/sam21')
                else:
                    cmd = encoder_command(args.family, model, precision, args.weights, batch)
                    if args.engine:
                        cmd[0] = str(args.engine.resolve())
                cmd += ['--batch', str(batch), '--warmup', str(args.warmup), '--iters', str(args.iters)]
                if current_input:
                    cmd += ['--input', str(current_input.resolve())]
                env = engine_environment()
                if args.family != 'sam21':
                    env.pop('NANO_LN_L1', None)
                result = subprocess.run(cmd, capture_output=True, text=True, env=env)
                if result.returncode:
                    raise RuntimeError(result.stdout + result.stderr)
                match = re.search(r'latency_ms(?:/batch)? mean=[\d.]+ p50=([\d.]+)', result.stdout)
                if not match:
                    raise RuntimeError(result.stdout)
                key = f'{model}/{precision}/b{batch}'
                measurements.setdefault(key, []).append(float(match[1]))
                print(f'{key} round {rep + 1}: {match[1]} ms', flush=True)
    summary = {key: {'round_p50_ms': values, 'median_ms': statistics.median(values)}
               for key, values in measurements.items()}
    args.output.write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps(summary, indent=2))
    temp.cleanup()

if __name__ == '__main__':
    main()
