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
    p.add_argument('--model', choices=[*MODELS, 'all'], default='all')
    p.add_argument('--precision', choices=['w4a4', 'w8a8', 'both'], default='both')
    p.add_argument('--seed', type=int, choices=SEEDS, default=SEEDS[0])
    p.add_argument('--batch', nargs='+', type=int, default=[1, 8])
    p.add_argument('--warmup', type=int, default=200)
    p.add_argument('--iters', type=int, default=200)
    p.add_argument('--rounds', type=int, default=6)
    inputs = p.add_mutually_exclusive_group()
    inputs.add_argument('--input', type=Path, help='float32 NCHW 1024x1024 tensor; omitted: zeros')
    inputs.add_argument('--image', type=Path, help='RGB image, preprocessed with SAM2Transforms')
    p.add_argument('--max-start-temperature', type=int, default=50)
    p.add_argument('--weights', type=Path, default=ROOT / 'weights')
    p.add_argument('--engine', type=Path, default=ROOT / 'build/sam21')
    p.add_argument('--output', type=Path, default=ROOT / 'outputs/benchmark.json')
    args = p.parse_args()
    if min(args.batch + [args.iters, args.rounds]) < 1 or args.warmup < 0:
        p.error('batch, iters and rounds must be positive; warmup must be nonnegative')
    models = list(MODELS) if args.model == 'all' else [args.model]
    precisions = ['w8a8', 'w4a4'] if args.precision == 'both' else [args.precision]
    measurements = {}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temp = tempfile.TemporaryDirectory(prefix='sghaquant-bench-')
    if args.image:
        from PIL import Image
        import numpy as np
        from sam2.utils.transforms import SAM2Transforms
        image = Image.open(args.image).convert('RGB')
        tensor = SAM2Transforms(resolution=1024, mask_threshold=0.0)(image)
        args.input = Path(temp.name) / 'input.bin'
        tensor.numpy().astype(np.float32).tofile(args.input)
    for batch in args.batch:
        cells = [(model, precision) for model in models for precision in precisions]
        for rep in range(args.rounds):
            for model, precision in cells if rep % 2 == 0 else reversed(cells):
                cool_gpu(args.max_start_temperature)
                cmd = engine_command(model, precision, args.seed, args.weights, args.engine)
                cmd += ['--batch', str(batch), '--warmup', str(args.warmup), '--iters', str(args.iters)]
                if args.input:
                    cmd += ['--input', str(args.input.resolve())]
                result = subprocess.run(cmd, capture_output=True, text=True, env=engine_environment())
                if result.returncode:
                    raise RuntimeError(result.stdout + result.stderr)
                match = re.search(r'latency_ms mean=[\d.]+ p50=([\d.]+)', result.stdout)
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
