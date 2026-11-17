"""Extract SAM or DINOv3 image features with the native CUDA engine."""
import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image

from sghaquant.encoders import ENCODERS, encode_array, preprocess
from sghaquant.runtime import ROOT

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--family', choices=ENCODERS, required=True)
    p.add_argument('--model', choices=['vit_b', 'vit_l', 'vit_h'])
    p.add_argument('--precision', choices=['w4a4', 'w8a8'], default='w4a4')
    inputs = p.add_mutually_exclusive_group(required=True)
    inputs.add_argument('--image', type=Path, nargs='+', help='one or more RGB images')
    inputs.add_argument('--input-npy', type=Path, help='preprocessed NCHW array')
    p.add_argument('--weights', type=Path, default=ROOT / 'weights')
    p.add_argument('--output', type=Path, default=ROOT / 'outputs/features.npz')
    p.add_argument('--eager', action='store_true', help='disable CUDA graphs')
    args = p.parse_args()
    model = args.model or ENCODERS[args.family][0]
    if model not in ENCODERS[args.family]:
        p.error(f'{args.family} models: {ENCODERS[args.family]}')
    metadata = {'family': args.family, 'model': model, 'precision': args.precision}
    if args.image:
        prepared = [preprocess(Image.open(path).convert('RGB'), args.family) for path in args.image]
        array = np.concatenate([item[0] for item in prepared], axis=0)
        metadata['images'] = [item[1] for item in prepared]
    else:
        array = np.load(args.input_npy, allow_pickle=False)
    features = encode_array(array, args.family, model, args.precision, args.weights, args.eager)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('wb') as stream:
        np.savez_compressed(stream, **features, metadata=json.dumps(metadata))
    print(args.output)
    for name, array in features.items():
        print(f'{name}: {list(array.shape)} {array.dtype}')

if __name__ == '__main__':
    main()
