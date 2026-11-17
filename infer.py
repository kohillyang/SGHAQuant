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

"""Segment an image with a point or box prompt."""
import argparse
from pathlib import Path

import numpy as np
from PIL import Image
import torch

from sghaquant.runtime import Encoder, MODELS, ROOT, SEEDS, load_predictor, native_forward

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model', choices=MODELS, default='tiny')
    p.add_argument('--precision', choices=['w4a4', 'w8a8'], default='w4a4')
    p.add_argument('--seed', type=int, choices=SEEDS, default=SEEDS[0])
    p.add_argument('--weights', type=Path, default=ROOT / 'weights')
    p.add_argument('--engine', type=Path, default=ROOT / 'build/sam21')
    p.add_argument('--image', type=Path, required=True)
    prompt = p.add_mutually_exclusive_group(required=True)
    prompt.add_argument('--point', nargs=2, type=float, metavar=('X', 'Y'))
    prompt.add_argument('--box', nargs=4, type=float, metavar=('X0', 'Y0', 'X1', 'Y1'))
    p.add_argument('--output', type=Path, default=ROOT / 'outputs/mask.png')
    args = p.parse_args()
    predictor = load_predictor(args.model, args.weights)
    image = np.array(Image.open(args.image).convert('RGB'))
    with Encoder(args.model, args.precision, args.seed, args.weights, args.engine) as encoder:
        predictor.model.forward_image = native_forward(predictor.model, encoder)
        with torch.no_grad():
            predictor.set_image(image)
            masks, scores, _ = predictor.predict(
                point_coords=np.asarray([args.point], dtype=np.float32) if args.point else None,
                point_labels=np.ones(1, dtype=np.int32) if args.point else None,
                box=np.asarray(args.box, dtype=np.float32) if args.box else None,
                multimask_output=True)
    best = int(np.argmax(scores))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray((masks[best] > 0).astype(np.uint8) * 255).save(args.output)
    print(f'{args.output}  score={scores[best]:.6f}')

if __name__ == '__main__':
    main()
