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

"""COCO val2017 point-prompt evaluation (Table 4)."""
import argparse
from contextlib import ExitStack
import csv
import json
from pathlib import Path
import time

import numpy as np
from PIL import Image
from pycocotools.coco import COCO
import torch

from sghaquant.runtime import Encoder, MODELS, ROOT, SEEDS, load_predictor, native_forward

def evaluate(args, model_name):
    coco = COCO(str(args.coco / 'annotations/instances_val2017.json'))
    image_ids = sorted(coco.getImgIds())[:args.images]
    predictor = load_predictor(model_name, args.weights)
    sam = predictor.model
    fp32_forward = sam.forward_image
    rows = []
    start = time.monotonic()
    seeds = SEEDS if args.seeds == 'all' else [int(x) for x in args.seeds.split(',')]
    if any(seed not in SEEDS for seed in seeds):
        raise ValueError(f'Available seeds: {SEEDS}')
    with ExitStack() as stack:
        forwards = {'fp32': fp32_forward}
        if args.precision in ('both', 'w4a4'):
            for seed in seeds:
                encoder = stack.enter_context(Encoder(model_name, 'w4a4', seed, args.weights, args.engine))
                forwards[f'w4a4_{seed}'] = native_forward(sam, encoder)
        if args.precision in ('both', 'w8a8'):
            encoder = stack.enter_context(Encoder(model_name, 'w8a8', weights=args.weights, engine=args.engine))
            forwards['w8a8'] = native_forward(sam, encoder)
        for index, image_id in enumerate(image_ids):
            info = coco.loadImgs(image_id)[0]
            anns = [a for a in coco.loadAnns(coco.getAnnIds(imgIds=image_id, iscrowd=False)) if a['area'] > 100]
            anns = sorted(anns, key=lambda a: -a['area'])[:8]
            if not anns:
                continue
            image = np.array(Image.open(args.coco / 'val2017' / info['file_name']).convert('RGB'))
            truth = np.stack([coco.annToMask(a) > 0 for a in anns])
            points = []
            for mask in truth:
                ys, xs = np.nonzero(mask)
                nearest = np.argmin((ys - ys.mean()) ** 2 + (xs - xs.mean()) ** 2)
                points.append([xs[nearest], ys[nearest]])
            points = np.asarray(points, dtype=np.float32)[:, None, :]
            labels = np.ones((len(anns), 1), dtype=np.int32)
            records = [{'img_id': image_id, 'ann_id': a['id']} for a in anns]
            for name, forward in forwards.items():
                sam.forward_image = forward
                with torch.no_grad():
                    predictor.set_image(image)
                    masks, scores, _ = predictor.predict(point_coords=points, point_labels=labels, multimask_output=True)
                masks, scores = np.asarray(masks), np.asarray(scores)
                if masks.ndim == 3:
                    masks, scores = masks[None], scores[None]
                selected = scores.argmax(1)
                masks = masks[np.arange(len(anns)), selected].reshape(truth.shape) > 0
                iou = (masks & truth).sum((1, 2)) / np.maximum((masks | truth).sum((1, 2)), 1)
                for i, row in enumerate(records):
                    row[f'{name}_iou_gt'] = float(iou[i])
                    row[f'{name}_idx'] = int(selected[i])
            rows.extend(records)
            if (index + 1) % 50 == 0:
                print(f'{model_name}: {index + 1}/{len(image_ids)} images, {len(rows)} instances, '
                      f'{time.monotonic() - start:.0f}s', flush=True)
        sam.forward_image = fp32_forward
    if not rows:
        raise ValueError('No eligible instances found')
    out = args.output / model_name
    out.mkdir(parents=True, exist_ok=True)
    with (out / 'instances.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    scores = {name: float(100 * np.mean([row[f'{name}_iou_gt'] for row in rows])) for name in forwards}
    summary = {'model': model_name, 'images': len(image_ids), 'instances': len(rows),
               'prompt': 'nearest mask pixel to centroid', 'max_instances': 8,
               'miou_percent': scores, 'seconds': time.monotonic() - start}
    values = [value for key, value in scores.items() if key.startswith('w4a4_')]
    if values:
        summary['w4a4_mean'] = float(np.mean(values))
        summary['w4a4_std'] = float(np.std(values, ddof=1)) if len(values) > 1 else None
    (out / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps(summary, indent=2), flush=True)
    del predictor, sam
    torch.cuda.empty_cache()
    return summary

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--coco', type=Path, required=True)
    p.add_argument('--model', choices=[*MODELS, 'all'], default='all')
    p.add_argument('--precision', choices=['w4a4', 'w8a8', 'both'], default='both')
    p.add_argument('--seeds', default='all', help='all or comma-separated calibration seeds')
    p.add_argument('--images', type=int, default=1000)
    p.add_argument('--weights', type=Path, default=ROOT / 'weights')
    p.add_argument('--engine', type=Path, default=ROOT / 'build/sam21')
    p.add_argument('--output', type=Path, default=ROOT / 'outputs/eval')
    args = p.parse_args()
    if args.images < 1:
        p.error('--images must be positive')
    for model in MODELS if args.model == 'all' else [args.model]:
        evaluate(args, model)

if __name__ == '__main__':
    main()
