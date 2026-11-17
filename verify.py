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

"""Compare a complete Table 4 evaluation with the published values."""
import argparse
import json
import math
from pathlib import Path

from sghaquant.runtime import MODELS, ROOT, SEEDS

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--results', type=Path, default=ROOT / 'outputs/eval')
    p.add_argument('--tolerance', type=float, default=0.01, help='absolute mIoU percentage points')
    args = p.parse_args()
    if not math.isfinite(args.tolerance) or args.tolerance < 0:
        p.error('--tolerance must be finite and nonnegative')
    expected = json.loads((ROOT / 'results/paper.json').read_text())
    failures = []
    for model in MODELS:
        path = args.results / model / 'summary.json'
        if not path.is_file():
            failures.append(f'{model}: missing {path}')
            continue
        actual = json.loads(path.read_text())
        ref = expected['models'][model]
        if actual['images'] != expected['images'] or actual['instances'] != expected['instances']:
            failures.append(f'{model}: expected 1000 images and 4629 instances')
        values = {'fp32': ref['fp32'], 'w8a8': ref['w8a8']}
        values.update({f'w4a4_{seed}': ref['w4a4']['seeds'][str(seed)] for seed in SEEDS})
        for key, value in values.items():
            observed = actual['miou_percent'].get(key)
            if observed is None or not math.isfinite(observed) or abs(observed - value) > args.tolerance:
                failures.append(f'{model}/{key}: {observed}, expected {value:.6f}')
        print(f'{model}: W4A4 {actual.get("w4a4_mean", 0):.4f}, W8A8 '
              f'{actual["miou_percent"].get("w8a8", 0):.4f}')
    if failures:
        raise SystemExit('\n'.join(failures))
    print('PASS: all 28 mIoU values match Table 4 within the specified tolerance.')

if __name__ == '__main__':
    main()
