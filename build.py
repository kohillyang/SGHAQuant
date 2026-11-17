"""Build the native CUDA encoders."""
import argparse
import os
from pathlib import Path
import shutil
import subprocess

ROOT = Path(__file__).resolve().parent

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--family', choices=['sam21', 'sam1', 'dinov3', 'all'], default='sam21')
    parser.add_argument('--arch', choices=['80', '86', '89'], default='89')
    parser.add_argument('--nvcc', default=os.environ.get('NVCC', 'nvcc'))
    args = parser.parse_args()
    if args.family in ('sam1', 'dinov3', 'all') and args.arch != '89':
        parser.error('SAM1 and DINOv3 static W8A8 require --arch 89 (Ada)')
    nvcc = shutil.which(args.nvcc)
    if not nvcc:
        parser.error('nvcc not found; set --nvcc to the CUDA 12.9 compiler')
    (ROOT / 'build').mkdir(exist_ok=True)
    targets = []
    if args.family in ('sam21', 'all'):
        targets.append(('sam21', 'src/engine.cu', []))
    if args.family in ('sam1', 'all'):
        targets += [(f'sam1_dh{dh}', 'src/sam1/nano_samv1.cu', [f'-DSAMV1_DH={dh}']) for dh in (64, 80)]
    if args.family in ('dinov3', 'all'):
        targets.append(('dinov3', 'src/dinov3/nano_vit.cu', []))
    for name, source, extra in targets:
        cmd = [nvcc, '-O3', '-std=c++17', f'-arch=sm_{args.arch}',
               '-diag-suppress', '550,128', *extra, str(ROOT / source),
               '-lcublasLt', '-o', str(ROOT / 'build' / name)]
        subprocess.run(cmd, check=True)
        print(ROOT / 'build' / name, flush=True)

if __name__ == '__main__':
    main()
