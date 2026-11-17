"""Build the SAM2.1 CUDA encoder."""
import argparse
import os
from pathlib import Path
import shutil
import subprocess

ROOT = Path(__file__).resolve().parent

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--arch', choices=['80', '86', '89'], default='89')
    parser.add_argument('--nvcc', default=os.environ.get('NVCC', 'nvcc'))
    args = parser.parse_args()
    nvcc = shutil.which(args.nvcc)
    if not nvcc:
        parser.error('nvcc not found; set --nvcc to the CUDA 12.9 compiler')
    (ROOT / 'build').mkdir(exist_ok=True)
    cmd = [nvcc, '-O3', '-std=c++17', f'-arch=sm_{args.arch}',
           '-diag-suppress', '550,128', str(ROOT / 'src/engine.cu'),
           '-lcublasLt', '-o', str(ROOT / 'build/sam21')]
    subprocess.run(cmd, check=True)
    print(ROOT / 'build/sam21')

if __name__ == '__main__':
    main()
