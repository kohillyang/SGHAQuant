# SGHAQuant

SAM2.1 W4A4 and W8A8 inference for *Localizing and Fixing the Instability of Hadamard Rotations on 4-bit Vision Transformers*.

The release includes Hiera-T/S/B+/L weights, CUDA encoder kernels, image segmentation and COCO evaluation. W4A4 uses grouped Hadamard rotations, channel scaling and output affine correction. The prompt encoder and mask decoder use the official SAM2.1 FP32 implementation.

## Setup

Linux, Python 3.11, CUDA 12.9, and an Ampere or Ada GPU. Paper measurements use an RTX 4090.

```bash
git clone https://github.com/kohillyang/SGHAQuant.git
cd SGHAQuant
python -m venv .venv
source .venv/bin/activate
pip install wheel
pip install torch==2.9.0 torchvision==0.24.0 --index-url https://download.pytorch.org/whl/cu128
SAM2_BUILD_CUDA=0 pip install --no-build-isolation -r requirements.txt
python build.py --arch 89
python download.py --model tiny
```

Use `--arch 86` for RTX 3090. Weights are downloaded from [Releases](https://github.com/kohillyang/SGHAQuant/releases/tag/sam21-v1) and checked against `weights.json`.

## Inference

```bash
python infer.py --model tiny --precision w4a4 --image image.jpg --point 320 240
python infer.py --model tiny --precision w8a8 --image image.jpg --box 100 80 500 400
```

The selected mask is saved to `outputs/mask.png`. Model names: `tiny`, `small`, `base_plus`, `large`.

## Reproduce Table 4

Place COCO `val2017/` and `annotations/instances_val2017.json` under `$COCO`.

```bash
python download.py --model all --seeds all
python evaluate.py --coco "$COCO" --model all
python verify.py
python benchmark.py --model all --image "$COCO/train2017/000000391895.jpg"
```

Evaluation uses the first 1,000 sorted COCO validation IDs, up to eight instances per image with area above 100 pixels, and one interior point nearest each mask's centroid: 4,629 instances. W4A4 results use seeds 20261010–20261014; standard deviation uses `ddof=1`.

| Model | W8A8 mIoU | W4A4 mIoU | W8A8 ms, B1 / B8 | W4A4 ms, B1 / B8 |
|---|---:|---:|---:|---:|
| Hiera-T | 60.71 | 60.74 ± 0.12 | 2.19 / 16.81 | 1.87 / 15.01 |
| Hiera-S | 62.15 | 62.40 ± 0.19 | 2.61 / 19.30 | 2.19 / 16.90 |
| Hiera-B+ | 60.40 | 60.62 ± 0.17 | 4.13 / 32.72 | 3.38 / 27.47 |
| Hiera-L | 68.63 | 68.34 ± 0.18 | 9.09 / 72.40 | 7.17 / 57.53 |

Latency covers the encoder and neck with CUDA graphs: 200 warmup iterations, 200 measured iterations, median of six rounds, GPU at or below 50°C before each run. The timing input above is from COCO train2017; omitting `--image` uses zeros. W4A4 uses per-token scales at qkv/fc1 and one static scale per fc2 layer; W8A8 uses the paper's GPTQ weights and dynamic fc2 scales.

Per-seed results: [paper](results/paper.json), [reproduction](results/validation.json).
