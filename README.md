# SGHAQuant

SAM2.1, SAM and DINOv3 W4A4/W8A8 inference for *Localizing and Fixing the Instability of Hadamard Rotations on 4-bit Vision Transformers*.

The release includes weights and native CUDA encoders. SAM2.1 also includes point/box segmentation and COCO evaluation with the official FP32 decoder.

| Family | Models | Encoder output |
|---|---|---|
| SAM2.1 | Hiera-T/S/B+/L | Multi-scale image features |
| SAM | ViT-B/L/H | `B × 256 × 64 × 64` embeddings |
| DINOv3 | ViT-L/16, 224 px | CLS, 4 registers, 196 patch tokens |

All three quantize qkv/fc1/fc2 with grouped Hadamard rotations. SAM2.1 W4A4 uses channel scaling and affine correction; SAM uses GPTQ weights; DINOv3 W4A4 uses RTN weights. DINOv3 keeps block 0's fc2 in FP16. SAM and DINOv3 use static fc2 scales at both precisions.

## Setup

Linux, Python 3.11, CUDA 12.9 and an Ada GPU (tested on RTX 4090).

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

SAM2.1 also supports `--arch 86` for RTX 3090. Weights are downloaded from [Releases](https://github.com/kohillyang/SGHAQuant/releases) and verified by SHA-256.

## Inference

```bash
python infer.py --model tiny --precision w4a4 --image image.jpg --point 320 240
python infer.py --model tiny --precision w8a8 --image image.jpg --box 100 80 500 400
```

The selected mask is saved to `outputs/mask.png`. Model names: `tiny`, `small`, `base_plus`, `large`.

SAM and DINOv3 encoder features:

```bash
python build.py --family sam1
python download.py --family sam1 --model vit_b
python encode.py --family sam1 --model vit_b --precision w4a4 --image image.jpg --output outputs/sam.npz

python build.py --family dinov3
python download.py --family dinov3 --model vit_l
python encode.py --family dinov3 --model vit_l --precision w8a8 --image image.jpg --output outputs/dino.npz

python benchmark.py --family sam1 --model vit_b
python benchmark.py --family dinov3 --model vit_l
```

Both encoders accept `--precision w4a4` or `w8a8`; SAM model names are `vit_b`, `vit_l`, `vit_h`. `download.py --model all` downloads every model in the selected family. SAM resizes the longest side to 1024, normalizes and pads; DINOv3 resizes the short side to 256 and center-crops to 224. `encode.py --input-npy` accepts preprocessed NCHW arrays (SAM: float32, 1024 px; DINOv3: uint8, 224 px).

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

Encoder checks: [SAM and DINOv3](results/encoders_validation.json). DINOv3 model files retain the [DINOv3 License](licenses/DINOv3-LICENSE.md).
