# Prompted Segmentation for Drywall QA

Text-conditioned binary segmentation of drywall defects using fine-tuned CLIPSeg.

**Random seed: 42**

---

## Task

Given an image and a natural-language prompt, produce a binary mask (PNG, 0/255) — no bounding box required:

| Prompt | Dataset |
|---|---|
| `segment crack` | Crack segmentation dataset |
| `segment taping area` | Drywall joint / tape dataset |

---

## Model

**CLIPSeg** (CIDAS/clipseg-rd64-refined) — a CLIP-based model that jointly processes image and text to directly output a segmentation mask. Unlike SAM, no bounding box or spatial prompt is needed — the text phrase alone conditions the segmentation.

---

## Results

| Prompt | Split | n | mIoU | Dice |
|---|---|---|---|---|
| segment crack | val | 186 | 0.4903 | 0.6465 |
| segment taping area | val | 250 | 0.5691 | 0.7116 |
| **Overall** | **val** | **436** | **0.5355** | **0.6838** |
| segment crack | test | 312 | 0.4857 | 0.6380 |

Avg inference time: **79ms/image** (val) | Model size: **~400 MB**

---

## Setup

### Requirements

- Python 3.12
- CUDA-capable GPU (tested on RTX 4060 Laptop GPU)

### Install

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install roboflow opencv-python pillow matplotlib numpy tqdm transformers
pip install git+https://github.com/facebookresearch/segment-anything.git
pip install pycocotools
```

---

## Project Structure

```
wall_cracks/
├── Drywall-Join-Detect-1/          # Dataset 1 (taping area)
├── crack-1/                        # Dataset 2 (cracks)
├── dataset/                        # Prepared dataset
│   ├── train/images/ + masks/
│   ├── val/images/ + masks/
│   └── test/images/ + masks/
├── checkpoints_clipseg/
│   ├── best/                       # Best val mIoU checkpoint
│   └── final/                      # Final epoch checkpoint
├── predictions_clipseg/            # Output masks (PNG, 0/255)
├── prepare_data.py
├── train_clipseg.py
├── evaluate_clipseg.py
├── training_curves_clipseg.png
├── visual_examples_clipseg.png
└── README.md
```

---

## Reproducing Results

### Step 1 — Download datasets

```python
from roboflow import Roboflow
rf = Roboflow(api_key="YOUR_KEY")

# Taping area
project1 = rf.workspace("objectdetect-pu6rn").project("drywall-join-detect")
project1.version(1).download("coco")

# Cracks (substitute — original dataset had no exportable versions)
project2 = rf.workspace("university-bswxt").project("crack-bphdr")
project2.version(1).download("coco-segmentation")
```

### Step 2 — Prepare data

```bash
python prepare_data.py
```

Fills bounding box rectangles as taping area GT masks. Rasterises crack polygons to binary masks. Takes ~1 minute.

### Step 3 — Train

```bash
python train_clipseg.py
```

Downloads CLIPSeg on first run (~400MB). Trains for 30 epochs (~1.1 hrs on RTX 4060). Saves best checkpoint to `checkpoints_clipseg/best/`.

### Step 4 — Evaluate & export masks

```bash
python evaluate_clipseg.py
```

Prints per-prompt mIoU and Dice on val and test sets. Saves prediction masks to `predictions_clipseg/`.

---

## Output Mask Format

- Single-channel PNG
- Same spatial size as source image
- Pixel values: `0` = background, `255` = foreground
- Filename: `{image_stem}__{prompt}.png`
  - Example: `10.rf.f0b182d5a78d6adc75e97f0f2314be9f__segment_crack.png`

---

## Training Details

| | Value |
|---|---|
| Base model | CLIPSeg clipseg-rd64-refined |
| Fine-tuning | Full model (all layers) |
| Loss | BCE + Dice (equal weight) |
| Optimizer | AdamW, LR=1e-4, weight_decay=1e-4 |
| Scheduler | Cosine annealing, 30 epochs |
| Batch size | 8 |
| Image size | 352x352 |
| Random seed | 42 |

---

## Hardware & Runtime

| | Value |
|---|---|
| GPU | NVIDIA RTX 4060 Laptop GPU |
| Training time | ~1.1 hours (30 epochs) |
| Inference time (val) | 79.0 ms/image |
| Inference time (test) | 142.6 ms/image |
| Model size | ~400 MB |
