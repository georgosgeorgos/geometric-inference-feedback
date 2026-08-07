# GIFT: Geometric Inference Feedback Tuning

Implementation of [GIFT](https://arxiv.org/abs/2603.27448) — a bootstrapping loop that converts test-time computation into training data for Image-to-CAD program synthesis.

## How GIFT Works

GIFT iteratively improves a vision-language model by generating candidate CadQuery programs for input images, evaluating them via 3D IoU, and classifying them into three buckets:

| Bucket | Condition | Training Pair |
|--------|-----------|---------------|
| **REJECT** (accept) | IoU >= 0.9 | (original image, generated code) |
| **FDA** (near-miss) | 0.5 < IoU < 0.9 | (rendered 3D output as 2D, ground truth code) |
| **DISCARD** | IoU <= 0.5 | Thrown away |

When ALL candidates for an image are discarded, the original (image, ground truth code) pair is oversampled to give the model more exposure to hard examples.

The augmented dataset is used to retrain the model, and the loop repeats.

## Setup

### Environment

Uses the shared `.cad` Python 3.10 environment:

```bash
# All commands use this Python
export CAD_PYTHON=/workspace/home/lab/gcg/dev/.cad/bin/python

# Or use uv
uv run --project /workspace/home/lab/gcg/dev/.cad python <script.py>
```

### System Dependencies

- `xorg-x11-server-Xvfb` — required for headless 3D rendering (installed)
- CUDA GPUs — required for VLLM inference and training

## Quick Start

### 1. Start the IoU Server

Computes 3D IoU between generated and ground truth CadQuery solids:

```bash
$CAD_PYTHON scripts/run_iou_server.py --port 5000 --num-workers 64
```

### 2. Start the VLLM Server

Serves the base model for candidate generation:

```bash
$CAD_PYTHON scripts/run_vllm_server.py \
    --model CADCODER/Qwen2.5-VL-7B-CadCoder \
    --port 8000 \
    --tensor_parallel_size 2
```

### 3. Run the GIFT Loop (Offline SFT)

The paper-faithful approach — generate all candidates, build augmented dataset, retrain:

```bash
$CAD_PYTHON scripts/run_gift_loop.py \
    --tau_accept 0.9 \
    --tau_reject 0.5 \
    --candidates_per_image 16 \
    --max_iterations 5 \
    --output_dir ./output
```

### 3b. Run Online RL (Alternative)

Train with IoU as reward using REINFORCE, GRPO, or DAPO:

```bash
# REINFORCE (vanilla policy gradient)
accelerate launch scripts/run_gift_rl.py \
    --algorithm reinforce \
    --n_samples 16 \
    --freeze_vision \
    --log_to_wandb

# GRPO (clipped, single inner step)
accelerate launch scripts/run_gift_rl.py \
    --algorithm grpo \
    --n_samples 16 \
    --freeze_vision

# DAPO (asymmetric clipping, multiple inner steps)
accelerate launch scripts/run_gift_rl.py \
    --algorithm dapo \
    --mu 4 \
    --epsilon_low 0.2 \
    --epsilon_high 0.28
```

GRPO is DAPO with `mu=1` and symmetric clipping — not a separate implementation.

### 4. Run the Demo

```bash
$CAD_PYTHON demo/app.py
# Opens at http://localhost:7860
```

Three tabs:
- **Single Inference**: Upload image, generate K candidates, see rendered output
- **GIFT Iteration**: Run one iteration on N samples, see bucket classification
- **History**: Load and view metrics from previous runs

## Project Structure

```
GIFT/
├── gift/
│   ├── config.py                  # GIFTConfig dataclass (all parameters)
│   ├── core/
│   │   ├── loop.py                # GIFTLoop orchestrator (generate -> evaluate -> classify -> build -> retrain)
│   │   ├── reject.py              # Classify candidates by IoU thresholds
│   │   ├── fda.py                 # Render near-misses to 2D, pair with ground truth
│   │   ├── renderer.py            # CadQuery -> STL -> PyVista -> PNG (under Xvfb)
│   │   └── dataset_builder.py     # Combine buckets into HuggingFace Dataset
│   ├── inference/
│   │   ├── __init__.py            # run_code(), compute_iou() helpers
│   │   ├── geom.py                # 3D IoU: shape alignment via inertia tensor + boolean ops
│   │   ├── processors.py          # IOUProcessor: parallel batch IoU with checkpointing
│   │   ├── iou_server.py          # FastAPI server for async IoU computation
│   │   ├── iou_client.py          # Async HTTP client for IoU server
│   │   └── vllm_client.py         # VLLM client with NCCL weight sync for RL
│   ├── training/
│   │   ├── sft_trainer.py         # SFTSyncedGrad: FSDP + KL regularization
│   │   ├── rl_trainer.py          # REINFORCE + DAPO trainers
│   │   └── collators.py           # QwenVLCollator (SFT) + QwenVLRLCollator (RL)
│   └── data/
│       ├── datasets.py            # MinimalImageCADDataset (SFT) + CADRLDataset (RL)
│       └── utils.py               # extract_code(), encode_image()
├── scripts/
│   ├── run_gift_loop.py           # CLI: full GIFT bootstrapping loop
│   ├── run_gift_rl.py             # CLI: online RL training
│   ├── run_iou_server.py          # CLI: start IoU server
│   └── run_vllm_server.py         # CLI: start VLLM server
├── demo/app.py                    # Gradio web UI
└── tests/                         # 15 tests
```

## IoU Computation

Three paths, all using `geom.py` (CadQuery-based):

| Path | When to Use |
|------|------------|
| `IOUClient` + `iou_server.py` | High-throughput async (GIFT loop, RL training) |
| `IOUProcessor` | Standalone batch with checkpointing (evaluation, long runs) |
| `compute_iou()` | Single-pair subprocess (quick checks) |

The IoU algorithm (`align_shapes`) normalizes scale via inertia tensor, tries 4 rotation alignments, and computes boolean intersection/union volume ratio.

## Configuration

All parameters live in `GIFTConfig` (`gift/config.py`):

| Parameter | Default | Description |
|-----------|---------|-------------|
| `tau_accept` | 0.9 | IoU threshold for accepting generated programs |
| `tau_reject` | 0.5 | IoU threshold below which candidates are discarded |
| `candidates_per_image` | 16 | K candidates generated per image |
| `max_iterations` | 5 | Number of GIFT loop iterations |
| `oversample_factor` | 3 | Repetitions for hard examples |
| `training_mode` | "sft" | Training mode: sft, reinforce, grpo, dapo |
| `use_kl` | True | KL regularization against base model |
| `freeze_vision` | True | Freeze vision encoder during training |

## Default Parameters (from paper)

- Base model: `CADCODER/Qwen2.5-VL-7B-CadCoder`
- Dataset: `CADCODER/DeepCAD-CQ-Vision-Paired`
- K=16 candidates, 5 iterations
- Temperature 0.6 for generation
- 6 GPU training with FSDP

## Tests

```bash
$CAD_PYTHON -m pytest tests/ -v
```

## Rendering

FDA rendering uses CadQuery -> STL -> PyVista under Xvfb:

1. Execute CadQuery code in subprocess (crash isolation)
2. Export solid to STL
3. Render via PyVista offscreen (`view_isometric`, white background)
4. Return as 448x448 PIL Image (matching training data resolution)

pythonocc is NOT installed and NOT needed. All geometry uses CadQuery's native OCP bindings.
