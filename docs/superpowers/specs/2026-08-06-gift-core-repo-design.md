# GIFT Core Repo — Design Specification

**Date:** 2026-08-06
**Paper:** [GIFT: Bootstrapping Image-to-CAD Program Synthesis via Geometric Feedback](https://arxiv.org/abs/2603.27448)
**Status:** Draft

## 1. Overview

GIFT (Geometric Inference Feedback Tuning) is a data augmentation framework that uses geometric feedback to bootstrap high-quality training data for Image-to-CAD program synthesis. It converts test-time computation into training data through two mechanisms:

- **GIFT-REJECT**: Soft rejection sampling — retains diverse generated programs that exceed an IoU threshold
- **GIFT-FDA**: Failure-driven augmentation — takes near-miss programs, renders their 3D output to 2D, and pairs the rendered images with ground-truth code

This repo implements the complete GIFT bootstrapping loop as a fresh codebase, copying essential infrastructure from CADRL (the existing training system) and adding the GIFT-specific orchestration, FDA rendering pipeline, and a Gradio web demo.

## 2. The GIFT Bootstrapping Loop

### Algorithm

```
Input: Base model M₀ (CADCODER/Qwen2.5-VL-7B-CadCoder), Dataset D₀
Parameters: τ_accept, τ_reject, K (candidates per image), max_iterations

For iteration n = 1, 2, ..., max_iterations:
  1. GENERATE
     - For each image in D₀, generate K candidate CadQuery programs using Mₙ₋₁
     - Via VLLM with temperature sampling

  2. EVALUATE
     - Execute each candidate program to produce a 3D solid
     - Compute 3D IoU against ground-truth solid (via IoU server)

  3. CLASSIFY candidates into three buckets per image:

     a. GIFT-REJECT (IoU ≥ τ_accept):
        → Training pair: (original_image, generated_code)
        → High-fidelity programs that may differ from ground truth

     b. GIFT-FDA (τ_reject < IoU < τ_accept):
        → Execute generated code → 3D solid → render to 2D image
        → Training pair: (rendered_2d_image, ground_truth_code)
        → Near-miss outputs become valid training data via novel viewpoints

     c. DISCARD (IoU ≤ τ_reject):
        → Thrown away — too far from target

  4. OVERSAMPLE (optional, when ALL K candidates for an image ≤ τ_reject):
     → Repeat (original_image, ground_truth_code) oversample_factor times
     → Gives the model more exposure to hard examples

  5. BUILD DATASET
     → Dₙ = D₀ ∪ REJECT_pairs ∪ FDA_pairs ∪ OVERSAMPLE_pairs
     → Save as versioned Arrow dataset

  6. RETRAIN
     → Fine-tune Mₙ₋₁ on Dₙ using SFT with optional KL regularization
     → Produces Mₙ

  7. CHECKPOINT
     → Save Mₙ, log metrics (mean IoU, VSR, bucket counts, dataset size)
     → Compare to previous iteration

Output: Final model M_final, augmented dataset D_final
```

### Default Parameters (from paper)
- `τ_accept = 0.9` — IoU threshold for accepting generated programs
- `τ_reject = 0.5` — IoU threshold below which candidates are discarded
- `K = 16` — candidates per image
- `max_iterations = 5`
- `oversample_factor = 3`

## 3. Architecture

### 3.1 Module Structure

```
GIFT/
├── gift/
│   ├── __init__.py
│   ├── config.py                # GIFTConfig dataclass
│   ├── core/
│   │   ├── __init__.py
│   │   ├── loop.py              # GIFTLoop orchestrator
│   │   ├── reject.py            # GIFT-REJECT: filter by τ_accept
│   │   ├── fda.py               # GIFT-FDA: render near-misses + pair with GT
│   │   ├── renderer.py          # 3D→2D rendering pipeline
│   │   └── dataset_builder.py   # Combine buckets into HF dataset
│   ├── inference/
│   │   ├── __init__.py
│   │   ├── vllm_client.py       # VLLM inference client
│   │   ├── iou_server.py        # IoU computation server
│   │   ├── geom.py              # 3D IoU with shape alignment
│   │   └── processors.py        # IOUProcessor (parallel IoU)
│   ├── training/
│   │   ├── __init__.py
│   │   ├── sft_trainer.py       # SFTSyncedGrad trainer
│   │   └── collators.py         # QwenVLCollator
│   └── data/
│       ├── __init__.py
│       ├── datasets.py          # MinimalImageCADDataset
│       └── utils.py             # extract_code, image encoding helpers
├── demo/
│   └── app.py                   # Gradio web UI
├── scripts/
│   ├── run_gift_loop.py         # CLI entry point for full loop
│   ├── run_iou_server.py        # Start IoU server
│   └── run_vllm_server.py       # Start VLLM server
├── CLAUDE.md
├── requirements.txt
└── pyproject.toml
```

### 3.2 Component Responsibilities

#### `gift/config.py` — GIFTConfig
Single dataclass holding all configuration:
- Model identity (`base_model`, `dataset`)
- GIFT loop parameters (`tau_accept`, `tau_reject`, `candidates_per_image`, `max_iterations`, `oversample_failures`, `oversample_factor`)
- Inference parameters (`temperature`, `top_p`, `max_tokens`)
- Training parameters (`lr`, `num_epochs`, `batch_size`, `use_kl`, `kl_weight`, `freeze_vision`)
- Infrastructure (`iou_server_url`, `iou_workers`, `render_size`)
- Paths (`output_dir`, `checkpoint_dir`)

#### `gift/core/loop.py` — GIFTLoop
The main orchestrator. Accepts a `GIFTConfig` and runs the full bootstrapping loop.
- `run()` — executes all iterations
- `run_iteration(n)` — runs a single iteration (generate → evaluate → classify → build → retrain)
- Tracks metrics per iteration and saves to JSON
- Supports resuming from a specific iteration

#### `gift/core/reject.py` — GIFT-REJECT
- `classify_candidates(ious, tau_accept, tau_reject)` → returns three lists: accepted, fda_eligible, discarded
- `build_reject_pairs(accepted_candidates, original_images)` → list of (image, code) pairs

#### `gift/core/fda.py` — GIFT-FDA
- `build_fda_pairs(fda_candidates, ground_truth_codes, renderer)` → list of (rendered_image, gt_code) pairs
- Calls renderer for each near-miss candidate
- Handles execution failures gracefully (skip if code doesn't produce valid solid)

#### `gift/core/renderer.py` — 3D→2D Rendering
- `render_code_to_image(cadquery_code, size=(448, 448))` → PIL Image
- Executes CadQuery code in subprocess → exports STEP → renders via headless OCC/VTK
- Isolated subprocess execution to handle segfaults from OCC
- Configurable render viewpoint and image size

#### `gift/core/dataset_builder.py` — Dataset Construction
- `build_augmented_dataset(original_data, reject_pairs, fda_pairs, oversample_pairs)` → HuggingFace Dataset
- Interleaves datasets with configurable mixing ratios
- Saves versioned Arrow files per iteration
- Supports loading previous iteration datasets for comparison

#### `gift/inference/` — Infrastructure (from CADRL)
- `vllm_client.py`: VLLM API client for batch inference. Generates K candidates per image.
- `iou_server.py`: FastAPI server that accepts CadQuery code pairs and returns IoU scores. Uses ProcessPoolExecutor with subprocess isolation.
- `geom.py`: `align_shapes()` — centers, scale-normalizes, and tries 4 rotation alignments to compute best 3D IoU.
- `processors.py`: `IOUProcessor` — parallel IoU computation with checkpointing and resume support. This is the subprocess-based path (each IoU runs in its own Python subprocess using CadQuery). Used as a fallback when the IoU server is not running.

**IoU computation path:** `GIFTLoop.run_iteration()` uses the IoU server (HTTP-based, `iou_server.py` + `geom_occ.py`) as the primary path for step 2 (EVALUATE). The `IOUProcessor` (subprocess-based, `processors.py` + `geom.py`) is available as a standalone fallback when no server is running. Config flag `use_iou_server: bool = True` controls which path is used.

#### `gift/training/` — Training Infrastructure (from CADRL)
- `sft_trainer.py`: `SFTSyncedGrad` — Accelerate-based SFT trainer with FSDP, manual gradient accumulation, global token-count normalization, KL regularization, WandB logging.
- `collators.py`: `QwenVLCollator` — tokenizes chat-format prompts via Qwen2VLProcessor, masks non-assistant tokens.

#### `gift/data/` — Data Handling (from CADRL)
- `datasets.py`: `MinimalImageCADDataset` — wraps HuggingFace datasets with image+code columns, handles lazy loading, augmentation, prompt construction.
- `utils.py`: `extract_code()` (regex extraction from markdown), `encode_image()` (PIL→base64).

### 3.3 Demo — Gradio Web UI

`demo/app.py` with three tabs:

**Tab 1: Single Image Inference**
- Upload an image of a 3D CAD model
- Generate K candidates with configurable temperature
- Display: input image, generated CadQuery code, IoU score, rendered 3D reconstruction
- Best-of-K selection highlighted

**Tab 2: GIFT Iteration Visualization**
- Run one GIFT iteration on a configurable batch (default: 20 samples)
- Show the three buckets visually:
  - REJECT accepted: original image + generated code + IoU score
  - FDA near-misses: original image vs rendered image + ground truth code
  - Discarded: original image + failed code + low IoU
  - Oversampled: hard examples flagged
- Summary metrics: accept rate, FDA rate, discard rate, mean IoU per bucket

**Tab 3: Iteration History**
- Load results from previous GIFT loop runs
- Plot IoU progression across iterations
- Compare dataset sizes, bucket distributions over time

**Prerequisites:** IoU server and VLLM server must be running.

## 4. Files Copied from CADRL

| CADRL Source | GIFT Destination | Changes |
|---|---|---|
| `Inference/Geom/_IOU.py` | `gift/inference/geom.py` | Keep `align_shapes`, `_compute_iou_centering`, `_compute_iou_centering_normalize`. Used by `processors.py` (subprocess path). |
| `Inference/Geom/_OCC_IOU.py` | `gift/inference/geom_occ.py` | OCC-based IoU with `align_shapes`, `load_step_file`. Used by `iou_server.py` (HTTP server path). |
| `Inference/processors.py` | `gift/inference/processors.py` | Keep `IOUProcessor` class and helper functions (`_run_gt_code`, `_run_gen_code`, `_compute_iou`). Update `PATH_TO_CAD_PYTHON` to be configurable. |
| `Inference/iou_serve.py` | `gift/inference/iou_server.py` | Keep server endpoints. Make host/port/workers configurable. |
| `Inference/VLLMClient.py` | `gift/inference/vllm_client.py` | Keep chat completion client. Remove NCCL weight sync methods (not needed for SFT loop — we retrain from checkpoint, not online). |
| `Trainers/SFT/__init__.py` | `gift/training/sft_trainer.py` | Keep `SFTSyncedGrad` only. Drop `SFT` class (redundant). Keep `selective_log_softmax`. |
| `DataUtils/Collators.py` | `gift/training/collators.py` | Keep `QwenVLCollator` only. Drop `QwenVLRLCollator` (RL not used in GIFT). |
| `DataUtils/Datasets.py` | `gift/data/datasets.py` | Keep `MinimalImageCADDataset` only. Drop `CADLMDataset` and `CADRLDataset`. Keep `extract_code` in utils.py. |
| `Inference/InferenceScaling/render_step_pyvista_simple.py` | `gift/core/renderer.py` | Merge all 3 render backends into one module. Add `render_cadquery_to_image()` wrapper. |
| `Inference/InferenceScaling/render.py` | `gift/core/renderer.py` | OCC Viewer3d high-quality backend (merged). |
| `Inference/InferenceScaling/render_step_headless.py` | `gift/core/renderer.py` | VTK fallback backend (merged). |

## 5. Rendering Pipeline (New)

The FDA rendering pipeline is a new component not present in CADRL's training loop, but CADRL already has three rendering backends in `Inference/InferenceScaling/`:

### Available Rendering Backends (from CADRL)

1. **OCC Viewer3d** (`render.py`): Highest quality — materials (shiny plastic/chrome), 3-point directional lighting, edge boundaries with `SetFaceBoundaryDraw(True)`, transparency. Uses `Viewer3d.Create()` offscreen. Requires EGL (`PYOPENGL_PLATFORM=egl`).

2. **VTK offscreen** (`render_step_headless.py`): STEP → STL via OCC meshing → VTK offscreen rendering. More reliable for headless servers. Simpler visuals (flat gray, white background).

3. **PyVista** (`render_step_pyvista_simple.py`): CadQuery → STL → `pv.Plotter(off_screen=True)`. Simplest API, most portable. Isometric view, smooth shading.

### Approach for FDA
1. Execute CadQuery code in a subprocess (isolated for crash safety, same pattern as `IOUProcessor._run_gen_code()`)
2. Export the resulting solid to a STEP file via `cq.exporters.export(solid, path)`
3. Render the STEP file to a 2D PNG using PyVista (default, most portable) with OCC Viewer3d as optional high-quality backend
4. Return as PIL Image at 448×448 (matching training data resolution)

### Key Files Copied for Rendering

| CADRL Source | GIFT Destination | Notes |
|---|---|---|
| `Inference/InferenceScaling/render_step_pyvista_simple.py` | `gift/core/renderer.py` | Primary renderer, cleaned up with PIL return |
| `Inference/InferenceScaling/render.py` | `gift/core/renderer.py` | OCC Viewer3d as optional high-quality backend |
| `Inference/InferenceScaling/render_step_headless.py` | `gift/core/renderer.py` | VTK fallback |

All three backends merged into a single `renderer.py` with a `render_cadquery_to_image(code, size, backend)` function that:
- Executes CadQuery code in subprocess → exports STEP
- Renders via selected backend (pyvista default, occ_viewer, vtk as fallbacks)
- Returns PIL Image

### Viewpoint Matching
The rendered FDA images should match the visual style of the training data (DeepCAD-CQ-Vision-Paired dataset). The training images use isometric views with white backgrounds. The PyVista renderer's `view_isometric()` with white background matches this.

### Parallel Rendering
`ProcessPoolExecutor` for batch rendering of FDA candidates, same subprocess isolation pattern as IoU computation. Each render is independent and crash-safe.

### Error Handling
- Code execution failure → skip this FDA candidate
- Rendering failure → skip this FDA candidate
- Timeout (30s per render) → skip

## 6. Data Flow

```
HuggingFace Dataset (image, code)
        │
        ▼
┌─── VLLM Server ───┐
│  Generate K        │
│  candidates/image  │
└────────┬───────────┘
         │ (image, [gen_code_1..K])
         ▼
┌─── IoU Server ─────┐
│  Evaluate each      │
│  candidate vs GT    │
└────────┬────────────┘
         │ (image, gen_code, iou_score)
         ▼
┌─── GIFT Classifier ─────────────────────────────────┐
│                                                       │
│  IoU ≥ τ_accept    →  REJECT bucket                  │
│  τ_reject < IoU    →  FDA bucket → Renderer → pair   │
│  IoU ≤ τ_reject    →  DISCARD                        │
│  All discarded     →  OVERSAMPLE bucket              │
│                                                       │
└────────┬─────────────────────────────────────────────┘
         │
         ▼
┌─── Dataset Builder ──┐
│  D₀ ∪ REJECT ∪ FDA  │
│  ∪ OVERSAMPLE        │
│  → Arrow files       │
└────────┬─────────────┘
         │
         ▼
┌─── SFT Trainer ──────┐
│  Fine-tune Mₙ₋₁      │
│  on augmented data    │
│  → Mₙ checkpoint     │
└───────────────────────┘
```

## 7. Dependencies

```
# Core
torch>=2.0
transformers>=4.40
accelerate
datasets
vllm

# Geometry
cadquery
OCP  # OpenCascade Python bindings (via cadquery)

# Rendering
pyvista  # or vtk for headless rendering

# Demo
gradio>=4.0

# Infrastructure
fastapi
uvicorn
aiohttp
joblib
tqdm
wandb
```

## 8. Environment

Uses the existing uv environment at `/workspace/home/lab/gcg/dev/.cad` (Python 3.10), same as CADRL.

## 9. Model

Base model: `CADCODER/Qwen2.5-VL-7B-CadCoder` (the pre-trained CadCoder model from HuggingFace).

The GIFT loop starts from this model and iteratively improves it. Each iteration produces a new checkpoint.

## 10. Success Criteria

1. The GIFT loop runs end-to-end on the DeepCAD dataset (at least one full iteration)
2. Each iteration produces a valid augmented dataset with REJECT + FDA + OVERSAMPLE pairs
3. IoU improves across iterations (matching paper's ~12% improvement claim)
4. The Gradio demo can:
   - Show single-image inference with K candidates
   - Visualize one GIFT iteration with bucket classification
   - Display FDA near-miss renders vs originals
5. The codebase is self-contained (no imports from CADRL at runtime)
