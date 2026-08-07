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

  6. RETRAIN (two modes)
     a. Offline SFT (`run_gift_loop.py`):
        → Fine-tune Mₙ₋₁ on Dₙ using SFTSyncedGrad with optional KL regularization (ref = M₀)
        → Produces Mₙ as a new checkpoint
     b. Online RL (`run_gift_rl.py`):
        → Train Mₙ₋₁ using REINFORCE/GRPO/DAPO with IoU as reward
        → Generates candidates online, computes reward, updates weights in-place
        → Syncs weights to VLLM server via PyNCCL after each step

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
│   │   ├── renderer.py          # 3D→2D rendering (PyVista + Xvfb)
│   │   └── dataset_builder.py   # Combine buckets into HF dataset
│   ├── inference/
│   │   ├── __init__.py          # run_code(), compute_iou() helpers
│   │   ├── vllm_client.py       # VLLM inference client (+ NCCL weight sync for RL)
│   │   ├── iou_client.py        # Async HTTP client for IoU server
│   │   ├── iou_server.py        # IoU computation server
│   │   ├── geom.py              # 3D IoU with shape alignment (CadQuery)
│   │   └── processors.py        # IOUProcessor (parallel IoU)
│   ├── training/
│   │   ├── __init__.py
│   │   ├── sft_trainer.py       # SFTSyncedGrad trainer (offline SFT)
│   │   ├── rl_trainer.py        # REINFORCE/GRPO/DAPO trainer (online RL)
│   │   └── collators.py         # QwenVLCollator + QwenVLRLCollator
│   └── data/
│       ├── __init__.py
│       ├── datasets.py          # MinimalImageCADDataset (SFT) + CADRLDataset (RL)
│       └── utils.py             # extract_code, image encoding helpers
├── demo/
│   └── app.py                   # Gradio web UI
├── scripts/
│   ├── run_gift_loop.py         # CLI entry point for full loop (offline SFT)
│   ├── run_gift_rl.py           # CLI entry point for RL-based GIFT loop
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
- Training parameters (`lr`, `num_epochs`, `batch_size`, `use_kl`, `kl_weight`, `freeze_vision`, `training_mode` [sft|reinforce|grpo|dapo])
- Infrastructure (`iou_server_url`, `iou_workers`, `use_iou_server`, `render_size`)
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
- Executes CadQuery code in subprocess → exports STL → renders via PyVista under Xvfb
- Isolated subprocess execution to handle segfaults
- Isometric view, white background (matching DeepCAD training images)

#### `gift/core/dataset_builder.py` — Dataset Construction
- `build_augmented_dataset(original_data, reject_pairs, fda_pairs, oversample_pairs)` → HuggingFace Dataset
- Interleaves datasets with configurable mixing ratios
- Saves versioned Arrow files per iteration
- Supports loading previous iteration datasets for comparison

#### `gift/inference/` — Infrastructure (from CADRL)
- `__init__.py`: `run_code(code)` — subprocess execution of CadQuery code. `compute_iou(gt_step, gen_step)` — subprocess IoU via `geom.py`. `extract_code(text)` — regex extraction of Python from markdown. These are the simple helpers used by `InferenceScaling` and the RL pipeline.
- `vllm_client.py`: VLLM API client for batch inference via OpenAI-compatible API. Generates K candidates per image. Includes NCCL weight sync methods (`init_communicator`, `update_model_params`) for the online RL path — the trainer pushes updated weights to the running VLLM server after each training step.
- `iou_client.py`: `IOUClient` — async HTTP client for the IoU server. Uses `httpx` with semaphore-based concurrency control. Returns `(ious, statuses)` arrays. Used by the RL pipeline's `QwenVLRLCollator` to compute rewards.
- `iou_server.py`: FastAPI server that accepts CadQuery code pairs and returns IoU scores. Uses ProcessPoolExecutor with subprocess isolation.
- `geom.py`: `align_shapes()` — CadQuery-based 3D IoU with center-of-mass alignment, scale normalization via inertia tensor, and 4 rotation alignments. This is the ONLY IoU backend (pythonocc is not installed).
- `processors.py`: `IOUProcessor` — parallel IoU computation with checkpointing and resume support. Each IoU runs in its own Python subprocess using CadQuery.

**IoU computation paths:**
1. **IoU server** (`iou_server.py` + `iou_client.py`): HTTP-based, async. Primary path for both the GIFT loop and the RL pipeline. `IOUClient` provides the async interface.
2. **Subprocess batch** (`processors.py`): Standalone parallel batch path with checkpointing. Fallback when no server is running.
3. **Simple helpers** (`__init__.py`): `compute_iou()` — single-pair subprocess IoU. Used by `InferenceScaling`.

All three paths use `geom.py` (CadQuery-based). Config flag `use_iou_server: bool = True` controls which path the loop uses.

#### `gift/training/` — Training Infrastructure (from CADRL)

Two training paths, one file each:

- `sft_trainer.py`: `SFTSyncedGrad` — Accelerate-based SFT trainer with FSDP, manual gradient accumulation, global token-count normalization, KL regularization (ref = M₀), WandB logging. Used by the **offline GIFT loop** (`run_gift_loop.py`): generate all candidates → build augmented dataset → retrain from checkpoint. This is the paper-faithful approach.
- `rl_trainer.py`: Two classes from two source files:
  - `REINFORCE` (from `Trainers/REINFORCE/`): vanilla policy gradient, loss = `-advantage * log_prob`, no clipping.
  - `DAPO` (from `Trainers/DAPO/`): importance-sampled policy gradient with asymmetric clipping (`epsilon_low`/`epsilon_high`), `mu` inner training steps per batch.
  - **GRPO is not a separate class** — it is `DAPO` with `mu=1` and `epsilon_high=epsilon_low` (symmetric clipping). This mapping is applied in `run_gift_rl.py`.
  - Both trainers sync weights to VLLM via PyNCCL after each step. Standardize on DAPO's `get_model_state_dict()` approach (uses FSDP2 API with CPU offload, explicit cache clearing).
  - Both trainers disable gradient checkpointing on frozen vision encoder to avoid recomputation issues under FSDP.
- `collators.py`: `QwenVLCollator` (SFT path — tokenizes chat prompts, masks non-assistant tokens) + `QwenVLRLCollator` (RL path — generates via VLLM, computes IoU reward via `IOUClient`, per-group reward normalization, builds RL training batch). Also include `NoDaemonProcess`/`NoDaemonContext` utilities (required for RL dataloader — the collator spawns child processes for IoU, and standard multiprocessing forbids daemonic processes from having children).

#### `gift/data/` — Data Handling (from CADRL)
- `datasets.py`: `MinimalImageCADDataset` (SFT path) + `CADRLDataset` (RL path — wraps prompts for VLLM generation with IoU reward).
- `utils.py`: `extract_code()` (regex extraction from markdown), `encode_image()` (PIL→base64).

### 3.3 Demo — Gradio Web UI

`demo/app.py` with three tabs:

**Tab 1: Single Image Inference**
- Upload an image of a 3D CAD model
- Generate K candidates with configurable temperature
- Display: input image, generated CadQuery code, IoU score, server-side rendered 2D image of 3D reconstruction
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
| `Inference/Geom/_IOU.py` | `gift/inference/geom.py` | Keep `align_shapes`, `_compute_iou_centering`, `_compute_iou_centering_normalize`. This is the ONLY IoU backend (CadQuery-based). Used by both `processors.py` and `iou_server.py`. |
| `Inference/processors.py` | `gift/inference/processors.py` | Keep `IOUProcessor` class and helper functions (`_run_gt_code`, `_run_gen_code`, `_compute_iou`). Update `PATH_TO_CAD_PYTHON` to be configurable. **Import path rewrite only** (already uses CadQuery-based `_IOU.py`): `from Inference.Geom._IOU import ...` → `from gift.inference.geom import ...`. |
| `Inference/iou_serve.py` | `gift/inference/iou_server.py` | Keep server endpoints. **Backend switch + path rewrite**: uses `_OCC_IOU.py` (pythonocc, not installed) → adapt to use `geom.py` (CadQuery-based). Rewrite `from Geom._OCC_IOU import ...` → `from gift.inference.geom import ...`. Make host/port/workers configurable. |
| `Inference/__init__.py` | `gift/inference/__init__.py` | Keep `run_code()`, `compute_iou()`, `extract_code()`. Update `PATH_TO_CAD_PYTHON` to be configurable. Rewrite import paths. Drop `compute_iou_serve()` (use `IOUClient` instead). |
| `Inference/VLLMClient.py` | `gift/inference/vllm_client.py` | Keep full client including NCCL weight sync methods (`init_communicator`, `update_model_params`) — needed for the RL path. Keep `chat()`, `EZ_chat()`, `sleep()`/`wake_up()`. |
| `Inference/IOUClient.py` | `gift/inference/iou_client.py` | Keep `IOUClient` as-is. Async HTTP client with retry logic and semaphore-based concurrency. |
| `Trainers/SFT/__init__.py` | `gift/training/sft_trainer.py` | Keep `SFTSyncedGrad` only. Drop `SFT` class (redundant). Keep `selective_log_softmax`. |
| `Trainers/REINFORCE/__init__.py` | `gift/training/rl_trainer.py` | `REINFORCE` class: vanilla policy gradient (`-advantage * log_prob`, no clipping). |
| `Trainers/DAPO/__init__.py` | `gift/training/rl_trainer.py` | `DAPO` class: importance-sampled policy gradient with asymmetric clipping (`epsilon_low`/`epsilon_high`), inner training loops (`mu`). **GRPO is not a separate class** — it is `DAPO` with `mu=1` and `epsilon_high=epsilon_low` (symmetric clipping), applied in the entry point script. Standardize on DAPO's FSDP weight sync approach (`get_model_state_dict` with CPU offload). |
| `CADCoderREINFORCE.py` | `scripts/run_gift_rl.py` | Adapt as CLI entry point. Keep algorithm dispatch (REINFORCE vs DAPO class, GRPO param mapping), dataset variant loading (base/rft/gift-fda/rft+gift-fda/all), vision freezing, `NoDaemonContext` setup. |
| `DataUtils/Collators.py` | `gift/training/collators.py` | Keep both `QwenVLCollator` (SFT path) and `QwenVLRLCollator` (RL path — generates via VLLM, computes IoU reward). |
| `DataUtils/Datasets.py` | `gift/data/datasets.py` | Keep `MinimalImageCADDataset` (SFT) + `CADRLDataset` (RL). Drop `CADLMDataset`. |
| `DataUtils/Datasets.py` (lines 19-25) | `gift/data/utils.py` | Extract `extract_code()` regex helper. Add `encode_image()` (PIL→base64 for VLLM API). **Note:** `extract_code()` is duplicated in 3 CADRL files (`Datasets.py`, `Collators.py`, `Inference/__init__.py`). Centralize here; all other modules import from `gift.data.utils`. |
| `Inference/InferenceScaling/render_step_pyvista_simple.py` | `gift/core/renderer.py` | Reuse PyVista rendering pattern (off_screen, view_isometric, white bg, STL mesh). New entry point: `render_code_to_image(code_str)` — executes CadQuery code string in subprocess, exports to STL, renders under Xvfb, returns PIL Image. The upstream pipeline (code execution → solid → STL) is new; the rendering itself reuses the CADRL pattern. |

**Not copied:** `Inference/Geom/_OCC_IOU.py` (requires pythonocc, not installed), `Inference/InferenceScaling/render.py` (OCC Viewer3d, requires pythonocc), `Inference/InferenceScaling/render_step_headless.py` (VTK direct, superseded by PyVista path).

## 5. Rendering Pipeline (New)

The FDA rendering pipeline is a new component not present in CADRL's training loop. It uses PyVista for headless 3D→2D rendering.

### Environment Constraints

- **pythonocc (`OCC.Core.*`) is NOT installed** — OCC Viewer3d rendering is unavailable
- **PyVista 0.46.5 + VTK 9.3.1 are installed** — works for offscreen rendering
- **Xvfb is required** — VTK's `vtkXOpenGLRenderWindow` needs an X display; `xvfb-run` provides a virtual one
- Verified end-to-end: CadQuery → STL export → PyVista offscreen render → PNG ✓

### Approach for FDA
1. Execute CadQuery code in a subprocess (isolated for crash safety, same pattern as `IOUProcessor._run_gen_code()`)
2. Export the resulting solid to STL via `cq.exporters.export(solid, path)`
3. Render via PyVista under Xvfb: `pv.Plotter(off_screen=True)`, isometric view, white background
4. Return as PIL Image at 448×448 (matching training data resolution)

### Renderer Implementation

Single `renderer.py` based on `Inference/InferenceScaling/render_step_pyvista_simple.py` with a `render_code_to_image(code, size)` function that:
- Runs CadQuery code in a subprocess (with timeout, crash isolation)
- Exports to temporary STL file
- Launches PyVista under `xvfb-run` for headless rendering
- Returns PIL Image
- Cleans up temp files

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
│  IoU ≥ τ_accept              →  REJECT bucket                  │
│  τ_reject < IoU < τ_accept   →  FDA bucket → Renderer → pair   │
│  IoU ≤ τ_reject              →  DISCARD                        │
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
┌─── Trainer ───────────────────────────────┐
│  Mode A (SFT): Fine-tune Mₙ₋₁ on Dₙ     │
│  Mode B (RL):  REINFORCE/GRPO/DAPO       │
│                with IoU as reward         │
│  → Mₙ checkpoint                         │
└───────────────────────────────────────────┘
```

## 7. Dependencies

### Python packages (all installed in `.cad` environment)

```
# Core
torch>=2.0
transformers>=4.40
accelerate
datasets
vllm

# Geometry & IoU
cadquery==2.6.0          # CadQuery with OCP bindings — INSTALLED
                          # (pythonocc/OCC.Core is NOT installed and NOT needed)

# Rendering
pyvista==0.46.5          # INSTALLED — PyVista offscreen rendering
vtk==9.3.1               # INSTALLED — backend for PyVista

# Demo
gradio>=4.0

# Infrastructure
fastapi
uvicorn
uvloop
httpx                    # Async HTTP client for IOUClient
openai                   # OpenAI-compatible API client for VLLMClient
requests                 # Sync HTTP for VLLMClient health checks
qwen_vl_utils            # Vision processing for QwenVLCollator
aiohttp
tqdm
wandb
```

### System dependencies

```
xorg-x11-server-Xvfb     # INSTALLED — required for headless PyVista rendering
                          # All rendering subprocesses must run under `xvfb-run -a`
```

### NOT available (and not needed)

- `pythonocc-core` / `OCC.Core.*` — not installed. We use CadQuery's native OCP bindings for IoU and CadQuery → STL → PyVista for rendering.
- `trimesh` — not installed, not needed.

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
