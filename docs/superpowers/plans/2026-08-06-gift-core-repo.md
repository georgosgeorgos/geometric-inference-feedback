# GIFT Core Repo Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the complete GIFT bootstrapping loop codebase by copying infrastructure from CADRL and adding GIFT-specific orchestration, FDA rendering, and a Gradio demo.

**Architecture:** Copy 15 files from CADRL with import path rewrites, then build 7 new GIFT-specific modules (config, reject, fda, renderer, dataset_builder, loop, demo). Two training paths: offline SFT via `run_gift_loop.py` and online RL via `run_gift_rl.py`.

**Tech Stack:** PyTorch, Transformers, Accelerate, VLLM, CadQuery 2.6, PyVista, FastAPI, Gradio

**Spec:** `docs/superpowers/specs/2026-08-06-gift-core-repo-design.md`

**Python env:** `uv run --project /workspace/home/lab/gcg/dev/.cad python`

**CADRL source:** `/workspace/home/lab/gcg/dev/CADRL/`

---

## File Map

| File | Source | Action |
|------|--------|--------|
| `gift/__init__.py` | New | Create (empty) |
| `gift/config.py` | New | Create |
| `gift/data/__init__.py` | New | Create (empty) |
| `gift/data/utils.py` | CADRL `DataUtils/Datasets.py` lines 19-25 + new | Extract + extend |
| `gift/data/datasets.py` | CADRL `DataUtils/Datasets.py` | Copy, drop `CADLMDataset`, rewrite imports |
| `gift/inference/__init__.py` | CADRL `Inference/__init__.py` | Copy, rewrite imports, configurable paths |
| `gift/inference/geom.py` | CADRL `Inference/Geom/_IOU.py` | Copy as-is |
| `gift/inference/processors.py` | CADRL `Inference/processors.py` | Copy, rewrite imports, configurable paths |
| `gift/inference/iou_server.py` | CADRL `Inference/iou_serve.py` | Copy, switch from `_OCC_IOU` to `geom` |
| `gift/inference/iou_client.py` | CADRL `Inference/IOUClient.py` | Copy as-is |
| `gift/inference/vllm_client.py` | CADRL `Inference/VLLMClient.py` | Copy as-is |
| `gift/training/__init__.py` | New | Create (empty) |
| `gift/training/collators.py` | CADRL `DataUtils/Collators.py` | Copy, rewrite `extract_code` import |
| `gift/training/sft_trainer.py` | CADRL `Trainers/SFT/__init__.py` | Copy, drop `SFT` class, keep `SFTSyncedGrad` |
| `gift/training/rl_trainer.py` | CADRL `Trainers/REINFORCE/` + `Trainers/DAPO/` | Merge both classes into one file |
| `gift/core/__init__.py` | New | Create (empty) |
| `gift/core/reject.py` | New | Create |
| `gift/core/fda.py` | New | Create |
| `gift/core/renderer.py` | CADRL `InferenceScaling/render_step_pyvista_simple.py` + new | Adapt pattern, new `render_code_to_image()` |
| `gift/core/dataset_builder.py` | New | Create |
| `gift/core/loop.py` | New | Create |
| `scripts/run_gift_loop.py` | New | Create |
| `scripts/run_gift_rl.py` | CADRL `CADCoderREINFORCE.py` | Adapt as CLI entry point |
| `scripts/run_iou_server.py` | New | Create (thin wrapper) |
| `scripts/run_vllm_server.py` | New | Create (thin wrapper) |
| `demo/app.py` | New | Create |
| `CLAUDE.md` | New | Create |
| `pyproject.toml` | New | Create |
| `requirements.txt` | New | Create |

---

## Task 1: Project Scaffolding

**Files:**
- Create: `CLAUDE.md`, `pyproject.toml`, `requirements.txt`
- Create: All `__init__.py` files

- [ ] **Step 1: Create directory structure and `__init__.py` files**

```bash
mkdir -p gift/{core,inference,training,data}
mkdir -p {scripts,demo,tests}
touch gift/__init__.py gift/core/__init__.py gift/inference/__init__.py.tmp gift/training/__init__.py gift/data/__init__.py
```

Note: `gift/inference/__init__.py` will be created in Task 4 (it has content). Create a temp placeholder.

- [ ] **Step 2: Create `CLAUDE.md`**

```markdown
# GIFT Project Instructions

## Python Environment

Use the uv environment at `/workspace/home/lab/gcg/dev/.cad` (Python 3.10).

- Run Python: `uv run --project /workspace/home/lab/gcg/dev/.cad python`
- Run scripts: `uv run --project /workspace/home/lab/gcg/dev/.cad python <script.py>`

## Overview

GIFT (Geometric Inference Feedback Tuning) bootstraps training data for Image-to-CAD via geometric feedback. Two training modes: offline SFT (`scripts/run_gift_loop.py`) and online RL (`scripts/run_gift_rl.py`).

## Key Paths

- Base model: `CADCODER/Qwen2.5-VL-7B-CadCoder`
- Dataset: `CADCODER/DeepCAD-CQ-Vision-Paired`
- Checkpoints: `/new_data/gcg-dev/cad/checkpoints/`

## Services

- IoU server: `python scripts/run_iou_server.py` (default: port 5000)
- VLLM server: `python scripts/run_vllm_server.py` (default: port 8000)
```

- [ ] **Step 3: Create `requirements.txt`**

```
torch>=2.0
transformers>=4.40
accelerate
datasets
vllm
cadquery==2.6.0
pyvista==0.46.5
vtk==9.3.1
gradio>=4.0
fastapi
uvicorn
uvloop
httpx
openai
requests
qwen_vl_utils
aiohttp
tqdm
wandb
numpy
tabulate
Pillow
```

- [ ] **Step 4: Create `pyproject.toml`**

```toml
[project]
name = "gift"
version = "0.1.0"
description = "GIFT: Geometric Inference Feedback Tuning for Image-to-CAD"
requires-python = ">=3.10"

[tool.pytest.ini_options]
testpaths = ["tests"]
```

- [ ] **Step 5: Commit**

```bash
git add -A
git commit -m "scaffold: project structure, CLAUDE.md, requirements"
```

---

## Task 2: Data Utilities

**Files:**
- Create: `gift/data/utils.py`
- Source: CADRL `DataUtils/Datasets.py` lines 19-25

- [ ] **Step 1: Write test for `extract_code` and `encode_image`**

Create `tests/test_data_utils.py`:

```python
from gift.data.utils import extract_code, encode_image
from PIL import Image
import base64

def test_extract_code_markdown():
    text = "Here is code:\n```python\nresult = cq.Workplane('XY').box(1,1,1)\nsolid = result\n```\nDone."
    assert "result = cq.Workplane" in extract_code(text)
    assert "solid = result" in extract_code(text)

def test_extract_code_no_block():
    assert extract_code("no code here") == "no code here"

def test_encode_image():
    img = Image.new("RGB", (10, 10), "red")
    b64 = encode_image(img)
    decoded = base64.b64decode(b64)
    assert len(decoded) > 0
```

- [ ] **Step 2: Run test to verify it fails**

```bash
uv run --project /workspace/home/lab/gcg/dev/.cad pytest tests/test_data_utils.py -v
```

Expected: FAIL (module not found)

- [ ] **Step 3: Implement `gift/data/utils.py`**

```python
import re
import base64
from io import BytesIO
from PIL import Image


def extract_code(text: str) -> str:
    """Extract Python code from markdown code blocks."""
    pattern = r"```(?:python)?\s*\n(.*?)```"
    matches = re.findall(pattern, text, re.DOTALL)
    return "\n".join(matches).strip() if matches else text.strip()


def encode_image(image: Image.Image, fmt: str = "PNG") -> str:
    """Encode PIL Image to base64 string."""
    buf = BytesIO()
    image.save(buf, format=fmt)
    return base64.b64encode(buf.getvalue()).decode("utf-8")
```

- [ ] **Step 4: Run test to verify it passes**

```bash
uv run --project /workspace/home/lab/gcg/dev/.cad pytest tests/test_data_utils.py -v
```

- [ ] **Step 5: Commit**

```bash
git add gift/data/utils.py tests/test_data_utils.py
git commit -m "feat: data utilities (extract_code, encode_image)"
```

---

## Task 3: Inference — Geometry (IoU)

**Files:**
- Create: `gift/inference/geom.py`
- Source: CADRL `Inference/Geom/_IOU.py` (102 lines)

- [ ] **Step 1: Copy `_IOU.py` to `gift/inference/geom.py`**

Copy the file as-is. No import changes needed — it only imports `cadquery`, `numpy`, and `typing`.

```bash
cp /workspace/home/lab/gcg/dev/CADRL/Inference/Geom/_IOU.py gift/inference/geom.py
```

- [ ] **Step 2: Verify import works**

```bash
uv run --project /workspace/home/lab/gcg/dev/.cad python -c "from gift.inference.geom import align_shapes, _compute_iou_centering, _compute_iou_centering_normalize; print('OK')"
```

- [ ] **Step 3: Commit**

```bash
git add gift/inference/geom.py
git commit -m "feat: copy IoU geometry module from CADRL"
```

---

## Task 4: Inference — Helpers (`__init__.py`)

**Files:**
- Create: `gift/inference/__init__.py`
- Source: CADRL `Inference/__init__.py` (209 lines)

- [ ] **Step 1: Copy and adapt**

Copy CADRL `Inference/__init__.py` to `gift/inference/__init__.py`. Make these changes:

1. Replace hardcoded `PATH_TO_CAD_PYTHON` with:
   ```python
   import shutil
   PATH_TO_CAD_PYTHON = os.environ.get(
       "CAD_PYTHON_PATH",
       shutil.which("python") or "/workspace/home/lab/gcg/dev/.cad/bin/python"
   )
   ```

2. In `OCC_IOU_TEMPLATE`, change:
   ```python
   # OLD: from Inference.Geom._IOU import align_shapes
   # NEW: from gift.inference.geom import align_shapes
   ```

3. Drop `compute_iou_serve()` function (replaced by `IOUClient`).

4. Change `extract_code` to import from centralized location:
   ```python
   from gift.data.utils import extract_code
   ```
   Remove the local `extract_code` definition. Re-export it for backward compat:
   ```python
   from gift.data.utils import extract_code  # noqa: F401
   ```

- [ ] **Step 2: Verify**

```bash
uv run --project /workspace/home/lab/gcg/dev/.cad python -c "from gift.inference import run_code, compute_iou, extract_code; print('OK')"
```

- [ ] **Step 3: Commit**

```bash
git add gift/inference/__init__.py
git commit -m "feat: inference helpers (run_code, compute_iou)"
```

---

## Task 5: Inference — Processors

**Files:**
- Create: `gift/inference/processors.py`
- Source: CADRL `Inference/processors.py` (370 lines)

- [ ] **Step 1: Copy and adapt**

Copy CADRL `Inference/processors.py` to `gift/inference/processors.py`. Changes:

1. Same `PATH_TO_CAD_PYTHON` fix as Task 4:
   ```python
   import shutil
   PATH_TO_CAD_PYTHON = os.environ.get(
       "CAD_PYTHON_PATH",
       shutil.which("python") or "/workspace/home/lab/gcg/dev/.cad/bin/python"
   )
   ```

2. In `OCC_IOU_TEMPLATE` and `_compute_iou()`, change import paths:
   ```python
   # OLD: from Inference.Geom._IOU import align_shapes
   # NEW: from gift.inference.geom import align_shapes
   ```
   Do the same for `_compute_iou_centering` and `_compute_iou_centering_normalize`.

3. In `_compute_iou()` function's `_iou_imports` dict, update all three entries:
   ```python
   _iou_imports = {
       "align": "from gift.inference.geom import align_shapes",
       "centering": "from gift.inference.geom import _compute_iou_centering",
       "centering_normalize": "from gift.inference.geom import _compute_iou_centering_normalize",
   }
   ```

- [ ] **Step 2: Verify import**

```bash
uv run --project /workspace/home/lab/gcg/dev/.cad python -c "from gift.inference.processors import IOUProcessor; print('OK')"
```

- [ ] **Step 3: Commit**

```bash
git add gift/inference/processors.py
git commit -m "feat: IoU processor with parallel computation"
```

---

## Task 6: Inference — IoU Server

**Files:**
- Create: `gift/inference/iou_server.py`
- Source: CADRL `Inference/iou_serve.py` (358 lines)

- [ ] **Step 1: Copy and adapt**

Copy CADRL `Inference/iou_serve.py` to `gift/inference/iou_server.py`. Changes:

1. **Template rewrite** — the CADRL server uses `_OCC_IOU.load_step_file()` (pythonocc, not installed). Replace with CadQuery's `importStep`, matching the pattern already used by `processors.py`:
   ```python
   # OLD (uses pythonocc load_step_file):
   OCC_IOU_TEMPLATE = """
   from Geom._OCC_IOU import align_shapes, load_step_file
   solid_gt = load_step_file("{ground_truth_step_path}")
   solid_gen = load_step_file("{generated_step_path}")
   IOU = align_shapes(solid_gen, solid_gt)[1]
   """

   # NEW (uses CadQuery importStep):
   OCC_IOU_TEMPLATE = """
   from gift.inference.geom import align_shapes
   import cadquery as cq
   solid_gt = cq.importers.importStep("{ground_truth_step_path}")
   solid_gen = cq.importers.importStep("{generated_step_path}")
   IOU = align_shapes(solid_gen, solid_gt)[1]
   """
   ```

2. In `_get_iou()` function: remove `load_step_file` import, use `cq.importers.importStep()` instead. Remove `load_step_file` from any `exec_globals` dict.

3. Update all remaining `_OCC_IOU` references to `gift.inference.geom`.

4. Make host/port/workers configurable via CLI args (verify `parse_args()` supports this).

- [ ] **Step 2: Create `scripts/run_iou_server.py`**

```python
"""Start the IoU computation server."""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from gift.inference.iou_server import create_app, parse_args
import uvicorn

if __name__ == "__main__":
    args = parse_args()
    app = create_app(num_workers=args.num_workers)
    uvicorn.run(app, host=args.host, port=args.port)
```

- [ ] **Step 3: Verify import**

```bash
uv run --project /workspace/home/lab/gcg/dev/.cad python -c "from gift.inference.iou_server import create_app; print('OK')"
```

- [ ] **Step 4: Commit**

```bash
git add gift/inference/iou_server.py scripts/run_iou_server.py
git commit -m "feat: IoU FastAPI server"
```

---

## Task 7: Inference — IoU Client

**Files:**
- Create: `gift/inference/iou_client.py`
- Source: CADRL `Inference/IOUClient.py` (94 lines)

- [ ] **Step 1: Copy as-is**

```bash
cp /workspace/home/lab/gcg/dev/CADRL/Inference/IOUClient.py gift/inference/iou_client.py
```

No changes needed — only imports `asyncio`, `httpx`, `numpy`, `tqdm`.

- [ ] **Step 2: Verify**

```bash
uv run --project /workspace/home/lab/gcg/dev/.cad python -c "from gift.inference.iou_client import IOUClient; print('OK')"
```

- [ ] **Step 3: Commit**

```bash
git add gift/inference/iou_client.py
git commit -m "feat: async IoU client"
```

---

## Task 8: Inference — VLLM Client

**Files:**
- Create: `gift/inference/vllm_client.py`
- Source: CADRL `Inference/VLLMClient.py` (400 lines)

- [ ] **Step 1: Copy as-is**

```bash
cp /workspace/home/lab/gcg/dev/CADRL/Inference/VLLMClient.py gift/inference/vllm_client.py
```

No internal import changes needed — uses `torch`, `openai`, `requests`, `vllm.distributed`, `asyncio`.

- [ ] **Step 2: Verify import**

```bash
uv run --project /workspace/home/lab/gcg/dev/.cad python -c "from gift.inference.vllm_client import VLLMClient; print('OK')"
```

- [ ] **Step 3: Commit**

```bash
git add gift/inference/vllm_client.py
git commit -m "feat: VLLM client with NCCL weight sync"
```

---

## Task 9: Data — Datasets

**Files:**
- Create: `gift/data/datasets.py`
- Source: CADRL `DataUtils/Datasets.py` (854 lines)

- [ ] **Step 1: Copy and adapt**

Copy CADRL `DataUtils/Datasets.py` to `gift/data/datasets.py`. Changes:

1. Remove `CADLMDataset` class entirely.
2. Remove the local `extract_code()` function definition.
3. Add import at top: `from gift.data.utils import extract_code`
4. Keep `MinimalImageCADDataset` and `CADRLDataset`.

- [ ] **Step 2: Verify**

```bash
uv run --project /workspace/home/lab/gcg/dev/.cad python -c "from gift.data.datasets import MinimalImageCADDataset, CADRLDataset; print('OK')"
```

- [ ] **Step 3: Commit**

```bash
git add gift/data/datasets.py
git commit -m "feat: SFT and RL dataset classes"
```

---

## Task 10: Training — Collators

**Files:**
- Create: `gift/training/collators.py`
- Source: CADRL `DataUtils/Collators.py` (234 lines)

- [ ] **Step 1: Copy and adapt**

Copy CADRL `DataUtils/Collators.py` to `gift/training/collators.py`. Changes:

1. Remove the local `extract_code()` definition.
2. Add: `from gift.data.utils import extract_code`
3. Keep: `QwenVLCollator`, `QwenVLRLCollator`, `NoDaemonProcess`, `NoDaemonContext`.

- [ ] **Step 2: Verify**

```bash
uv run --project /workspace/home/lab/gcg/dev/.cad python -c "from gift.training.collators import QwenVLCollator, QwenVLRLCollator, NoDaemonContext; print('OK')"
```

- [ ] **Step 3: Commit**

```bash
git add gift/training/collators.py
git commit -m "feat: SFT and RL collators with NoDaemonContext"
```

---

## Task 11: Training — SFT Trainer

**Files:**
- Create: `gift/training/sft_trainer.py`
- Source: CADRL `Trainers/SFT/__init__.py` (1480 lines)

- [ ] **Step 1: Copy and adapt**

Copy CADRL `Trainers/SFT/__init__.py` to `gift/training/sft_trainer.py`. Changes:

1. Remove the `SFT` class entirely (keep only `SFTSyncedGrad`).
2. Keep `selective_log_softmax` function.
3. No import path changes needed — only uses external packages.

- [ ] **Step 2: Verify**

```bash
uv run --project /workspace/home/lab/gcg/dev/.cad python -c "from gift.training.sft_trainer import SFTSyncedGrad, selective_log_softmax; print('OK')"
```

- [ ] **Step 3: Commit**

```bash
git add gift/training/sft_trainer.py
git commit -m "feat: SFTSyncedGrad trainer"
```

---

## Task 12: Training — RL Trainer

**Files:**
- Create: `gift/training/rl_trainer.py`
- Source: CADRL `Trainers/REINFORCE/__init__.py` (468 lines) + `Trainers/DAPO/__init__.py` (677 lines)

- [ ] **Step 1: Merge both files**

Create `gift/training/rl_trainer.py` by combining both source files:

1. Copy `REINFORCE` class from `Trainers/REINFORCE/__init__.py`.
2. Copy `DAPO` class from `Trainers/DAPO/__init__.py`.
3. Keep only one copy of `selective_log_softmax` (shared by both).
4. No import path changes needed.

Structure:
```python
# gift/training/rl_trainer.py
"""REINFORCE and DAPO trainers for online RL."""

# ... shared imports ...

def selective_log_softmax(logits, index):
    ...

class REINFORCE:
    ...

class DAPO:
    ...
```

- [ ] **Step 2: Verify**

```bash
uv run --project /workspace/home/lab/gcg/dev/.cad python -c "from gift.training.rl_trainer import REINFORCE, DAPO; print('OK')"
```

- [ ] **Step 3: Commit**

```bash
git add gift/training/rl_trainer.py
git commit -m "feat: REINFORCE and DAPO RL trainers"
```

---

## Task 13: GIFT Config

**Files:**
- Create: `gift/config.py`

- [ ] **Step 1: Write test**

Create `tests/test_config.py`:

```python
from gift.config import GIFTConfig

def test_defaults():
    cfg = GIFTConfig()
    assert cfg.tau_accept == 0.9
    assert cfg.tau_reject == 0.5
    assert cfg.candidates_per_image == 16
    assert cfg.max_iterations == 5

def test_override():
    cfg = GIFTConfig(tau_accept=0.8, candidates_per_image=8)
    assert cfg.tau_accept == 0.8
    assert cfg.candidates_per_image == 8
```

- [ ] **Step 2: Run test — verify failure**

```bash
uv run --project /workspace/home/lab/gcg/dev/.cad pytest tests/test_config.py -v
```

- [ ] **Step 3: Implement `gift/config.py`**

```python
from dataclasses import dataclass, field


@dataclass
class GIFTConfig:
    """All configuration for the GIFT bootstrapping loop."""

    # Model
    base_model: str = "CADCODER/Qwen2.5-VL-7B-CadCoder"
    dataset: str = "CADCODER/DeepCAD-CQ-Vision-Paired"

    # GIFT loop
    tau_accept: float = 0.9
    tau_reject: float = 0.5
    candidates_per_image: int = 16
    max_iterations: int = 5
    oversample_failures: bool = True
    oversample_factor: int = 3

    # Inference
    temperature: float = 0.6
    top_p: float = 0.9
    max_tokens: int = 4096

    # Training
    training_mode: str = "sft"  # sft | reinforce | grpo | dapo
    lr: float = 1e-5
    num_epochs: int = 1
    batch_size: int = 4
    use_kl: bool = True
    kl_weight: float = 0.04
    freeze_vision: bool = True

    # Infrastructure
    vllm_server_url: str = "http://localhost:8000"
    iou_server_url: str = "http://localhost:5000"
    iou_workers: int = 64
    use_iou_server: bool = True
    render_size: int = 448

    # Paths
    output_dir: str = "./output"
    checkpoint_dir: str = "./checkpoints"
```

- [ ] **Step 4: Run test — verify pass**

```bash
uv run --project /workspace/home/lab/gcg/dev/.cad pytest tests/test_config.py -v
```

- [ ] **Step 5: Commit**

```bash
git add gift/config.py tests/test_config.py
git commit -m "feat: GIFTConfig dataclass"
```

---

## Task 14: Core — GIFT-REJECT

**Files:**
- Create: `gift/core/reject.py`

- [ ] **Step 1: Write test**

Create `tests/test_reject.py`:

```python
from gift.core.reject import classify_candidates, build_reject_pairs

def test_classify_basic():
    ious = [0.95, 0.7, 0.3, 0.91, 0.45]
    accepted, fda, discarded = classify_candidates(ious, tau_accept=0.9, tau_reject=0.5)
    assert accepted == [0, 3]       # indices with IoU >= 0.9
    assert fda == [1]               # indices with 0.5 < IoU < 0.9
    assert discarded == [2, 4]      # indices with IoU <= 0.5

def test_classify_boundary():
    # Exact boundary values
    ious = [0.9, 0.5]
    accepted, fda, discarded = classify_candidates(ious, tau_accept=0.9, tau_reject=0.5)
    assert accepted == [0]          # IoU >= tau_accept
    assert fda == []
    assert discarded == [1]         # IoU <= tau_reject

def test_build_reject_pairs():
    accepted_indices = [0, 2]
    images = ["img_a", "img_b", "img_c"]
    codes = ["code_a", "code_b", "code_c"]
    pairs = build_reject_pairs(accepted_indices, images, codes)
    assert pairs == [("img_a", "code_a"), ("img_c", "code_c")]
```

- [ ] **Step 2: Run test — verify failure**

```bash
uv run --project /workspace/home/lab/gcg/dev/.cad pytest tests/test_reject.py -v
```

- [ ] **Step 3: Implement `gift/core/reject.py`**

```python
"""GIFT-REJECT: classify candidates by IoU thresholds."""
from typing import List, Tuple, Any


def classify_candidates(
    ious: List[float],
    tau_accept: float = 0.9,
    tau_reject: float = 0.5,
) -> Tuple[List[int], List[int], List[int]]:
    """Classify candidate indices into accepted, fda-eligible, and discarded.

    Returns (accepted_indices, fda_indices, discarded_indices).
    """
    accepted, fda, discarded = [], [], []
    for i, iou in enumerate(ious):
        if iou >= tau_accept:
            accepted.append(i)
        elif iou > tau_reject:
            fda.append(i)
        else:
            discarded.append(i)
    return accepted, fda, discarded


def build_reject_pairs(
    accepted_indices: List[int],
    images: List[Any],
    codes: List[str],
) -> List[Tuple[Any, str]]:
    """Build (image, generated_code) training pairs from accepted candidates."""
    return [(images[i], codes[i]) for i in accepted_indices]
```

- [ ] **Step 4: Run test — verify pass**

```bash
uv run --project /workspace/home/lab/gcg/dev/.cad pytest tests/test_reject.py -v
```

- [ ] **Step 5: Commit**

```bash
git add gift/core/reject.py tests/test_reject.py
git commit -m "feat: GIFT-REJECT candidate classification"
```

---

## Task 15: Core — Renderer

**Files:**
- Create: `gift/core/renderer.py`
- Reference: CADRL `Inference/InferenceScaling/render_step_pyvista_simple.py`

- [ ] **Step 1: Implement `gift/core/renderer.py`**

This is a new module that adapts the PyVista rendering pattern. The key function `render_code_to_image()` runs CadQuery code in a subprocess, exports STL, and renders via PyVista under Xvfb.

```python
"""3D-to-2D rendering via CadQuery + PyVista + Xvfb."""
import os
import subprocess
import tempfile
import shutil
from typing import Optional, Tuple
from PIL import Image

CAD_PYTHON = os.environ.get(
    "CAD_PYTHON_PATH",
    shutil.which("python") or "/workspace/home/lab/gcg/dev/.cad/bin/python",
)


def render_code_to_image(
    code: str,
    size: Tuple[int, int] = (448, 448),
    timeout: int = 30,
) -> Optional[Image.Image]:
    """Execute CadQuery code, render the result to a 2D image.

    Returns PIL Image or None on failure.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        stl_path = os.path.join(tmpdir, "output.stl")
        png_path = os.path.join(tmpdir, "output.png")

        # Step 1: Execute CadQuery code → STL
        if not _execute_to_stl(code, stl_path, timeout):
            return None

        # Step 2: Render STL → PNG via PyVista under Xvfb
        if not _render_stl_to_png(stl_path, png_path, size, timeout):
            return None

        return Image.open(png_path).copy()


def _execute_to_stl(code: str, stl_path: str, timeout: int) -> bool:
    """Run CadQuery code in subprocess, export to STL."""
    script = f"""
import cadquery as cq
{code}
cq.exporters.export(solid, r"{stl_path}")
"""
    try:
        result = subprocess.run(
            [CAD_PYTHON, "-c", script],
            capture_output=True, text=True, timeout=timeout,
        )
        return result.returncode == 0 and os.path.exists(stl_path)
    except (subprocess.TimeoutExpired, Exception):
        return False


def _render_stl_to_png(
    stl_path: str, png_path: str, size: Tuple[int, int], timeout: int
) -> bool:
    """Render STL file to PNG via PyVista offscreen under Xvfb."""
    script = f"""
import pyvista as pv
pv.OFF_SCREEN = True
mesh = pv.read(r"{stl_path}")
plotter = pv.Plotter(off_screen=True, window_size={list(size)})
plotter.set_background("white")
plotter.add_mesh(mesh, color="steelblue", smooth_shading=True)
plotter.view_isometric()
plotter.screenshot(r"{png_path}")
plotter.close()
"""
    try:
        result = subprocess.run(
            ["xvfb-run", "-a", CAD_PYTHON, "-c", script],
            capture_output=True, text=True, timeout=timeout,
        )
        return result.returncode == 0 and os.path.exists(png_path)
    except (subprocess.TimeoutExpired, Exception):
        return False


def render_stl_to_image(
    stl_path: str,
    size: Tuple[int, int] = (448, 448),
    timeout: int = 30,
) -> Optional[Image.Image]:
    """Render an existing STL file to image. Convenience wrapper."""
    with tempfile.TemporaryDirectory() as tmpdir:
        png_path = os.path.join(tmpdir, "output.png")
        if not _render_stl_to_png(stl_path, png_path, size, timeout):
            return None
        return Image.open(png_path).copy()
```

- [ ] **Step 2: Write smoke test**

Create `tests/test_renderer.py`:

```python
import pytest
from gift.core.renderer import render_code_to_image

def test_render_simple_box():
    code = "solid = cq.Workplane('XY').box(10, 10, 10)"
    img = render_code_to_image(code, size=(224, 224), timeout=30)
    assert img is not None
    assert img.size == (224, 224)

def test_render_bad_code():
    img = render_code_to_image("raise ValueError('bad')", timeout=10)
    assert img is None
```

- [ ] **Step 3: Run test**

```bash
uv run --project /workspace/home/lab/gcg/dev/.cad pytest tests/test_renderer.py -v --timeout=60
```

- [ ] **Step 4: Commit**

```bash
git add gift/core/renderer.py tests/test_renderer.py
git commit -m "feat: 3D-to-2D renderer (CadQuery + PyVista + Xvfb)"
```

---

## Task 16: Core — GIFT-FDA

**Files:**
- Create: `gift/core/fda.py`

- [ ] **Step 1: Write test**

Create `tests/test_fda.py`:

```python
from unittest.mock import patch, MagicMock
from PIL import Image
from gift.core.fda import build_fda_pairs

def test_build_fda_pairs_with_mock_renderer():
    fake_img = Image.new("RGB", (448, 448), "blue")
    fda_indices = [1, 3]
    gen_codes = ["code_0", "code_1", "code_2", "code_3"]
    gt_codes = ["gt_0", "gt_1", "gt_2", "gt_3"]

    with patch("gift.core.fda.render_code_to_image") as mock_render:
        mock_render.return_value = fake_img
        pairs = build_fda_pairs(fda_indices, gen_codes, gt_codes)

    assert len(pairs) == 2
    assert pairs[0] == (fake_img, "gt_1")
    assert pairs[1] == (fake_img, "gt_3")

def test_build_fda_pairs_render_failure():
    with patch("gift.core.fda.render_code_to_image") as mock_render:
        mock_render.return_value = None  # render fails
        pairs = build_fda_pairs([0], ["bad_code"], ["gt_0"])

    assert len(pairs) == 0  # failed renders are skipped
```

- [ ] **Step 2: Run test — verify failure**

```bash
uv run --project /workspace/home/lab/gcg/dev/.cad pytest tests/test_fda.py -v
```

- [ ] **Step 3: Implement `gift/core/fda.py`**

```python
"""GIFT-FDA: failure-driven augmentation via 3D rendering."""
from typing import List, Tuple, Any, Optional
from PIL import Image
from gift.core.renderer import render_code_to_image


def build_fda_pairs(
    fda_indices: List[int],
    gen_codes: List[str],
    gt_codes: List[str],
    render_size: int = 448,
) -> List[Tuple[Image.Image, str]]:
    """Render near-miss generated code to 2D, pair with ground truth code.

    For each FDA-eligible candidate:
    1. Execute generated code → 3D solid → render to 2D image
    2. Pair rendered image with ground truth code

    Skips candidates where rendering fails.
    """
    pairs = []
    for idx in fda_indices:
        rendered = render_code_to_image(
            gen_codes[idx], size=(render_size, render_size)
        )
        if rendered is not None:
            pairs.append((rendered, gt_codes[idx]))
    return pairs
```

- [ ] **Step 4: Run test — verify pass**

```bash
uv run --project /workspace/home/lab/gcg/dev/.cad pytest tests/test_fda.py -v
```

- [ ] **Step 5: Commit**

```bash
git add gift/core/fda.py tests/test_fda.py
git commit -m "feat: GIFT-FDA failure-driven augmentation"
```

---

## Task 17: Core — Dataset Builder

**Files:**
- Create: `gift/core/dataset_builder.py`

- [ ] **Step 1: Write test**

Create `tests/test_dataset_builder.py`:

```python
from PIL import Image
from gift.core.dataset_builder import build_augmented_dataset

def test_build_augmented_dataset():
    base_data = [
        {"image": Image.new("RGB", (10, 10)), "code": "base_code_0"},
        {"image": Image.new("RGB", (10, 10)), "code": "base_code_1"},
    ]
    reject_pairs = [(Image.new("RGB", (10, 10)), "reject_code")]
    fda_pairs = [(Image.new("RGB", (10, 10)), "fda_code")]
    oversample_pairs = [(Image.new("RGB", (10, 10)), "oversample_code")] * 2

    ds = build_augmented_dataset(base_data, reject_pairs, fda_pairs, oversample_pairs)
    assert len(ds) == 2 + 1 + 1 + 2  # base + reject + fda + oversample
    assert "image" in ds.column_names
    assert "code" in ds.column_names
    assert "source" in ds.column_names

def test_empty_augmentation():
    base_data = [{"image": Image.new("RGB", (10, 10)), "code": "c"}]
    ds = build_augmented_dataset(base_data, [], [], [])
    assert len(ds) == 1
```

- [ ] **Step 2: Run test — verify failure**

```bash
uv run --project /workspace/home/lab/gcg/dev/.cad pytest tests/test_dataset_builder.py -v
```

- [ ] **Step 3: Implement `gift/core/dataset_builder.py`**

```python
"""Build augmented HuggingFace datasets from GIFT loop outputs."""
import os
from typing import List, Tuple, Any, Dict
from PIL import Image
from datasets import Dataset


def build_augmented_dataset(
    base_data: List[Dict],
    reject_pairs: List[Tuple[Image.Image, str]],
    fda_pairs: List[Tuple[Image.Image, str]],
    oversample_pairs: List[Tuple[Image.Image, str]],
) -> Dataset:
    """Combine base data with GIFT-generated pairs into a HuggingFace Dataset."""
    images, codes, sources = [], [], []

    for item in base_data:
        images.append(item["image"])
        codes.append(item["code"])
        sources.append("base")

    for img, code in reject_pairs:
        images.append(img)
        codes.append(code)
        sources.append("reject")

    for img, code in fda_pairs:
        images.append(img)
        codes.append(code)
        sources.append("fda")

    for img, code in oversample_pairs:
        images.append(img)
        codes.append(code)
        sources.append("oversample")

    return Dataset.from_dict({"image": images, "code": codes, "source": sources})


def save_dataset(dataset: Dataset, output_dir: str, iteration: int) -> str:
    """Save dataset as Arrow files, versioned by iteration."""
    path = os.path.join(output_dir, f"iteration_{iteration}")
    os.makedirs(path, exist_ok=True)
    dataset.save_to_disk(path)
    return path
```

- [ ] **Step 4: Run test — verify pass**

```bash
uv run --project /workspace/home/lab/gcg/dev/.cad pytest tests/test_dataset_builder.py -v
```

- [ ] **Step 5: Commit**

```bash
git add gift/core/dataset_builder.py tests/test_dataset_builder.py
git commit -m "feat: augmented dataset builder"
```

---

## Task 18: Core — GIFT Loop Orchestrator

**Files:**
- Create: `gift/core/loop.py`

- [ ] **Step 1: Write test**

Create `tests/test_loop.py`:

```python
from gift.core.loop import GIFTLoop
from gift.config import GIFTConfig

def test_loop_init():
    cfg = GIFTConfig(max_iterations=2)
    loop = GIFTLoop(cfg)
    assert loop.config.max_iterations == 2
    assert loop.iteration == 0
```

Full integration test deferred — requires running VLLM and IoU servers.

- [ ] **Step 2: Run test — verify failure**

```bash
uv run --project /workspace/home/lab/gcg/dev/.cad pytest tests/test_loop.py -v
```

- [ ] **Step 3: Implement `gift/core/loop.py`**

```python
"""GIFT bootstrapping loop orchestrator."""
import os
import json
import logging
from typing import Optional, Dict, List
from datasets import load_dataset
from gift.config import GIFTConfig
from gift.core.reject import classify_candidates, build_reject_pairs
from gift.core.fda import build_fda_pairs
from gift.core.dataset_builder import build_augmented_dataset, save_dataset

logger = logging.getLogger(__name__)


class GIFTLoop:
    """Runs the full GIFT bootstrapping loop (offline SFT mode)."""

    def __init__(self, config: GIFTConfig):
        self.config = config
        self.iteration = 0
        self.metrics: List[Dict] = []

    def run(self, start_iteration: int = 0):
        """Run all iterations of the GIFT loop."""
        self.iteration = start_iteration
        for n in range(start_iteration, self.config.max_iterations):
            logger.info(f"=== GIFT Iteration {n + 1}/{self.config.max_iterations} ===")
            metrics = self.run_iteration(n)
            self.metrics.append(metrics)
            self._save_metrics()
            self.iteration = n + 1

    def run_iteration(self, n: int) -> Dict:
        """Run a single GIFT iteration: generate → evaluate → classify → build → retrain."""
        from gift.inference.vllm_client import VLLMClient
        from gift.inference.iou_client import IOUClient
        import asyncio

        # Load dataset
        data = load_dataset(self.config.dataset, split="train")

        # Init clients
        vllm = VLLMClient(base_url=self.config.vllm_server_url)
        iou_client = IOUClient(server_url=self.config.iou_server_url)

        # Step 1: Generate K candidates per image
        logger.info(f"Generating {self.config.candidates_per_image} candidates per image...")
        all_images, all_gt_codes, all_gen_codes, all_ious = [], [], [], []

        for idx in range(len(data)):
            sample = data[idx]
            image = sample["image"]
            gt_code = sample["code"]

            prompts = [self._build_prompt(image)] * self.config.candidates_per_image
            responses = vllm.EZ_chat(prompts)
            from gift.data.utils import extract_code
            gen_codes = [extract_code(r) for r in responses]

            # Step 2: Evaluate via IoU
            jobs = [
                {"ground_truth": gt_code, "generated": gc}
                for gc in gen_codes
            ]
            ious, statuses = asyncio.run(iou_client.main(jobs))

            all_images.append(image)
            all_gt_codes.append(gt_code)
            all_gen_codes.append(gen_codes)
            all_ious.append(ious.tolist())

        # Step 3: Classify
        reject_pairs, fda_pairs, oversample_pairs = [], [], []
        n_accepted, n_fda, n_discarded = 0, 0, 0

        for i in range(len(all_images)):
            accepted, fda, discarded = classify_candidates(
                all_ious[i], self.config.tau_accept, self.config.tau_reject,
            )
            n_accepted += len(accepted)
            n_fda += len(fda)
            n_discarded += len(discarded)

            # REJECT pairs
            reject_pairs.extend(
                build_reject_pairs(accepted, [all_images[i]] * len(all_gen_codes[i]), all_gen_codes[i])
            )

            # FDA pairs
            fda_pairs.extend(
                build_fda_pairs(fda, all_gen_codes[i], [all_gt_codes[i]] * len(all_gen_codes[i]),
                                render_size=self.config.render_size)
            )

            # Oversample: when ALL candidates are discarded
            if len(accepted) == 0 and len(fda) == 0 and self.config.oversample_failures:
                oversample_pairs.extend(
                    [(all_images[i], all_gt_codes[i])] * self.config.oversample_factor
                )

        # Step 4: Build augmented dataset
        base_data = [{"image": data[i]["image"], "code": data[i]["code"]} for i in range(len(data))]
        augmented = build_augmented_dataset(base_data, reject_pairs, fda_pairs, oversample_pairs)
        dataset_path = save_dataset(augmented, self.config.output_dir, n)

        # Step 5: Retrain via SFT
        logger.info(f"Retraining on augmented dataset ({len(augmented)} samples)...")
        self._retrain(augmented, n)

        metrics = {
            "iteration": n,
            "n_accepted": n_accepted,
            "n_fda": n_fda,
            "n_discarded": n_discarded,
            "n_oversample": len(oversample_pairs),
            "dataset_size": len(augmented),
            "mean_iou": sum(sum(ious) for ious in all_ious) / max(sum(len(ious) for ious in all_ious), 1),
        }
        logger.info(f"Iteration {n} metrics: {json.dumps(metrics, indent=2)}")
        return metrics

    def _retrain(self, augmented_dataset, iteration: int):
        """Fine-tune the model on the augmented dataset using SFTSyncedGrad."""
        from transformers import AutoModelForImageTextToText, AutoProcessor
        from gift.training.sft_trainer import SFTSyncedGrad
        from gift.training.collators import QwenVLCollator
        from gift.data.datasets import MinimalImageCADDataset

        # Load model (from previous checkpoint or base)
        ckpt_dir = self.config.checkpoint_dir
        if iteration > 0:
            prev_ckpt = os.path.join(ckpt_dir, f"iteration_{iteration - 1}")
            model_path = prev_ckpt if os.path.exists(prev_ckpt) else self.config.base_model
        else:
            model_path = self.config.base_model

        processor = AutoProcessor.from_pretrained(self.config.base_model)
        model = AutoModelForImageTextToText.from_pretrained(model_path)

        if self.config.freeze_vision:
            for attr in ("vision_model", "visual"):
                if hasattr(model, attr):
                    getattr(model, attr).requires_grad_(False)
                    break

        collate_fn = QwenVLCollator(processor, assistant_only=True)
        dataset = MinimalImageCADDataset(augmented_dataset, collate_fn=collate_fn)

        trainer = SFTSyncedGrad(
            model=model,
            lr=self.config.lr,
            max_steps=len(dataset) // self.config.batch_size * self.config.num_epochs,
            ref_kl=self.config.use_kl,
            ref_kl_weight=self.config.kl_weight,
        )

        trainer.train(
            dataset=dataset,
            batch_size=self.config.batch_size,
            epochs=self.config.num_epochs,
        )

        # Save checkpoint
        save_path = os.path.join(ckpt_dir, f"iteration_{iteration}")
        os.makedirs(save_path, exist_ok=True)
        model.save_pretrained(save_path)
        processor.save_pretrained(save_path)
        logger.info(f"Saved checkpoint to {save_path}")

    def _build_prompt(self, image) -> dict:
        """Build a VLLM-compatible prompt for image-to-CAD generation."""
        from gift.data.utils import encode_image
        return {
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{encode_image(image)}"}},
                {"type": "text", "text": "Generate the CADQuery code needed to create the CAD for the provided image."},
            ],
        }

    def _save_metrics(self):
        """Save accumulated metrics to JSON."""
        os.makedirs(self.config.output_dir, exist_ok=True)
        path = os.path.join(self.config.output_dir, "gift_metrics.json")
        with open(path, "w") as f:
            json.dump(self.metrics, f, indent=2)
```

- [ ] **Step 4: Run test — verify pass**

```bash
uv run --project /workspace/home/lab/gcg/dev/.cad pytest tests/test_loop.py -v
```

- [ ] **Step 5: Commit**

```bash
git add gift/core/loop.py tests/test_loop.py
git commit -m "feat: GIFT loop orchestrator"
```

---

## Task 19: Scripts — Entry Points

**Files:**
- Create: `scripts/run_gift_loop.py`
- Create: `scripts/run_gift_rl.py` (from CADRL `CADCoderREINFORCE.py`)
- Create: `scripts/run_vllm_server.py`

- [ ] **Step 1: Create `scripts/run_gift_loop.py`**

```python
"""Run the full GIFT bootstrapping loop (offline SFT mode)."""
import argparse
import logging
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from gift.config import GIFTConfig
from gift.core.loop import GIFTLoop

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


def main():
    parser = argparse.ArgumentParser(description="GIFT bootstrapping loop (offline SFT)")
    parser.add_argument("--base_model", default="CADCODER/Qwen2.5-VL-7B-CadCoder")
    parser.add_argument("--dataset", default="CADCODER/DeepCAD-CQ-Vision-Paired")
    parser.add_argument("--tau_accept", type=float, default=0.9)
    parser.add_argument("--tau_reject", type=float, default=0.5)
    parser.add_argument("--candidates_per_image", type=int, default=16)
    parser.add_argument("--max_iterations", type=int, default=5)
    parser.add_argument("--oversample_factor", type=int, default=3)
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--vllm_server_url", default="http://localhost:8000")
    parser.add_argument("--iou_server_url", default="http://localhost:5000")
    parser.add_argument("--output_dir", default="./output")
    parser.add_argument("--start_iteration", type=int, default=0)
    args = parser.parse_args()

    config = GIFTConfig(**{k: v for k, v in vars(args).items() if k != "start_iteration"})
    loop = GIFTLoop(config)
    loop.run(start_iteration=args.start_iteration)


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Create `scripts/run_gift_rl.py`**

Copy CADRL `CADCoderREINFORCE.py` and adapt:

1. Add `sys.path.insert(0, ...)` at top for GIFT imports.
2. Change all imports:
   - `from DataUtils.Collators import ...` → `from gift.training.collators import ...`
   - `from DataUtils.Datasets import ...` → `from gift.data.datasets import ...`
   - `from Inference.VLLMClient import ...` → `from gift.inference.vllm_client import ...`
   - `from Inference.IOUClient import ...` → `from gift.inference.iou_client import ...`
   - `from Trainers.REINFORCE import REINFORCE` → `from gift.training.rl_trainer import REINFORCE`
   - `from Trainers.DAPO import DAPO` → `from gift.training.rl_trainer import DAPO`
3. Keep all argument parsing and logic as-is.

- [ ] **Step 3: Create `scripts/run_vllm_server.py`**

```python
"""Start VLLM server for GIFT inference."""
import argparse
import subprocess
import sys

def main():
    parser = argparse.ArgumentParser(description="Start VLLM server")
    parser.add_argument("--model", default="CADCODER/Qwen2.5-VL-7B-CadCoder")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--tensor_parallel_size", type=int, default=1)
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.9)
    args = parser.parse_args()

    cmd = [
        sys.executable, "-m", "vllm.entrypoints.openai.api_server",
        "--model", args.model,
        "--port", str(args.port),
        "--tensor-parallel-size", str(args.tensor_parallel_size),
        "--gpu-memory-utilization", str(args.gpu_memory_utilization),
        "--trust-remote-code",
    ]
    subprocess.run(cmd)


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Verify script imports**

```bash
uv run --project /workspace/home/lab/gcg/dev/.cad python -c "
import sys; sys.path.insert(0, '.')
from gift.config import GIFTConfig
from gift.core.loop import GIFTLoop
print('run_gift_loop imports OK')
"
```

- [ ] **Step 5: Commit**

```bash
git add scripts/
git commit -m "feat: CLI entry points (gift loop, RL, IoU server, VLLM server)"
```

---

## Task 20: Demo — Gradio App

**Files:**
- Create: `demo/app.py`

- [ ] **Step 1: Implement `demo/app.py`**

```python
"""GIFT demo — Gradio web UI with 3 tabs."""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import gradio as gr
import numpy as np
from PIL import Image
from gift.config import GIFTConfig
from gift.data.utils import extract_code, encode_image
from gift.core.reject import classify_candidates
from gift.core.renderer import render_code_to_image


def create_demo():
    config = GIFTConfig()

    # --- Tab 1: Single Image Inference ---
    def run_inference(image, k, temperature):
        if image is None:
            return "No image provided", None, None
        from gift.inference.vllm_client import VLLMClient

        vllm = VLLMClient(base_url=config.vllm_server_url)
        b64 = encode_image(image)
        prompt = {
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}},
                {"type": "text", "text": "Generate the CADQuery code needed to create the CAD for the provided image."},
            ],
        }

        vllm.set_sampling_params(temperature=temperature, max_tokens=4096, n=int(k))
        responses = vllm.EZ_chat([prompt])
        codes = [extract_code(r) for r in responses]

        # Render best candidate
        results = []
        for i, code in enumerate(codes):
            rendered = render_code_to_image(code)
            results.append({
                "index": i,
                "code": code,
                "rendered": rendered,
            })

        best = results[0] if results else None
        best_code = best["code"] if best else "No results"
        best_render = best["rendered"] if best else None
        summary = f"Generated {len(codes)} candidates"
        return best_code, best_render, summary

    # --- Tab 2: GIFT Iteration ---
    def run_iteration(n_samples, tau_accept, tau_reject, k):
        from gift.inference.vllm_client import VLLMClient
        from gift.inference.iou_client import IOUClient
        from datasets import load_dataset
        import asyncio

        data = load_dataset(config.dataset, split="train")
        n_samples = min(int(n_samples), len(data))
        data = data.select(range(n_samples))

        vllm = VLLMClient(base_url=config.vllm_server_url)
        iou_client = IOUClient(server_url=config.iou_server_url)

        all_results = []
        for idx in range(n_samples):
            sample = data[idx]
            image = sample["image"]
            gt_code = sample["code"]

            b64 = encode_image(image)
            prompt = {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}},
                    {"type": "text", "text": "Generate the CADQuery code needed to create the CAD for the provided image."},
                ],
            }

            vllm.set_sampling_params(temperature=config.temperature, max_tokens=4096, n=int(k))
            responses = vllm.EZ_chat([prompt])
            codes = [extract_code(r) for r in responses]

            jobs = [{"ground_truth": gt_code, "generated": c} for c in codes]
            ious, _ = asyncio.run(iou_client.main(jobs))

            accepted, fda, discarded = classify_candidates(
                ious.tolist(), tau_accept, tau_reject
            )
            all_results.append({
                "image_idx": idx,
                "n_accepted": len(accepted),
                "n_fda": len(fda),
                "n_discarded": len(discarded),
                "best_iou": float(max(ious)) if len(ious) > 0 else 0,
                "mean_iou": float(np.mean(ious[ious >= 0])) if np.any(ious >= 0) else 0,
            })

        total = len(all_results)
        summary = {
            "samples": total,
            "accept_rate": sum(r["n_accepted"] for r in all_results) / max(total * int(k), 1),
            "fda_rate": sum(r["n_fda"] for r in all_results) / max(total * int(k), 1),
            "discard_rate": sum(r["n_discarded"] for r in all_results) / max(total * int(k), 1),
            "mean_best_iou": np.mean([r["best_iou"] for r in all_results]),
        }
        return json.dumps(summary, indent=2), json.dumps(all_results[:10], indent=2)

    # --- Tab 3: History ---
    def load_history(metrics_path):
        if not os.path.exists(metrics_path):
            return "No metrics file found", None
        with open(metrics_path) as f:
            metrics = json.load(f)

        text = json.dumps(metrics, indent=2)
        return text, None

    # --- Build UI ---
    with gr.Blocks(title="GIFT Demo") as demo:
        gr.Markdown("# GIFT: Geometric Inference Feedback Tuning")

        with gr.Tab("Single Inference"):
            with gr.Row():
                input_image = gr.Image(type="pil", label="Input Image")
                with gr.Column():
                    k_slider = gr.Slider(1, 32, value=16, step=1, label="K (candidates)")
                    temp_slider = gr.Slider(0.1, 1.5, value=0.6, step=0.1, label="Temperature")
                    run_btn = gr.Button("Generate")
            with gr.Row():
                output_code = gr.Textbox(label="Best Code", lines=15)
                output_render = gr.Image(type="pil", label="Rendered Output")
            output_summary = gr.Textbox(label="Summary")
            run_btn.click(run_inference, [input_image, k_slider, temp_slider],
                         [output_code, output_render, output_summary])

        with gr.Tab("GIFT Iteration"):
            with gr.Row():
                n_samples = gr.Slider(1, 100, value=20, step=1, label="Samples")
                tau_a = gr.Slider(0.5, 1.0, value=0.9, step=0.05, label="τ_accept")
                tau_r = gr.Slider(0.0, 0.9, value=0.5, step=0.05, label="τ_reject")
                k_iter = gr.Slider(1, 32, value=16, step=1, label="K")
            iter_btn = gr.Button("Run Iteration")
            iter_summary = gr.Textbox(label="Summary", lines=10)
            iter_details = gr.Textbox(label="Per-sample Details", lines=15)
            iter_btn.click(run_iteration, [n_samples, tau_a, tau_r, k_iter],
                          [iter_summary, iter_details])

        with gr.Tab("History"):
            metrics_path = gr.Textbox(value="./output/gift_metrics.json", label="Metrics File")
            load_btn = gr.Button("Load")
            history_text = gr.Textbox(label="Metrics", lines=20)
            history_plot = gr.Plot(label="IoU Progression")
            load_btn.click(load_history, [metrics_path], [history_text, history_plot])

    return demo


if __name__ == "__main__":
    demo = create_demo()
    demo.launch(server_name="0.0.0.0", server_port=7860)
```

- [ ] **Step 2: Verify import**

```bash
uv run --project /workspace/home/lab/gcg/dev/.cad python -c "
import sys; sys.path.insert(0, '.')
from demo.app import create_demo
print('Demo imports OK')
"
```

- [ ] **Step 3: Commit**

```bash
git add demo/app.py
git commit -m "feat: Gradio demo with 3 tabs"
```

---

## Task 21: Final Integration Check

- [ ] **Step 1: Run all tests**

```bash
uv run --project /workspace/home/lab/gcg/dev/.cad pytest tests/ -v --timeout=120
```

- [ ] **Step 2: Verify all imports**

```bash
uv run --project /workspace/home/lab/gcg/dev/.cad python -c "
# Core
from gift.config import GIFTConfig
from gift.core.reject import classify_candidates, build_reject_pairs
from gift.core.fda import build_fda_pairs
from gift.core.renderer import render_code_to_image
from gift.core.dataset_builder import build_augmented_dataset
from gift.core.loop import GIFTLoop

# Data
from gift.data.utils import extract_code, encode_image
from gift.data.datasets import MinimalImageCADDataset, CADRLDataset

# Inference
from gift.inference import run_code, compute_iou
from gift.inference.geom import align_shapes
from gift.inference.processors import IOUProcessor
from gift.inference.iou_client import IOUClient
from gift.inference.vllm_client import VLLMClient

# Training
from gift.training.collators import QwenVLCollator, QwenVLRLCollator, NoDaemonContext
from gift.training.sft_trainer import SFTSyncedGrad
from gift.training.rl_trainer import REINFORCE, DAPO

print('All imports OK')
"
```

- [ ] **Step 3: Final commit if any fixes needed**

```bash
git add -A
git status
# Only commit if there are changes
```
