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
