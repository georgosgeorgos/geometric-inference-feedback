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
