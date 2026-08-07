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
