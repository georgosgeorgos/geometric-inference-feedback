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

        results = []
        for i, code in enumerate(codes):
            rendered = render_code_to_image(code)
            results.append({"index": i, "code": code, "rendered": rendered})

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
        return json.dumps(metrics, indent=2), None

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
                tau_a = gr.Slider(0.5, 1.0, value=0.9, step=0.05, label="tau_accept")
                tau_r = gr.Slider(0.0, 0.9, value=0.5, step=0.05, label="tau_reject")
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
