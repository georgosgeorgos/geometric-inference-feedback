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

_vllm_client = None
_iou_client = None


def _get_vllm(config):
    global _vllm_client
    if _vllm_client is None:
        from gift.inference.vllm_client import VLLMClient
        _vllm_client = VLLMClient(base_url=config.vllm_server_url)
    return _vllm_client


def _get_iou(config):
    global _iou_client
    if _iou_client is None:
        from gift.inference.iou_client import IOUClient
        _iou_client = IOUClient(server_url=config.iou_server_url)
    return _iou_client


def _build_prompt(image):
    """Build a VLLM-compatible prompt for image-to-CAD generation."""
    b64 = encode_image(image)
    return {
        "role": "user",
        "content": [
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}},
            {"type": "text", "text": "Generate the CADQuery code needed to create the CAD for the provided image."},
        ],
    }


def create_demo():
    config = GIFTConfig()

    # --- Tab 1: Single Image Inference ---
    def run_inference(image, gt_code, k, temperature):
        if image is None:
            return "No image provided", None, None, None
        from gift.core.renderer import render_code_to_image

        vllm = _get_vllm(config)
        prompt = _build_prompt(image)
        vllm.set_sampling_params(temperature=temperature, max_tokens=4096, n=int(k))
        responses = vllm.EZ_chat([prompt])
        codes = [extract_code(r) for r in responses]

        # If ground truth provided, compute IoU and rank
        if gt_code and gt_code.strip():
            import asyncio
            iou_client = _get_iou(config)
            jobs = [{"ground_truth": gt_code.strip(), "generated": c} for c in codes]
            ious, _ = asyncio.run(iou_client.main(jobs))
            ious = ious.tolist()
            ranked = sorted(enumerate(codes), key=lambda x: ious[x[0]], reverse=True)
        else:
            ious = None
            ranked = list(enumerate(codes))

        # Render top-3
        gallery_images = []
        best_code = None
        best_render = None
        for rank, (orig_idx, code) in enumerate(ranked[:3]):
            rendered = render_code_to_image(code)
            iou_str = f" (IoU: {ious[orig_idx]:.3f})" if ious else ""
            label = f"#{rank + 1}{iou_str}"
            if rendered:
                gallery_images.append((rendered, label))
            if rank == 0:
                best_code = code
                best_render = rendered

        iou_summary = ""
        if ious:
            iou_summary = f"Best IoU: {max(ious):.3f} | Mean IoU: {np.mean(ious):.3f} | "
        summary = f"{iou_summary}Generated {len(codes)} candidates, rendered top {len(gallery_images)}"

        return best_code, best_render, gallery_images, summary

    # --- Tab 2: GIFT Iteration ---
    def run_iteration(n_samples, tau_accept, tau_reject, k, progress=gr.Progress()):
        import asyncio
        from datasets import load_dataset

        data = load_dataset(config.dataset, split="train")
        n_samples = min(int(n_samples), len(data))
        data = data.select(range(n_samples))
        K = int(k)

        vllm = _get_vllm(config)
        iou_client = _get_iou(config)

        # Batch all prompts in one call
        progress(0.1, desc="Building prompts...")
        all_prompts = []
        for idx in range(n_samples):
            prompt = _build_prompt(data[idx]["image"])
            all_prompts.extend([prompt] * K)

        progress(0.2, desc=f"Generating {len(all_prompts)} candidates...")
        vllm.set_sampling_params(temperature=config.temperature, max_tokens=4096, n=1)
        all_responses = vllm.EZ_chat(all_prompts)
        all_codes = [extract_code(r) for r in all_responses]

        # Batch IoU
        progress(0.6, desc="Evaluating IoU...")
        all_jobs = []
        for idx in range(n_samples):
            gt_code = data[idx]["code"]
            for ki in range(K):
                all_jobs.append({"ground_truth": gt_code, "generated": all_codes[idx * K + ki]})
        all_ious_flat, _ = asyncio.run(iou_client.main(all_jobs))

        # Classify per-sample
        progress(0.9, desc="Classifying...")
        all_results = []
        for idx in range(n_samples):
            start = idx * K
            ious = all_ious_flat[start:start + K].tolist()
            accepted, fda, discarded = classify_candidates(ious, tau_accept, tau_reject)
            best_iou = max(ious) if ious else 0
            valid_ious = [v for v in ious if v >= 0]
            all_results.append({
                "image_idx": idx,
                "n_accepted": len(accepted),
                "n_fda": len(fda),
                "n_discarded": len(discarded),
                "best_iou": round(best_iou, 4),
                "mean_iou": round(np.mean(valid_ious), 4) if valid_ious else 0,
            })

        total = len(all_results)
        total_k = total * K
        summary = {
            "samples": total,
            "candidates_per_sample": K,
            "accept_rate": round(sum(r["n_accepted"] for r in all_results) / max(total_k, 1), 4),
            "fda_rate": round(sum(r["n_fda"] for r in all_results) / max(total_k, 1), 4),
            "discard_rate": round(sum(r["n_discarded"] for r in all_results) / max(total_k, 1), 4),
            "mean_best_iou": round(float(np.mean([r["best_iou"] for r in all_results])), 4),
            "mean_iou": round(float(np.mean([r["mean_iou"] for r in all_results])), 4),
        }
        return json.dumps(summary, indent=2), json.dumps(all_results[:20], indent=2)

    # --- Tab 3: History ---
    def load_history(metrics_path):
        if not os.path.exists(metrics_path):
            return "No metrics file found", None
        with open(metrics_path) as f:
            metrics = json.load(f)

        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(8, 5))
        iters = [m["iteration"] for m in metrics]
        mean_ious = [m.get("mean_iou", 0) for m in metrics]
        sizes = [m.get("dataset_size", 0) for m in metrics]

        ax.plot(iters, mean_ious, "b-o", label="Mean IoU", linewidth=2)
        ax.set_xlabel("Iteration")
        ax.set_ylabel("Mean IoU", color="b")
        ax.tick_params(axis="y", labelcolor="b")
        ax.set_ylim(0, 1)

        ax2 = ax.twinx()
        ax2.bar(iters, sizes, alpha=0.3, color="gray", label="Dataset Size")
        ax2.set_ylabel("Dataset Size", color="gray")
        ax2.tick_params(axis="y", labelcolor="gray")

        ax.set_title("GIFT Loop Progression")
        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2, loc="lower right")
        fig.tight_layout()

        return json.dumps(metrics, indent=2), fig

    # --- Build UI ---
    with gr.Blocks(title="GIFT Demo") as demo:
        gr.Markdown("# GIFT: Geometric Inference Feedback Tuning")

        with gr.Tab("Single Inference"):
            with gr.Row():
                input_image = gr.Image(type="pil", label="Input Image")
                with gr.Column():
                    gt_code_input = gr.Textbox(label="Ground Truth Code (optional, for IoU ranking)", lines=5)
                    k_slider = gr.Slider(1, 32, value=16, step=1, label="K (candidates)")
                    temp_slider = gr.Slider(0.1, 1.5, value=0.6, step=0.1, label="Temperature")
                    run_btn = gr.Button("Generate", variant="primary")
            with gr.Row():
                output_code = gr.Textbox(label="Best Code", lines=15)
                output_render = gr.Image(type="pil", label="Best Rendered Output")
            output_gallery = gr.Gallery(label="Top-3 Candidates", columns=3, height=300)
            output_summary = gr.Textbox(label="Summary")
            run_btn.click(run_inference, [input_image, gt_code_input, k_slider, temp_slider],
                         [output_code, output_render, output_gallery, output_summary])

        with gr.Tab("GIFT Iteration"):
            with gr.Row():
                n_samples = gr.Slider(1, 100, value=20, step=1, label="Samples")
                tau_a = gr.Slider(0.5, 1.0, value=0.9, step=0.05, label="tau_accept")
                tau_r = gr.Slider(0.0, 0.9, value=0.5, step=0.05, label="tau_reject")
                k_iter = gr.Slider(1, 32, value=16, step=1, label="K")
            iter_btn = gr.Button("Run Iteration", variant="primary")
            iter_summary = gr.Textbox(label="Summary", lines=10)
            iter_details = gr.Textbox(label="Per-sample Details (first 20)", lines=15)
            iter_btn.click(run_iteration, [n_samples, tau_a, tau_r, k_iter],
                          [iter_summary, iter_details])

        with gr.Tab("History"):
            metrics_path = gr.Textbox(value="./output/gift_metrics.json", label="Metrics File")
            load_btn = gr.Button("Load", variant="primary")
            history_text = gr.Textbox(label="Metrics JSON", lines=20)
            history_plot = gr.Plot(label="IoU Progression")
            load_btn.click(load_history, [metrics_path], [history_text, history_plot])

    return demo


if __name__ == "__main__":
    demo = create_demo()
    demo.launch(server_name="0.0.0.0", server_port=7860)
