"""GIFT bootstrapping loop orchestrator."""
import os
import json
import logging
from typing import Dict, List
from datasets import load_dataset
from gift.config import GIFTConfig
from gift.core.reject import classify_candidates, build_reject_pairs
from gift.core.fda import build_fda_pairs
from gift.core.dataset_builder import build_augmented_dataset, save_dataset
from gift.data.utils import extract_code

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
        data = load_dataset(self.config.dataset, split="train")
        for n in range(start_iteration, self.config.max_iterations):
            logger.info(f"=== GIFT Iteration {n + 1}/{self.config.max_iterations} ===")
            metrics = self.run_iteration(n, data)
            self.metrics.append(metrics)
            self._save_metrics()
            self.iteration = n + 1

    def run_iteration(self, n: int, data=None) -> Dict:
        """Run a single GIFT iteration: generate -> evaluate -> classify -> build -> retrain."""
        from gift.inference.vllm_client import VLLMClient
        from gift.inference.iou_client import IOUClient
        import asyncio

        if data is None:
            data = load_dataset(self.config.dataset, split="train")

        K = self.config.candidates_per_image
        N = len(data)
        vllm = VLLMClient(base_url=self.config.vllm_server_url)
        iou_client = IOUClient(server_url=self.config.iou_server_url)

        # Step 1: Batch-generate K candidates per image (one HTTP call)
        logger.info(f"Generating {K} candidates x {N} samples ({N * K} prompts)...")
        all_prompts = []
        for idx in range(N):
            prompt = self._build_prompt(data[idx]["image"])
            all_prompts.extend([prompt] * K)
        all_responses = vllm.EZ_chat(all_prompts)
        all_gen_codes = [extract_code(r) for r in all_responses]

        # Step 2: Batch IoU evaluation (one asyncio.run)
        logger.info(f"Evaluating IoU for {len(all_gen_codes)} candidates...")
        all_jobs = []
        for idx in range(N):
            gt_code = data[idx]["code"]
            for k in range(K):
                all_jobs.append({"ground_truth": gt_code, "generated": all_gen_codes[idx * K + k]})
        all_ious_flat, _ = asyncio.run(iou_client.main(all_jobs))

        # Reshape into per-sample IoU lists and gen_codes
        per_sample_ious = []
        per_sample_codes = []
        for idx in range(N):
            start = idx * K
            per_sample_ious.append(all_ious_flat[start:start + K].tolist())
            per_sample_codes.append(all_gen_codes[start:start + K])

        # Step 3: Classify
        reject_pairs, fda_pairs, oversample_pairs = [], [], []
        n_accepted, n_fda, n_discarded = 0, 0, 0

        for i in range(N):
            image = data[i]["image"]
            gt_code = data[i]["code"]
            accepted, fda, discarded = classify_candidates(
                per_sample_ious[i], self.config.tau_accept, self.config.tau_reject,
            )
            n_accepted += len(accepted)
            n_fda += len(fda)
            n_discarded += len(discarded)

            reject_pairs.extend(
                build_reject_pairs(accepted, [image] * len(per_sample_codes[i]), per_sample_codes[i])
            )
            fda_pairs.extend(
                build_fda_pairs(fda, per_sample_codes[i], [gt_code] * len(per_sample_codes[i]),
                                render_size=self.config.render_size)
            )
            if len(accepted) == 0 and len(fda) == 0 and self.config.oversample_failures:
                oversample_pairs.extend([(image, gt_code)] * self.config.oversample_factor)

        # Step 4: Build augmented dataset
        augmented = build_augmented_dataset(data, reject_pairs, fda_pairs, oversample_pairs)
        save_dataset(augmented, self.config.output_dir, n)

        # Step 5: Retrain via SFT
        logger.info(f"Retraining on augmented dataset ({len(augmented)} samples)...")
        self._retrain(augmented, n)

        total_ious = sum(sum(ious) for ious in per_sample_ious)
        total_count = sum(len(ious) for ious in per_sample_ious)
        metrics = {
            "iteration": n,
            "n_accepted": n_accepted,
            "n_fda": n_fda,
            "n_discarded": n_discarded,
            "n_oversample": len(oversample_pairs),
            "dataset_size": len(augmented),
            "mean_iou": total_ious / max(total_count, 1),
        }
        logger.info(f"Iteration {n} metrics: {json.dumps(metrics, indent=2)}")
        return metrics

    def _retrain(self, augmented_dataset, iteration: int):
        """Fine-tune the model on the augmented dataset using SFTSyncedGrad."""
        from transformers import AutoModelForImageTextToText, AutoProcessor
        from gift.training.sft_trainer import SFTSyncedGrad
        from gift.training.collators import QwenVLCollator
        from gift.data.datasets import MinimalImageCADDataset

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
