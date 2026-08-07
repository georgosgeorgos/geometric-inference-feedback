import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

os.environ["TOKENIZERS_PARALLELISM"] = "false"
import multiprocessing as mp
mp.set_start_method('forkserver', force=True)
import torch
from gift.training.collators import QwenVLRLCollator, NoDaemonContext
from gift.data.datasets import CADRLDataset
from gift.inference.vllm_client import VLLMClient
from gift.inference.iou_client import IOUClient
from transformers import AutoModelForImageTextToText, AutoProcessor
from datasets import load_dataset
import argparse
torch.set_float32_matmul_precision('high')

args = argparse.ArgumentParser(description="RL finetuning of CAD-Coder (REINFORCE / GRPO / DAPO)")

# Algorithm selection
args.add_argument("--algorithm", type=str, default="reinforce", choices=["reinforce", "grpo", "dapo"],
                  help="RL algorithm to use. reinforce=vanilla policy gradient, grpo=clipped with single inner step, dapo=full DAPO. Default: reinforce")

# Model and Data
args.add_argument("--base_model", type=str, default="CADCODER/Qwen2.5-VL-7B-CadCoder")
args.add_argument("--dataset", type=str, default="CADCODER/DeepCAD-CQ-Vision-Paired")
args.add_argument("--dataset_variant", type=str, default="base",
                  choices=["base", "rft", "gift-fda", "rft+gift-fda", "all"],
                  help="Dataset variant: base=original train, rft=RFT filtered (IoU>0.9), "
                       "gift-fda=GIFT-FDA balanced, rft+gift-fda=both RFT variants, "
                       "all=original + rft + gift-fda interleaved. Default: base")
args.add_argument("--dataset_base_path", type=str,
                  default="/home/lab/gcg/dev/CADRL/Inference/InferenceScaling",
                  help="Base path for local arrow datasets (RFT/GIFT-FDA variants). Default: InferenceScaling dir")
args.add_argument("--max_samples", type=int, default=None,
                  help="Subsample the dataset to this many prompts. Default: None (use all)")
args.add_argument("--assistant_only", action="store_true")
args.add_argument("--max_len", type=int, default=4096)
args.add_argument("--system_prompt", type=str, default="You are a helpful assistant")
args.add_argument("--user_prompt", type=str, default="Generate the CADQuery code needed to create the CAD for the provided image.")

# Trainer
args.add_argument("--lr", type=float, default=1e-5)
args.add_argument("--weight_decay", type=float, default=0.0)
args.add_argument("--schedule_type", type=str, default="constant_with_warmup")
args.add_argument("--lr_final", type=float, default=1e-6)
args.add_argument("--warmup_steps", type=int, default=100)
args.add_argument("--resume", action="store_true")
args.add_argument("--resume_path", type=str, default=None)
args.add_argument("--optimizer_type", type=str, default="AdamW")
args.add_argument("--gradient_checkpointing", action="store_true")
args.add_argument("--max_grad_norm", type=float, default=1.0)
args.add_argument("--use_kl", action="store_true")
args.add_argument("--beta", type=float, default=0.04)
args.add_argument("--freeze_vision", action="store_true")
args.add_argument("--per_gpu_batch_size", type=int, default=2)
args.add_argument("--throughput_batch_size", type=int, default=4)
args.add_argument("--num_epochs", type=int, default=1)
args.add_argument("--num_workers", type=int, default=4)
args.add_argument("--prefetch_factor", type=int, default=1)
args.add_argument("--max_train_steps", type=int, default=None,
                  help="Maximum number of training steps (overrides epochs). Default: None (train full epochs)")
args.add_argument("--checkpoint_steps", type=int, default=50)
args.add_argument("--checkpoint_dir", type=str, default=None)
args.add_argument("--n_last_checkpoints", type=int, default=3)
args.add_argument("--keep_old_checkpoints", action="store_true")

# DAPO/GRPO specific
args.add_argument("--mu", type=int, default=1, help="Inner training steps (DAPO only, GRPO uses 1). Default: 1")
args.add_argument("--epsilon_high", type=float, default=0.28, help="High clipping epsilon (DAPO). Default: 0.28")
args.add_argument("--epsilon_low", type=float, default=0.2, help="Low clipping epsilon (DAPO/GRPO). Default: 0.2")

# VLLM server
args.add_argument("--server_url", default="http://localhost:8000", type=str)
args.add_argument("--model_name", type=str, default=None)
args.add_argument("--group_port", type=int, default=51216,
                  help="Port for NCCL weight sync between trainer and VLLM. Must be unique per concurrent experiment.")

# Reward server
args.add_argument("--reward_server_url", type=str, default="http://localhost:5000")
args.add_argument("--iou_timeout", type=int, default=30)

# Generation
args.add_argument("--n_samples", type=int, default=16, help="Group size: number of samples per prompt. Default: 16")
args.add_argument("--temperature", type=float, default=0.5)
args.add_argument("--top_p", type=float, default=0.8)
args.add_argument("--max_new_tokens", type=int, default=4096)

# Wandb
args.add_argument("--log_to_wandb", action="store_true")
args.add_argument("--wandb_project", type=str, default="RL-CADCoder")
args.add_argument("--wandb_name", type=str, default=None)
args.add_argument("--log_gradient_norms", action="store_true")

args.add_argument("--reward_type", type=str, default="iou", choices=["iou", "syntax"])

args = args.parse_args()


def main():
    # Auto-set defaults based on algorithm
    if args.checkpoint_dir is None:
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.checkpoint_dir = f"/new_data/gcg-dev/cad/checkpoints/{args.algorithm}-baseline_{timestamp}"

    if args.wandb_name is None:
        args.wandb_name = f"{args.algorithm.upper()}_Experiment"

    if args.resume_path is None and args.resume:
        args.resume_path = os.path.join(args.checkpoint_dir, "last")

    # For GRPO: force mu=1, symmetric clipping
    if args.algorithm == "grpo":
        args.mu = 1
        args.epsilon_high = args.epsilon_low  # symmetric clipping

    print(f"Algorithm: {args.algorithm.upper()}")
    print(f"Group size (n_samples): {args.n_samples}")
    print(f"Reward type: {args.reward_type}")

    processor = AutoProcessor.from_pretrained(args.base_model)
    model = AutoModelForImageTextToText.from_pretrained(args.base_model)

    if args.freeze_vision:
        print("Freezing vision model parameters...")
        for attr in ("vision_model", "vision_encoder", "vision_tower", "vision", "visual"):
            if hasattr(model, attr):
                getattr(model, attr).requires_grad_(False)
                break

    iou_client = IOUClient(server_url=args.reward_server_url, timeout=args.iou_timeout)
    vllm_client = VLLMClient(base_url=args.server_url, model_name=args.model_name, group_port=args.group_port)

    collate_fn = QwenVLRLCollator(
        processor,
        vllm_client=vllm_client,
        iou_client=iou_client,
        max_batch_size=args.throughput_batch_size,
        n=args.n_samples,
        max_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        assistant_only=args.assistant_only,
        system_prompt=args.system_prompt,
        user_prompt=args.user_prompt,
        max_len=args.max_len,
        reward_type=args.reward_type,
    )

    # Load dataset based on variant
    base_path = args.dataset_base_path
    dataset_name = args.dataset.replace("/", "/")  # e.g. CADCODER/DeepCAD-CQ-Vision-Paired

    if args.dataset_variant == "base":
        data = load_dataset(args.dataset, split="train")
        print(f"Loaded base dataset: {len(data)} samples")

    elif args.dataset_variant == "rft":
        data = load_dataset(
            "arrow",
            data_files=f"{base_path}/{dataset_name}-RFT/train-rft/*.arrow",
            split="train", keep_in_memory=False,
        )
        print(f"Loaded RFT dataset (IoU>0.9): {len(data)} samples")

    elif args.dataset_variant == "gift-fda":
        data = load_dataset(
            "arrow",
            data_files=f"{base_path}/{dataset_name}-GIFT-FDA-Balanced/train-gift-fda-balanced/*.arrow",
            split="train", keep_in_memory=False,
        )
        print(f"Loaded GIFT-FDA balanced dataset: {len(data)} samples")

    elif args.dataset_variant == "rft+gift-fda":
        from datasets import interleave_datasets
        data_rft = load_dataset(
            "arrow",
            data_files=f"{base_path}/{dataset_name}-RFT/train-rft/*.arrow",
            split="train", keep_in_memory=False,
        )
        data_gift = load_dataset(
            "arrow",
            data_files=f"{base_path}/{dataset_name}-GIFT-FDA-Balanced/train-gift-fda-balanced/*.arrow",
            split="train", keep_in_memory=False,
        )
        data_gift = data_gift.cast(data_rft.features)
        total = len(data_rft) + len(data_gift)
        data = interleave_datasets(
            [data_rft, data_gift],
            probabilities=[len(data_rft) / total, len(data_gift) / total],
            seed=42, stopping_strategy="all_exhausted",
        )
        print(f"Loaded RFT+GIFT-FDA interleaved: {len(data)} samples (rft={len(data_rft)}, gift={len(data_gift)})")

    elif args.dataset_variant == "all":
        from datasets import interleave_datasets
        data_base = load_dataset(
            "arrow",
            data_files=f"{base_path}/{dataset_name}-RFT/train/*.arrow",
            split="train", keep_in_memory=False,
        )
        data_rft = load_dataset(
            "arrow",
            data_files=f"{base_path}/{dataset_name}-RFT/train-rft/*.arrow",
            split="train", keep_in_memory=False,
        )
        data_gift = load_dataset(
            "arrow",
            data_files=f"{base_path}/{dataset_name}-GIFT-FDA-Balanced/train-gift-fda-balanced/*.arrow",
            split="train", keep_in_memory=False,
        )
        data_rft = data_rft.cast(data_base.features)
        data_gift = data_gift.cast(data_base.features)
        total = len(data_base) + len(data_rft) + len(data_gift)
        data = interleave_datasets(
            [data_base, data_rft, data_gift],
            probabilities=[len(data_base) / total, len(data_rft) / total, len(data_gift) / total],
            seed=42, stopping_strategy="all_exhausted",
        )
        print(f"Loaded all interleaved: {len(data)} samples (base={len(data_base)}, rft={len(data_rft)}, gift={len(data_gift)})")

    # Subsample dataset if requested
    if args.max_samples is not None and args.max_samples < len(data):
        data = data.shuffle(seed=42).select(range(args.max_samples))
        print(f"Subsampled to {len(data)} prompts")

    dataset = CADRLDataset(
        data,
        collate_fn=collate_fn,
        system_prompt=args.system_prompt,
        user_prompt=args.user_prompt,
    )

    max_steps = (len(dataset) + args.throughput_batch_size - 1) // args.throughput_batch_size * args.num_epochs

    # Handle checkpoint resume
    resume_path = None
    if args.resume and args.resume_path:
        if "/last" in args.resume_path:
            ckpt_dir = args.resume_path.replace("/last", "/")
            if os.path.isdir(ckpt_dir):
                checkpoints = [f for f in os.listdir(ckpt_dir) if f.endswith(".pt")]
                if checkpoints:
                    checkpoints.sort(key=lambda x: os.path.getmtime(os.path.join(ckpt_dir, x)), reverse=True)
                    resume_path = os.path.join(ckpt_dir, checkpoints[0])
                    print(f"Resuming from: {resume_path}")
        else:
            resume_path = args.resume_path

    # Build trainer based on algorithm
    if args.algorithm == "reinforce":
        from gift.training.rl_trainer import REINFORCE
        trainer = REINFORCE(
            model=model,
            vllm_client=vllm_client,
            lr=args.lr,
            weight_decay=args.weight_decay,
            schedule_type=args.schedule_type,
            lr_final=args.lr_final,
            max_steps=max_steps,
            warmup_steps=args.warmup_steps,
            checkpoint_path=resume_path,
            optimizer_type=args.optimizer_type,
            gradient_checkpointing=args.gradient_checkpointing,
            max_grad_norm=args.max_grad_norm,
            scale_scheduler_by_porc_count=True,
            temperature=args.temperature,
            log_to_wandb=args.log_to_wandb,
            wandb_project=args.wandb_project,
            wandb_run_name=args.wandb_name,
            ref_kl=args.use_kl,
            beta=args.beta,
        )
    else:
        # GRPO and DAPO both use the DAPO trainer
        from gift.training.rl_trainer import DAPO
        trainer = DAPO(
            model=model,
            vllm_client=vllm_client,
            lr=args.lr,
            weight_decay=args.weight_decay,
            schedule_type=args.schedule_type,
            lr_final=args.lr_final,
            max_steps=max_steps,
            warmup_steps=args.warmup_steps,
            checkpoint_path=resume_path,
            optimizer_type=args.optimizer_type,
            gradient_checkpointing=args.gradient_checkpointing,
            max_grad_norm=args.max_grad_norm,
            mu=args.mu,
            epsilon_high=args.epsilon_high,
            epsilon_low=args.epsilon_low,
            scale_scheduler_by_porc_count=True,
            temperature=args.temperature,
            log_to_wandb=args.log_to_wandb,
            wandb_project=args.wandb_project,
            wandb_run_name=args.wandb_name,
            ref_kl=args.use_kl,
            beta=args.beta,
            log_gradient_norms=args.log_gradient_norms,
        )

    ctx = NoDaemonContext() if args.num_workers > 0 else None

    trainer.train(
        dataset=dataset,
        batch_size=args.per_gpu_batch_size,
        epochs=args.num_epochs,
        checkpoint_steps=args.checkpoint_steps,
        checkpoint_dir=args.checkpoint_dir,
        verbose=True,
        n_total_checkpoints_to_keep=args.n_last_checkpoints,
        remove_old_checkpoints=not args.keep_old_checkpoints,
        dataloader_num_workers=args.num_workers,
        dataloader_prefetch_factor=args.prefetch_factor if args.num_workers > 0 else None,
        dataloader_multiprocessing_context=ctx,
        max_steps=args.max_train_steps,
    )


if __name__ == "__main__":
    main()
