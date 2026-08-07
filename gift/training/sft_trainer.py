import os
import torch
import numpy as np
from tqdm.auto import tqdm
from typing import Dict, List, Optional, Union, Any, Tuple
from transformers import (
    get_linear_schedule_with_warmup,
    get_cosine_schedule_with_warmup,
    get_constant_schedule_with_warmup,
)
from accelerate import Accelerator, DistributedDataParallelKwargs
from accelerate.utils import DistributedType
from copy import deepcopy
import torch.nn.functional as F
from accelerate.utils.fsdp_utils import fsdp2_prepare_model
import wandb
from datetime import datetime
import json


def selective_log_softmax(logits, index) -> torch.Tensor:
    """
    A memory-efficient implementation of the common `log_softmax -> gather` operation.

    This function is equivalent to the following naive implementation:
    ```python
    logps = torch.gather(logits.log_softmax(-1), dim=-1, index=index.unsqueeze(-1)).squeeze(-1)
    ```

    Args:
        logits (`torch.Tensor`):
            Logits tensor of shape `(..., num_classes)`.
        index (`torch.Tensor`):
            Index tensor of shape `(...)`, specifying the positions to gather from the log-softmax output.

    Returns:
        `torch.Tensor`:
            Gathered log probabilities with the same shape as `index`.
    """
    if logits.dtype in [torch.float32, torch.float64]:
        selected_logits = torch.gather(
            logits, dim=-1, index=index.unsqueeze(-1)
        ).squeeze(-1)
        # loop to reduce peak mem consumption
        logsumexp_values = torch.stack([torch.logsumexp(lg, dim=-1) for lg in logits])
        per_token_logps = (
            selected_logits - logsumexp_values
        )  # log_softmax(x_i) = x_i - logsumexp(x)
    else:
        # logsumexp approach is unstable with bfloat16, fall back to slightly less efficient approach
        per_token_logps = []
        for row_logits, row_labels in zip(
            logits, index
        ):  # loop to reduce peak mem consumption
            row_logps = F.log_softmax(row_logits, dim=-1)
            row_per_token_logps = row_logps.gather(
                dim=-1, index=row_labels.unsqueeze(-1)
            ).squeeze(-1)
            per_token_logps.append(row_per_token_logps)
        per_token_logps = torch.stack(per_token_logps)
    return per_token_logps


class SFTSyncedGrad:
    def __init__(
        self,
        model: torch.nn.Module,
        lr: float = 2e-5,
        weight_decay: float = 0.01,
        schedule_type: str = "cosine_with_warmup",
        lr_final: float = 1e-6,
        max_steps: int = 1000,
        warmup_steps: int = 100,
        compile: bool = False,
        compile_backend: Optional[str] = None,
        checkpoint_path: Optional[str] = None,
        optimizer_type: str = "AdamW",
        gradient_checkpointing: bool = True,
        max_grad_norm: float = 1.0,
        gradient_accumulation_steps: int = 1,
        custom_loss: Optional[callable] = None,
        ref_kl: bool = False,
        ref_kl_weight: float = 0.01,
        scale_scheduler_by_porc_count: bool = True,
        log_to_wandb: bool = False,
        wandb_project: str = "SFT-CADCoder",
        wandb_run_name: str = "SFT_Experiment",
        wandb_id: Optional[str] = None,
        set_to_none: bool = False,
    ):
        """
        Initialize the SFT trainer for transformer models.
        """
        self.optimizer_type = optimizer_type
        self.gradient_checkpointing = gradient_checkpointing
        self.max_grad_norm = max_grad_norm
        self.custom_loss = custom_loss
        self.ref_kl = ref_kl
        self.ref_kl_weight = ref_kl_weight
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.log_to_wandb = log_to_wandb
        self.wandb_project = wandb_project
        self.wandb_run_name = wandb_run_name
        self.wandb_id = wandb_id
        self.set_to_none = set_to_none

        # Store training configuration and start time
        self.training_start_time = datetime.now()
        self.training_config = {
            "lr": lr,
            "weight_decay": weight_decay,
            "schedule_type": schedule_type,
            "lr_final": lr_final,
            "max_steps": max_steps,
            "warmup_steps": warmup_steps,
            "compile": compile,
            "compile_backend": compile_backend,
            "optimizer_type": optimizer_type,
            "gradient_checkpointing": gradient_checkpointing,
            "max_grad_norm": max_grad_norm,
            "gradient_accumulation_steps": gradient_accumulation_steps,
            "ref_kl": ref_kl,
            "ref_kl_weight": ref_kl_weight,
            "scale_scheduler_by_porc_count": scale_scheduler_by_porc_count,
            "log_to_wandb": log_to_wandb,
            "wandb_project": wandb_project,
            "wandb_run_name": wandb_run_name,
            "set_to_none": set_to_none,
            "training_start_time": self.training_start_time.strftime("%Y-%m-%d_%H-%M-%S"),
        }

        # Initialize accelerator with gradient accumulation config
        self.accelerator = Accelerator(
            gradient_accumulation_steps=1,
            kwargs_handlers=[
                DistributedDataParallelKwargs(find_unused_parameters=True)
            ],
        )

        if scale_scheduler_by_porc_count:
            np = self.accelerator.state.num_processes
            max_steps = (max_steps + np - 1) // np

        # Setup model
        self.model = model

        # if ref_kl is True, make a copy of the model for reference KL loss
        if self.ref_kl:
            print(
                "Using reference KL loss, creating a copy of the model for reference."
            )
            self.ref_model = deepcopy(model)
            self.ref_model.eval()  # Ensure reference model is in eval mode
            self.ref_model.requires_grad_(
                False
            )  # Disable gradients for reference model
            self.ref_accelerator = Accelerator(
                kwargs_handlers=[
                    DistributedDataParallelKwargs(find_unused_parameters=True)
                ]
            )

            if self.ref_accelerator.distributed_type == DistributedType.DEEPSPEED:
                if (
                    self.ref_accelerator.deepspeed_plugin.deepspeed_config[
                        "zero_optimization"
                    ]["stage"]
                    != 3
                ):
                    self.ref_accelerator.deepspeed_plugin.deepspeed_config[
                        "zero_optimization"
                    ]["stage"] = 0

        # Note: Gradient checkpointing should be enabled BEFORE passing model to trainer
        # Enabling it here can override selective checkpointing (e.g., language_model only)
        # and cause memory issues by checkpointing frozen components
        # if self.gradient_checkpointing and hasattr(
        #     self.model, "gradient_checkpointing_enable"
        # ):
        #     self.model.gradient_checkpointing_enable()

        # Apply torch.compile if requested (requires PyTorch 2.0+)
        if hasattr(torch, "compile") and compile:
            compile_kwargs = {}
            if compile_backend is not None:
                compile_kwargs["backend"] = compile_backend
            self.model = torch.compile(self.model, **compile_kwargs)
            if self.ref_kl:
                self.ref_model = torch.compile(self.ref_model, **compile_kwargs)

        # Set up optimizer
        self.lr = lr
        self.weight_decay = weight_decay
        self.setup_optimizer()

        # Set up learning rate scheduler
        self.schedule_type = schedule_type
        self.lr_final = lr_final
        self.max_steps = max_steps
        self.warmup_steps = warmup_steps
        self.setup_scheduler()

        self.current_epoch = 0
        self.global_step = 0

        # Load checkpoint if provided
        if checkpoint_path:
            self.load_checkpoint(checkpoint_path)

        # Clear CUDA cache
        torch.cuda.empty_cache()

    def setup_optimizer(self):
        """Set up the optimizer for training."""
        param_list = [p for p in self.model.parameters() if p.requires_grad]

        if self.optimizer_type == "Adam":
            self.optimizer = torch.optim.Adam(
                param_list, lr=self.lr, weight_decay=self.weight_decay, fused=True
            )
        elif self.optimizer_type == "AdamW":
            # Use fused=True for maximum performance on H100/A100 GPUs
            self.optimizer = torch.optim.AdamW(
                param_list, lr=self.lr, weight_decay=self.weight_decay, fused=True
            )
        elif self.optimizer_type == "SGD":
            self.optimizer = torch.optim.SGD(
                param_list, lr=self.lr, weight_decay=self.weight_decay
            )
        elif self.optimizer_type == "Lion":
            try:
                import lion_pytorch

                self.optimizer = lion_pytorch.Lion(
                    param_list, lr=self.lr, weight_decay=self.weight_decay
                )
            except ImportError:
                print("Lion optimizer not available. Falling back to AdamW.")
                self.optimizer = torch.optim.AdamW(
                    param_list, lr=self.lr, weight_decay=self.weight_decay
                )
        elif self.optimizer_type == "PagedAdamW8bit":
            try:
                import bitsandbytes as bnb

                self.optimizer = bnb.optim.PagedAdamW8bit(
                    param_list, lr=self.lr, weight_decay=self.weight_decay
                )
            except ImportError:
                print("bitsandbytes not available. Falling back to AdamW.")
                self.optimizer = torch.optim.AdamW(
                    param_list, lr=self.lr, weight_decay=self.weight_decay
                )
        else:
            # Default to AdamW
            self.optimizer = torch.optim.AdamW(
                param_list, lr=self.lr, weight_decay=self.weight_decay
            )

    def setup_scheduler(self):
        """Set up the learning rate scheduler."""
        if self.schedule_type == "linear_with_warmup":
            self.scheduler = get_linear_schedule_with_warmup(
                self.optimizer,
                num_warmup_steps=self.warmup_steps,
                num_training_steps=self.max_steps,
            )
        elif self.schedule_type == "cosine_with_warmup":
            self.scheduler = get_cosine_schedule_with_warmup(
                self.optimizer,
                num_warmup_steps=self.warmup_steps,
                num_training_steps=self.max_steps,
                num_cycles=0.5,
            )
        elif self.schedule_type == "constant_with_warmup":
            self.scheduler = get_constant_schedule_with_warmup(
                self.optimizer, num_warmup_steps=self.warmup_steps
            )
        else:
            self.scheduler = None

    def save_checkpoint(self, path):
        """Save a checkpoint."""

        if (
            self.accelerator.distributed_type == DistributedType.DEEPSPEED
            and self.accelerator.deepspeed_config["zero_optimization"]["stage"] == 3
        ):
            # obtain state dict from DeepSpeed engine
            model_state_dict = self.accelerator.get_state_dict(self.model)
            checkpoint = {
                "model_state_dict": model_state_dict,
                "current_epoch": self.current_epoch,
                "global_step": self.global_step,
            }
            self.accelerator.wait_for_everyone()
            if self.accelerator.is_main_process:
                self.accelerator.save(checkpoint, path)

            # Wait for save to complete before continuing
            self.accelerator.wait_for_everyone()

            # release memory
            del model_state_dict
            del checkpoint
        elif self.accelerator.distributed_type == DistributedType.FSDP:
            # Ensure all ranks are ready to save
            self.accelerator.wait_for_everyone()

            if self.accelerator.is_main_process:
                print(f"Starting checkpoint save to {path}...")

            # Get state dict - this is a collective operation that all ranks must participate in
            model_state_dict = self.accelerator.get_state_dict(
                self.model,
                unwrap=True
            )

            # Wait for all ranks to finish gathering state
            self.accelerator.wait_for_everyone()

            # Only main process saves to avoid I/O contention
            if self.accelerator.is_main_process:
                checkpoint = {
                    "model_state_dict": model_state_dict,
                    "current_epoch": self.current_epoch,
                    "global_step": self.global_step,
                }
                # Use torch.save directly for better control
                torch.save(checkpoint, path)
                print(f"Checkpoint saved successfully to {path}")

                # Release memory on main process
                del checkpoint
                del model_state_dict
            else:
                # Non-main processes just delete their gathered state
                del model_state_dict

            # Critical: Wait for main process to finish saving before all ranks continue
            self.accelerator.wait_for_everyone()

            # Release cached memory after FSDP gather to prevent fragmentation
            torch.cuda.empty_cache()

        elif self.accelerator.is_main_process:
            # Get unwrapped model
            unwrapped_model = self.accelerator.unwrap_model(self.model)

            checkpoint = {
                "model_state_dict": unwrapped_model.state_dict(),
                "current_epoch": self.current_epoch,
                "global_step": self.global_step,
            }

            # Use accelerator to save the checkpoint
            self.accelerator.save(checkpoint, path)

    def load_checkpoint(self, path):
        """Load a checkpoint."""
        checkpoint = torch.load(path)

        # Load model state dict
        if hasattr(self.model, "load_state_dict"):
            try:
                self.model.load_state_dict(checkpoint["model_state_dict"], strict=True)
            except:
                print("Model state dict not found in checkpoint or incompatible.")

        self.current_epoch = checkpoint.get("current_epoch", 0)
        self.global_step = checkpoint.get("global_step", 0)

    def reset_optimizer(self):
        """Reset the optimizer and scheduler."""
        self.setup_optimizer()
        self.setup_scheduler()

        # Prepare model, optimizer and scheduler with accelerator
        self.model, self.optimizer, self.scheduler = self.accelerator.prepare(
            self.model, self.optimizer, self.scheduler
        )

    def train(
        self,
        dataset,
        train_indices=None,
        batch_size=8,
        epochs=3,
        continue_loop=True,
        verbose=True,
        checkpoint_steps=1000,
        checkpoint_dir="checkpoints",
        dataloader_num_workers=4,
        remove_old_checkpoints=True,
        n_total_checkpoints_to_keep=3,
    ):
        """
        Train the model using Accelerate.

        Args:
            dataset: The full dataset to train on
            train_indices: Optional specific indices to use from dataset (if None, uses all)
            batch_size: Batch size
            epochs: Number of epochs to train
            continue_loop: Whether to continue from the last epoch or reset
            verbose: Whether to show progress during training
            checkpoint_steps: Save checkpoint every N steps
            checkpoint_dir: Directory to save checkpoints
            eval_dataset: Dataset for evaluation (if None, no evaluation is performed)
            eval_indices: Optional specific indices for evaluation dataset
            eval_batch_size: Batch size for evaluation (defaults to train batch_size if None)
            eval_steps: How often to run evaluation (in steps)
            max_steps: Maximum number of steps to train for (overrides epochs if provided)
            save_best_only: Only save checkpoints when evaluation loss improves
            early_stopping_patience: Number of evaluations with no improvement after which to stop
            dataloader_num_workers: Number of workers for data loading
        """
        if not continue_loop:
            self.model.train()
            self.current_epoch = 0
            self.global_step = 0
            self.reset_optimizer()

        os.makedirs(checkpoint_dir, exist_ok=True)

        # Save training configuration to JSON (only on main process)
        if self.accelerator.is_main_process:
            # Update config with training runtime parameters
            runtime_config = self.training_config.copy()
            runtime_config.update({
                "batch_size": batch_size,
                "epochs": epochs,
                "checkpoint_steps": checkpoint_steps,
                "dataloader_num_workers": dataloader_num_workers,
                "num_training_samples": len(dataset) if train_indices is None else len(train_indices),
                "distributed_type": str(self.accelerator.distributed_type),
                "num_processes": self.accelerator.num_processes,
                "mixed_precision": str(self.accelerator.mixed_precision),
            })

            config_path = os.path.join(checkpoint_dir, "training_config.json")
            with open(config_path, 'w') as f:
                json.dump(runtime_config, f, indent=2)

            if verbose:
                print(f"Training configuration saved to {config_path}")

        # Set up indices for training
        if train_indices is None:
            train_indices = list(range(len(dataset)))

        # Setup dataloaders
        collate_fn = getattr(dataset, "_collate_fn", None)
        train_sampler = torch.utils.data.SubsetRandomSampler(train_indices)

        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            sampler=train_sampler,
            num_workers=dataloader_num_workers,
            collate_fn=collate_fn,
            pin_memory=True,
            persistent_workers=True if dataloader_num_workers > 0 else False,
            prefetch_factor=4 if dataloader_num_workers > 0 else None,  # Increased for better GPU utilization
        )
        data_iterator = iter(dataloader)

        # Prepare model, optimizer, scheduler, and dataloaders with accelerator
        self.model, self.optimizer, self.scheduler, dataloader = (
            self.accelerator.prepare(
                self.model, self.optimizer, self.scheduler, dataloader
            )
        )
        if self.ref_kl:
            if self.accelerator.distributed_type == DistributedType.FSDP:
                self.ref_model.eval()
                self.ref_model.requires_grad_(False)
                self.ref_model = fsdp2_prepare_model(self.accelerator, self.ref_model)
                self.ref_model.eval()
                self.ref_model.requires_grad_(False)
            else:
                self.ref_model = self.ref_accelerator.prepare(self.ref_model)
                self.ref_model.eval()
                self.ref_model.requires_grad_(False)

        # Set up step tracking
        epoch = self.current_epoch
        total_steps_done = self.global_step
        steps_per_epoch = (
            len(dataloader) + self.gradient_accumulation_steps - 1
        ) // self.gradient_accumulation_steps

        if self.accelerator.is_main_process:
            if verbose:
                print(
                    f"Training with {len(train_indices)} samples in {steps_per_epoch} steps per epoch"
                )
                print(
                    f"Training for {epochs} epochs ({epochs * steps_per_epoch} steps)"
                )
            if self.log_to_wandb:
                wandb.init(
                    project=self.wandb_project,
                    name=self.wandb_run_name,
                    id=self.wandb_id,
                    config={"learning_rate": self.lr},
                )

        loss_fct = torch.nn.CrossEntropyLoss(reduction="sum")

        self.accelerator.register_for_checkpointing(self.scheduler)

        while epoch < self.current_epoch + epochs:
            if verbose and self.accelerator.is_main_process:
                print(f"Epoch {epoch + 1}")
                # prog = tqdm(dataloader, total=len(dataloader), disable=not self.accelerator.is_main_process)
                prog = tqdm(
                    range(steps_per_epoch),
                    total=steps_per_epoch,
                    disable=not self.accelerator.is_main_process,
                )
            else:
                # prog = dataloader
                prog = range(steps_per_epoch)

            accumulated_loss = 0.0

            self.model.train()
            for step in prog:
                step_loss = 0.0
                step_kl_loss = 0.0
                batches = []
                local_total_items = 0

                # Collect batches and count local items WITHOUT collective ops
                for b_i in range(self.gradient_accumulation_steps):
                    if b_i + step * self.gradient_accumulation_steps >= len(dataloader):
                        break
                    try:
                        batch = next(data_iterator)
                    except StopIteration:
                        # Dataloader exhausted - exit gracefully
                        break
                    batch = batch.to(self.accelerator.device)
                    local_total_items += batch["labels"].ne(-100).sum().item()
                    batches.append(batch)

                # Skip step if no batches were collected (end of epoch)
                if len(batches) == 0:
                    break

                # Single gather operation for all ranks to participate in
                # Convert to tensor for gather operation
                local_total_items_tensor = torch.tensor(
                    [local_total_items],
                    device=self.accelerator.device,
                    dtype=torch.long
                )
                total_num_items_in_batch = (
                    self.accelerator.gather(local_total_items_tensor).sum().item()
                )

                # Accumulate local losses before gathering to avoid mismatched collective ops
                local_step_loss = 0.0
                local_step_kl_loss = 0.0

                for batch in batches:
                    labels = batch.pop("labels", None)

                    outputs = self.model(**batch)
                    # Efficient loss computation with in-place operations
                    shift_logits = outputs.logits[..., :-1, :].contiguous()
                    shift_labels = labels[..., 1:].contiguous()

                    # Flatten the tokens - use view instead of reshape when possible
                    shift_labels = shift_labels.view(-1)
                    B = shift_labels.size(0)
                    shift_logits = shift_logits.view(B, -1)

                    # Release logits memory immediately after use
                    del outputs

                    loss = (
                        loss_fct(shift_logits, shift_labels)
                        / total_num_items_in_batch
                        * self.accelerator.num_processes
                    )

                    if self.ref_kl:
                        shift_logits_extracted = shift_logits[shift_labels != -100]
                        shift_labels_extracted = shift_labels[shift_labels != -100]
                        per_token_logps = selective_log_softmax(
                            shift_logits_extracted, shift_labels_extracted
                        )
                        with torch.no_grad():
                            ref_outputs = self.ref_model(**batch)
                            ref_shift_logits = ref_outputs.logits[
                                ..., :-1, :
                            ].contiguous()
                            ref_shift_logits = ref_shift_logits.view(B, -1)
                            ref_shift_logits_extracted = ref_shift_logits[
                                shift_labels != -100
                            ]
                            ref_per_token_logps = selective_log_softmax(
                                ref_shift_logits_extracted, shift_labels_extracted
                            )

                        per_token_kl = (
                            torch.exp(ref_per_token_logps - per_token_logps)
                            - (ref_per_token_logps - per_token_logps)
                            - 1
                        )

                        kl_loss = (
                            per_token_kl.sum()
                            / total_num_items_in_batch
                            * self.accelerator.num_processes
                        )

                        full_loss = loss + self.ref_kl_weight * kl_loss

                        local_step_kl_loss += kl_loss.item()

                        # Backward pass handled by accelerator
                        self.accelerator.backward(full_loss)
                    else:
                        # Backward pass handled by accelerator
                        self.accelerator.backward(loss)

                    local_step_loss += loss.item()

                # Gather losses once per step, not per batch - ensures all ranks participate
                local_step_loss_tensor = torch.tensor(
                    [local_step_loss],
                    device=self.accelerator.device
                )
                gathered_step_loss = self.accelerator.gather(local_step_loss_tensor).sum().item()
                accumulated_loss += gathered_step_loss
                step_loss = gathered_step_loss

                if self.ref_kl:
                    local_step_kl_loss_tensor = torch.tensor(
                        [local_step_kl_loss],
                        device=self.accelerator.device
                    )
                    step_kl_loss = self.accelerator.gather(local_step_kl_loss_tensor).sum().item()

                if self.max_grad_norm > 0:
                    self.accelerator.clip_grad_norm_(
                        self.model.parameters(), self.max_grad_norm
                    )

                # Step optimizer
                self.optimizer.step()
                self.optimizer.zero_grad(set_to_none=self.set_to_none)

                # Step scheduler if we're using it
                if self.scheduler is not None:
                    self.scheduler.step()

                # Increment global step counter
                total_steps_done += 1
                self.global_step = total_steps_done

                # Step-based checkpoint saving
                should_save = total_steps_done % checkpoint_steps == 0

                if should_save:
                    checkpoint_path = os.path.join(
                        checkpoint_dir, f"checkpoint_step_{total_steps_done}.pt"
                    )
                    self.save_checkpoint(checkpoint_path)

                    if verbose and self.accelerator.is_main_process:
                        print(
                            f"\nSaved checkpoint at step {total_steps_done} to {checkpoint_path}"
                        )

                    # Optional: Remove older checkpoints to save disk space
                    if remove_old_checkpoints and self.accelerator.is_main_process:
                        old_checkpoint = os.path.join(
                            checkpoint_dir,
                            f"checkpoint_step_{total_steps_done - (n_total_checkpoints_to_keep * checkpoint_steps)}.pt",
                        )
                        if os.path.exists(old_checkpoint):
                            os.remove(old_checkpoint)

                # Update progress bar
                if self.accelerator.is_main_process:
                    if verbose:
                        current_lr = self.optimizer.param_groups[0]["lr"]
                        avg_loss = accumulated_loss / (step + 1)
                        prog.set_postfix(
                            {
                                "epoch loss": f"{avg_loss:.5f}",
                                "loss": f"{step_loss:.5f}",
                                "lr": f"{current_lr:.7f}",
                                "step": total_steps_done,
                                "kl loss": f"{step_kl_loss:.5f}"
                                if self.ref_kl
                                else "N/A",
                            }
                        )
                    if self.log_to_wandb:
                        log_dict = {
                            "loss": step_loss,
                            "kl_loss": step_kl_loss if self.ref_kl else None,
                            "step": total_steps_done,
                            "lr": current_lr,
                        }
                        # Add GPU memory monitoring if CUDA is available
                        if torch.cuda.is_available():
                            log_dict.update({
                                "gpu_memory_allocated_gb": torch.cuda.memory_allocated() / 1e9,
                                "gpu_memory_reserved_gb": torch.cuda.memory_reserved() / 1e9,
                                "gpu_memory_max_allocated_gb": torch.cuda.max_memory_allocated() / 1e9,
                            })
                        wandb.log(log_dict)

            # Ensure all ranks complete the epoch together before moving to next epoch
            self.accelerator.wait_for_everyone()

            # Increment epoch counter
            epoch += 1
            self.current_epoch = epoch

            # Print epoch summary
            if verbose and self.accelerator.is_main_process:
                print(
                    f"Epoch {epoch} completed. Average loss: {accumulated_loss / steps_per_epoch:.5f}"
                )

            # save epoch checkpoint
            epoch_checkpoint_path = os.path.join(
                checkpoint_dir, f"checkpoint_epoch_{epoch}.pt"
            )
            self.save_checkpoint(epoch_checkpoint_path)

        # Save final model
        final_path = os.path.join(checkpoint_dir, "final_model.pt")
        self.save_checkpoint(final_path)

        if verbose and self.accelerator.is_main_process:
            print(f"Training completed. Final model saved to {final_path}")

        # Make sure all processes reach here
        self.accelerator.wait_for_everyone()

        # Release cached memory after training completes
        torch.cuda.empty_cache()
