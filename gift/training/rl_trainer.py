import os
import time
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
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from accelerate.utils import is_peft_model
from accelerate.utils.fsdp_utils import fsdp2_prepare_model
from contextlib import nullcontext
import wandb


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
        selected_logits = torch.gather(logits, dim=-1, index=index.unsqueeze(-1)).squeeze(-1)
        # loop to reduce peak mem consumption
        logsumexp_values = torch.stack([torch.logsumexp(lg, dim=-1) for lg in logits])
        per_token_logps = selected_logits - logsumexp_values  # log_softmax(x_i) = x_i - logsumexp(x)
    else:
        # logsumexp approach is unstable with bfloat16, fall back to slightly less efficient approach
        per_token_logps = []
        for row_logits, row_labels in zip(logits, index):  # loop to reduce peak mem consumption
            row_logps = F.log_softmax(row_logits, dim=-1)
            row_per_token_logps = row_logps.gather(dim=-1, index=row_labels.unsqueeze(-1)).squeeze(-1)
            per_token_logps.append(row_per_token_logps)
        per_token_logps = torch.stack(per_token_logps)
    return per_token_logps


class REINFORCE:
    """
    Vanilla REINFORCE trainer for CAD code generation.

    Compared to DAPO:
    - No importance sampling ratio (no old_logps)
    - No clipping (no epsilon_low/epsilon_high)
    - No inner training loops (no mu)
    - Loss = -advantage * log_prob (simple policy gradient)
    """

    def __init__(
        self,
        model: torch.nn.Module,
        vllm_client,
        lr: float = 1e-5,
        weight_decay: float = 0.0,
        schedule_type: str = "constant_with_warmup",
        lr_final: float = 1e-6,
        max_steps: int = 1000,
        warmup_steps: int = 100,
        checkpoint_path: Optional[str] = None,
        optimizer_type: str = "AdamW",
        gradient_checkpointing: bool = True,
        max_grad_norm: float = 1.0,
        ref_kl: bool = False,
        beta: float = 0.04,
        scale_scheduler_by_porc_count: bool = True,
        temperature: float = 1.0,
        log_to_wandb: bool = False,
        wandb_project: str = "REINFORCE-CADCoder",
        wandb_run_name: str = "REINFORCE_Experiment",
        wandb_id: Optional[str] = None,
    ):
        self.optimizer_type = optimizer_type
        self.gradient_checkpointing = gradient_checkpointing
        self.max_grad_norm = max_grad_norm
        self.ref_kl = ref_kl
        self.beta = beta
        self.vllm_client = vllm_client
        self.temperature = temperature
        self.log_to_wandb = log_to_wandb
        self.wandb_project = wandb_project
        self.wandb_run_name = wandb_run_name
        self.wandb_id = wandb_id

        self.accelerator = Accelerator(gradient_accumulation_steps=1)

        self.is_fsdp_enabled = self.accelerator.distributed_type == DistributedType.FSDP

        if self.is_fsdp_enabled:
            print("Using FSDP for distributed training.")

        if self.accelerator.distributed_type == DistributedType.DEEPSPEED:
            raise NotImplementedError("REINFORCE does not support DeepSpeed. Use FSDP or DDP.")

        if scale_scheduler_by_porc_count:
            np_ = self.accelerator.state.num_processes
            max_steps = (max_steps + np_ - 1) // np_

        self.model = model

        if self.ref_kl:
            print("Using reference KL loss, creating a copy of the model.")
            self.ref_model = deepcopy(model)
            self.ref_model.eval()
            self.ref_model.requires_grad_(False)
            self.ref_accelerator = Accelerator()

        if self.gradient_checkpointing and hasattr(self.model, "gradient_checkpointing_enable"):
            self.model.gradient_checkpointing_enable()

        self.lr = lr
        self.weight_decay = weight_decay
        self.setup_optimizer()

        self.schedule_type = schedule_type
        self.lr_final = lr_final
        self.max_steps = max_steps
        self.warmup_steps = warmup_steps
        self.setup_scheduler()

        self.current_epoch = 0
        self.global_step = 0

        if checkpoint_path:
            self.load_checkpoint(checkpoint_path)

        torch.cuda.empty_cache()

    def setup_optimizer(self):
        param_list = [p for p in self.model.parameters() if p.requires_grad]
        if self.optimizer_type == "AdamW":
            self.optimizer = torch.optim.AdamW(param_list, lr=self.lr, weight_decay=self.weight_decay)
        elif self.optimizer_type == "Adam":
            self.optimizer = torch.optim.Adam(param_list, lr=self.lr, weight_decay=self.weight_decay)
        elif self.optimizer_type == "SGD":
            self.optimizer = torch.optim.SGD(param_list, lr=self.lr, weight_decay=self.weight_decay)
        else:
            self.optimizer = torch.optim.AdamW(param_list, lr=self.lr, weight_decay=self.weight_decay)

    def setup_scheduler(self):
        if self.schedule_type == "linear_with_warmup":
            self.scheduler = get_linear_schedule_with_warmup(
                self.optimizer, num_warmup_steps=self.warmup_steps, num_training_steps=self.max_steps
            )
        elif self.schedule_type == "cosine_with_warmup":
            self.scheduler = get_cosine_schedule_with_warmup(
                self.optimizer, num_warmup_steps=self.warmup_steps, num_training_steps=self.max_steps, num_cycles=0.5
            )
        elif self.schedule_type == "constant_with_warmup":
            self.scheduler = get_constant_schedule_with_warmup(
                self.optimizer, num_warmup_steps=self.warmup_steps
            )
        else:
            self.scheduler = None

    def save_checkpoint(self, path):
        if self.is_fsdp_enabled:
            model_state_dict = {}
            for param_name, param in self.model.named_parameters():
                full_name = param_name
                for extra in ("_fsdp_wrapped_module.", "_checkpoint_wrapped_module.", "_orig_mod."):
                    full_name = full_name.replace(extra, "")
                data = param.full_tensor()
                if self.accelerator.is_main_process:
                    model_state_dict[full_name] = data.to("cpu")

            checkpoint = {
                "model_state_dict": model_state_dict,
                "current_epoch": self.current_epoch,
                "global_step": self.global_step,
                "wandb_id": self.wandb_id if self.log_to_wandb else None,
            }
            self.accelerator.wait_for_everyone()
            if self.accelerator.is_main_process:
                self.accelerator.save(checkpoint, path)
            del model_state_dict
            torch.cuda.empty_cache()
        elif self.accelerator.is_main_process:
            unwrapped_model = self.accelerator.unwrap_model(self.model)
            checkpoint = {
                "model_state_dict": unwrapped_model.state_dict(),
                "current_epoch": self.current_epoch,
                "global_step": self.global_step,
                "wandb_id": self.wandb_id if self.log_to_wandb else None,
            }
            self.accelerator.save(checkpoint, path)

    def _move_model_to_vllm(self):
        if self.accelerator.is_main_process:
            self.vllm_client.sleep()

        if self.is_fsdp_enabled:
            for param_name, param in self.model.named_parameters():
                full_name = param_name
                for extra in ("_fsdp_wrapped_module.", "_checkpoint_wrapped_module.", "_orig_mod."):
                    full_name = full_name.replace(extra, "")
                data = param.full_tensor()
                if self.accelerator.is_main_process:
                    self.vllm_client.update_named_param(full_name, data)
        else:
            for name, param in self.model.named_parameters():
                if self.accelerator.is_main_process:
                    self.vllm_client.update_named_param(name, param.data)

        if self.accelerator.is_main_process:
            self.vllm_client.wake_up()

    def load_checkpoint(self, path):
        checkpoint = torch.load(path)
        if hasattr(self.model, "load_state_dict"):
            try:
                self.model.load_state_dict(checkpoint["model_state_dict"], strict=True)
                print(f"Checkpoint loaded from {path}")
            except Exception:
                print("Model state dict incompatible, starting fresh.")
        self.current_epoch = checkpoint.get("current_epoch", 0)
        self.global_step = checkpoint.get("global_step", 0)
        self.wandb_id = checkpoint.get("wandb_id", None)

    def reset_optimizer(self):
        self.setup_optimizer()
        self.setup_scheduler()
        self.model, self.optimizer, self.scheduler = self.accelerator.prepare(
            self.model, self.optimizer, self.scheduler
        )

    def train(
        self,
        dataset,
        train_indices=None,
        batch_size=8,
        epochs=1,
        continue_loop=True,
        verbose=True,
        checkpoint_steps=50,
        checkpoint_dir="checkpointsREINFORCE",
        dataloader_num_workers=4,
        dataloader_prefetch_factor=1,
        remove_old_checkpoints=True,
        n_total_checkpoints_to_keep=3,
        dataloader_multiprocessing_context=None,
    ):
        if not continue_loop:
            self.model.train()
            self.current_epoch = 0
            self.global_step = 0
            self.reset_optimizer()

        if self.accelerator.is_main_process:
            self.vllm_client.init_communicator()
            if self.log_to_wandb:
                wandb.init(
                    project=self.wandb_project,
                    name=self.wandb_run_name,
                    id=self.wandb_id,
                    resume="allow",
                    config={
                        "algorithm": "REINFORCE",
                        "batch_size": batch_size,
                        "learning_rate": self.lr,
                        "temperature": self.temperature,
                    },
                )
                self.wandb_id = wandb.run.id
        else:
            self.vllm_client = None

        os.makedirs(checkpoint_dir, exist_ok=True)

        if train_indices is None:
            train_indices = list(range(len(dataset)))

        collate_fn = getattr(dataset, "_collate_fn", None)
        train_sampler = torch.utils.data.SubsetRandomSampler(train_indices)

        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            sampler=train_sampler,
            num_workers=dataloader_num_workers,
            collate_fn=collate_fn,
            prefetch_factor=dataloader_prefetch_factor,
            pin_memory=False,
            multiprocessing_context=dataloader_multiprocessing_context,
        )

        self.model, self.optimizer, self.scheduler, dataloader = self.accelerator.prepare(
            self.model, self.optimizer, self.scheduler, dataloader
        )

        if self.ref_kl:
            if self.is_fsdp_enabled:
                self.ref_model.eval()
                self.ref_model.requires_grad_(False)
                self.ref_model = fsdp2_prepare_model(self.accelerator, self.ref_model)
                self.ref_model.eval()
                self.ref_model.requires_grad_(False)
            else:
                self.ref_model = self.ref_accelerator.prepare(self.ref_model)
                self.ref_model.eval()
                self.ref_model.requires_grad_(False)

        epoch = self.current_epoch
        total_steps_done = self.global_step
        steps_per_epoch = len(dataloader)

        if verbose and self.accelerator.is_main_process:
            print(f"Training with {len(train_indices)} samples in {steps_per_epoch} steps per epoch")
            print(f"Training for {epochs} epochs ({epochs * steps_per_epoch} total steps)")

        while epoch < self.current_epoch + epochs:
            if verbose and self.accelerator.is_main_process:
                print(f"Epoch {epoch + 1}")
                prog = tqdm(dataloader, total=len(dataloader), disable=not self.accelerator.is_main_process)
            else:
                prog = dataloader

            accumulated_loss = 0.0
            self.model.train()
            self._move_model_to_vllm()
            step_times = []
            train_times = []
            sync_times = []

            for step, batches in enumerate(prog):
                step_start_time = time.perf_counter()
                step_loss = 0.0
                step_kl_loss = 0.0
                rewards_mean = 0.0
                total_num_items_in_batch = 0

                # Accumulate gradients across sub-batches
                for b_i in range(len(batches)):
                    batches[b_i] = batches[b_i].to(self.accelerator.device)
                    local_num_items = batches[b_i]["labels"].ne(-100).sum()
                    total_num_items_in_batch += self.accelerator.gather(local_num_items).sum().item()

                    r = batches[b_i].pop("rewards", None)
                    rewards_mean += self.accelerator.gather(torch.mean(r)).mean().item()

                    labels = batches[b_i].pop("labels", None)
                    advantage = batches[b_i].pop("advantage", None)
                    tiled_advantage = advantage.unsqueeze(-1).expand_as(labels)

                    outputs = self.model(**batches[b_i])

                    shift_logits = outputs.logits[..., :-1, :].contiguous() / self.temperature
                    shift_labels = labels[..., 1:].contiguous()

                    shift_labels = shift_labels.view(-1)
                    shift_advantage = tiled_advantage[..., 1:].contiguous().view(-1)
                    B = shift_labels.size(0)
                    shift_logits = shift_logits.view(B, -1)

                    shift_logits_extracted = shift_logits[shift_labels != -100]
                    shift_labels_extracted = shift_labels[shift_labels != -100]
                    shift_advantage_extracted = shift_advantage[shift_labels != -100]

                    per_token_logps = selective_log_softmax(shift_logits_extracted, shift_labels_extracted)

                    # REINFORCE: loss = -advantage * log_prob
                    per_token_loss = -shift_advantage_extracted * per_token_logps
                    loss = per_token_loss.sum() / total_num_items_in_batch * self.accelerator.num_processes

                    l_gathered = self.accelerator.gather(loss).sum().item() / self.accelerator.num_processes
                    accumulated_loss += l_gathered
                    step_loss += l_gathered

                    if self.ref_kl:
                        with torch.no_grad():
                            ref_outputs = self.ref_model(**batches[b_i])
                            ref_shift_logits = ref_outputs.logits[..., :-1, :].contiguous() / self.temperature
                            ref_shift_logits = ref_shift_logits.view(B, -1)
                            ref_shift_logits_extracted = ref_shift_logits[shift_labels != -100]
                            ref_per_token_logps = selective_log_softmax(ref_shift_logits_extracted, shift_labels_extracted)

                        per_token_kl = torch.exp(ref_per_token_logps - per_token_logps) - (ref_per_token_logps - per_token_logps) - 1
                        kl_loss = per_token_kl.sum() / total_num_items_in_batch * self.accelerator.num_processes
                        loss = loss + self.beta * kl_loss

                        kl_gathered = self.accelerator.gather(kl_loss).sum().item() / self.accelerator.num_processes
                        step_kl_loss += kl_gathered

                    self.accelerator.backward(loss)

                rewards_mean = rewards_mean / len(batches)

                if self.max_grad_norm > 0:
                    grad_norm = self.accelerator.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)

                self.optimizer.step()
                self.optimizer.zero_grad()

                train_time = time.perf_counter() - step_start_time
                train_times.append(train_time)

                total_steps_done += 1
                self.global_step = total_steps_done

                # Sync updated weights to VLLM for next batch generation
                sync_start_time = time.perf_counter()
                self._move_model_to_vllm()
                sync_time = time.perf_counter() - sync_start_time
                sync_times.append(sync_time)

                step_time = time.perf_counter() - step_start_time
                step_times.append(step_time)

                if verbose and self.accelerator.is_main_process:
                    current_lr = self.optimizer.param_groups[0]["lr"]
                    avg_loss = accumulated_loss / (step + 1)
                    prog.set_postfix({
                        "epoch loss": f"{avg_loss:.5f}",
                        "loss": f"{step_loss:.5f}",
                        "lr": f"{current_lr:.7f}",
                        "step": total_steps_done,
                        "rewards mean": f"{rewards_mean:.5f}",
                        "kl loss": f"{step_kl_loss:.5f}" if self.ref_kl else "N/A",
                        "step_time": f"{step_time:.1f}s",
                    })

                if self.log_to_wandb and self.accelerator.is_main_process:
                    wandb.log({
                        "loss": step_loss,
                        "kl_loss": step_kl_loss if self.ref_kl else None,
                        "mean_reward": rewards_mean,
                        "learning_rate": current_lr,
                        "timing/step_time": step_time,
                        "timing/train_time": train_time,
                        "timing/sync_time": sync_time,
                    })

                if self.scheduler is not None:
                    self.scheduler.step()

                if total_steps_done % checkpoint_steps == 0:
                    checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_step_{total_steps_done}.pt")
                    self.save_checkpoint(checkpoint_path)
                    if verbose and self.accelerator.is_main_process:
                        print(f"\nSaved checkpoint at step {total_steps_done}")
                    if remove_old_checkpoints and self.accelerator.is_main_process:
                        old_ckpt = os.path.join(
                            checkpoint_dir,
                            f"checkpoint_step_{total_steps_done - (n_total_checkpoints_to_keep * checkpoint_steps)}.pt",
                        )
                        if os.path.exists(old_ckpt):
                            os.remove(old_ckpt)

            epoch += 1
            self.current_epoch = epoch

            if verbose and self.accelerator.is_main_process:
                st = np.array(step_times)
                tt = np.array(train_times)
                sy = np.array(sync_times)
                print(f"Epoch {epoch} completed. Average loss: {accumulated_loss / steps_per_epoch:.5f}")
                print(f"  Step time:  mean={st.mean():.2f}s  median={np.median(st):.2f}s  total={st.sum()/3600:.2f}h")
                print(f"  Train time: mean={tt.mean():.2f}s  median={np.median(tt):.2f}s")
                print(f"  Sync time:  mean={sy.mean():.2f}s  median={np.median(sy):.2f}s")

            epoch_path = os.path.join(checkpoint_dir, f"checkpoint_epoch_{epoch}.pt")
            self.save_checkpoint(epoch_path)

        final_path = os.path.join(checkpoint_dir, "final_model.pt")
        self.save_checkpoint(final_path)

        if verbose and self.accelerator.is_main_process:
            print(f"Training completed. Final model saved to {final_path}")

        self.accelerator.wait_for_everyone()


class DAPO:
    def __init__(
        self,
        model: torch.nn.Module,
        vllm_client,
        lr: float = 2e-5,
        weight_decay: float = 0.01,
        schedule_type: str = "cosine_with_warmup",
        lr_final: float = 1e-6,
        max_steps: int = 1000,
        warmup_steps: int = 100,
        checkpoint_path: Optional[str] = None,
        optimizer_type: str = "AdamW",
        gradient_checkpointing: bool = True,
        max_grad_norm: float = 1.0,
        ref_kl: bool = False,
        beta: float = 0.04,
        scale_scheduler_by_porc_count: bool = True,
        mu: int = 4,
        epsilon_low: float = 0.2,
        epsilon_high: float = 0.28,
        temperature: float = 0.5,
        log_to_wandb: bool = False,
        wandb_project: str = "DAPO-CADCoder",
        wandb_run_name: str = "RL_Experiment",
        wandb_id: Optional[str] = None,
        log_gradient_norms: bool = False
    ):
        """
        Initialize the DAPO trainer for transformer models.
        """
        self.optimizer_type = optimizer_type
        self.gradient_checkpointing = gradient_checkpointing
        self.max_grad_norm = max_grad_norm
        self.ref_kl = ref_kl
        self.beta = beta
        self.mu = mu
        self.epsilon_low = epsilon_low
        self.epsilon_high = epsilon_high
        self.vllm_client = vllm_client
        self.temperature = temperature
        self.log_to_wandb = log_to_wandb
        self.wandb_project = wandb_project
        self.wandb_run_name = wandb_run_name
        self.wandb_id = wandb_id
        self.log_gradient_norms = log_gradient_norms

        # Initialize accelerator with gradient accumulation config
        self.accelerator = Accelerator(
            gradient_accumulation_steps=1
        )

        self.is_fsdp_enabled = self.accelerator.distributed_type == DistributedType.FSDP

        if self.is_fsdp_enabled:
            print("Using Fully Sharded Data Parallel (FSDP) for distributed training.")

        if self.accelerator.distributed_type == DistributedType.DEEPSPEED:
            raise NotImplementedError(
                "DAPO does not support DeepSpeed training yet. Please use FSDP or standard DDP."
            )

        if scale_scheduler_by_porc_count:
            np = self.accelerator.state.num_processes
            max_steps = (max_steps + np - 1) // np

        # Setup model
        self.model = model

        # if ref_kl is True, make a copy of the model for reference KL loss
        if self.ref_kl:
            print("Using reference KL loss, creating a copy of the model for reference.")
            self.ref_model = deepcopy(model)
            self.ref_model.eval()  # Ensure reference model is in eval mode
            self.ref_model.requires_grad_(False)  # Disable gradients for reference model
            self.ref_accelerator = Accelerator()

            if self.ref_accelerator.distributed_type == DistributedType.DEEPSPEED:
                if self.ref_accelerator.deepspeed_plugin.deepspeed_config["zero_optimization"]["stage"] != 3:
                    self.ref_accelerator.deepspeed_plugin.deepspeed_config["zero_optimization"]["stage"] = 0

        # Enable gradient checkpointing if requested
        if self.gradient_checkpointing and hasattr(self.model, "gradient_checkpointing_enable"):
            self.model.gradient_checkpointing_enable()
            # Disable gradient checkpointing on frozen vision encoder to avoid
            # recomputation issues with empty pixel_values under FSDP
            for attr in ("visual", "vision_model", "vision_encoder", "vision_tower"):
                if hasattr(self.model, attr):
                    vis = getattr(self.model, attr)
                    if hasattr(vis, "gradient_checkpointing_disable"):
                        vis.gradient_checkpointing_disable()
                    break

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
            self.optimizer = torch.optim.Adam(param_list, lr=self.lr, weight_decay=self.weight_decay)
        elif self.optimizer_type == "AdamW":
            self.optimizer = torch.optim.AdamW(param_list, lr=self.lr, weight_decay=self.weight_decay)
        elif self.optimizer_type == "SGD":
            self.optimizer = torch.optim.SGD(param_list, lr=self.lr, weight_decay=self.weight_decay)
        elif self.optimizer_type == "Lion":
            try:
                import lion_pytorch
                self.optimizer = lion_pytorch.Lion(param_list, lr=self.lr, weight_decay=self.weight_decay)
            except ImportError:
                print("Lion optimizer not available. Falling back to AdamW.")
                self.optimizer = torch.optim.AdamW(param_list, lr=self.lr, weight_decay=self.weight_decay)
        else:
            # Default to AdamW
            self.optimizer = torch.optim.AdamW(param_list, lr=self.lr, weight_decay=self.weight_decay)

    def setup_scheduler(self):
        """Set up the learning rate scheduler."""
        if self.schedule_type == "linear_with_warmup":
            self.scheduler = get_linear_schedule_with_warmup(
                self.optimizer,
                num_warmup_steps=self.warmup_steps,
                num_training_steps=self.max_steps
            )
        elif self.schedule_type == "cosine_with_warmup":
            self.scheduler = get_cosine_schedule_with_warmup(
                self.optimizer,
                num_warmup_steps=self.warmup_steps,
                num_training_steps=self.max_steps,
                num_cycles=0.5
            )
        elif self.schedule_type == "constant_with_warmup":
            self.scheduler = get_constant_schedule_with_warmup(
                self.optimizer,
                num_warmup_steps=self.warmup_steps
            )
        else:
            self.scheduler = None

    def save_checkpoint(self, path):
        """Save a checkpoint."""

        if self.accelerator.distributed_type == DistributedType.DEEPSPEED and self.accelerator.deepspeed_config["zero_optimization"]["stage"] == 3:
            # obtain state dict from DeepSpeed engine
            model_state_dict = self.accelerator.get_state_dict(self.model)
            checkpoint = {
                    'model_state_dict': model_state_dict,
                    'current_epoch': self.current_epoch,
                    'global_step': self.global_step,
                    'wandb_id': self.wandb_id if self.log_to_wandb else None
                }
            self.accelerator.wait_for_everyone()
            if self.accelerator.is_main_process:
                self.accelerator.save(checkpoint, path)

            del model_state_dict
            torch.cuda.empty_cache()

        elif self.accelerator.distributed_type == DistributedType.FSDP:
            model_state_dict = {}

            for param_name, param in self.model.named_parameters():
                full_name = param_name
                for extra in ("_fsdp_wrapped_module.", "_checkpoint_wrapped_module.", "_orig_mod."):
                    full_name = full_name.replace(extra, "")
                data = param.full_tensor()
                if self.accelerator.is_main_process:
                    model_state_dict[full_name] = data.to('cpu')

            checkpoint = {
                'model_state_dict': model_state_dict,
                'current_epoch': self.current_epoch,
                'global_step': self.global_step,
                'wandb_id': self.wandb_id if self.log_to_wandb else None
            }
            self.accelerator.wait_for_everyone()
            if self.accelerator.is_main_process:
                self.accelerator.save(checkpoint, path)

            del model_state_dict
            torch.cuda.empty_cache()

        elif self.accelerator.is_main_process:
            unwrapped_model = self.accelerator.unwrap_model(self.model)

            checkpoint = {
                'model_state_dict': unwrapped_model.state_dict(),
                'current_epoch': self.current_epoch,
                'global_step': self.global_step,
                'wandb_id': self.wandb_id if self.log_to_wandb else None
            }

            self.accelerator.save(checkpoint, path)

    def _move_model_to_vllm(self):

        # Pause Server
        if self.accelerator.is_main_process:
            self.vllm_client.sleep()
        # For DeepSpeed ZeRO-3 and FSDP, we need to gather all parameters before operations
        deepspeed_plugin = self.accelerator.state.deepspeed_plugin
        zero_stage_3 = deepspeed_plugin is not None and deepspeed_plugin.zero_stage == 3
        if zero_stage_3:
            import deepspeed

            gather_if_zero3 = deepspeed.zero.GatheredParameters
        else:
            gather_if_zero3 = nullcontext

        if self.is_fsdp_enabled:
            from torch.distributed.checkpoint.state_dict import get_model_state_dict, StateDictOptions
            # Use the proper FSDP2 API to gather full state dict without corrupting DTensor internals
            full_sd = get_model_state_dict(
                self.model,
                options=StateDictOptions(full_state_dict=True, cpu_offload=True)
            )
            if self.accelerator.is_main_process:
                for name, data in full_sd.items():
                    self.vllm_client.update_named_param(name, data.to(self.accelerator.device))
            del full_sd
            torch.cuda.empty_cache()
        else:
            for name, param in self.model.named_parameters():
                with gather_if_zero3([param]):
                    data = param.data
                    if self.accelerator.is_main_process:
                        self.vllm_client.update_named_param(name, data)

        # Resume Server
        if self.accelerator.is_main_process:
            self.vllm_client.wake_up()

    def load_checkpoint(self, path):
        """Load a checkpoint."""
        checkpoint = torch.load(path)

        # Load model state dict
        if hasattr(self.model, "load_state_dict"):
            try:
                self.model.load_state_dict(checkpoint['model_state_dict'], strict=True)
                print(f"Model state dict loaded successfully from {path}")
            except:
                print("Model state dict not found in checkpoint or incompatible.")

        self.current_epoch = checkpoint.get('current_epoch', 0)
        self.global_step = checkpoint.get('global_step', 0)

        self.wandb_id = checkpoint.get('wandb_id', None)

    def reset_optimizer(self):
        """Reset the optimizer and scheduler."""
        self.setup_optimizer()
        self.setup_scheduler()

        # Prepare model, optimizer and scheduler with accelerator
        self.model, self.optimizer, self.scheduler = self.accelerator.prepare(
            self.model, self.optimizer, self.scheduler
        )

    def train(self,
              dataset,
              train_indices=None,
              batch_size=8,
              epochs=3,
              continue_loop=True,
              verbose=True,
              checkpoint_steps=1000,
              checkpoint_dir='checkpoints',
              dataloader_num_workers=4,
              dataloader_prefetch_factor=1,
              remove_old_checkpoints=True,
              n_total_checkpoints_to_keep=3,
              dataloader_multiprocessing_context=None,
              max_steps=None):
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

        if self.accelerator.is_main_process:
            self.vllm_client.init_communicator()
            if self.log_to_wandb:
                wandb.init(project=self.wandb_project, name=self.wandb_run_name, id=self.wandb_id,
                           resume="allow",
                           config={
                    "batch_size": batch_size,
                    "learning_rate": self.lr,
                    "schedule_type": self.schedule_type,
                    "lr_final": self.lr_final,
                    "warmup_steps": self.warmup_steps,
                    "epsilon_low": self.epsilon_low,
                    "epsilon_high": self.epsilon_high,
                    "temperature": self.temperature
                })

                self.wandb_id = wandb.run.id
            pass
        else:
            self.vllm_client = None

        os.makedirs(checkpoint_dir, exist_ok=True)

        # Set up indices for training
        if train_indices is None:
            train_indices = list(range(len(dataset)))

        # Setup dataloaders
        collate_fn = getattr(dataset, '_collate_fn', None)
        train_sampler = torch.utils.data.SubsetRandomSampler(train_indices)

        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            sampler=train_sampler,
            num_workers=dataloader_num_workers,
            collate_fn=collate_fn,
            prefetch_factor=dataloader_prefetch_factor,
            pin_memory=False,
            multiprocessing_context=dataloader_multiprocessing_context
        )

        # Prepare model, optimizer, scheduler, and dataloaders with accelerator
        self.model, self.optimizer, self.scheduler, dataloader = self.accelerator.prepare(
            self.model, self.optimizer, self.scheduler, dataloader
        )
        if self.ref_kl:
            if self.accelerator.distributed_type == DistributedType.FSDP:
                self.ref_model.eval()
                self.ref_model.requires_grad_(False)
                # self.ref_model = torch.compile(self.ref_model, **self.accelerator.state.dynamo_plugin.to_kwargs())
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
        steps_per_epoch = len(dataloader)

        if verbose and self.accelerator.is_main_process:
            print(f"Training with {len(train_indices)} samples in {steps_per_epoch} steps per epoch")
            if max_steps:
                print(f"Training for max {max_steps} steps (overrides {epochs} epochs setting)")
            else:
                print(f"Training for {epochs} epochs ({epochs * steps_per_epoch} steps)")

        self.accelerator.register_for_checkpointing(self.scheduler)

        while (max_steps is None and epoch < self.current_epoch + epochs) or (
            max_steps is not None and total_steps_done < max_steps
        ):

            if verbose and self.accelerator.is_main_process:
                print(f"Epoch {epoch + 1}")
                prog = tqdm(dataloader, total=len(dataloader), disable=not self.accelerator.is_main_process)
            else:
                prog = dataloader
            accumulated_loss = 0.0

            self.model.train()
            self._move_model_to_vllm()
            step_times = []
            train_times = []
            sync_times = []
            for step, batches in enumerate(prog):
                step_start_time = time.perf_counter()
                step_loss = 0.0
                step_kl_loss = 0.0
                total_num_items_in_batch = 0
                local_batch_size = 0
                rewards_mean = 0.0
                labels_batch = []
                advantage_batch = []
                for b_i in range(len(batches)):
                    batches[b_i] = batches[b_i].to(self.accelerator.device)
                    local_num_items_in_batch = batches[b_i]['labels'].ne(-100).sum()
                    local_batch_size += local_num_items_in_batch.item()
                    total_num_items_in_batch += self.accelerator.gather(local_num_items_in_batch).sum().item()

                    r = batches[b_i].pop('rewards', None)
                    rewards_s = torch.mean(r)

                    rewards_mean += self.accelerator.gather(rewards_s).mean().item()

                    labels_batch.append(batches[b_i].pop('labels', None))
                    advantage_batch.append(batches[b_i].pop('advantage', None))

                rewards_mean = rewards_mean / len(batches)

                ref_logps = []
                old_logps = []
                for inner_step in range(self.mu):
                    inner_step_loss = 0.0
                    inner_step_kl_loss = 0.0
                    for batch_idx, batch in enumerate(batches):
                        labels = labels_batch[batch_idx]
                        advantage = advantage_batch[batch_idx]

                        tiled_advantage = advantage.unsqueeze(-1).expand_as(labels)

                        outputs = self.model(**batch)

                        # loss = outputs.loss
                        shift_logits = outputs.logits[..., :-1, :].contiguous()/self.temperature
                        shift_labels = labels[..., 1:].contiguous()

                        # Flatten the tokens
                        shift_labels = shift_labels.view(-1)
                        shift_advantage = tiled_advantage[..., 1:].contiguous().view(-1)
                        B = shift_labels.size(0)
                        shift_logits = shift_logits.view(B, -1)

                        # gather labeled logits
                        shift_logits_extracted = shift_logits[shift_labels != -100]
                        shift_labels_extracted = shift_labels[shift_labels != -100]
                        shift_advantage_extracted = shift_advantage[shift_labels != -100]

                        per_token_logps = selective_log_softmax(shift_logits_extracted, shift_labels_extracted)
                        # per_token_logps =  torch.gather(shift_logits_extracted.log_softmax(-1), dim=-1, index=shift_labels_extracted.unsqueeze(-1)).squeeze(-1)

                        if inner_step == 0:
                            old_logps.append(per_token_logps.detach())

                            if self.ref_kl:
                                with torch.no_grad():
                                    ref_outputs = self.ref_model(**batch)
                                    ref_shift_logits = ref_outputs.logits[..., :-1, :].contiguous()/self.temperature
                                    ref_shift_logits = ref_shift_logits.view(B, -1)
                                    ref_shift_logits_extracted = ref_shift_logits[shift_labels != -100]
                                    ref_per_token_logps = selective_log_softmax(ref_shift_logits_extracted, shift_labels_extracted)
                                    ref_logps.append(ref_per_token_logps.detach())

                        old_per_token_logps = old_logps[batch_idx]

                        coef_1 = torch.exp(per_token_logps - old_per_token_logps)
                        coef_2 = torch.clamp(coef_1, 1 - self.epsilon_low, 1 + self.epsilon_high)

                        per_token_loss1 = coef_1 * shift_advantage_extracted
                        per_token_loss2 = coef_2 * shift_advantage_extracted
                        per_token_loss = -torch.min(per_token_loss1, per_token_loss2)

                        loss = per_token_loss.sum() / total_num_items_in_batch * self.accelerator.num_processes

                        l_gathered = self.accelerator.gather(loss).sum().item() / self.accelerator.num_processes
                        accumulated_loss += l_gathered
                        inner_step_loss += l_gathered
                        step_loss += l_gathered

                        if self.ref_kl:
                            ref_per_token_logps = ref_logps[batch_idx]
                            per_token_kl = (
                                torch.exp(ref_per_token_logps - per_token_logps) - (ref_per_token_logps - per_token_logps) - 1
                            )
                            kl_loss = per_token_kl.sum() / total_num_items_in_batch * self.accelerator.num_processes

                            loss += self.beta * kl_loss

                            kl_gathered = self.accelerator.gather(kl_loss).sum().item() / self.accelerator.num_processes
                            inner_step_kl_loss += kl_gathered
                            step_kl_loss += kl_gathered

                        self.accelerator.backward(loss)

                    if self.max_grad_norm > 0:
                        grad_norm = self.accelerator.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)

                    self.optimizer.step()
                    self.optimizer.zero_grad()

                    if verbose and self.accelerator.is_main_process:
                        current_lr = self.optimizer.param_groups[0]['lr']
                        avg_loss = accumulated_loss / (step * self.mu + inner_step + 1)
                        prog.set_postfix({
                            'epoch loss': f"{avg_loss:.5f}",
                            'loss': f"{inner_step_loss:.5f}",
                            'lr': f"{current_lr:.7f}",
                            'step': total_steps_done,
                            'inner step': inner_step + 1,
                            'rewards mean': f"{rewards_mean:.5f}",
                            'kl loss': f"{inner_step_kl_loss:.5f}" if self.ref_kl else "N/A"
                        })


                train_time = time.perf_counter() - step_start_time
                train_times.append(train_time)

                # Increment global step counter
                total_steps_done += 1
                self.global_step = total_steps_done

                # Check if we've reached max_steps
                if max_steps is not None and total_steps_done >= max_steps:
                    if verbose and self.accelerator.is_main_process:
                        print(f"\nReached max_steps={max_steps}, stopping training.")
                    # Save a final checkpoint
                    checkpoint_path = os.path.join(checkpoint_dir, f'checkpoint_step_{total_steps_done}.pt')
                    self.save_checkpoint(checkpoint_path)
                    if verbose and self.accelerator.is_main_process:
                        print(f"Saved final checkpoint at step {total_steps_done} to {checkpoint_path}")
                    break

                # update vllm model
                sync_start_time = time.perf_counter()
                self._move_model_to_vllm()
                sync_time = time.perf_counter() - sync_start_time
                sync_times.append(sync_time)

                step_time = time.perf_counter() - step_start_time
                step_times.append(step_time)

                # Log to wandb if enabled
                if self.log_to_wandb and self.accelerator.is_main_process:
                    wandb.log({
                        "loss": step_loss,
                        "kl_loss": step_kl_loss if self.ref_kl else None,
                        "mean_reward": rewards_mean,
                        "grad_norm": grad_norm.item() if self.log_gradient_norms else None,
                        "learning_rate": current_lr,
                        "timing/step_time": step_time,
                        "timing/train_time": train_time,
                        "timing/sync_time": sync_time,
                    })

                # Update learning rate scheduler
                if self.scheduler is not None:
                    self.scheduler.step()

                # Step-based checkpoint saving
                should_save = (total_steps_done % checkpoint_steps == 0)

                if should_save:
                    checkpoint_path = os.path.join(checkpoint_dir, f'checkpoint_step_{total_steps_done}.pt')
                    self.save_checkpoint(checkpoint_path)

                    if verbose and self.accelerator.is_main_process:
                        print(f"\nSaved checkpoint at step {total_steps_done} to {checkpoint_path}")

                    # Optional: Remove older checkpoints to save disk space
                    if remove_old_checkpoints and self.accelerator.is_main_process:
                        old_checkpoint = os.path.join(checkpoint_dir, f'checkpoint_step_{total_steps_done - (n_total_checkpoints_to_keep * checkpoint_steps)}.pt')
                        if os.path.exists(old_checkpoint):
                            os.remove(old_checkpoint)

            # Break out of epoch loop if max_steps reached
            if max_steps is not None and total_steps_done >= max_steps:
                break

            # Increment epoch counter
            epoch += 1
            self.current_epoch = epoch

            # Print epoch summary
            if verbose and self.accelerator.is_main_process:
                st = np.array(step_times)
                tt = np.array(train_times)
                sy = np.array(sync_times)
                print(f"Epoch {epoch} completed. Average loss: {accumulated_loss / steps_per_epoch:.5f}")
                print(f"  Step time:  mean={st.mean():.2f}s  median={np.median(st):.2f}s  total={st.sum()/3600:.2f}h")
                print(f"  Train time: mean={tt.mean():.2f}s  median={np.median(tt):.2f}s")
                print(f"  Sync time:  mean={sy.mean():.2f}s  median={np.median(sy):.2f}s")

            # save epoch checkpoint
            epoch_checkpoint_path = os.path.join(checkpoint_dir, f'checkpoint_epoch_{epoch}.pt')
            self.save_checkpoint(epoch_checkpoint_path)

        # Save final model
        final_path = os.path.join(checkpoint_dir, 'final_model.pt')
        self.save_checkpoint(final_path)

        if verbose and self.accelerator.is_main_process:
            print(f"Training completed. Final model saved to {final_path}")

        # Make sure all processes reach here
        self.accelerator.wait_for_everyone()
