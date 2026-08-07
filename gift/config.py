from dataclasses import dataclass


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
