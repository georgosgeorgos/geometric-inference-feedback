"""Build augmented HuggingFace datasets from GIFT loop outputs."""
import os
from typing import List, Tuple, Dict
from PIL import Image
from datasets import Dataset


def build_augmented_dataset(
    base_data: List[Dict],
    reject_pairs: List[Tuple[Image.Image, str]],
    fda_pairs: List[Tuple[Image.Image, str]],
    oversample_pairs: List[Tuple[Image.Image, str]],
) -> Dataset:
    """Combine base data with GIFT-generated pairs into a HuggingFace Dataset."""
    images, codes, sources = [], [], []

    for item in base_data:
        images.append(item["image"])
        codes.append(item["code"])
        sources.append("base")

    for img, code in reject_pairs:
        images.append(img)
        codes.append(code)
        sources.append("reject")

    for img, code in fda_pairs:
        images.append(img)
        codes.append(code)
        sources.append("fda")

    for img, code in oversample_pairs:
        images.append(img)
        codes.append(code)
        sources.append("oversample")

    return Dataset.from_dict({"image": images, "code": codes, "source": sources})


def save_dataset(dataset: Dataset, output_dir: str, iteration: int) -> str:
    """Save dataset as Arrow files, versioned by iteration."""
    path = os.path.join(output_dir, f"iteration_{iteration}")
    os.makedirs(path, exist_ok=True)
    dataset.save_to_disk(path)
    return path
