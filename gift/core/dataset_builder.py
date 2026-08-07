"""Build augmented HuggingFace datasets from GIFT loop outputs."""
import os
from typing import List, Tuple, Dict, Union
from PIL import Image
from datasets import Dataset, concatenate_datasets


def _pairs_to_dataset(pairs: List[Tuple[Image.Image, str]], source: str) -> Dataset:
    """Convert (image, code) pairs to a HF Dataset with source column."""
    if not pairs:
        return None
    images, codes = zip(*pairs)
    return Dataset.from_dict({
        "image": list(images), "code": list(codes),
        "source": [source] * len(pairs),
    })


def build_augmented_dataset(
    base_data: Union[Dataset, List[Dict]],
    reject_pairs: List[Tuple[Image.Image, str]],
    fda_pairs: List[Tuple[Image.Image, str]],
    oversample_pairs: List[Tuple[Image.Image, str]],
) -> Dataset:
    """Combine base data with GIFT-generated pairs into a HuggingFace Dataset.

    base_data can be an HF Dataset (zero-copy) or a list of dicts (legacy).
    """
    if isinstance(base_data, Dataset):
        base_ds = base_data
        if "source" not in base_ds.column_names:
            base_ds = base_ds.add_column("source", ["base"] * len(base_ds))
    else:
        base_ds = Dataset.from_dict({
            "image": [d["image"] for d in base_data],
            "code": [d["code"] for d in base_data],
            "source": ["base"] * len(base_data),
        })

    parts = [base_ds]
    for pairs, source in [(reject_pairs, "reject"), (fda_pairs, "fda"), (oversample_pairs, "oversample")]:
        ds = _pairs_to_dataset(pairs, source)
        if ds is not None:
            parts.append(ds)

    if len(parts) == 1:
        return parts[0]
    return concatenate_datasets(parts)


def save_dataset(dataset: Dataset, output_dir: str, iteration: int) -> str:
    """Save dataset as Arrow files, versioned by iteration."""
    path = os.path.join(output_dir, f"iteration_{iteration}")
    os.makedirs(path, exist_ok=True)
    dataset.save_to_disk(path)
    return path
