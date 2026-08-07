"""GIFT-FDA: failure-driven augmentation via 3D rendering."""
from typing import List, Tuple
from PIL import Image
from gift.core.renderer import render_code_to_image


def build_fda_pairs(
    fda_indices: List[int],
    gen_codes: List[str],
    gt_codes: List[str],
    render_size: int = 448,
) -> List[Tuple[Image.Image, str]]:
    """Render near-miss generated code to 2D, pair with ground truth code.

    Skips candidates where rendering fails.
    """
    pairs = []
    for idx in fda_indices:
        rendered = render_code_to_image(
            gen_codes[idx], size=(render_size, render_size)
        )
        if rendered is not None:
            pairs.append((rendered, gt_codes[idx]))
    return pairs
