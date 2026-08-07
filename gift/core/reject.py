"""GIFT-REJECT: classify candidates by IoU thresholds."""
from typing import List, Tuple, Any


def classify_candidates(
    ious: List[float],
    tau_accept: float = 0.9,
    tau_reject: float = 0.5,
) -> Tuple[List[int], List[int], List[int]]:
    """Classify candidate indices into accepted, fda-eligible, and discarded.

    Returns (accepted_indices, fda_indices, discarded_indices).
    """
    accepted, fda, discarded = [], [], []
    for i, iou in enumerate(ious):
        if iou >= tau_accept:
            accepted.append(i)
        elif iou > tau_reject:
            fda.append(i)
        else:
            discarded.append(i)
    return accepted, fda, discarded


def build_reject_pairs(
    accepted_indices: List[int],
    images: List[Any],
    codes: List[str],
) -> List[Tuple[Any, str]]:
    """Build (image, generated_code) training pairs from accepted candidates."""
    return [(images[i], codes[i]) for i in accepted_indices]
