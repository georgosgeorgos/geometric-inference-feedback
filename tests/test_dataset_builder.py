from PIL import Image
from gift.core.dataset_builder import build_augmented_dataset

def test_build_augmented_dataset():
    base_data = [
        {"image": Image.new("RGB", (10, 10)), "code": "base_code_0"},
        {"image": Image.new("RGB", (10, 10)), "code": "base_code_1"},
    ]
    reject_pairs = [(Image.new("RGB", (10, 10)), "reject_code")]
    fda_pairs = [(Image.new("RGB", (10, 10)), "fda_code")]
    oversample_pairs = [(Image.new("RGB", (10, 10)), "oversample_code")] * 2

    ds = build_augmented_dataset(base_data, reject_pairs, fda_pairs, oversample_pairs)
    assert len(ds) == 2 + 1 + 1 + 2
    assert "image" in ds.column_names
    assert "code" in ds.column_names
    assert "source" in ds.column_names

def test_empty_augmentation():
    base_data = [{"image": Image.new("RGB", (10, 10)), "code": "c"}]
    ds = build_augmented_dataset(base_data, [], [], [])
    assert len(ds) == 1
