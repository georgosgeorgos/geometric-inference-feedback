from unittest.mock import patch
from PIL import Image
from gift.core.fda import build_fda_pairs

def test_build_fda_pairs_with_mock_renderer():
    fake_img = Image.new("RGB", (448, 448), "blue")
    fda_indices = [1, 3]
    gen_codes = ["code_0", "code_1", "code_2", "code_3"]
    gt_codes = ["gt_0", "gt_1", "gt_2", "gt_3"]

    with patch("gift.core.fda.render_code_to_image") as mock_render:
        mock_render.return_value = fake_img
        pairs = build_fda_pairs(fda_indices, gen_codes, gt_codes)

    assert len(pairs) == 2
    assert pairs[0] == (fake_img, "gt_1")
    assert pairs[1] == (fake_img, "gt_3")

def test_build_fda_pairs_render_failure():
    with patch("gift.core.fda.render_code_to_image") as mock_render:
        mock_render.return_value = None
        pairs = build_fda_pairs([0], ["bad_code"], ["gt_0"])

    assert len(pairs) == 0
