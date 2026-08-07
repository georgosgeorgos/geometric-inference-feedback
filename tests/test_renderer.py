import pytest
from gift.core.renderer import render_code_to_image

def test_render_simple_box():
    code = "solid = cq.Workplane('XY').box(10, 10, 10)"
    img = render_code_to_image(code, size=(224, 224), timeout=30)
    assert img is not None
    assert img.size == (224, 224)

def test_render_bad_code():
    img = render_code_to_image("raise ValueError('bad')", timeout=10)
    assert img is None
