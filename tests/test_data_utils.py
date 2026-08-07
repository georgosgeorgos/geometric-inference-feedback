from gift.data.utils import extract_code, encode_image
from PIL import Image
import base64

def test_extract_code_markdown():
    text = "Here is code:\n```python\nresult = cq.Workplane('XY').box(1,1,1)\nsolid = result\n```\nDone."
    assert "result = cq.Workplane" in extract_code(text)
    assert "solid = result" in extract_code(text)

def test_extract_code_no_block():
    assert extract_code("no code here") == "no code here"

def test_encode_image():
    img = Image.new("RGB", (10, 10), "red")
    b64 = encode_image(img)
    decoded = base64.b64decode(b64)
    assert len(decoded) > 0
