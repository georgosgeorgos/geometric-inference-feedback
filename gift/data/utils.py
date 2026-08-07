import re
import base64
from io import BytesIO
from PIL import Image


def extract_code(text: str) -> str:
    """Extract Python code from markdown code blocks."""
    pattern = r"```(?:python)?\s*\n(.*?)```"
    matches = re.findall(pattern, text, re.DOTALL)
    return "\n".join(matches).strip() if matches else text.strip()


def encode_image(image: Image.Image, fmt: str = "PNG") -> str:
    """Encode PIL Image to base64 string."""
    buf = BytesIO()
    image.save(buf, format=fmt)
    return base64.b64encode(buf.getvalue()).decode("utf-8")
