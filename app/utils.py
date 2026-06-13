import io
import base64
from PIL import Image

def img_to_base64(img: Image.Image) -> str:
    \"\"\"
    Converts a PIL Image to a base64 encoded PNG string.
    \"\"\"
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")
