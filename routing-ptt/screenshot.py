#pip install mss pillow
import io, base64
import mss, mss.tools
from PIL import Image

def capture_fullscreen_b64(max_width: int = 16000, quality: int = 85) -> str:
    """
    Grabs the primary monitor, optionally downsizes to save tokens,
    and returns a base64-encoded PNG string.
    """
    with mss.mss() as sct:
        shot = sct.grab(sct.monitors[0])  # full primary screen
        png_bytes = mss.tools.to_png(shot.rgb, shot.size)

    # Downsize to keep cost reasonable (optional but recommended)
    im = Image.open(io.BytesIO(png_bytes))
    if im.width > max_width:
        new_h = int(im.height * (max_width / im.width))
        im = im.resize((max_width, new_h), Image.LANCZOS)

    buf = io.BytesIO()
    im.save(buf, format="PNG", optimize=True, compress_level=9)
    return base64.b64encode(buf.getvalue()).decode("utf-8")
