"""The NaVIDA frame contract, shared by training and the live driver.

Upstream resizes every frame to a fixed size before the Qwen processor so that
one frame lands on a 22x18 grid of visual tokens; the size depends on the
backbone's patch size, so it is a property of the model type and every frame
producer reads it from here.
"""
import io

import numpy as np
from PIL import Image

# Sensor-frame downscale before the resize (upstream eval.sh --resolution-ratio);
# the Qwen3-VL base is evaluated and trained at 1.0.
RESOLUTION_RATIO = 1.0
NAVIDA_MAX_HISTORY = 200        # upstream eval.sh --max-action-history
# Processor max_pixels the judge is trained, scored and deployed with (640 * 28 * 28).
JUDGE_MAX_PIXELS = 501760

IMG_SIZES = {"qwen2_5_vl": (308, 252), "qwen3_vl": (352, 288)}   # (width, height)

_active_model_type = None


def img_size_for(model_type: str):
    if model_type not in IMG_SIZES:
        raise ValueError(f"no image contract for model_type {model_type!r}; "
                         f"known: {sorted(IMG_SIZES)}")
    return IMG_SIZES[model_type]


def configure(model_type: str) -> None:
    """Select the frame contract for this process; the host calls it on load."""
    global _active_model_type
    img_size_for(model_type)
    _active_model_type = model_type


def img_size():
    if _active_model_type is None:
        raise RuntimeError("no frame contract selected: call navida_io.configure(model_type) "
                           "or load the backbone through moon.harness.host first")
    return IMG_SIZES[_active_model_type]


def resize_navida(img: Image.Image) -> Image.Image:
    return img.resize(img_size())


def jpeg_bytes(img: Image.Image) -> bytes:
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return buf.getvalue()


def jpeg_roundtrip(img: Image.Image) -> Image.Image:
    return Image.open(io.BytesIO(jpeg_bytes(img))).convert("RGB")


def frame_from_sensor(rgb: np.ndarray,
                      resolution_ratio: float = RESOLUTION_RATIO) -> Image.Image:
    """Sensor frame -> policy input: downscale, then resize to the frame contract.
    The pixels go straight to the processor, as in the upstream evaluation."""
    rgb = np.ascontiguousarray(rgb[:, :, :3])
    if resolution_ratio < 1:
        import cv2
        rgb = cv2.resize(rgb, (0, 0), fx=resolution_ratio, fy=resolution_ratio)
    return resize_navida(Image.fromarray(rgb.astype("uint8")).convert("RGB"))


def load_navida_image(path) -> Image.Image:
    return resize_navida(Image.open(path).convert("RGB"))


def judge_frame_from_sensor(rgb: np.ndarray) -> Image.Image:
    """Sensor frame -> judge input. The judge's training rows were full-resolution
    JPEG dumps resized by load_navida_image, so deployment reproduces that chain
    rather than the policy's downscale."""
    img = Image.fromarray(np.asarray(rgb).astype("uint8")).convert("RGB")
    return resize_navida(jpeg_roundtrip(img))

