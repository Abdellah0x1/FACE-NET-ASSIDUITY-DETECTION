"""
face_utils.py
Shared parameters and helper functions for the whole project.
The model engineer should import this file too, so that enrollment crops and
live webcam crops are produced with exactly the same settings.
"""
import cv2
import numpy as np
import torch
from PIL import Image, ImageOps
from facenet_pytorch import MTCNN, InceptionResnetV1

# ---------- Shared parameters (agree on these with the model engineer) ----------
IMAGE_SIZE = 160          # FaceNet input size
MARGIN = 20               # margin around the face box (MTCNN units)
MIN_CONFIDENCE = 0.95     # minimum MTCNN detection confidence
MAX_SIDE = 1024           # big photos are downscaled before detection (speed)
WEIGHTS = "vggface2"      # FaceNet pretrained weights
EXTS = {".jpg", ".jpeg", ".png"}
# --------------------------------------------------------------------------------

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_image(path):
    """Open a photo as RGB, apply the phone's EXIF rotation, downscale if huge."""
    img = ImageOps.exif_transpose(Image.open(path)).convert("RGB")
    w, h = img.size
    scale = MAX_SIDE / max(w, h)
    if scale < 1:
        img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
    return img


def get_mtcnn():
    return MTCNN(image_size=IMAGE_SIZE, margin=MARGIN, select_largest=True,
                 post_process=False, device=DEVICE)


def detect_and_crop(mtcnn, img):
    """PIL RGB image -> (crop as 160x160x3 uint8 array, confidence).
    Returns (None, 0.0) when no face is found."""
    face, prob = mtcnn(img, return_prob=True)
    if face is None or prob is None:
        return None, 0.0
    crop = face.permute(1, 2, 0).clamp(0, 255).byte().cpu().numpy()
    return crop, float(np.asarray(prob, dtype=float).max())


def crop_blur(crop_rgb):
    """Laplacian variance of a face crop (comparable across cameras)."""
    gray = cv2.cvtColor(crop_rgb, cv2.COLOR_RGB2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def get_facenet():
    return InceptionResnetV1(pretrained=WEIGHTS).eval().to(DEVICE)


@torch.no_grad()
def embed(model, crops, batch_size=32):
    """List of 160x160x3 uint8 crops -> array (N, 512), L2-normalized by FaceNet."""
    out = []
    for i in range(0, len(crops), batch_size):
        x = torch.from_numpy(np.stack(crops[i:i + batch_size])).permute(0, 3, 1, 2).float()
        x = (x - 127.5) / 128.0           # FaceNet's standard normalization
        out.append(model(x.to(DEVICE)).cpu().numpy())
    return np.concatenate(out)