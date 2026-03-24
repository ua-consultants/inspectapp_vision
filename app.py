"""
InspectApp Vision Service
=========================
Compares two images to check if they show the same content.
Uses only Pillow + numpy — no extra CV libraries needed.

Algorithm:
  1. Resize both images to 64x64 grayscale
  2. Compute mean absolute difference (MAD) of pixel values → 0=identical, 1=completely different
  3. Compute histogram intersection similarity → 1=identical distributions, 0=completely different
  4. Combine both → final similarity score 0-1

Endpoints:
  POST /compare   { "image_a": "<base64>", "image_b": "<base64>", "type": "barcode"|"assembly" }
  GET  /health    → { "status": "ok" }
"""

import os, io, base64, logging
import numpy as np
from PIL import Image, ImageFile, ImageOps
from flask import Flask, request, jsonify
from flask_cors import CORS

ImageFile.LOAD_TRUNCATED_IMAGES = True
logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app, origins=os.environ.get("ALLOWED_ORIGINS", "*"))

MAX_IMAGE_BYTES = 10 * 1024 * 1024  # 10 MB

# Lower threshold = more strict. 0.0=identical, 1.0=totally different
# If difference score > threshold → flag as mismatch
THRESHOLDS = {
    "barcode":  0.15,   # strict — barcode content must match closely
    "assembly": 0.35,   # looser — sheet may be photographed at angle/different lighting
    "general":  0.25,
}


def decode_image(b64: str) -> Image.Image:
    """Decode base64 data URI or raw base64 to PIL Image."""
    if "," in b64:
        b64 = b64.split(",", 1)[1]
    b64 = b64.strip().replace("\n","").replace("\r","").replace(" ","")
    try:
        raw = base64.b64decode(b64)
    except Exception as e:
        raise ValueError(f"Base64 decode error: {e}")
    if len(raw) > MAX_IMAGE_BYTES:
        raise ValueError("Image exceeds 10 MB limit")
    buf = io.BytesIO(raw)
    buf.seek(0)
    try:
        img = Image.open(buf)
        img.load()
        return img.convert("RGB")
    except Exception as e:
        raise ValueError(f"Image decode error: {e}")


def to_gray_array(img: Image.Image, size: int = 64) -> np.ndarray:
    """Resize to square, convert to grayscale, return float32 array 0-1."""
    resized = img.resize((size, size), Image.LANCZOS)
    gray    = ImageOps.grayscale(resized)
    arr     = np.array(gray, dtype=np.float32) / 255.0
    return arr


def mean_diff_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """
    1 - mean absolute difference.
    Returns 1.0 for identical images, 0.0 for maximally different.
    """
    diff = np.abs(a.astype(np.float32) - b.astype(np.float32))
    mad  = float(np.mean(diff))          # 0.0 – 1.0
    return 1.0 - mad


def histogram_similarity(a: np.ndarray, b: np.ndarray, bins: int = 64) -> float:
    """
    Histogram intersection similarity.
    Returns 1.0 for identical histograms, lower for different distributions.
    Robust to brightness shifts.
    """
    hist_a, _ = np.histogram(a, bins=bins, range=(0, 1))
    hist_b, _ = np.histogram(b, bins=bins, range=(0, 1))
    # Normalize
    hist_a = hist_a.astype(np.float32) / (hist_a.sum() + 1e-9)
    hist_b = hist_b.astype(np.float32) / (hist_b.sum() + 1e-9)
    # Intersection
    intersection = np.minimum(hist_a, hist_b).sum()
    return float(intersection)


def block_hash_similarity(img: Image.Image, size: int = 16) -> np.ndarray:
    """Simple block hash — resize to size×size, return binary array."""
    small = img.resize((size, size), Image.LANCZOS).convert("L")
    arr   = np.array(small, dtype=np.float32)
    return (arr > arr.mean()).flatten()


def hash_similarity(img_a: Image.Image, img_b: Image.Image) -> float:
    """Proportion of matching bits in block hash. 1.0=identical, 0.0=opposite."""
    h_a = block_hash_similarity(img_a)
    h_b = block_hash_similarity(img_b)
    matching = np.sum(h_a == h_b)
    return float(matching) / float(len(h_a))


def compare_images(img_a: Image.Image, img_b: Image.Image, ctype: str) -> dict:
    """Run comparison and return structured result."""
    threshold = THRESHOLDS.get(ctype, THRESHOLDS["general"])

    arr_a = to_gray_array(img_a)
    arr_b = to_gray_array(img_b)

    mad_sim  = mean_diff_similarity(arr_a, arr_b)   # 0–1, higher = more similar
    hist_sim = histogram_similarity(arr_a, arr_b)    # 0–1, higher = more similar
    hash_sim = hash_similarity(img_a, img_b)         # 0–1, higher = more similar

    # Weighted combination — MAD is most reliable for content comparison
    combined = 0.50 * mad_sim + 0.25 * hist_sim + 0.25 * hash_sim

    # Difference score: 0=identical, 1=completely different
    diff_score = 1.0 - combined

    match = diff_score <= threshold

    if combined >= 0.90:
        confidence = "high"
    elif combined >= 0.70:
        confidence = "medium"
    else:
        confidence = "low"

    if match:
        if combined >= 0.95:
            message = "Images match — content is consistent."
        else:
            message = "Images match — minor visual differences likely due to lighting or angle."
    else:
        if diff_score >= 0.60:
            message = "Images do not match — content appears significantly different."
        elif diff_score >= 0.40:
            message = "Images differ — moderate differences detected. This may be a different document or label."
        else:
            message = "Images have minor differences — could be the same content under different conditions, but exceeds the acceptable threshold."

    return {
        "match":       bool(match),
        "confidence":  confidence,
        "diff_score":  round(float(diff_score), 4),
        "combined":    round(float(combined), 4),
        "mad_sim":     round(float(mad_sim), 4),
        "hist_sim":    round(float(hist_sim), 4),
        "hash_sim":    round(float(hash_sim), 4),
        "threshold":   threshold,
        "message":     message,
        "type":        ctype,
    }


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "service": "InspectApp Vision"})


@app.route("/compare", methods=["POST"])
def compare():
    data = request.get_json(force=True, silent=True)
    if not data:
        return jsonify({"error": "JSON body required"}), 400

    # Optional secret check
    secret = os.environ.get("API_SECRET", "")
    if secret and data.get("secret") != secret:
        return jsonify({"error": "Unauthorized"}), 401

    a_b64  = data.get("image_a", "")
    b_b64  = data.get("image_b", "")
    ctype  = data.get("type", "general")

    if not a_b64 or not b_b64:
        return jsonify({"error": "Both image_a and image_b are required"}), 400

    try:
        img_a = decode_image(a_b64)
    except Exception as e:
        return jsonify({"error": f"image_a decode failed: {e}"}), 422

    try:
        img_b = decode_image(b_b64)
    except Exception as e:
        return jsonify({"error": f"image_b decode failed: {e}"}), 422

    try:
        result = compare_images(img_a, img_b, ctype)
        log.info(f"[{ctype}] match={result['match']} diff={result['diff_score']} conf={result['confidence']}")
        return jsonify(result)
    except Exception as e:
        log.error(f"Comparison error: {e}", exc_info=True)
        return jsonify({"error": f"Comparison failed: {e}"}), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
