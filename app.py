"""
InspectApp Vision Service
=========================
Compares two images to determine if they show the same content.
Used for:
  - Assembly instruction sheet: admin upload vs inspector photo
  - Barcode label: admin upload vs inspector photo

Deployed on Render (free tier).
Uses perceptual hashing (imagehash) + structural similarity (scikit-image)
— no paid AI API needed.

Endpoints:
  POST /compare   { "image_a": "<base64>", "image_b": "<base64>", "type": "barcode"|"assembly" }
  GET  /health    → { "status": "ok" }
"""

import os
import io
import base64
import logging
from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image
import imagehash
import numpy as np
from skimage.metrics import structural_similarity as ssim
from skimage.color import rgb2gray
from skimage.transform import resize

# ── Config ────────────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app, origins=os.environ.get("ALLOWED_ORIGINS", "*"))

# Similarity thresholds (0–1, higher = more similar)
THRESHOLDS = {
    "barcode":  {"hash": 0.85, "ssim": 0.60},   # barcodes need stricter match
    "assembly": {"hash": 0.70, "ssim": 0.45},   # assembly sheets may be photographed at angle
    "default":  {"hash": 0.75, "ssim": 0.50},
}

MAX_IMAGE_BYTES = 10 * 1024 * 1024  # 10 MB


# ── Helpers ───────────────────────────────────────────────────────────────────
def decode_image(b64_string: str) -> Image.Image:
    """Decode a base64 image string (with or without data URI prefix)."""
    if "," in b64_string:
        b64_string = b64_string.split(",", 1)[1]
    raw = base64.b64decode(b64_string)
    if len(raw) > MAX_IMAGE_BYTES:
        raise ValueError("Image too large (max 10 MB)")
    return Image.open(io.BytesIO(raw)).convert("RGB")


def image_to_array(img: Image.Image, size=(256, 256)) -> np.ndarray:
    """Resize and convert PIL image to float numpy array."""
    img_resized = img.resize(size, Image.LANCZOS)
    return np.array(img_resized, dtype=np.float64) / 255.0


def phash_similarity(img_a: Image.Image, img_b: Image.Image) -> float:
    """
    Perceptual hash similarity [0, 1].
    1.0 = identical, 0.0 = completely different.
    Uses average hash + difference hash for robustness.
    """
    # Average hash
    ah_a = imagehash.average_hash(img_a, hash_size=16)
    ah_b = imagehash.average_hash(img_b, hash_size=16)
    ah_dist = (ah_a - ah_b) / (16 * 16)

    # Difference hash
    dh_a = imagehash.dhash(img_a, hash_size=16)
    dh_b = imagehash.dhash(img_b, hash_size=16)
    dh_dist = (dh_a - dh_b) / (16 * 16)

    # Perceptual hash
    ph_a = imagehash.phash(img_a, hash_size=16)
    ph_b = imagehash.phash(img_b, hash_size=16)
    ph_dist = (ph_a - ph_b) / (16 * 16)

    # Weighted average of the three (lower distance = more similar)
    combined_dist = 0.3 * ah_dist + 0.4 * ph_dist + 0.3 * dh_dist
    return float(1.0 - combined_dist)


def ssim_similarity(img_a: Image.Image, img_b: Image.Image) -> float:
    """
    Structural Similarity Index [0, 1].
    1.0 = identical structures, 0.0 = completely different.
    Works on grayscale to be illumination-independent.
    """
    size = (256, 256)
    arr_a = rgb2gray(resize(np.array(img_a), size, anti_aliasing=True))
    arr_b = rgb2gray(resize(np.array(img_b), size, anti_aliasing=True))
    score, _ = ssim(arr_a, arr_b, full=True, data_range=1.0)
    return float(max(0.0, score))


def compare_images(img_a: Image.Image, img_b: Image.Image, compare_type: str) -> dict:
    """
    Run both comparisons and return a verdict with details.
    """
    thresh = THRESHOLDS.get(compare_type, THRESHOLDS["default"])

    hash_score = phash_similarity(img_a, img_b)
    ssim_score = ssim_similarity(img_a, img_b)

    # Combined score (weighted)
    combined = 0.55 * hash_score + 0.45 * ssim_score

    # Match requires BOTH to pass their thresholds
    hash_pass = hash_score >= thresh["hash"]
    ssim_pass = ssim_score >= thresh["ssim"]
    match = hash_pass and ssim_pass

    # Confidence label
    if combined >= 0.85:
        confidence = "high"
    elif combined >= 0.65:
        confidence = "medium"
    else:
        confidence = "low"

    # Human-readable message
    if match:
        msg = "Images match — content appears consistent."
    elif hash_pass and not ssim_pass:
        msg = "Partial match — overall layout is similar but structural details differ. This may indicate a different version or angle."
    elif not hash_pass and ssim_pass:
        msg = "Partial match — structure is similar but visual content differs. Possible lighting or quality difference."
    else:
        msg = "Images do not match — content appears significantly different."

    return {
        "match":       match,
        "confidence":  confidence,
        "combined":    round(combined, 4),
        "hash_score":  round(hash_score, 4),
        "ssim_score":  round(ssim_score, 4),
        "message":     msg,
        "type":        compare_type,
        "thresholds":  thresh,
    }


# ── Routes ────────────────────────────────────────────────────────────────────
@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "service": "InspectApp Vision"})


@app.route("/compare", methods=["POST"])
def compare():
    """
    Body (JSON):
      image_a   string  Base64-encoded image (admin upload)
      image_b   string  Base64-encoded image (inspector upload)
      type      string  "barcode" | "assembly" | "general"
      secret    string  Shared secret for basic auth (optional but recommended)
    """
    data = request.get_json(force=True, silent=True)
    if not data:
        return jsonify({"error": "JSON body required"}), 400

    # Optional shared-secret check
    expected_secret = os.environ.get("API_SECRET", "")
    if expected_secret and data.get("secret") != expected_secret:
        return jsonify({"error": "Unauthorized"}), 401

    image_a_b64 = data.get("image_a", "")
    image_b_b64 = data.get("image_b", "")
    compare_type = data.get("type", "general")

    if not image_a_b64 or not image_b_b64:
        return jsonify({"error": "Both image_a and image_b are required"}), 400

    try:
        img_a = decode_image(image_a_b64)
        img_b = decode_image(image_b_b64)
    except Exception as e:
        log.warning(f"Image decode error: {e}")
        return jsonify({"error": f"Image decode failed: {str(e)}"}), 422

    try:
        result = compare_images(img_a, img_b, compare_type)
        log.info(f"Compared [{compare_type}]: match={result['match']} combined={result['combined']}")
        return jsonify(result)
    except Exception as e:
        log.error(f"Comparison error: {e}", exc_info=True)
        return jsonify({"error": f"Comparison failed: {str(e)}"}), 500


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
