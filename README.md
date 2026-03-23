# InspectApp Vision Service

Python image comparison microservice. Deployed on Render (free tier).

## How it works

Compares two images using:
1. **Perceptual hashing** (average + difference + phash) — fast fingerprint of visual content
2. **SSIM** (Structural Similarity Index) — measures structural/texture similarity

Both scores are combined. If either score falls below the threshold for that comparison type, it flags a mismatch.

## Comparison types

| Type | Use case | Hash threshold | SSIM threshold |
|---|---|---|---|
| `barcode` | Barcode label admin vs inspector | 0.85 | 0.60 |
| `assembly` | Assembly sheet admin vs inspector | 0.70 | 0.45 |
| `general` | Fallback | 0.75 | 0.50 |

Barcode is stricter because two barcodes that look similar but encode different data must be caught.
Assembly is looser because the inspector photographs the sheet at an angle under factory lighting.

## Deploy to Render

1. Push this folder to a GitHub repo
2. Go to [render.com](https://render.com) → New → Web Service → connect your repo
3. Render auto-detects `render.yaml` — click Deploy
4. After deploy, copy the service URL (e.g. `https://inspectapp-vision.onrender.com`)
5. In your PHP config (`includes/db.php` or a new `config.php`), set:
   ```php
   define('VISION_SERVICE_URL', 'https://inspectapp-vision.onrender.com');
   define('VISION_API_SECRET',  'your-secret-from-render-env-vars');
   ```

## API

### POST /compare
```json
{
  "image_a": "<base64 string>",
  "image_b": "<base64 string>",
  "type":    "barcode",
  "secret":  "your-api-secret"
}
```

### Response
```json
{
  "match":      true,
  "confidence": "high",
  "combined":   0.91,
  "hash_score": 0.94,
  "ssim_score": 0.87,
  "message":    "Images match — content appears consistent.",
  "type":       "barcode"
}
```

### GET /health
```json
{ "status": "ok", "service": "InspectApp Vision" }
```

## Notes on Render free tier

- Free instances **spin down after 15 min of inactivity** — the first request after spin-down takes ~30 seconds
- This is fine for InspectApp: comparisons happen when the inspector clicks "Next" after uploading an image, so a brief wait is acceptable
- The service uses ~200 MB RAM — within the free tier limit
