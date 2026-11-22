"""Cliente REST mínimo para el endpoint de predicción de Azure Custom Vision.

Variables necesarias (.env):
    CV_PREDICTION_ENDPOINT  p. ej. https://eastus.api.cognitive.microsoft.com/
    CV_PREDICTION_KEY       clave del recurso de predicción
    CV_PROJECT_ID           GUID del proyecto
    CV_PUBLISH_NAME         nombre con el que se publicó la iteración (p. ej. iter3-advanced)

Si CV_MOCK=1 se devuelven probabilidades simuladas (solo para probar la app sin Azure).
"""
from __future__ import annotations

import hashlib
import io
import time

import requests

from . import config


class PredictionError(RuntimeError):
    pass


def is_mock() -> bool:
    return config.get("CV_MOCK", "0") == "1"


def prediction_url(publish_name: str | None = None) -> str:
    endpoint = (config.get("CV_PREDICTION_ENDPOINT") or "").rstrip("/")
    project = config.get("CV_PROJECT_ID")
    publish = publish_name or config.get("CV_PUBLISH_NAME")
    if not (endpoint and project and publish):
        raise PredictionError(
            "Faltan CV_PREDICTION_ENDPOINT, CV_PROJECT_ID o CV_PUBLISH_NAME en .env"
        )
    return f"{endpoint}/customvision/v3.0/Prediction/{project}/classify/iterations/{publish}/image"


def _mock_predict(image_bytes: bytes) -> dict[str, float]:
    # Determinista por imagen: la misma foto siempre da el mismo resultado.
    seed = int(hashlib.md5(image_bytes).hexdigest()[:8], 16)
    p = (seed % 1000) / 1000
    return {config.POSITIVE_TAG: p, config.NEGATIVE_TAG: 1 - p}


def predict(image_bytes: bytes, publish_name: str | None = None,
            max_retries: int = 6, timeout: int = 30) -> dict[str, float]:
    """Envía una imagen al endpoint y devuelve {tag: probabilidad}."""
    if is_mock():
        return _mock_predict(image_bytes)

    key = config.get("CV_PREDICTION_KEY")
    if not key:
        raise PredictionError("Falta CV_PREDICTION_KEY en .env")
    url = prediction_url(publish_name)
    headers = {"Prediction-Key": key, "Content-Type": "application/octet-stream"}

    for attempt in range(max_retries):
        resp = requests.post(url, headers=headers, data=image_bytes, timeout=timeout)
        if resp.status_code == 429 or resp.status_code >= 500:
            # Límite de transacciones por segundo (F0: 2 TPS) -> backoff exponencial
            time.sleep(min(2 ** attempt, 20))
            continue
        if resp.status_code != 200:
            raise PredictionError(f"HTTP {resp.status_code}: {resp.text[:300]}")
        preds = resp.json().get("predictions", [])
        out = {t: 0.0 for t in config.TAGS}
        for p in preds:
            out[p["tagName"]] = float(p["probability"])
        return out
    raise PredictionError("Se agotaron los reintentos (HTTP 429/5xx)")


def prepare_image(image_bytes: bytes, max_side: int = 1600) -> bytes:
    """Normaliza a JPEG RGB y limita el lado mayor (límite de Custom Vision: 4 MB por predicción)."""
    from PIL import Image, ImageOps

    im = Image.open(io.BytesIO(image_bytes))
    im = ImageOps.exif_transpose(im).convert("RGB")
    if max(im.size) > max_side:
        im.thumbnail((max_side, max_side))
    buf = io.BytesIO()
    im.save(buf, format="JPEG", quality=92)
    return buf.getvalue()
