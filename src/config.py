"""Configuración compartida del proyecto (lee variables de .env o de st.secrets)."""
from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
RESULTS_DIR = ROOT / "resultados"

# Clases del modelo (tag de Custom Vision -> etiqueta legible)
POSITIVE_TAG = "danado"      # clase positiva: vehículo con daño
NEGATIVE_TAG = "sin_dano"    # clase negativa: vehículo sin daño
TAGS = [POSITIVE_TAG, NEGATIVE_TAG]
TAG_LABELS = {POSITIVE_TAG: "Con daño", NEGATIVE_TAG: "Sin daño"}

try:
    from dotenv import load_dotenv

    load_dotenv(ROOT / ".env")
except ImportError:  # python-dotenv es opcional
    pass


def get(name: str, default: str | None = None) -> str | None:
    """Busca una variable en el entorno y, si existe, en st.secrets (Streamlit Cloud)."""
    value = os.getenv(name)
    if value:
        return value
    try:
        import streamlit as st

        if name in st.secrets:
            return str(st.secrets[name])
    except Exception:
        pass
    return default
