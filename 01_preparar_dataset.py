"""Paso 1 — Preparación del dataset "Car Damage Detection" (Kaggle: anujms/car-damage-detection).

Qué hace:
  1. Lee raw/extracted/data1a/{training,validation}/{00-damage,01-whole}.
  2. Normaliza cada imagen: corrige orientación EXIF, convierte a RGB, limita el lado mayor
     a 1600 px y guarda en JPEG (calidad 95). Así ninguna supera el límite de 6 MB de Custom Vision.
  3. Elimina duplicados exactos (MD5 del píxel) y casi duplicados (dHash 64 bits, Hamming <= 4):
       - duplicado entre train y test  -> se quita de train (evita fuga de información);
       - duplicado con etiquetas distintas -> se quitan todas las copias (etiqueta ambigua);
       - duplicado dentro del mismo split y clase -> se conserva una sola copia.
  4. Escribe data/train/<tag>/ y data/test/<tag>/, un manifiesto CSV y un resumen JSON,
     y genera figuras para el informe (distribución de clases, tamaños, muestra de imágenes).

El split de Kaggle "validation" se usa como conjunto de PRUEBA independiente: nunca se sube
a Custom Vision, así las métricas finales no están sesgadas por el entrenamiento.

Uso:
    python 01_preparar_dataset.py
"""
from __future__ import annotations

import csv
import hashlib
import json
import shutil
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image, ImageOps

from src import config

RAW = config.ROOT / "raw" / "extracted" / "data1a"
OUT = config.DATA_DIR
FIG = config.RESULTS_DIR / "figuras_dataset"
FOLDER_TO_TAG = {"00-damage": config.POSITIVE_TAG, "01-whole": config.NEGATIVE_TAG}
SPLIT_MAP = {"training": "train", "validation": "test"}
MAX_SIDE = 1600
HAMMING_MAX = 4


def dhash(im: Image.Image, size: int = 8) -> int:
    g = im.convert("L").resize((size + 1, size), Image.LANCZOS)
    px = list(g.getdata())
    bits = 0
    for r in range(size):
        for c in range(size):
            bits = (bits << 1) | (px[r * (size + 1) + c] > px[r * (size + 1) + c + 1])
    return bits


def load_records():
    recs = []
    for split_src, split in SPLIT_MAP.items():
        for folder, tag in FOLDER_TO_TAG.items():
            for p in sorted((RAW / split_src / folder).iterdir()):
                if not p.is_file():
                    continue
                im = ImageOps.exif_transpose(Image.open(p)).convert("RGB")
                recs.append({
                    "src": p, "split": split, "tag": tag, "im": im,
                    "w": im.size[0], "h": im.size[1],
                    "md5": hashlib.md5(im.tobytes()).hexdigest(),
                    "dhash": dhash(im),
                })
    return recs


def group_duplicates(recs):
    """Union-find sobre MD5 idéntico o dHash a distancia de Hamming <= HAMMING_MAX."""
    parent = list(range(len(recs)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(a, b):
        parent[find(a)] = find(b)

    by_md5 = defaultdict(list)
    for i, r in enumerate(recs):
        by_md5[r["md5"]].append(i)
    for idx in by_md5.values():
        for j in idx[1:]:
            union(idx[0], j)
    hashes = [r["dhash"] for r in recs]
    for i in range(len(recs)):
        for j in range(i + 1, len(recs)):
            if bin(hashes[i] ^ hashes[j]).count("1") <= HAMMING_MAX:
                union(i, j)
    groups = defaultdict(list)
    for i in range(len(recs)):
        groups[find(i)].append(i)
    return [g for g in groups.values() if len(g) > 1]


def resolve(recs, groups):
    drop = {}
    for g in groups:
        tags = {recs[i]["tag"] for i in g}
        splits = {recs[i]["split"] for i in g}
        if len(tags) > 1:
            for i in g:
                drop[i] = "etiqueta_ambigua"
            continue
        if "test" in splits:
            keep_test = [i for i in g if recs[i]["split"] == "test"]
            for i in g:
                if recs[i]["split"] == "train":
                    drop[i] = "fuga_train_test"
            for i in keep_test[1:]:
                drop[i] = "duplicado_test"
        else:
            for i in g[1:]:
                drop[i] = "duplicado_train"
    return drop


def save_figures(kept, all_recs):
    FIG.mkdir(parents=True, exist_ok=True)
    counts = Counter((r["split"], r["tag"]) for r in kept)

    fig, ax = plt.subplots(figsize=(5.2, 3.6), dpi=150)
    x = range(2)
    for k, tag in enumerate(config.TAGS):
        vals = [counts[("train", tag)], counts[("test", tag)]]
        bars = ax.bar([i + (k - 0.5) * 0.38 for i in x], vals, width=0.38,
                      color=["#e8833a", "#2a6fdb"][k], label=config.TAG_LABELS[tag])
        ax.bar_label(bars, fontsize=8)
    ax.set_xticks(list(x), ["Entrenamiento", "Prueba"])
    ax.set_ylabel("Imágenes")
    ax.set_title("Distribución de clases después de la limpieza", fontsize=11)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG / "distribucion_clases.png")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(5.2, 3.6), dpi=150)
    ax.hist([min(r["w"], r["h"]) for r in all_recs], bins=40, color="#2a6fdb")
    ax.axvline(256, ls="--", color="#333", lw=1, label="256 px (mínimo recomendado)")
    ax.set_xlabel("Lado menor de la imagen original (px)")
    ax.set_ylabel("Imágenes")
    ax.set_title("Resolución de las imágenes originales", fontsize=11)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG / "resoluciones.png")
    plt.close(fig)

    fig, axes = plt.subplots(2, 6, figsize=(12, 4.4), dpi=130)
    for row, tag in enumerate(config.TAGS):
        sample = [r for r in kept if r["tag"] == tag and r["split"] == "train"][::97][:6]
        for col, r in enumerate(sample):
            ax = axes[row, col]
            ax.imshow(r["im"])
            ax.axis("off")
            if col == 0:
                ax.set_title(config.TAG_LABELS[tag], loc="left", fontsize=11)
    fig.suptitle("Ejemplos del conjunto de entrenamiento", fontsize=12)
    fig.tight_layout()
    fig.savefig(FIG / "muestra_imagenes.png")
    plt.close(fig)


def main():
    if not RAW.exists():
        raise SystemExit(f"No existe {RAW}. Descarga y descomprime el dataset primero (ver README).")
    print("Leyendo imágenes...")
    recs = load_records()
    print(f"  {len(recs)} imágenes leídas")

    groups = group_duplicates(recs)
    drop = resolve(recs, groups)
    print(f"  grupos de duplicados: {len(groups)} | imágenes descartadas: {len(drop)}")
    print("  motivos:", dict(Counter(drop.values())))

    for split in ("train", "test"):
        shutil.rmtree(OUT / split, ignore_errors=True)
    kept, manifest = [], []
    counters = Counter()
    for i, r in enumerate(recs):
        reason = drop.get(i, "")
        row = {"origen": str(r["src"].relative_to(config.ROOT)), "split": r["split"],
               "tag": r["tag"], "ancho": r["w"], "alto": r["h"], "descartada": reason, "archivo": ""}
        if not reason:
            counters[(r["split"], r["tag"])] += 1
            dest = OUT / r["split"] / r["tag"] / f"{r['tag']}_{r['split']}_{counters[(r['split'], r['tag'])]:04d}.jpg"
            dest.parent.mkdir(parents=True, exist_ok=True)
            im = r["im"].copy()
            if max(im.size) > MAX_SIDE:
                im.thumbnail((MAX_SIDE, MAX_SIDE))
            im.save(dest, "JPEG", quality=95)
            row["archivo"] = str(dest.relative_to(config.ROOT))
            kept.append(r)
        manifest.append(row)

    OUT.mkdir(parents=True, exist_ok=True)
    with open(OUT / "manifiesto.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(manifest[0].keys()))
        w.writeheader()
        w.writerows(manifest)

    short = [min(r["w"], r["h"]) for r in recs]
    summary = {
        "fuente": "Kaggle — anujms/car-damage-detection (data1a)",
        "imagenes_originales": len(recs),
        "grupos_duplicados": len(groups),
        "descartadas": dict(Counter(drop.values())),
        "final": {f"{s}/{t}": counters[(s, t)] for s in ("train", "test") for t in config.TAGS},
        "total_final": len(kept),
        "lado_menor_px": {"min": min(short), "mediana": sorted(short)[len(short) // 2], "max": max(short)},
        "lado_menor_menor_256": sum(s < 256 for s in short),
    }
    (OUT / "resumen_dataset.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    save_figures(kept, recs)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"Figuras en {FIG}")


if __name__ == "__main__":
    main()
