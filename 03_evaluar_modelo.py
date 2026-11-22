"""Paso 3 — Evaluación independiente del modelo publicado contra el conjunto de PRUEBA.

Custom Vision solo reporta precision, recall y AP sobre su propia partición interna.
Este script envía cada imagen de data/test/ (nunca usada en entrenamiento) al endpoint de
predicción y calcula: precision, recall, F1, especificidad, exactitud, matriz de confusión,
curva ROC + AUC (con IC 95 % por bootstrap), curva Precision-Recall + AP, barrido de umbral,
umbrales óptimos (Youden, F1 máx., recall >= 95 %) y galería de errores.

Uso:
    python 03_evaluar_modelo.py --modelos iter1-baseline iter2-full iter3-advanced
    python 03_evaluar_modelo.py --modelos demo --mock      # prueba sin Azure (resultados aleatorios)

El ÚLTIMO modelo de la lista se toma como modelo final para las figuras detalladas.
Las predicciones se guardan en resultados/predicciones/<modelo>.csv y se reutilizan
(si se interrumpe, se reanuda donde quedó).
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from src import config, cv_client, metricas

PRED_DIR = config.RESULTS_DIR / "predicciones"
FIG_DIR = config.RESULTS_DIR / "figuras_evaluacion"


def test_files():
    items = []
    for tag in config.TAGS:
        for p in sorted((config.DATA_DIR / "test" / tag).glob("*.jpg")):
            items.append((p, 1 if tag == config.POSITIVE_TAG else 0))
    if not items:
        raise SystemExit("No hay imágenes en data/test. Ejecuta 01_preparar_dataset.py")
    return items


def run_predictions(model: str, tps: float) -> list[dict]:
    PRED_DIR.mkdir(parents=True, exist_ok=True)
    path = PRED_DIR / f"{model}.csv"
    rows = {}
    if path.exists():
        with open(path, encoding="utf-8") as f:
            rows = {r["archivo"]: r for r in csv.DictReader(f)}
    items = test_files()
    pending = [(p, y) for p, y in items if p.name not in rows]
    print(f"[{model}] {len(items)} imágenes de prueba, {len(pending)} por predecir")
    fields = ["archivo", "y_true", "p_danado", "p_sin_dano", "latencia_ms"]
    with open(path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        if not rows:
            w.writeheader()
        for k, (p, y) in enumerate(pending, 1):
            t0 = time.perf_counter()
            probs = cv_client.predict(p.read_bytes(), publish_name=model)
            ms = (time.perf_counter() - t0) * 1000
            row = {"archivo": p.name, "y_true": y, "p_danado": f"{probs[config.POSITIVE_TAG]:.6f}",
                   "p_sin_dano": f"{probs[config.NEGATIVE_TAG]:.6f}", "latencia_ms": f"{ms:.0f}"}
            w.writerow(row)
            f.flush()
            rows[p.name] = row
            if k % 25 == 0 or k == len(pending):
                print(f"   {k}/{len(pending)}", flush=True)
            if not cv_client.is_mock():
                time.sleep(max(0.0, 1.0 / tps - ms / 1000))
    order = {p.name: i for i, (p, _) in enumerate(items)}
    return sorted((r for r in rows.values() if r["archivo"] in order), key=lambda r: order[r["archivo"]])


def error_gallery(rows, threshold, out_path, k=6):
    fp = sorted([r for r in rows if int(r["y_true"]) == 0 and float(r["p_danado"]) >= threshold],
                key=lambda r: -float(r["p_danado"]))[:k]
    fn = sorted([r for r in rows if int(r["y_true"]) == 1 and float(r["p_danado"]) < threshold],
                key=lambda r: float(r["p_danado"]))[:k]
    if not fp and not fn:
        return False
    from PIL import ImageOps

    fig, axes = plt.subplots(2, k, figsize=(2.2 * k, 5.0), dpi=130)
    for row_i, (title, group, tag) in enumerate([
        ("Falsos positivos (real: sin daño → predicho: con daño)", fp, config.NEGATIVE_TAG),
        ("Falsos negativos (real: con daño → predicho: sin daño)", fn, config.POSITIVE_TAG),
    ]):
        for j in range(k):
            ax = axes[row_i, j]
            ax.axis("off")
            if j < len(group):
                r = group[j]
                im = Image.open(config.DATA_DIR / "test" / tag / r["archivo"]).convert("RGB")
                ax.imshow(ImageOps.fit(im, (320, 240)))
                ax.set_title(f"P(daño) = {float(r['p_danado']):.2f}", fontsize=8)
    fig.tight_layout(rect=(0, 0, 1, 0.93), h_pad=3.0)
    fig.text(0.01, 0.97, "Falsos positivos (real: sin daño → predicho: con daño)", fontsize=11, weight="bold")
    fig.text(0.01, 0.49, "Falsos negativos (real: con daño → predicho: sin daño)", fontsize=11, weight="bold")
    fig.savefig(out_path)
    plt.close(fig)
    return True


def fmt(x):
    return f"{x:.3f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--modelos", nargs="+", default=[config.get("CV_PUBLISH_NAME") or "modelo"],
                    help="nombres de publicación (publishName) de las iteraciones a comparar")
    ap.add_argument("--umbral", type=float, default=0.5)
    ap.add_argument("--tps", type=float, default=float(config.get("CV_TPS", "1.8")),
                    help="transacciones por segundo (F0 = 2, S0 = 10)")
    ap.add_argument("--mock", action="store_true", help="simular el endpoint")
    args = ap.parse_args()
    if args.mock:
        os.environ["CV_MOCK"] = "1"

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    results, curves = {}, {}
    for model in args.modelos:
        rows = run_predictions(model, args.tps)
        y = np.array([int(r["y_true"]) for r in rows])
        s = np.array([float(r["p_danado"]) for r in rows])
        lat = np.array([float(r["latencia_ms"]) for r in rows])
        m = metricas.compute_metrics(y, s, args.umbral)
        m["roc_auc_ic95"] = metricas.bootstrap_auc_ci(y, s)
        m["umbrales"] = metricas.threshold_analysis(y, s)
        m["metricas_umbral_youden"] = metricas.compute_metrics(y, s, m["umbrales"]["youden"])
        m["latencia_ms"] = {"mediana": float(np.median(lat)), "p95": float(np.percentile(lat, 95))}
        results[model] = m
        curves[model] = (y, s)

    final = args.modelos[-1]
    y, s = curves[final]
    mf = results[final]
    yt = mf["umbrales"]["youden"]
    metricas.plot_roc(curves).savefig(FIG_DIR / "roc.png")
    metricas.plot_pr(curves).savefig(FIG_DIR / "precision_recall.png")
    metricas.plot_confusion(y, s, args.umbral, f"Matriz de confusión — {final}").savefig(FIG_DIR / "matriz_confusion.png")
    metricas.plot_confusion(y, s, yt, f"Matriz de confusión — {final} (umbral Youden)").savefig(
        FIG_DIR / "matriz_confusion_youden.png")
    marks = {"0.5": 0.5, "Youden": yt, "Recall 95%": mf["umbrales"]["recall_95"]}
    metricas.plot_threshold_sweep(y, s, marks).savefig(FIG_DIR / "barrido_umbral.png")
    metricas.plot_score_hist(y, s, args.umbral).savefig(FIG_DIR / "histograma_probabilidades.png")
    error_gallery(rows_for(final), args.umbral, FIG_DIR / "errores.png")
    plt.close("all")

    # métricas internas de Custom Vision (si existen)
    cv_internal = {}
    for p in sorted((config.RESULTS_DIR / "custom_vision").glob("*.json")):
        d = json.loads(p.read_text(encoding="utf-8"))
        perf = d["performance_threshold_0.5"]
        cv_internal[d["iteracion"]["name"]] = {
            "tipo": d["iteracion"].get("trainingType"), "precision": perf["precision"],
            "recall": perf["recall"], "average_precision": perf["averagePrecision"],
        }

    # McNemar exacto: ¿el modelo final se equivoca en imágenes distintas a los otros?
    from scipy.stats import binomtest

    mcnemar = {}
    ok_final = (s >= args.umbral).astype(int) == y
    for name in args.modelos[:-1]:
        yo, so = curves[name]
        ok_other = (so >= args.umbral).astype(int) == yo
        b = int(np.sum(ok_final & ~ok_other))
        c = int(np.sum(~ok_final & ok_other))
        p = binomtest(b, b + c, 0.5).pvalue if b + c else 1.0
        mcnemar[name] = {"solo_final": b, "solo_otro": c, "p_valor": float(p)}

    out = {"umbral": args.umbral, "modelo_final": final, "modelos": results,
           "mcnemar_vs_final": mcnemar, "custom_vision_interno": cv_internal, "mock": args.mock}
    (config.RESULTS_DIR / "metricas.json").write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")

    lines = [f"# Resultados en el conjunto de prueba (n = {mf['n']})", "",
             f"Umbral de decisión: {args.umbral}. Clase positiva: con daño.", "",
             "| Modelo | Accuracy | Precision | Recall | Especificidad | F1 | F1 macro | ROC AUC (IC 95 %) | AP |",
             "|---|---|---|---|---|---|---|---|---|"]
    for name, m in results.items():
        lo, hi = m["roc_auc_ic95"]
        lines.append(f"| {name} | {fmt(m['accuracy'])} | {fmt(m['precision'])} | {fmt(m['recall'])} | "
                     f"{fmt(m['specificity'])} | {fmt(m['f1'])} | {fmt(m['f1_macro'])} | "
                     f"{fmt(m['roc_auc'])} ({fmt(lo)}–{fmt(hi)}) | {fmt(m['average_precision'])} |")
    lines += ["", f"## Modelo final: {final}", "",
              f"Matriz (umbral {args.umbral}): TP={mf['tp']} FP={mf['fp']} TN={mf['tn']} FN={mf['fn']}", "",
              "Umbrales candidatos: " + ", ".join(f"{k} = {v:.3f}" for k, v in mf["umbrales"].items()), ""]
    my = mf["metricas_umbral_youden"]
    lines.append(f"Con umbral Youden ({yt:.3f}): precision={fmt(my['precision'])}, recall={fmt(my['recall'])}, "
                 f"F1={fmt(my['f1'])}, especificidad={fmt(my['specificity'])}")
    lines.append(f"Latencia del endpoint: mediana {mf['latencia_ms']['mediana']:.0f} ms, "
                 f"p95 {mf['latencia_ms']['p95']:.0f} ms")
    if mcnemar:
        lines += ["", f"## Prueba de McNemar (exacta) contra {final}, umbral {args.umbral}", "",
                  "| Modelo | Solo acierta el final | Solo acierta este modelo | p-valor |", "|---|---|---|---|"]
        for name, d in mcnemar.items():
            lines.append(f"| {name} | {d['solo_final']} | {d['solo_otro']} | {d['p_valor']:.3f} |")
    if cv_internal:
        lines += ["", "## Métricas internas de Custom Vision (umbral 0.5, partición interna)", "",
                  "| Iteración | Tipo | Precision | Recall | AP |", "|---|---|---|---|---|"]
        for name, c in cv_internal.items():
            lines.append(f"| {name} | {c['tipo']} | {fmt(c['precision'])} | {fmt(c['recall'])} | "
                         f"{fmt(c['average_precision'])} |")
    if args.mock:
        lines.insert(1, "\n> AVISO: resultados SIMULADOS (--mock). No usar en el informe.\n")
    (config.RESULTS_DIR / "tabla_metricas.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
    print(f"\nFiguras en {FIG_DIR}")


def rows_for(model: str):
    with open(PRED_DIR / f"{model}.csv", encoding="utf-8") as f:
        return list(csv.DictReader(f))


if __name__ == "__main__":
    main()
