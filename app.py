"""PeritIA — Triage de daños en vehículos para siniestros de autos.

App web (Streamlit) que consume el endpoint de Azure Custom Vision.
Ejecutar:  streamlit run app.py
"""
from __future__ import annotations

import io
import json
import zipfile
from pathlib import PurePosixPath

import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

from src import config, cv_client, metricas

st.set_page_config(page_title="PeritIA · Daños en vehículos", page_icon="🚗", layout="wide")

IMG_EXT = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
FOLDER_ALIASES = {
    config.POSITIVE_TAG: 1, "00-damage": 1, "damage": 1, "con_dano": 1, "danado": 1,
    config.NEGATIVE_TAG: 0, "01-whole": 0, "whole": 0, "sin_dano": 0, "no_damage": 0,
}

st.markdown(
    """
    <style>
      .verdict {padding: .55rem .8rem; border-radius: .5rem; font-weight: 600; margin: .3rem 0 .4rem 0;}
      .v-dano {background: #fdecea; color: #a12a1f; border-left: 5px solid #d93025;}
      .v-ok   {background: #e8f5ec; color: #1e6b3a; border-left: 5px solid #2e9d57;}
      .v-rev  {background: #fff6e0; color: #8a5a00; border-left: 5px solid #f2a900;}
      .small  {font-size: .82rem; color: #666;}
    </style>
    """,
    unsafe_allow_html=True,
)


# ----------------------------------------------------------------------------- utilidades

def load_json(path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


@st.cache_data(show_spinner=False, max_entries=2000)
def cached_predict(image_bytes: bytes, publish_name: str, mock: bool) -> dict:
    return cv_client.predict(image_bytes, publish_name=publish_name)


def decide(p: float, thr: float, band: float):
    if p >= thr + band:
        return "dano", "Daño detectado · continuar reclamación"
    if p < thr - band:
        return "ok", "Sin daño visible · validar coherencia del reclamo"
    return "rev", "Zona de duda · enviar a perito"


def images_from_zip(data: bytes):
    """Devuelve [(nombre, bytes, etiqueta|None)] leyendo carpetas danado/sin_dano si existen."""
    out = []
    with zipfile.ZipFile(io.BytesIO(data)) as z:
        for info in z.infolist():
            path = PurePosixPath(info.filename)
            if info.is_dir() or path.suffix.lower() not in IMG_EXT or "__MACOSX" in path.parts:
                continue
            label = None
            for part in path.parts[:-1]:
                if part.lower() in FOLDER_ALIASES:
                    label = FOLDER_ALIASES[part.lower()]
            out.append((str(path), z.read(info), label))
    return out


# ----------------------------------------------------------------------------- barra lateral

metrics_file = load_json(config.RESULTS_DIR / "metricas.json")
default_thr = 0.5
if metrics_file and not metrics_file.get("mock"):
    final = metrics_file["modelo_final"]
    default_thr = round(metrics_file["modelos"][final]["umbrales"]["youden"], 2)

with st.sidebar:
    st.title("PeritIA")
    st.caption("Triage automático de fotos de siniestros de autos")
    publish = st.text_input("Iteración publicada (publishName)", config.get("CV_PUBLISH_NAME") or "")
    mock = cv_client.is_mock()
    try:
        cv_client.prediction_url(publish or None)
        conectado = True
    except cv_client.PredictionError:
        conectado = False
    if mock:
        st.warning("Modo DEMO: probabilidades simuladas (CV_MOCK=1).")
    elif conectado:
        st.success("Conectado al endpoint de Azure Custom Vision")
    else:
        st.error("Falta configurar el endpoint en .env")
    st.divider()
    thr = st.slider("Umbral de decisión P(con daño)", 0.05, 0.95, float(default_thr), 0.01,
                    help="Por defecto: umbral de Youden calculado en el conjunto de prueba.")
    band = st.slider("Margen de 'zona de duda' (±)", 0.0, 0.30, 0.10, 0.01,
                     help="Las fotos con probabilidad cercana al umbral se envían a revisión humana.")
    st.caption(f"Con daño si P ≥ {thr + band:.2f} · Sin daño si P < {thr - band:.2f}")

if not (conectado or mock):
    st.info("Configura `.env` (ver README) o ejecuta con `CV_MOCK=1` para una demostración.")

tab_fotos, tab_eval, tab_info = st.tabs(
    ["Analizar fotos del siniestro", "Evaluar un dataset etiquetado", "Acerca del modelo"])

# ----------------------------------------------------------------------------- pestaña 1

with tab_fotos:
    st.subheader("Cargue una o varias fotos del vehículo")
    st.write("Acepta JPG/PNG/WEBP o un archivo .zip con varias fotos. "
             "El modelo indica si la imagen muestra daño visible y con qué probabilidad.")
    files = st.file_uploader("Fotos o .zip", type=["jpg", "jpeg", "png", "webp", "bmp", "zip"],
                             accept_multiple_files=True, key="fotos")
    items = []
    for f in files or []:
        data = f.getvalue()
        if f.name.lower().endswith(".zip"):
            items += [(n, b) for n, b, _ in images_from_zip(data)]
        else:
            items.append((f.name, data))

    if items and (conectado or mock):
        rows = []
        prog = st.progress(0.0, text="Consultando el modelo…")
        for i, (name, data) in enumerate(items):
            try:
                img_bytes = cv_client.prepare_image(data)
                probs = cached_predict(img_bytes, publish, mock)
                p = probs[config.POSITIVE_TAG]
                kind, text = decide(p, thr, band)
                rows.append({"archivo": name, "p_con_dano": p, "decision": text, "_kind": kind,
                             "_img": img_bytes})
            except Exception as e:  # imagen corrupta o error HTTP
                rows.append({"archivo": name, "p_con_dano": np.nan, "decision": f"Error: {e}",
                             "_kind": "err", "_img": None})
            prog.progress((i + 1) / len(items), text=f"Consultando el modelo… {i + 1}/{len(items)}")
        prog.empty()

        kinds = [r["_kind"] for r in rows]
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Fotos analizadas", len(rows))
        c2.metric("Con daño", kinds.count("dano"))
        c3.metric("Sin daño", kinds.count("ok"))
        c4.metric("Revisión de perito", kinds.count("rev") + kinds.count("err"))

        cols = st.columns(3)
        for i, r in enumerate(rows):
            with cols[i % 3]:
                with st.container(border=True):
                    if r["_img"] is not None:
                        st.image(r["_img"], use_container_width=True)
                    st.markdown(f"<div class='small'>{r['archivo']}</div>", unsafe_allow_html=True)
                    css = {"dano": "v-dano", "ok": "v-ok", "rev": "v-rev"}.get(r["_kind"], "v-rev")
                    st.markdown(f"<div class='verdict {css}'>{r['decision']}</div>", unsafe_allow_html=True)
                    if not np.isnan(r["p_con_dano"]):
                        st.progress(float(r["p_con_dano"]),
                                    text=f"P(con daño) = {r['p_con_dano']:.1%} · "
                                         f"P(sin daño) = {1 - r['p_con_dano']:.1%}")

        df = pd.DataFrame([{k: v for k, v in r.items() if not k.startswith("_")} for r in rows])
        with st.expander("Tabla de resultados"):
            st.dataframe(df.style.format({"p_con_dano": "{:.3f}"}), use_container_width=True)
        st.download_button("Descargar resultados (CSV)", df.to_csv(index=False).encode("utf-8"),
                           "resultados_peritia.csv", "text/csv")

# ----------------------------------------------------------------------------- pestaña 2

with tab_eval:
    st.subheader("Evalúe el modelo con su propio dataset etiquetado")
    st.write("Suba un .zip con dos carpetas: `danado/` y `sin_dano/` "
             "(también se aceptan `00-damage/` y `01-whole/`). "
             "Se calculan precision, recall, F1, matriz de confusión, curva ROC y AUC.")
    zf = st.file_uploader("Dataset .zip", type=["zip"], key="dataset")
    if zf and (conectado or mock):
        data = [(n, b, y) for n, b, y in images_from_zip(zf.getvalue()) if y is not None]
        if not data:
            st.error("No se encontraron carpetas danado/ y sin_dano/ dentro del .zip.")
        elif st.button(f"Evaluar {len(data)} imágenes", type="primary"):
            ys, ps, names = [], [], []
            prog = st.progress(0.0)
            for i, (n, b, y) in enumerate(data):
                try:
                    probs = cached_predict(cv_client.prepare_image(b), publish, mock)
                    ys.append(y)
                    ps.append(probs[config.POSITIVE_TAG])
                    names.append(n)
                except Exception as e:
                    st.warning(f"{n}: {e}")
                prog.progress((i + 1) / len(data), text=f"{i + 1}/{len(data)}")
            prog.empty()
            if len(set(ys)) < 2:
                st.error("Se necesitan imágenes de ambas clases para calcular ROC/AUC.")
            else:
                m = metricas.compute_metrics(ys, ps, thr)
                c = st.columns(6)
                c[0].metric("Accuracy", f"{m['accuracy']:.3f}")
                c[1].metric("Precision", f"{m['precision']:.3f}")
                c[2].metric("Recall", f"{m['recall']:.3f}")
                c[3].metric("F1", f"{m['f1']:.3f}")
                c[4].metric("Especificidad", f"{m['specificity']:.3f}")
                c[5].metric("ROC AUC", f"{m['roc_auc']:.3f}")
                g1, g2 = st.columns(2)
                g1.pyplot(metricas.plot_confusion(ys, ps, thr))
                g2.pyplot(metricas.plot_roc({"Modelo": (ys, ps)}))
                g3, g4 = st.columns(2)
                g3.pyplot(metricas.plot_pr({"Modelo": (ys, ps)}))
                g4.pyplot(metricas.plot_threshold_sweep(ys, ps, {"umbral": thr}))
                rep = pd.DataFrame(m["classification_report"]).T
                if "accuracy" in rep.index:  # la fila accuracy repite el mismo valor en todas las columnas
                    rep = rep.drop(index="accuracy")
                rep["support"] = rep["support"].astype(int)
                st.dataframe(rep.style.format({"precision": "{:.3f}", "recall": "{:.3f}", "f1-score": "{:.3f}"}),
                             use_container_width=True)
                out = pd.DataFrame({"archivo": names, "real": ys, "p_con_dano": ps,
                                    "prediccion": (np.array(ps) >= thr).astype(int)})
                st.download_button("Descargar predicciones (CSV)", out.to_csv(index=False).encode("utf-8"),
                                   "evaluacion_peritia.csv", "text/csv")

# ----------------------------------------------------------------------------- pestaña 3

with tab_info:
    st.subheader("Contexto de negocio")
    st.write(
        "En una aseguradora de autos, cada reclamación llega con fotos del vehículo. Un perito revisa "
        "manualmente si el daño es visible antes de continuar el trámite. PeritIA hace un primer filtro: "
        "clasifica cada foto como **con daño** o **sin daño**, prioriza los casos claros y envía a un "
        "perito solo los dudosos o los reclamos cuyas fotos no muestran daño (posible inconsistencia o fraude)."
    )
    ds = load_json(config.DATA_DIR / "resumen_dataset.json")
    if ds:
        st.subheader("Dataset")
        st.write(f"Fuente: {ds['fuente']}. {ds['imagenes_originales']} imágenes originales; "
                 f"{sum(ds['descartadas'].values())} descartadas por duplicados o fuga train/test; "
                 f"{ds['total_final']} finales.")
        st.table(pd.Series(ds["final"], name="imágenes"))
    if metrics_file:
        st.subheader("Desempeño en el conjunto de prueba independiente")
        if metrics_file.get("mock"):
            st.warning("Las métricas guardadas son simuladas. Ejecute 03_evaluar_modelo.py con el endpoint real.")
        rows = []
        for name, m in metrics_file["modelos"].items():
            rows.append({"modelo": name, "accuracy": m["accuracy"], "precision": m["precision"],
                         "recall": m["recall"], "F1": m["f1"], "ROC AUC": m["roc_auc"],
                         "AP": m["average_precision"]})
        st.dataframe(pd.DataFrame(rows).set_index("modelo").style.format("{:.3f}"), use_container_width=True)
        figdir = config.RESULTS_DIR / "figuras_evaluacion"
        a, b = st.columns(2)
        for col, fname in [(a, "roc.png"), (b, "matriz_confusion.png"),
                           (a, "precision_recall.png"), (b, "barrido_umbral.png")]:
            if (figdir / fname).exists():
                col.image(str(figdir / fname), use_container_width=True)
    st.caption("Modelo: Azure Custom Vision (clasificación multiclase, dominio General [A2]). "
               "Proyecto final — Visión por Computador, Maestría en IA, Universidad de La Salle.")
