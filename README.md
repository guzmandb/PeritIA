# PeritIA — Detección de daños en vehículos para siniestros de autos

**App en línea:** https://peritia-evaluador-danios.streamlit.app/

**Video de sustentación:** https://youtu.be/Q6f0-3j1pEY

Proyecto final, Visión por Computador (Maestría en IA, Universidad de La Salle). Actividad 6.

**Problema:** en una aseguradora de autos, cada reclamación trae fotos del vehículo. PeritIA clasifica
cada foto como **con daño** o **sin daño** usando un modelo de Azure Custom Vision publicado como API.
Así se priorizan los casos claros y solo se envían a un perito los dudosos o los que no muestran daño.

| Elemento | Detalle |
|---|---|
| Área | Reconocimiento de objetos (clasificación de imágenes binaria) |
| Dataset | Kaggle `anujms/car-damage-detection`: 2.300 imágenes que quedan en 2.276 después de la limpieza |
| Herramienta | Azure Custom Vision, dominio *General [A2]*, multiclase |
| Despliegue | Endpoint REST de predicción de Custom Vision (East US) |
| App | Streamlit (`app.py`), publicada en Streamlit Community Cloud: carga una o varias fotos, o un `.zip` etiquetado para evaluarlo |
| Evaluación | `03_evaluar_modelo.py`: precision, recall, F1, matriz de confusión, ROC/AUC, PR, umbrales |

## Estructura

```
01_preparar_dataset.py   limpieza, deduplicación, split train/test, figuras del dataset
02_custom_vision.py      crear proyecto, subir imágenes, entrenar, publicar (API REST v3.3)
03_evaluar_modelo.py     evaluación contra data/test usando el endpoint publicado
app.py                   app web Streamlit
src/                     configuración, cliente del endpoint y métricas
data/train, data/test    imágenes procesadas (danado / sin_dano)
resultados/              métricas, figuras y predicciones
docs/                    informe (Markdown, PDF y Word) y guion del video
```

## Paso a paso

### 0. Instalar dependencias

```powershell
pip install -r requirements.txt
```

Copie `.env.example` como `.env`. Complete las claves con los datos del Portal de Azure:
- **Endpoint:** recurso → *Keys and Endpoint*. Use el endpoint que aparece ahí. Puede ser `https://<nombre>.cognitiveservices.azure.com/`.
- **Seguridad:** `.env` está en `.gitignore`. No lo muestre en el video.

### 1. Dataset (ya ejecutado)

```powershell
python 01_preparar_dataset.py
```

### 2. Entrenamiento en Custom Vision (3 iteraciones comparables)

```powershell
python 02_custom_vision.py crear-proyecto --nombre DanosVehiculares
```
Copie el `CV_PROJECT_ID` que imprime en `.env`. Luego:

```powershell
python 02_custom_vision.py subir --max-por-clase 150
python 02_custom_vision.py entrenar --tipo Regular
python 02_custom_vision.py subir
python 02_custom_vision.py entrenar --tipo Regular
python 02_custom_vision.py entrenar --tipo Advanced --horas 1 --force
python 02_custom_vision.py estado
```

Qué produce cada comando:
- Primer `subir` + `entrenar` → **Iteration 1**, línea base con 150 imágenes por clase.
- Segundo `subir` + `entrenar` → **Iteration 2**, con todas las imágenes de entrenamiento (≈1.817).
- `entrenar --tipo Advanced` → **Iteration 3**, entrenamiento avanzado de 1 hora con los mismos datos.

Publique las tres iteraciones con nombres distintos. Use los IDs que imprime `estado`:

```powershell
python 02_custom_vision.py publicar --iteracion <ID1> --nombre iter1-baseline
```
```powershell
python 02_custom_vision.py publicar --iteracion <ID2> --nombre iter2-full
```
```powershell
python 02_custom_vision.py publicar --iteracion <ID3> --nombre iter3-advanced
```

> Si el recurso es **F0 (gratis)**: el entrenamiento avanzado puede no estar disponible o agotar la cuota
> mensual de horas. En ese caso, use Iteration 2 como modelo final. El "ajuste" del modelo queda así:
> (a) más datos, de la Iteration 1 a la 2; (b) optimización del umbral con la curva ROC (Youden).
> El recurso F0 también limita las predicciones a 2 por segundo. El script ya respeta ese límite.

### 3. Evaluación independiente

```powershell
python 03_evaluar_modelo.py --modelos iter1-baseline iter2-full iter3-advanced
```

Genera `resultados/metricas.json`, `resultados/tabla_metricas.md` y las figuras en `resultados/figuras_evaluacion/`.

### 4. App web

```powershell
streamlit run app.py
```

Para una demostración sin Azure: `$env:CV_MOCK="1"; streamlit run app.py`.

**Despliegue opcional de la app en la nube** (Streamlit Community Cloud):
1. Suba el repositorio a GitHub sin `.env` ni `raw/`.
2. Cree la app en share.streamlit.io.
3. Pegue las variables de `.env.example` en *Secrets*, en formato TOML.

`src/config.py` lee también de `st.secrets`.

## Despliegue de la app web (Streamlit Community Cloud)

1. Subir este repositorio a GitHub (el `.env` y las imágenes de `data/train` y `data/test` no se suben).
2. Entrar a https://share.streamlit.io con la cuenta de GitHub y elegir **Create app**.
3. Repositorio: este; rama `main`; archivo principal `app.py`; Python 3.11.
4. En **Advanced settings → Secrets**, pegar el contenido de `.streamlit/secrets.toml.example` con los valores reales.
5. **Deploy**. La app queda en una URL pública `https://<nombre>.streamlit.app`.
