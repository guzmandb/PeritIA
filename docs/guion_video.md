# Guion del video (máximo 10 minutos, objetivo 9:00)

Formato sugerido: grabar la pantalla con OBS o Clipchamp, con la cámara en una esquina.
- Resolución 1080p, micrófono con poco eco.
- Antes de grabar: cerrar pestañas con claves y **no mostrar el archivo `.env`** ni la página *Keys and Endpoint*.
- Subir a YouTube como "No listado" y pegar el enlace en la portada del informe.

| Tiempo | Sección | En pantalla | Qué decir (resumen) |
|---|---|---|---|
| 0:00–0:40 | Presentación | Diapositiva de portada | Nombre, maestría, curso, título del proyecto: "PeritIA: detección de daños en vehículos para siniestros de autos". |
| 0:40–1:50 | Contexto y problema | Diapositiva con el flujo de reclamación | Trabajo como científico de datos en una aseguradora. El perito revisa a mano las fotos de cada reclamo. Objetivo: un primer filtro automático (triage) con 3 salidas: con daño, sin daño y enviar a perito. Área: reconocimiento de objetos / clasificación. |
| 1:50–3:00 | Dataset | `muestra_imagenes.png`, `distribucion_clases.png`, `resoluciones.png` | Kaggle, 2.300 imágenes, 2 clases balanceadas. EDA: el 67 % tiene menos de 256 px. Limpieza con MD5 + dHash: 24 descartadas, **8 eran fuga entre entrenamiento y prueba**. Test independiente de 459 imágenes que nunca se suben a Azure. |
| 3:00–4:30 | Entrenamiento | Portal de Custom Vision: proyecto, etiquetas, imágenes, pestaña Performance de las 3 iteraciones. Terminal con `02_custom_vision.py estado` | Transfer learning, dominio General [A2]. Tres iteraciones: base con 300 imágenes, todos los datos, entrenamiento avanzado de 1 h. Todo automatizado con la API REST para que sea reproducible. Mostrar precision/recall del portal. |
| 4:30–5:15 | Despliegue | Diagrama `resultados/figuras_app/arquitectura.png`; portal: *Publish* y *Prediction URL* (con la clave tapada) | Dos capas en la nube: el modelo publicado como endpoint REST en Azure y la app web en Streamlit Community Cloud, conectada al repositorio de GitHub. Las claves van en los *secrets* de la plataforma, nunca en el código. Explicar la petición (imagen en bytes) y la respuesta JSON con probabilidades. |
| 5:15–7:15 | **Demo de la app** | La app publicada, abierta desde su enlace público en el navegador | (1) Subir las 6 fotos de la lista de abajo: dos con daño claro, dos sin daño y dos dudosas. Mostrar el semáforo, la barra de probabilidad y el CSV. (2) Mover el umbral y el margen, y mostrar cómo cambian las decisiones. (3) Pestaña "Evaluar dataset": subir `data/demo_eval.zip` y mostrar las métricas, la matriz y la ROC calculadas en vivo. |
| 7:15–8:50 | Resultados y análisis | `tabla_metricas.md`, `roc.png`, `matriz_confusion.png`, `barrido_umbral.png`, `errores.png` | Comparar las 3 iteraciones (AUC con IC 95 %). Leer la matriz de confusión: qué significan los FP y FN en seguros. Umbral de Youden vs. 0,5. Mostrar 2 o 3 errores y explicar por qué fallan. Explicar por qué no basta con las métricas del portal. |
| 8:50–9:30 | Limitaciones y conclusiones | Diapositiva de cierre | Sesgo de encuadre y de dominio; Ley 1581 (datos personales); el modelo apoya y no reemplaza al perito. Trabajo futuro: detección del área dañada y severidad. Conclusión principal con el número de AUC/F1. |

## Frases clave para justificar (por si el profesor pregunta)

- **Por qué Custom Vision:**
  - El servicio aplica *transfer learning* sobre redes preentrenadas, lo que permite buen desempeño con ~1.800 imágenes.
  - El despliegue como API es inmediato.
  - Ya existía la infraestructura de la Actividad 4.
- **Por qué un test propio:**
  - El portal solo da precision, recall y AP sobre una partición interna aleatoria. No da F1, ROC ni AUC.
  - El conjunto de prueba es independiente y está libre de duplicados.
- **AUC:** es la probabilidad de que una foto con daño elegida al azar reciba mayor puntaje que una sin daño. No depende del umbral.
- **Youden:** es el punto de la ROC que maximiza sensibilidad + especificidad − 1, y equilibra los dos tipos de error.
- **Zona de duda:** el costo de un error es asimétrico y regulado. Por eso los casos inciertos van a un humano y no hay rechazo automático.
- **dHash:** es un hash perceptual. Imágenes casi iguales (reescaladas o recomprimidas) dan hashes con pocos bits distintos. Se usó una distancia de Hamming ≤ 4 sobre 64 bits.

## Cifras finales para decir en el video

Modelo final: `iter3-advanced` (entrenamiento avanzado de 1 hora). Conjunto de prueba independiente de 459 imágenes (229 con daño, 230 sin daño) que nunca se subió a Custom Vision.

| Modelo | Accuracy | Precision | Recall | F1 | ROC AUC (IC 95 %) |
|---|---|---|---|---|---|
| iter1-baseline (300 imágenes) | 0,948 | 0,960 | 0,934 | 0,947 | 0,989 (0,981–0,995) |
| iter2-full (1.817 imágenes) | 0,954 | 0,964 | 0,943 | 0,954 | 0,990 (0,983–0,996) |
| **iter3-advanced** | **0,954** | 0,960 | 0,948 | **0,954** | **0,991 (0,985–0,997)** |

Frases clave:

- "Con solo 300 fotos ya llegamos a AUC 0,989. Eso muestra el poder del *transfer learning*: el modelo base ya sabe qué es una abolladura."
- "Más datos y el entrenamiento avanzado mejoran poco. McNemar da p = 0,71 y p = 1,0: las diferencias no son significativas. Lo digo con honestidad."
- "Elegí la iteración 3 porque tiene el mejor AUC y el mejor AP, y porque con su umbral óptimo deja pasar menos daños."
- "Con umbral 0,5 la matriz es VP 217, FP 9, VN 221, FN 12. Con el umbral de Youden, 0,42, el recall sube a 0,969: solo 7 de 229 daños se escapan, a cambio de 2 falsos positivos más."
- "En seguros, un falso positivo cuesta tiempo de un perito. Un falso negativo puede objetar mal un reclamo. Por eso priorizo el recall, y la app nunca objeta sola."
- "AUC 0,991: si tomo al azar una foto con daño y otra sin daño, el modelo pone más alta la de daño el 99,1 % de las veces."
- "La zona de duda manda a revisión solo el 2 % de las fotos. Pero 12 de los 21 errores son con mucha seguridad, así que además propongo auditar por muestreo."
- "El portal decía precision 0,92. Mi evaluación independiente da 0,96. El portal usa una partición pequeña y aleatoria; por eso hice mi propia evaluación con 459 imágenes y sin fuga."
- "Latencia del endpoint: mediana 561 ms por foto, en el nivel gratuito."

Errores para mostrar en `errores.png`:

- **Falsos positivos (P = 0,99):** auto mojado con baldes y agua en el piso; auto en un patio con grúas; mecánico agachado junto a un Mercedes. El modelo aprendió el **contexto de taller** como pista de daño.
- **Falsos negativos (P = 0,00–0,17):** Honda rojo con el frente destrozado visto de lejos; auto negro con daño de bajo contraste; auto contra un muro de ladrillos. Es el **sesgo de encuadre**: en el dataset, el daño casi siempre es un primer plano.

Demo de la app (las mismas fotos de la figura 5 del informe, en `data/test/`):

- `danado/danado_test_0060.jpg` → 99,8 % de daño (daño detectado)
- `danado/danado_test_0100.jpg` → 99,8 % (daño detectado)
- `danado/danado_test_0202.jpg` → 32,4 % (zona de duda: el daño se ve de lejos y el fondo es un patio)
- `sin_dano/sin_dano_test_0060.jpg` → 0,0 % (sin daño visible)
- `sin_dano/sin_dano_test_0100.jpg` → 0,0 % (sin daño visible)
- `sin_dano/sin_dano_test_0080.jpg` → 50,3 % (zona de duda: vista trasera con reflejos)

Luego subir `data/demo_eval.zip` en la pestaña "Evaluar un dataset etiquetado" (30 imágenes: exactitud 0,967, AUC 1,000).
