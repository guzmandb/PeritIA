<div class="portada" markdown="1">

# PeritIA

<p><strong>Clasificación automática de daños en vehículos como apoyo a la gestión de siniestros de autos</strong></p>

<p>Actividad 6 — Proyecto final de Visión por Computador</p>
<p>Maestría en Inteligencia Artificial — Universidad de La Salle</p>
<br>
<p>Jherson Guzmán</p>
<p>Noviembre de 2026</p>
<br>
<p>Aplicación web: [ENLACE APP]</p>
<p>Repositorio del código: [ENLACE REPOSITORIO]</p>
<p>Video de sustentación: [ENLACE YOUTUBE]</p>

</div>

## Resumen

En este trabajo se desarrolló PeritIA, un sistema de visión por computador que clasifica fotografías de vehículos en dos categorías, **con daño** y **sin daño**, con el fin de apoyar la primera revisión de las reclamaciones de un seguro de autos. El modelo se entrenó en **Azure Custom Vision** a partir de un conjunto de datos público de Kaggle que, después de un proceso de depuración, quedó con 2.276 imágenes. La iteración final se publicó como un **servicio REST en la nube** y se integró en una **aplicación web desarrollada en Streamlit**, desplegada en Streamlit Community Cloud, en la que un usuario puede cargar una o varias fotos, o un conjunto de imágenes etiquetadas, y obtener la decisión del modelo junto con sus métricas.

Se compararon tres iteraciones: una línea base con 300 imágenes, una con todo el conjunto de entrenamiento y una con entrenamiento avanzado. Todas se evaluaron sobre un **conjunto de prueba independiente de 459 imágenes** que nunca se cargó en el servicio. Con el umbral por defecto de 0,5, el modelo final (`iter3-advanced`) obtuvo una exactitud de 0,954, precision de 0,960, recall de 0,948, F1 de 0,954 y un **área bajo la curva ROC de 0,991** (IC 95 %: 0,985–0,997). A partir de la curva ROC se definió además un umbral operativo de 0,42 que eleva el recall a 0,969, y una "zona de duda" que remite a un perito los casos en los que el modelo no tiene certeza suficiente.

## 1. Contexto y área de aplicación

### 1.1 Contexto

El autor se desempeña como científico de datos en una compañía de seguros en Colombia. En el ramo de autos, cuando un asegurado reporta un siniestro, normalmente adjunta fotografías del vehículo a través de la aplicación móvil, del correo electrónico o de WhatsApp. Un analista o un perito revisa esas imágenes para confirmar que existe un daño visible; si lo hay, el caso se asigna a un taller o a un perito de campo, se estima el costo de la reparación y, finalmente, la reclamación se aprueba o se objeta.

La revisión inicial de las fotos es una tarea manual y repetitiva, y su carga crece al mismo ritmo que el volumen de reclamaciones, en especial en temporadas de alta siniestralidad. Además, una parte de las imágenes recibidas no muestra ningún daño: hay fotos incompletas, fotos de otro vehículo o reclamaciones inconsistentes. Estas imágenes consumen tiempo de los peritos sin aportar información útil para el proceso.

### 1.2 Propuesta de valor

PeritIA se plantea como un primer filtro visual que no reemplaza al perito, sino que organiza su trabajo. Cuando el modelo identifica con alta probabilidad que una foto muestra daño, el caso puede avanzar directamente a la asignación de taller, lo que reduce el tiempo de respuesta al asegurado. Cuando identifica con alta probabilidad que no hay daño visible, el caso se marca para validar la coherencia de la reclamación, ya que puede ser una señal temprana de inconsistencia o de fraude; la guía de la unidad menciona precisamente este uso de la visión por computador en banca y seguros, dentro de la gestión de riesgos y el cumplimiento. Por último, las fotos con una probabilidad cercana al umbral se envían a revisión humana. Con este esquema se espera un mejor aprovechamiento del tiempo de los peritos, una mayor capacidad para absorber picos de demanda y la trazabilidad de cada decisión.

### 1.3 Área de aplicación

El proyecto se ubica en el área de **reconocimiento de objetos mediante clasificación de imágenes**. El sistema no solo reconoce un vehículo, sino que determina su estado: distingue entre un vehículo con daño visible (abolladuras, vidrios rotos, deformaciones o raspones) y uno en buen estado. Para ello se utilizó *transfer learning* sobre una red convolucional preentrenada, una técnica que la guía de la unidad destaca porque permite construir soluciones empresariales con relativamente pocos datos.

## 2. Dataset

### 2.1 Fuente y criterios de selección

Se utilizó el conjunto **Car Damage Detection**, publicado en Kaggle (`anujms/car-damage-detection`, carpeta `data1a`). Contiene fotografías reales de vehículos obtenidas de internet, organizadas en dos clases y dos particiones:

| Partición original | 00-damage (con daño) | 01-whole (sin daño) |
|---|---|---|
| training | 920 | 920 |
| validation | 230 | 230 |

Este conjunto se eligió porque es de acceso público y puede descargarse sin restricciones, tiene clases balanceadas y bien definidas, lo que permite un análisis ROC directo, y su tamaño es compatible con los límites del nivel gratuito de Custom Vision (5.000 imágenes por proyecto). La figura 1 muestra algunos ejemplos de cada clase.

![Ejemplos del conjunto de entrenamiento](resultados/figuras_dataset/muestra_imagenes.png){: .wide}
<p class="caption">Figura 1. Ejemplos del conjunto de entrenamiento por clase.</p>

### 2.2 Análisis exploratorio

Las 2.300 imágenes originales están en formato JPEG y todas, salvo una en escala de grises, son RGB. Su resolución es muy variable: el lado menor va de 74 a 3.264 píxeles, con una mediana de apenas 194 píxeles, y 1.543 imágenes (el 67 %) tienen el lado menor por debajo de 256 píxeles, que es el mínimo que recomienda Custom Vision (figura 2). El servicio reescala estas imágenes de forma automática, pero su baja resolución limita el nivel de detalle disponible para detectar daños pequeños.

En la revisión visual también se observó un posible **sesgo de encuadre**: muchas fotos sin daño son imágenes comerciales del vehículo completo, mientras que muchas fotos con daño son primeros planos tomados en talleres o patios. Esto implica el riesgo de que el modelo aprenda a reconocer el encuadre o el entorno en lugar del daño mismo, un aspecto que se retoma en el análisis de errores de la sección 5.

![Resoluciones](resultados/figuras_dataset/resoluciones.png)
<p class="caption">Figura 2. Distribución del lado menor de las imágenes originales.</p>

### 2.3 Preparación de los datos

La preparación se automatizó en el script `01_preparar_dataset.py` y comprendió tres etapas.

**Normalización.** Se corrigió la orientación según los metadatos EXIF, se convirtieron todas las imágenes a RGB, se limitó el lado mayor a 1.600 píxeles y se guardaron en JPEG con calidad 95. Con ello ninguna imagen supera el límite de 6 MB que impone el servicio (antes del proceso había una que lo excedía).

**Deduplicación.** Se buscaron duplicados exactos mediante un hash MD5 de los píxeles y casi duplicados (imágenes reescaladas o recomprimidas) mediante un *hash* perceptual dHash de 64 bits, considerando duplicadas las imágenes con una distancia de Hamming menor o igual a 4. Se encontraron 23 grupos de duplicados y se descartaron 24 imágenes:

| Motivo del descarte | Imágenes |
|---|---|
| Duplicado dentro de entrenamiento | 15 |
| **Fuga entre entrenamiento y prueba** (la misma foto en ambas particiones) | **8** |
| Duplicado dentro de prueba | 1 |

La eliminación de la fuga fue especialmente importante: si una misma imagen aparece en entrenamiento y en prueba, el modelo ya la conoce y las métricas resultan artificialmente altas.

**Partición.** La partición `validation` de Kaggle se reservó como **conjunto de prueba independiente** y nunca se cargó en Custom Vision. La partición `training` se cargó en el servicio, que a su vez separa internamente una porción para calcular las métricas que muestra en su portal.

| | Con daño | Sin daño | Total |
|---|---|---|---|
| Entrenamiento | 898 | 919 | 1.817 |
| Prueba | 229 | 230 | 459 |
| **Total** | **1.127** | **1.149** | **2.276** |

Para garantizar la trazabilidad, el archivo `data/manifiesto.csv` registra el origen, la partición, la etiqueta, el tamaño y, cuando aplica, el motivo de descarte de cada imagen. La figura 3 muestra la distribución final de clases.

![Distribución de clases](resultados/figuras_dataset/distribucion_clases.png)
<p class="caption">Figura 3. Distribución de clases después de la limpieza.</p>

## 3. Metodología

### 3.1 Herramienta y modelo

El modelo se construyó con **Azure Custom Vision**, en la región East US y con el nivel gratuito F0, reutilizando la infraestructura creada en la Actividad 4 (grupo de recursos `ComputerVisionImages`, recurso de entrenamiento `AnimalIdentificationJGR` y recurso de predicción `AnimalIdentificationJGR-Prediction`). Se creó el proyecto `DanosVehiculares` de tipo *Classification – Multiclass*, en el que cada imagen recibe una sola etiqueta (`danado` o `sin_dano`), con el dominio *General [A2]*. Custom Vision aplica *transfer learning*: parte de una red convolucional preentrenada con millones de imágenes y ajusta sus capas finales con las imágenes del proyecto.

Para que el proceso fuera reproducible, todo el flujo se automatizó con la **API REST de entrenamiento v3.3** en el script `02_custom_vision.py`, que crea el proyecto y las etiquetas, carga las imágenes en lotes de 64, lanza el entrenamiento, consulta las métricas y publica cada iteración.

### 3.2 Experimentos y ajustes

Se diseñaron tres iteraciones, cada una orientada a responder una pregunta concreta:

| Iteración | Imágenes de entrenamiento | Tipo de entrenamiento | Pregunta |
|---|---|---|---|
| Iteration 1 (`iter1-baseline`) | 150 + 150 (muestra estratificada, semilla 42) | Regular | ¿Qué desempeño se alcanza con pocos datos? |
| Iteration 2 (`iter2-full`) | 898 + 919 | Regular | ¿Cuánto mejora el modelo con seis veces más datos? |
| Iteration 3 (`iter3-advanced`) | 898 + 919 | Avanzado, presupuesto de 1 hora | ¿Cuánto aporta un entrenamiento más largo con búsqueda de arquitectura? |

Como ajuste adicional se estudió el **umbral de decisión**. Custom Vision clasifica por defecto con un umbral de 0,5, pero ese valor no siempre es el más adecuado para el negocio. Con las probabilidades del conjunto de prueba se calcularon tres alternativas: el umbral que maximiza el índice de Youden (J = TPR − FPR) sobre la curva ROC, el que maximiza el F1 y el mayor umbral que garantiza un recall de al menos 95 %.

### 3.3 Despliegue en la nube

La solución quedó desplegada en dos capas, como se resume en la figura 4.

**Modelo como servicio.** Cada iteración se publicó en el recurso de predicción con un nombre propio (`publishName`), de modo que cada una quedó disponible como un endpoint REST independiente. La aplicación consume la iteración final mediante una petición como la siguiente:

```
POST https://animalidentificationjgr-prediction.cognitiveservices.azure.com/customvision/v3.0/
     Prediction/<projectId>/classify/iterations/iter3-advanced/image
Headers: Prediction-Key: <clave>   Content-Type: application/octet-stream
Body:    bytes de la imagen
Respuesta: {"predictions":[{"tagName":"danado","probability":0.97},{"tagName":"sin_dano","probability":0.03}]}
```

**Aplicación web.** La interfaz se publicó en **Streamlit Community Cloud** a partir del repositorio del proyecto en GitHub. Las credenciales del endpoint no se guardan en el código: se configuran como *secrets* de la plataforma y la aplicación las lee en tiempo de ejecución. De esta manera, cualquier persona con el enlace puede usar el sistema desde el navegador, sin instalar nada.

![Arquitectura](resultados/figuras_app/arquitectura.png){: .wide}
<p class="caption">Figura 4. Arquitectura de la solución: construcción del modelo y operación en la nube.</p>

### 3.4 Aplicación web

La aplicación, implementada en `app.py`, está organizada en tres pestañas y una barra lateral.

En la pestaña **Analizar fotos del siniestro**, el usuario carga una o varias fotos, o un archivo `.zip`. Para cada imagen la aplicación muestra la foto, la probabilidad de daño y una decisión con código de colores: *Daño detectado* (rojo), *Sin daño visible* (verde) o *Zona de duda, enviar a perito* (amarillo). En la parte superior resume cuántas fotos quedaron en cada categoría, y al final permite descargar los resultados en CSV (figura 5).

![Pestaña de análisis de fotos](resultados/figuras_app/app_analizar_recorte.png){: .screen}
<p class="caption">Figura 5. Análisis de seis fotos del conjunto de prueba con la iteración final. Dos quedan en la zona de duda y se remiten a un perito.</p>

En la pestaña **Evaluar un dataset etiquetado**, el usuario sube un `.zip` con las carpetas `danado/` y `sin_dano/`. La aplicación envía cada imagen al endpoint y calcula en el momento la exactitud, la precision, el recall, el F1, la especificidad y el AUC, además de dibujar la matriz de confusión, las curvas ROC y precision-recall y el barrido de umbral (figura 6). Esta funcionalidad permite que el área de negocio valide el modelo con sus propias fotos antes de adoptarlo.

![Pestaña de evaluación de un dataset](resultados/figuras_app/app_evaluar_recorte.png){: .screen}
<p class="caption">Figura 6. Evaluación de un conjunto etiquetado de 30 imágenes desde la aplicación.</p>

La pestaña **Acerca del modelo** presenta el contexto del problema, el resumen del dataset y las métricas de la evaluación independiente. En la barra lateral se pueden ajustar el umbral de decisión, que por defecto es el de Youden, y el margen de la zona de duda, de modo que el área de siniestros calibre el equilibrio entre automatización y control humano.

En la implementación se aplicaron algunas buenas prácticas: las claves se leen del archivo `.env` o de los *secrets* de Streamlit y nunca se escriben en el código; las imágenes se normalizan antes de enviarlas (orientación EXIF, RGB y lado mayor de hasta 1.600 píxeles); ante una respuesta HTTP 429 se reintenta con espera exponencial, porque el nivel F0 admite dos transacciones por segundo; y existe un modo de demostración (`CV_MOCK=1`) que permite probar la interfaz sin conexión a Azure.

### 3.5 Protocolo de evaluación

Custom Vision solo informa precision, recall y AP calculados sobre su partición interna; no reporta F1, curva ROC ni AUC. Por esa razón se implementó una evaluación propia en el script `03_evaluar_modelo.py`. Las 459 imágenes de prueba se enviaron al endpoint de cada iteración y se registró la probabilidad asignada a la clase *con daño*. Tomando esa clase como positiva, se calcularon con scikit-learn la exactitud, la precision, el recall (sensibilidad), la especificidad, el F1 y el F1 macro, la matriz de confusión, la curva ROC con su AUC y un intervalo de confianza del 95 % estimado por *bootstrap* con 2.000 remuestreos, y la curva precision-recall con su AP. Se realizó también un barrido de umbral y se construyó una galería con los falsos positivos y falsos negativos más llamativos. Finalmente, las iteraciones se compararon con la **prueba exacta de McNemar**, que evalúa si dos modelos se equivocan en imágenes distintas con una frecuencia mayor que la esperable por azar.

## 4. Resultados

### 4.1 Métricas internas de Custom Vision

| Iteración | Tipo | Minutos de entrenamiento | Precision | Recall | AP |
|---|---|---|---|---|---|
| Iteration 1 (`iter1-baseline`) | Regular | 3 | 0,883 | 0,883 | 0,981 |
| Iteration 2 (`iter2-full`) | Regular | 9 | 0,923 | 0,923 | 0,972 |
| Iteration 3 (`iter3-advanced`) | Avanzado | 60 | 0,923 | 0,923 | 0,972 |

Por etiqueta (precision / recall), la iteración 1 obtuvo 0,926 / 0,833 en *danado* y 0,848 / 0,933 en *sin_dano*; la iteración 2, 0,937 / 0,906 y 0,911 / 0,940; y la iteración 3, 0,904 / 0,944 y 0,943 / 0,902. Estas cifras se calculan sobre una partición interna pequeña que el servicio selecciona de forma automática, por lo que no permiten una comparación rigurosa ni incluyen F1 o ROC. La evaluación que sustenta las conclusiones de este informe es la del conjunto de prueba independiente.

### 4.2 Conjunto de prueba independiente

Con el umbral por defecto de 0,5 y n = 459, se obtuvieron los siguientes resultados:

| Modelo | Accuracy | Precision | Recall | Especificidad | F1 | ROC AUC (IC 95 %) | AP | VP | FP | VN | FN |
|---|---|---|---|---|---|---|---|---|---|---|---|
| iter1-baseline | 0,948 | 0,960 | 0,934 | 0,961 | 0,947 | 0,989 (0,981–0,995) | 0,988 | 214 | 9 | 221 | 15 |
| iter2-full | 0,954 | 0,964 | 0,943 | 0,965 | 0,954 | 0,990 (0,983–0,996) | 0,990 | 216 | 8 | 222 | 13 |
| **iter3-advanced** | 0,954 | 0,960 | 0,948 | 0,961 | **0,954** | **0,991** (0,985–0,997) | 0,991 | 217 | 9 | 221 | 12 |

El F1 macro fue de 0,948, 0,954 y 0,954, respectivamente. El endpoint del modelo final respondió con una latencia mediana de 561 ms y un percentil 95 de 1.851 ms por imagen, medidos desde Colombia con el nivel gratuito.

![Curva ROC](resultados/figuras_evaluacion/roc.png)
<p class="caption">Figura 7. Curvas ROC de las tres iteraciones sobre el conjunto de prueba.</p>

![Curva PR](resultados/figuras_evaluacion/precision_recall.png)
<p class="caption">Figura 8. Curvas precision-recall de las tres iteraciones.</p>

![Matriz de confusión](resultados/figuras_evaluacion/matriz_confusion.png)
<p class="caption">Figura 9. Matriz de confusión del modelo final con umbral 0,5.</p>

### 4.3 Comparación estadística entre iteraciones

| Comparación contra `iter3-advanced` | Imágenes que solo acierta `iter3-advanced` | Imágenes que solo acierta el otro modelo | p-valor (exacto) |
|---|---|---|---|
| `iter1-baseline` | 16 | 13 | 0,711 |
| `iter2-full` | 6 | 6 | 1,000 |

Ninguno de los p-valores es inferior a 0,05. Las tres iteraciones aciertan y fallan prácticamente en las mismas imágenes, y las diferencias de exactitud observadas (de 0,948 a 0,954) están dentro de la variación esperable en una muestra de 459 imágenes.

### 4.4 Umbral operativo

| Umbral | Valor | Precision | Recall | Especificidad | F1 | FP | FN |
|---|---|---|---|---|---|---|---|
| Por defecto | 0,500 | 0,960 | 0,948 | 0,961 | 0,954 | 9 | 12 |
| **Youden (coincide con el F1 máximo)** | **0,416** | 0,953 | **0,969** | 0,952 | **0,961** | 11 | **7** |
| Mayor umbral con recall ≥ 95 % | 0,483 | 0,956 | 0,952 | 0,957 | 0,954 | 10 | 11 |

Se adoptó el umbral de Youden (0,42) como valor por defecto de la aplicación. Frente al umbral de 0,5, permite detectar cinco daños adicionales (los falsos negativos bajan de 12 a 7) a cambio de dos falsos positivos más (de 9 a 11). En el contexto de los siniestros este intercambio resulta conveniente, porque un falso positivo solo implica que un perito revise un caso, mientras que un falso negativo puede llevar a cerrar de forma equivocada una reclamación con daño real. Debe tenerse en cuenta que estos umbrales se calcularon sobre el mismo conjunto de prueba, por lo que su desempeño es ligeramente optimista; en un piloto deberían recalibrarse con un conjunto de validación propio de la aseguradora.

![Barrido de umbral](resultados/figuras_evaluacion/barrido_umbral.png)
<p class="caption">Figura 10. Precision, recall y F1 del modelo final en función del umbral.</p>

![Histograma](resultados/figuras_evaluacion/histograma_probabilidades.png)
<p class="caption">Figura 11. Distribución de la probabilidad de daño asignada por el modelo final, según la clase real.</p>

## 5. Análisis de resultados

### 5.1 Efecto de la cantidad de datos

Al pasar de 300 a 1.817 imágenes de entrenamiento, la exactitud subió de 0,948 a 0,954 y los falsos negativos bajaron de 15 a 13. Es una mejora real, pero pequeña. Con solo 150 fotos por clase, el *transfer learning* de Custom Vision ya alcanzaba un AUC de 0,989, lo que indica que la red preentrenada ya cuenta con representaciones útiles para reconocer abolladuras, vidrios rotos o láminas deformadas, y que las imágenes adicionales sirven sobre todo para afinar los casos límite. Desde el punto de vista del negocio, este hallazgo sugiere que un piloto con unos pocos cientos de fotos etiquetadas de la propia compañía ya sería viable.

### 5.2 Entrenamiento avanzado frente a regular

El entrenamiento avanzado (60 minutos frente a 9) no modificó la exactitud con umbral 0,5, que se mantuvo en 0,954, pero sí mejoró ligeramente la **capacidad del modelo para ordenar los casos por riesgo**: el AUC pasó de 0,990 a 0,991 y el AP de 0,990 a 0,991. Con el umbral de 0,5 solo cambió la naturaleza de un error (un falso negativo menos y un falso positivo más). La diferencia se hace visible al ajustar el umbral: con el criterio de Youden, la iteración 3 alcanza un recall de 0,969 y deja sin detectar solo 7 daños, frente a 13 de la iteración 2 en su propio umbral óptimo. Por estas razones se seleccionó la iteración 3 como modelo final. Aun así, la prueba de McNemar indica que la diferencia no es estadísticamente significativa, de modo que la elección es razonable pero no concluyente.

### 5.3 Interpretación de la curva ROC

Un AUC de 0,991 significa que, si se toman al azar una foto con daño y una sin daño, el modelo asigna una probabilidad de daño mayor a la primera en el 99,1 % de los casos. El intervalo de confianza (0,985–0,997) está muy por encima de 0,5, que corresponde a un clasificador aleatorio. Las curvas de las tres iteraciones prácticamente se superponen (figura 7), lo que coincide con el resultado de McNemar. Por su parte, el histograma de la figura 11 muestra que las probabilidades se concentran cerca de 0 y de 1, es decir, que el modelo suele ser muy seguro en sus respuestas.

### 5.4 Costo de los errores en el contexto de seguros

Los dos tipos de error no tienen el mismo costo. Un **falso positivo** (un vehículo sin daño clasificado como dañado) envía el caso a un perito, que descarta el daño; cuesta tiempo de revisión, pero no genera pagos indebidos. Un **falso negativo** (un vehículo con daño clasificado como sin daño) es más delicado: si la decisión se automatizara, podría llevar a objetar injustamente una reclamación, con la consecuente queja del cliente y el riesgo regulatorio ante la Superintendencia Financiera. Por eso el recall es la métrica prioritaria en este caso y el sistema se diseñó para que nunca objete una reclamación de forma automática: la etiqueta "sin daño visible" solo sirve para priorizar la cola de revisión.

### 5.5 Patrones de error

La figura 12 muestra los errores con mayor confianza del modelo final. Entre los **falsos positivos** aparecen vehículos sucios o mojados, charcos y manchas en el pavimento, un mecánico trabajando junto a un auto y vehículos fotografiados en talleres o patios. Todo indica que el modelo aprendió **atajos de contexto**: en el dataset, las fotos con daño suelen tomarse precisamente en esos entornos. Entre los **falsos negativos** predominan daños frontales vistos a distancia en una toma completa del vehículo, daños sobre carrocerías oscuras con poco contraste y daños parciales de lámina. Como en el dataset los daños aparecen casi siempre en primer plano, el modelo tiende a asociar la vista completa del vehículo con la ausencia de daño, lo que confirma el **sesgo de encuadre** identificado en el análisis exploratorio.

Conviene destacar que 12 de los 21 errores cometidos con umbral 0,5 tienen una probabilidad mayor que 0,8 o menor que 0,2; es decir, el modelo también se equivoca con seguridad, lo que limita lo que puede lograrse solo con la zona de duda.

![Errores](resultados/figuras_evaluacion/errores.png){: .wide}
<p class="caption">Figura 12. Falsos positivos y falsos negativos con mayor confianza del modelo final.</p>

### 5.6 Zona de duda

Con el umbral de Youden y un margen de ±0,10, solo 9 imágenes (el 2,0 %) caen en la zona de duda y se remiten a revisión humana, y la exactitud sobre las 450 imágenes que se deciden de forma automática es de 0,964. Dado que muchos errores se cometen con alta confianza, ampliar el margen apenas ayuda: con ±0,15 la zona solo crece a 10 imágenes. Por ello se recomienda **combinar la zona de duda con una auditoría por muestreo** de los casos automatizados, en particular de los marcados como "sin daño visible".

### 5.7 Métricas del portal frente a la evaluación independiente

Las métricas del portal de Custom Vision (precision y recall entre 0,883 y 0,923) resultaron más bajas que las del conjunto de prueba independiente (entre 0,94 y 0,97). Esta diferencia puede explicarse porque la partición interna del servicio es pequeña y aleatoria, y porque probablemente incluye casi duplicados difíciles que la depuración descrita en la sección 2.3 eliminó del conjunto de prueba. La evaluación propia, realizada sobre 459 imágenes deduplicadas y sin fuga, es la referencia confiable del desempeño del sistema.

## 6. Limitaciones y consideraciones éticas

**Dominio de los datos.** Las imágenes provienen de internet y corresponden principalmente a vehículos y entornos de Estados Unidos, el Reino Unido y la India. Antes de cualquier piloto es necesario validar el modelo con fotos reales de reclamaciones colombianas, que suelen tomarse con celulares de gama media, en condiciones de poca luz, con placas locales y por los propios asegurados.

**Atajos de contexto y encuadre.** El análisis de errores mostró que el modelo utiliza el fondo y el encuadre como pistas. Para mitigarlo se propone ampliar el conjunto con vistas completas de vehículos dañados y primeros planos de vehículos sin daño, y aplicar una etapa previa de detección que recorte el vehículo antes de clasificarlo.

**Alcance.** La clasificación es binaria: el sistema no localiza el daño ni estima su severidad o su costo. Como trabajo futuro se plantean la detección de objetos (con Custom Vision Object Detection o YOLO) para ubicar la zona afectada, la clasificación por severidad (leve, moderada o grave) y la identificación de las piezas comprometidas.

**Privacidad.** Las fotografías pueden contener placas, rostros u otros datos personales. En Colombia aplica la Ley Estatutaria 1581 de 2012, que exige contar con la autorización del titular, definir la finalidad del tratamiento y limitar los datos a los estrictamente necesarios. En lo posible, las placas y los rostros deberían anonimizarse antes de enviar las imágenes a la nube.

**Gobierno del modelo.** El sistema está concebido para **apoyar** la decisión y no para reemplazarla; nunca rechaza una reclamación de forma automática. Su operación requiere monitorear la deriva de los datos (*drift*), reentrenar periódicamente, auditar por muestreo los casos automatizados y registrar cada predicción.

**Capacidad y costos.** El nivel F0 limita el servicio a dos predicciones por segundo y 10.000 al mes, suficiente para un prototipo pero no para producción, donde se requeriría el nivel S0.

## 7. Conclusiones

El objetivo del proyecto se cumplió. El modelo final, `iter3-advanced`, clasifica fotografías de vehículos con y sin daño con un F1 de 0,954 y un AUC de 0,991 (IC 95 %: 0,985–0,997) sobre un conjunto de prueba independiente y depurado. Con el umbral de Youden alcanza un recall de 0,969, lo que significa que solo 7 de los 229 vehículos con daño quedan sin detectar.

El *transfer learning* demostró ser muy eficiente en el uso de datos: con 300 imágenes ya se obtenía un AUC de 0,989, y ni el aumento de datos ni el entrenamiento avanzado produjeron mejoras estadísticamente significativas (McNemar, p > 0,7). Esto sugiere que, para un piloto, el esfuerzo debe concentrarse en conseguir fotos del dominio real y no en acumular más imágenes genéricas.

La evaluación propia resultó indispensable. Custom Vision no reporta F1, ROC ni AUC, y su partición interna subestimó el desempeño. Además, la deduplicación con *hash* perceptual evitó que 8 imágenes repetidas entre entrenamiento y prueba inflaran los resultados.

El umbral de decisión es, en última instancia, una decisión de negocio. Ajustarlo a partir de la curva ROC tuvo más impacto en el resultado operativo que cambiar de iteración, y por eso la aplicación permite que el área de siniestros lo calibre según su tolerancia al riesgo.

Finalmente, el sistema está listo como prototipo de apoyo al triage, pero no como herramienta de decisión autónoma. Los atajos de contexto y de encuadre, junto con el origen no colombiano de las imágenes, hacen necesario validarlo con reclamaciones reales, mantener la revisión humana y auditar por muestreo. Como pasos siguientes se proponen la detección del área dañada, la estimación de severidad, la anonimización de placas y rostros conforme a la Ley 1581 de 2012 y el despliegue en el nivel S0 con monitoreo continuo.

## Referencias

- Anuj M. S. (s. f.). *Car Damage Detection* [Conjunto de datos]. Kaggle. https://www.kaggle.com/datasets/anujms/car-damage-detection
- Congreso de Colombia. (2012). *Ley Estatutaria 1581 de 2012, por la cual se dictan disposiciones generales para la protección de datos personales.*
- Espejo, A. (s. f.). *Aplicaciones y desafíos actuales en visión por computador* [Material de clase, Unidad 3]. Universidad de La Salle.
- Fawcett, T. (2006). An introduction to ROC analysis. *Pattern Recognition Letters, 27*(8), 861–874.
- McNemar, Q. (1947). Note on the sampling error of the difference between correlated proportions or percentages. *Psychometrika, 12*(2), 153–157.
- Microsoft. (2024). *What is Custom Vision?* Microsoft Learn. https://learn.microsoft.com/azure/ai-services/custom-vision-service/overview
- Microsoft. (2024). *Custom Vision REST API reference.* Microsoft Learn. https://learn.microsoft.com/rest/api/customvision/
- Pedregosa, F., Varoquaux, G., Gramfort, A., Michel, V., Thirion, B., Grisel, O., … Duchesnay, É. (2011). Scikit-learn: Machine learning in Python. *Journal of Machine Learning Research, 12*, 2825–2830.
- Streamlit Inc. (2024). *Streamlit Community Cloud documentation.* https://docs.streamlit.io/deploy/streamlit-community-cloud
- Viso.ai. (s. f.). *Computer vision business value.* https://viso.ai/deep-learning/computer-vision-business-value/
- Youden, W. J. (1950). Index for rating diagnostic tests. *Cancer, 3*(1), 32–35.
