# Guion del video de sustentación — PeritIA

Duración objetivo: **9 minutos** (límite 10). A un ritmo natural de unas 130 palabras por minuto, la narración de abajo dura alrededor de 8 minutos y deja margen para las transiciones y la demo.

## 1. Preparación antes de grabar

### Configuración de OBS Studio (gratuito: https://obsproject.com)

1. **Escena "Pantalla":** fuente *Captura de pantalla* (monitor completo) o *Captura de ventana* del navegador.
2. **Cámara (opcional):** fuente *Dispositivo de captura de video*, pequeña, en la esquina inferior derecha. Mostrar la cara da cercanía.
3. **Audio:** en *Configuración → Audio* deje solo el micrófono y agréguele el filtro *Supresión de ruido (RNNoise)*.
4. **Salida:** *Configuración → Salida → Grabación*: formato **MP4**, calidad "Alta". *Video*: 1920×1080 a 30 FPS.
5. Haga una grabación de prueba de 20 segundos y escúchela con audífonos.

Alternativa más simple: **Clipchamp** (incluido en Windows 11) o la grabadora de pantalla de **PowerPoint**.

### Pestañas abiertas y en orden (navegador a pantalla completa, zoom 110–125 %)

1. El informe PDF abierto en la portada.
2. Portal de Azure → grupo de recursos `ComputerVisionImages` (sin abrir *Claves y punto de conexión*).
3. Custom Vision → proyecto `DanosVehiculares`, pestaña *Training Images*. **Entre desde el portal de Azure:** recurso `AnimalIdentificationJGR` → *Información general* → enlace *Portal de Custom Vision*. Si entra directo a customvision.ai, el inicio de sesión con la cuenta de La Salle falla.
4. La misma página en la pestaña *Performance*, con *Iteration 3* seleccionada.
5. La app: https://peritia-evaluador-danios.streamlit.app/ (ábrala unos minutos antes para que "despierte").
6. El repositorio: https://github.com/guzmandb/PeritIA

Tenga en una carpeta las 6 fotos de la demo y `demo_eval.zip` (sección 4).

### Seguridad

- No muestre el archivo `.env`, la página *Claves y punto de conexión* ni el diálogo *Prediction URL* de Custom Vision, porque muestran la clave.
- Active el modo *No molestar* de Windows y cierre pestañas personales.

### Consejos para hablar

- Hable como si le explicara el proyecto a un colega de la aseguradora: con calma y sin leer palabra por palabra. El guion es un apoyo, no un teleprompter.
- Grabe **por bloques** (una toma por sección) y luego únalas. Si se equivoca, repite solo ese bloque. OBS permite pausar la grabación.
- Cada vez que mencione una cifra, señálela con el mouse.
- **Sí mencione dónde trabaja y por qué eligió el tema:** es el "contexto y relación con el trabajo" que pide la rúbrica y le da sentido al proyecto. No es necesario decir el nombre de la empresa.
- **Sí muestre Azure:** es la evidencia de que el modelo se entrenó y se desplegó en la nube. Bastan 1 a 2 minutos.

## 2. Estructura y tiempos

| Tiempo | Bloque | En pantalla |
|---|---|---|
| 0:00–0:40 | Presentación | Portada del informe |
| 0:40–1:50 | Contexto y motivación | Sección 1 del informe |
| 1:50–3:00 | Dataset y preparación | Figuras 1 a 3 y tabla de duplicados |
| 3:00–4:30 | Entrenamiento en Azure | Portal de Azure y customvision.ai |
| 4:30–5:15 | Despliegue | Figura de arquitectura y repositorio |
| 5:15–7:15 | Demo de la app | App publicada |
| 7:15–8:40 | Resultados y análisis | Curva ROC, matriz, barrido de umbral y errores |
| 8:40–9:20 | Limitaciones y cierre | Conclusiones del informe |

## 3. Narración

### 0:00 — Presentación (portada del informe)

> Hola, profesor. Mi nombre es Jherson Guzmán, estudiante de la Maestría en Inteligencia Artificial de la Universidad de La Salle. En este video presento mi proyecto final de Visión por Computador: **PeritIA**, un sistema que clasifica fotografías de vehículos como *con daño* o *sin daño* para apoyar la gestión de siniestros de autos. Voy a mostrar el contexto, el dataset, el entrenamiento y el despliegue en Azure, la aplicación web funcionando y, al final, el análisis de las métricas.

### 0:40 — Contexto y motivación (sección 1)

> Trabajo como científico de datos en una compañía de seguros en Colombia. En el ramo de autos, cuando un cliente reporta un siniestro, envía fotos del vehículo por la app, por correo o por WhatsApp, y un perito las revisa una por una para confirmar si hay daño visible antes de continuar el trámite.
>
> Esa primera revisión es manual, repetitiva y crece con el volumen de reclamaciones. Además, una parte de las fotos no muestra ningún daño: son fotos incompletas, de otro carro o reclamos inconsistentes. Por eso elegí este tema: es un problema real de mi sector en el que la visión por computador aporta valor directo.
>
> La propuesta es un primer filtro con tres salidas: si el modelo está seguro de que hay daño, el caso avanza a taller; si está seguro de que no hay daño, se marca para validar la coherencia del reclamo, que puede ser una señal de fraude; y si tiene dudas, va a un perito. El área de aplicación es el reconocimiento de objetos mediante clasificación de imágenes con *transfer learning*.

### 1:50 — Dataset (figuras 1, 2 y 3)

> Usé el dataset público *Car Damage Detection* de Kaggle: 2.300 fotos reales en dos clases balanceadas. En el análisis exploratorio encontré dos cosas importantes. Primero, las imágenes son pequeñas: el 67 % tiene el lado menor por debajo de 256 píxeles. Segundo, hay un posible sesgo de encuadre: las fotos con daño suelen ser primeros planos en talleres, y las sin daño, fotos comerciales del carro completo.
>
> Antes de entrenar limpié los datos. Además de normalizar las imágenes, busqué duplicados con un hash MD5 y con un *hash* perceptual, dHash, que detecta fotos casi iguales aunque estén reescaladas. Encontré 24 duplicados y, lo más importante, **8 fotos que estaban a la vez en entrenamiento y en prueba**. Si no se eliminan, el modelo se evalúa con fotos que ya vio y las métricas salen infladas. Al final quedaron 1.817 imágenes para entrenar y 459 para una prueba independiente que **nunca se subió a Azure**.

### 3:00 — Entrenamiento en Azure (portal y customvision.ai)

*(Mostrar el grupo de recursos en el portal de Azure.)*

> Para el modelo usé Azure Custom Vision. Reutilicé los recursos que había creado en la Actividad 4: el grupo de recursos `ComputerVisionImages`, con un recurso de entrenamiento y uno de predicción, ambos en el nivel gratuito. Lo importante no es el nombre del recurso, sino que sobre él creé un proyecto nuevo, exclusivo para este problema.

*(Cambiar a customvision.ai, pestaña Training Images.)*

> Este es el proyecto `DanosVehiculares`, de clasificación multiclase con el dominio *General A2*, y estas son las dos etiquetas: `danado` y `sin_dano`. Todo el flujo —crear el proyecto, subir las imágenes, entrenar y publicar— lo automaticé con la API REST de Custom Vision en Python, para que sea reproducible.

*(Pestaña Performance; recorrer las tres iteraciones.)*

> Entrené tres iteraciones para responder tres preguntas. La primera, con solo 300 imágenes, para ver qué tan lejos se llega con pocos datos. La segunda, con todas las imágenes. Y la tercera, con el entrenamiento avanzado de una hora, que hace una búsqueda de arquitectura más profunda. Aquí se ven la precision y el recall que reporta el portal. Pero el portal solo los calcula sobre una partición interna pequeña y no da F1, curva ROC ni AUC. Por eso hice mi propia evaluación, que muestro más adelante.

### 4:30 — Despliegue (figura de arquitectura y repositorio)

> El despliegue tiene dos capas en la nube. La primera es el modelo: cada iteración quedó publicada como un endpoint REST en Azure; la aplicación le envía la imagen y recibe un JSON con la probabilidad de cada clase. La segunda es la aplicación web, desarrollada en Streamlit y publicada en Streamlit Community Cloud desde este repositorio de GitHub. Las claves del endpoint no están en el código: se configuran como *secrets* en la plataforma.

### 5:15 — Demo de la app (app publicada)

> Esta es la aplicación, abierta desde su enlace público. En la barra lateral se ve que está conectada al endpoint de Azure con la iteración final, junto con el umbral de decisión y el margen de la zona de duda.

*(Subir las 6 fotos.)*

> Voy a cargar seis fotos del conjunto de prueba. Las dos primeras tienen daño evidente y el modelo las clasifica con 99,8 % de probabilidad. Estas dos no tienen daño y el modelo da prácticamente cero. Y estas dos son interesantes: este carro tiene daño, pero se ve de lejos y en un patio, y esta foto trasera sin daño tiene reflejos. El modelo da 32 % y 50 %, y como están cerca del umbral, la aplicación no decide sola: las envía a un perito.

*(Llevar el margen de la zona de duda a 0.)*

> Si quito la zona de duda, el modelo decide todo y se equivoca en esas dos fotos: una sale como falso negativo y la otra como falso positivo. Eso muestra por qué la zona de duda es útil.

*(Pestaña Evaluar un dataset etiquetado; subir `demo_eval.zip` y pulsar Evaluar.)*

> La segunda pestaña permite evaluar el modelo con un conjunto etiquetado. Subo un zip con 30 fotos en dos carpetas y la aplicación calcula en vivo la exactitud, la precision, el recall, el F1, la matriz de confusión y la curva ROC. Esto sirve para que el área de negocio valide el modelo con sus propias fotos antes de usarlo.

### 7:15 — Resultados y análisis (secciones 4 y 5 del informe)

*(Curva ROC.)*

> Ahora los resultados sobre las 459 imágenes de prueba. El modelo final, la iteración avanzada, logró una exactitud de 95,4 %, un F1 de 0,954 y un **AUC de 0,991**, con un intervalo de confianza entre 0,985 y 0,997. Un AUC de 0,991 significa que, si tomo al azar una foto con daño y una sin daño, el modelo le da mayor probabilidad de daño a la correcta el 99,1 % de las veces.
>
> Un hallazgo interesante: con solo 300 fotos ya se llegaba a 0,989. Más datos y el entrenamiento avanzado mejoran poco, y la prueba de McNemar confirma que la diferencia no es estadísticamente significativa.

*(Matriz de confusión y barrido de umbral.)*

> En seguros los dos errores no cuestan lo mismo. Un falso positivo solo hace que un perito revise un caso. Un falso negativo puede llevar a objetar una reclamación con daño real. Por eso ajusté el umbral con la curva ROC: con el índice de Youden, en 0,42, el recall sube a 96,9 % y solo se escapan 7 de 229 daños, a cambio de dos falsos positivos más.

*(Galería de errores.)*

> Y aquí están los errores. Los falsos positivos son carros mojados, con charcos o en talleres: el modelo aprendió el contexto como pista. Los falsos negativos son daños vistos de lejos o en carros oscuros: es el sesgo de encuadre que había detectado en el análisis exploratorio.

### 8:40 — Limitaciones y cierre (conclusiones)

> Como limitaciones, las fotos vienen de internet y no de reclamaciones colombianas, así que antes de un piloto habría que validar con fotos reales. Además, las imágenes pueden tener placas o rostros, por lo que aplica la Ley 1581 de protección de datos. Por eso el sistema apoya al perito, no lo reemplaza, y nunca objeta un reclamo de forma automática.
>
> En conclusión, PeritIA clasifica daños con un AUC de 0,991, está desplegado en la nube y tiene una aplicación web lista para usar. Como trabajo futuro propongo detectar la zona dañada y estimar su severidad. Muchas gracias por su atención.

## 4. Material para la demo

Fotos (en `data/test/` de la carpeta local del proyecto):

- `danado/danado_test_0060.jpg` → 99,8 % de daño (daño detectado)
- `danado/danado_test_0100.jpg` → 99,8 % (daño detectado)
- `danado/danado_test_0202.jpg` → 32,4 % (zona de duda: daño visto de lejos, fondo de patio)
- `sin_dano/sin_dano_test_0060.jpg` → 0,0 % (sin daño visible)
- `sin_dano/sin_dano_test_0100.jpg` → 0,0 % (sin daño visible)
- `sin_dano/sin_dano_test_0080.jpg` → 50,3 % (zona de duda: vista trasera con reflejos)

Dataset de evaluación: `data/demo_eval.zip` (30 imágenes; exactitud 0,967, AUC 1,000).

## 5. Preguntas que pueden surgir

- **¿Por qué Custom Vision?** Aplica *transfer learning* sobre redes preentrenadas, da buen desempeño con pocos miles de imágenes, el despliegue como API es inmediato y ya existía la infraestructura de la Actividad 4.
- **¿Por qué una evaluación propia?** El portal solo da precision, recall y AP sobre una partición interna aleatoria; no da F1, ROC ni AUC. El conjunto de prueba propio es independiente y está libre de duplicados.
- **¿Qué es el índice de Youden?** Es el punto de la curva ROC que maximiza sensibilidad + especificidad − 1; equilibra los dos tipos de error.
- **¿Qué es dHash?** Un *hash* perceptual: imágenes casi iguales producen *hashes* con pocos bits distintos. Se usó una distancia de Hamming de hasta 4 sobre 64 bits.
- **¿Por qué el portal da 0,92 y la prueba 0,96?** La partición interna del portal es pequeña y probablemente contiene casi duplicados difíciles que se eliminaron del conjunto de prueba.

## 6. Lista de verificación final

- [ ] Video de máximo 10 minutos, en 1080p, con audio claro.
- [ ] No aparecen claves, `.env` ni la página de claves de Azure.
- [ ] Subido a YouTube como **No listado**; el enlace funciona en una ventana de incógnito.
- [ ] Enlace pegado en la portada del informe y PDF regenerado.
- [ ] PDF entregado antes de las 23:59 del 22 de noviembre.
