# Resultados en el conjunto de prueba (n = 459)

Umbral de decisión: 0.5. Clase positiva: con daño.

| Modelo | Accuracy | Precision | Recall | Especificidad | F1 | F1 macro | ROC AUC (IC 95 %) | AP |
|---|---|---|---|---|---|---|---|---|
| iter1-baseline | 0.948 | 0.960 | 0.934 | 0.961 | 0.947 | 0.948 | 0.989 (0.981–0.995) | 0.988 |
| iter2-full | 0.954 | 0.964 | 0.943 | 0.965 | 0.954 | 0.954 | 0.990 (0.983–0.996) | 0.990 |
| iter3-advanced | 0.954 | 0.960 | 0.948 | 0.961 | 0.954 | 0.954 | 0.991 (0.985–0.997) | 0.991 |

## Modelo final: iter3-advanced

Matriz (umbral 0.5): TP=217 FP=9 TN=221 FN=12

Umbrales candidatos: youden = 0.416, max_f1 = 0.416, recall_95 = 0.483

Con umbral Youden (0.416): precision=0.953, recall=0.969, F1=0.961, especificidad=0.952
Latencia del endpoint: mediana 561 ms, p95 1851 ms

## Prueba de McNemar (exacta) contra iter3-advanced, umbral 0.5

| Modelo | Solo acierta el final | Solo acierta este modelo | p-valor |
|---|---|---|---|
| iter1-baseline | 16 | 13 | 0.711 |
| iter2-full | 6 | 6 | 1.000 |

## Métricas internas de Custom Vision (umbral 0.5, partición interna)

| Iteración | Tipo | Precision | Recall | AP |
|---|---|---|---|---|
| Iteration 1 | Regular | 0.883 | 0.883 | 0.981 |
| Iteration 2 | Regular | 0.923 | 0.923 | 0.972 |
| Iteration 3 | Advanced | 0.923 | 0.923 | 0.972 |