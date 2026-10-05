"""
Sección 5 del notebook: K-means con búsqueda automática de K.

Criterio:
1. Se prueban K de 4 a 9 y se calcula el Silhouette Score de cada uno.
2. Se descartan los K donde el cluster más pequeño tiene menos del 2 % de los píxeles.
3. Entre los K cuyo score está a menos de `tolerancia` del mejor, se elige el K más alto
   (a igual calidad estadística, más clusters capturan más matices; caso del rojo de
   la túnica en San Jerónimo de Lorenzo Lotto).
"""
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

PROPORCION_MINIMA = 0.02


def buscar_mejor_k(pixels_lab, k_range=range(4, 10), sample_size=4000, seed=10, tolerancia=0.01):
    """Busca el número óptimo de clusters K para una imagen, usando Silhouette Score.
    El score se calcula sobre una submuestra de píxeles por eficiencia (silhouette
    es O(n^2)). Se descartan K con clusters muy pequeños (< 2% de los píxeles).
    Entre los K cuyo score esté dentro de `tolerancia` del mejor score encontrado, se
    prefiere el K más alto."""
    n = pixels_lab.shape[0]
    rs = np.random.RandomState(seed)
    idx_sample = rs.choice(n, size=min(sample_size, n), replace=False)
    sample = pixels_lab[idx_sample]

    resultados = []
    for k in k_range:
        km = KMeans(n_clusters=k, n_init=10, random_state=seed)
        labels_full = km.fit_predict(pixels_lab)
        labels_sample = km.predict(sample)
        score = silhouette_score(sample, labels_sample)
        _, counts = np.unique(labels_full, return_counts=True)
        prop_min = counts.min() / n
        resultados.append({"k": k, "score": score, "prop_min_cluster": prop_min,
                           "modelo": km, "labels": labels_full})

    validos = [r for r in resultados if r["prop_min_cluster"] >= PROPORCION_MINIMA]
    candidatos = validos if validos else resultados

    mejor_score = max(r["score"] for r in candidatos)
    empatados = [r for r in candidatos if mejor_score - r["score"] <= tolerancia]
    mejor = max(empatados, key=lambda r: r["k"])
    return mejor, resultados
