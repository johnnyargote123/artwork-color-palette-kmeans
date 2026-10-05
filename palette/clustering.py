"""
2) AGRUPACIÓN
Mismo procedimiento del notebook (sección 5): K-means sobre los píxeles en CIELAB,
eligiendo K con Silhouette Score y descartando K con clusters de menos del 3 %.
"""
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

PROPORCION_MINIMA = 0.03  # un cluster con menos del 3 % de los píxeles se considera ruido


def buscar_mejor_k(pixels_lab, k_range=range(2, 11), sample_size=4000, seed=42):
    """Prueba cada K del rango y devuelve el de mejor silhouette entre los válidos.

    - Silhouette se calcula sobre una submuestra de píxeles, porque su costo es O(n^2).
    - n_init se fija en 10 de forma explícita: el valor por defecto cambió entre
      versiones de scikit-learn y podría alterar los resultados sin avisar.
    """
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
    mejor = max(candidatos, key=lambda r: r["score"])
    return mejor, resultados
