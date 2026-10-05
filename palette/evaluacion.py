"""
Secciones 7, 8 y 10.3 del notebook: t-SNE, reconstrucción, ΔE CIEDE2000 y discriminabilidad.
"""
import numpy as np
from skimage.color import deltaE_ciede2000, lab2rgb
from sklearn.manifold import TSNE

from .preparacion import convertir_a_lab, redimensionar_proporcional

UMBRAL_DISCRIMINABILIDAD = 10  # distancia LAB por debajo de la cual dos colores se confunden


def visualizar_tsne(pixels_lab, sample_size=2000, perplexity=30, seed=0):
    """Reduce una muestra de píxeles LAB a 2D con t-SNE, para visualizar
    la distribución de colores de la imagen."""
    n = pixels_lab.shape[0]
    rs = np.random.RandomState(seed)
    idx = rs.choice(n, size=min(sample_size, n), replace=False)

    tsne = TSNE(n_components=2, perplexity=perplexity, random_state=seed, verbose=0)
    emb = tsne.fit_transform(pixels_lab[idx])
    colores_pixel = np.clip(lab2rgb(pixels_lab[idx].reshape(-1, 1, 3)).reshape(-1, 3), 0, 1)

    return emb, colores_pixel


def reconstruir_imagen_alta_resolucion(modelo, img_original, lado_max=800):
    """Entrena a 150 px (rápido) pero reconstruye a 800 px prediciendo el cluster de cada
    píxel con el modelo ya entrenado; predict() es mucho más barato que fit()."""
    img_grande = redimensionar_proporcional(img_original, lado_max=lado_max)
    pixels_lab_grande = convertir_a_lab(img_grande)
    labels_grande = modelo.predict(pixels_lab_grande)

    centroides_lab = modelo.cluster_centers_
    centroides_rgb = lab2rgb(centroides_lab.reshape(-1, 1, 3)).reshape(-1, 3)
    centroides_rgb = np.clip(centroides_rgb, 0, 1)

    ancho, alto = img_grande.size
    return centroides_rgb[labels_grande].reshape(alto, ancho, 3)


def evaluar_reconstruccion(modelo, pixels_lab):
    """Calcula el error perceptual (ΔE00) entre cada píxel original y el
    color de su centroide asignado, para evaluar qué tan fiel es la
    reconstrucción a la imagen real.
    Se usa ΔE CIEDE2000 en vez de la distancia euclidiana en LAB porque corrige
    distorsiones conocidas de esta y mide mejor lo que percibe una persona."""
    labels = modelo.labels_
    centroides_lab = modelo.cluster_centers_
    pixels_reconstruidos = centroides_lab[labels]

    delta_e00 = deltaE_ciede2000(
        pixels_lab.reshape(-1, 1, 3),
        pixels_reconstruidos.reshape(-1, 1, 3)
    ).reshape(-1)

    return {
        "media": float(np.mean(delta_e00)),
        "mediana": float(np.median(delta_e00)),
        "p95": float(np.percentile(delta_e00, 95)),
        "delta_e00": delta_e00,
    }


def evaluar_discriminabilidad(centroides_lab):
    """Mide qué tan fácil es distinguir cada color de la paleta de los demás,
    calculando la distancia euclidiana mínima en espacio LAB entre cualquier
    par de colores, la misma métrica que ya usa K-means para formar los clusters."""
    n = len(centroides_lab)
    distancias = []
    pares = []
    for i in range(n):
        for j in range(i + 1, n):
            d = float(np.linalg.norm(centroides_lab[i] - centroides_lab[j]))
            distancias.append(d)
            pares.append((i, j))

    idx_min = int(np.argmin(distancias))
    return {
        "distancia_minima": distancias[idx_min],
        "distancia_promedio": float(np.mean(distancias)),
        "par_mas_parecido": pares[idx_min],
    }
