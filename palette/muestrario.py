"""
Sección 6 del notebook: convertir los centroides en un muestrario ordenado por luminosidad,
con su proporción en la imagen y su equivalente CMYK.
"""
import numpy as np
from skimage.color import lab2rgb


def generar_muestrario(modelo, pixels_lab):
    """Convierte los centroides (LAB) a RGB y los ordena por luminosidad.
    Devuelve los colores RGB, la proporción de píxeles de cada uno y el orden aplicado."""
    centroides_lab = modelo.cluster_centers_
    centroides_rgb = lab2rgb(centroides_lab.reshape(-1, 1, 3)).reshape(-1, 3)
    centroides_rgb = np.clip(centroides_rgb, 0, 1)

    labels = modelo.labels_
    _, counts = np.unique(labels, return_counts=True)
    proporciones = counts / counts.sum()

    orden = np.argsort(-centroides_lab[:, 0])  # de más claro a más oscuro
    return centroides_rgb[orden], proporciones[orden], orden


def rgb_a_cmyk(rgb):
    """Convierte un color RGB (0-1) a su aproximación CMYK (0-1),
    la proporción de tinta usada en impresión para reproducir ese color."""
    r, g, b = rgb
    k = 1 - max(r, g, b)
    if k == 1:
        return 0.0, 0.0, 0.0, 1.0
    c = (1 - r - k) / (1 - k)
    m = (1 - g - k) / (1 - k)
    y = (1 - b - k) / (1 - k)
    return c, m, y, k


def rgb_a_hex(rgb):
    """Convierte un color RGB (0-1) a código HEX."""
    r, g, b = (np.round(np.asarray(rgb) * 255)).astype(int)
    return f"#{r:02X}{g:02X}{b:02X}"
