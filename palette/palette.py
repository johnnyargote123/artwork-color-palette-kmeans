"""
3) PALETA Y EVALUACIÓN
Convierte los centroides en un muestrario ordenado por proporción y mide qué tan bien
representa la paleta a la imagen (reconstrucción + ΔE CIEDE2000).
"""
from dataclasses import dataclass

import numpy as np
from skimage.color import deltaE_ciede2000, lab2rgb


@dataclass
class Color:
    """Un color de la paleta."""
    hex: str
    rgb: tuple[int, int, int]
    lab: tuple[float, float, float]
    proporcion: float  # fracción de píxeles de la imagen que pertenecen a este color


def _lab_a_rgb255(lab: np.ndarray) -> np.ndarray:
    """Convierte colores CIELAB de forma (n, 3) a RGB en 0-255."""
    rgb = lab2rgb(lab.reshape(1, -1, 3)).reshape(-1, 3)  # lab2rgb espera una "imagen"
    return np.clip(np.round(rgb * 255), 0, 255).astype(int)


def construir_paleta(centroides_lab: np.ndarray, labels: np.ndarray) -> list[Color]:
    """Devuelve los colores ordenados de mayor a menor proporción en la imagen.
    El número de cada cluster es arbitrario; lo que importa es su color y su peso."""
    conteos = np.bincount(labels, minlength=len(centroides_lab))
    proporciones = conteos / conteos.sum()
    rgbs = _lab_a_rgb255(centroides_lab)

    colores = [
        Color(
            hex="#{:02X}{:02X}{:02X}".format(*rgb),
            rgb=tuple(int(v) for v in rgb),
            lab=tuple(round(float(v), 1) for v in lab),
            proporcion=float(p),
        )
        for lab, rgb, p in zip(centroides_lab, rgbs, proporciones)
    ]
    return sorted(colores, key=lambda c: c.proporcion, reverse=True)


def reconstruir_imagen(centroides_lab, labels, alto, ancho) -> np.ndarray:
    """Pinta cada píxel con el color de su cluster. Devuelve RGB en [0, 1]."""
    lab_reconstruido = centroides_lab[labels].reshape(alto, ancho, 3)
    return np.clip(lab2rgb(lab_reconstruido), 0, 1)


def delta_e_promedio(pixels_lab, centroides_lab, labels) -> float:
    """ΔE CIEDE2000 promedio entre cada píxel original y el color que lo reemplaza.
    Menor es mejor: < 1 imperceptible, 2-10 perceptible de cerca, > 10 se nota a simple vista."""
    return float(np.mean(deltaE_ciede2000(pixels_lab, centroides_lab[labels])))
