"""
Sección 10.2 del notebook: armonía de color según el modelo de Itten.

El tono (hue) se calcula a partir de los canales a y b de LAB. Los colores con croma baja
(grises, negros, blancos casi neutros) tienen un tono numéricamente inestable, porque
arctan2(b, a) con a y b cercanos a cero cambia mucho con variaciones mínimas; por eso se
excluyen antes de evaluar la armonía.
"""
import numpy as np

CHROMA_MINIMO = 10

PLANTILLAS_ARMONIA = {
    "complementaria": [180],
    "analoga": [30],
    "triada": [120, 240],
    "tetrada": [90, 180, 270],
}

PLANTILLAS_OFFSETS = {
    "complementaria": [180],
    "analoga": [30, -30],
    "triada": [120, 240],
    "tetrada": [90, 180, 270],
}


def calcular_hue_lab(lab_color):
    """Calcula el ángulo de tono (hue, en grados 0-360) a partir de los canales
    a y b de un color en LAB."""
    L, a, b = lab_color
    return np.degrees(np.arctan2(b, a)) % 360


def calcular_chroma(centroides_lab):
    """Saturación de cada color en LAB: distancia al eje neutro."""
    return np.sqrt(centroides_lab[:, 1] ** 2 + centroides_lab[:, 2] ** 2)


def evaluar_armonia(centroides_lab):
    """Calcula, para cada par de colores de la paleta, la diferencia de hue,
    y mide qué tan cerca está esa diferencia de alguna plantilla clásica de
    armonía. Devuelve la plantilla más cercana y su error promedio en grados
    (0 = encaja perfecto, valores más altos = menos armonioso según la teoría)."""
    hues = [calcular_hue_lab(c) for c in centroides_lab]
    n = len(hues)

    diferencias = []
    for i in range(n):
        for j in range(i + 1, n):
            diff = abs(hues[i] - hues[j])
            diff = min(diff, 360 - diff)  # distancia circular
            diferencias.append(diff)

    mejor_plantilla, menor_error = None, float("inf")
    for nombre, angulos_objetivo in PLANTILLAS_ARMONIA.items():
        errores = [min(abs(d - ang) for ang in angulos_objetivo) for d in diferencias]
        error_promedio = np.mean(errores)
        if error_promedio < menor_error:
            menor_error = error_promedio
            mejor_plantilla = nombre

    return mejor_plantilla, float(menor_error)


def evaluar_armonia_saturados(centroides_lab, chroma_minimo=CHROMA_MINIMO):
    """Evalúa armonía solo sobre colores con suficiente saturación (chroma),
    excluyendo grises/negros/blancos casi neutros, cuyo hue es inestable."""
    chromas = calcular_chroma(centroides_lab)
    centroides_saturados = centroides_lab[chromas >= chroma_minimo]

    if len(centroides_saturados) < 2:
        return None, None, len(centroides_saturados)

    plantilla, error = evaluar_armonia(centroides_saturados)
    return plantilla, error, len(centroides_saturados)


def sugerir_color_armonico(color_base_lab, plantilla="complementaria"):
    """Dado un color base (en LAB), calcula qué color(es) agregar para
    completar la plantilla de armonía elegida, manteniendo la misma
    luminosidad y saturación (chroma) del color base; solo cambia el hue."""
    L, a, b = color_base_lab
    chroma = np.sqrt(a ** 2 + b ** 2)
    hue_base = np.degrees(np.arctan2(b, a)) % 360

    if plantilla not in PLANTILLAS_OFFSETS:
        raise ValueError(f"Plantilla '{plantilla}' no reconocida. Opciones: {list(PLANTILLAS_OFFSETS)}")

    colores_sugeridos = []
    for offset in PLANTILLAS_OFFSETS[plantilla]:
        hue_nuevo = (hue_base + offset) % 360
        a_nuevo = chroma * np.cos(np.radians(hue_nuevo))
        b_nuevo = chroma * np.sin(np.radians(hue_nuevo))
        colores_sugeridos.append(np.array([L, a_nuevo, b_nuevo]))

    return colores_sugeridos
