"""
Secciones 3 y 4 del notebook: redimensionar conservando la proporción y convertir a CIELAB,
encadenados en un Pipeline de scikit-learn.
"""
import numpy as np
from PIL import Image
from skimage.color import rgb2lab
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer

LADO_ENTRENAMIENTO = 150  # px del lado más largo para entrenar (rápido)


def redimensionar_proporcional(img, lado_max=150):
    """Redimensiona una imagen para que su lado más largo mida `lado_max` px,
    conservando la proporción original (sin deformar, sin recortar,
    sin agregar píxeles de relleno)."""
    ancho_original, alto_original = img.size

    if ancho_original >= alto_original:
        nuevo_ancho = lado_max
        nuevo_alto = int(alto_original * (lado_max / ancho_original))
    else:
        nuevo_alto = lado_max
        nuevo_ancho = int(ancho_original * (lado_max / alto_original))

    return img.resize((nuevo_ancho, nuevo_alto), Image.LANCZOS)


def convertir_a_lab(img):
    """Convierte una imagen PIL (RGB) a un array de píxeles en espacio CIELAB.
    Devuelve una matriz de forma (n_píxeles, 3).
    Los píxeles vienen en 0-255, pero rgb2lab trabaja en escala [0, 1]."""
    rgb_array = np.asarray(img) / 255.0
    lab_array = rgb2lab(rgb_array)
    return lab_array.reshape(-1, 3)


pipeline_preparacion = Pipeline([
    ("resize", FunctionTransformer(lambda img: redimensionar_proporcional(img, lado_max=LADO_ENTRENAMIENTO))),
    ("a_lab", FunctionTransformer(convertir_a_lab)),
])
