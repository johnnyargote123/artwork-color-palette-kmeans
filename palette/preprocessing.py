"""
1) PREPROCESAMIENTO
Mismo procedimiento del notebook (secciones 3 y 4):
redimensionar conservando la proporción y convertir de RGB a CIELAB.
"""
import numpy as np
from PIL import Image
from skimage.color import rgb2lab


def redimensionar_proporcional(img: Image.Image, lado_max: int = 150) -> Image.Image:
    """Deja el lado más largo en `lado_max` px y ajusta el otro en proporción,
    sin deformar, sin recortar y sin agregar relleno."""
    ancho_original, alto_original = img.size

    if ancho_original >= alto_original:
        nuevo_ancho = lado_max
        nuevo_alto = int(alto_original * (lado_max / ancho_original))
    else:
        nuevo_alto = lado_max
        nuevo_ancho = int(ancho_original * (lado_max / alto_original))

    return img.resize((nuevo_ancho, nuevo_alto), Image.LANCZOS)


def convertir_a_lab(img: Image.Image) -> np.ndarray:
    """Convierte una imagen PIL (RGB) a una matriz de píxeles CIELAB de forma (n_píxeles, 3).
    rgb2lab espera valores en [0, 1], por eso se divide entre 255."""
    rgb_array = np.asarray(img.convert("RGB")) / 255.0
    lab_array = rgb2lab(rgb_array)
    return lab_array.reshape(-1, 3)
