"""
Sección 9 del notebook: función integradora.
No reimplementa lógica: encadena las funciones de las secciones 4 a 8.
La única diferencia con el notebook es que recibe una imagen ya abierta en lugar
de una ruta, porque en la app la imagen llega desde el navegador.
"""
import os

from .agrupacion import buscar_mejor_k
from .evaluacion import reconstruir_imagen_alta_resolucion
from .muestrario import generar_muestrario
from .preparacion import pipeline_preparacion


def extraer_paleta(img_original, lado_reconstruccion=800, **kwargs_busqueda):
    """De una imagen PIL (RGB) a los resultados principales del método."""
    # Preparación (sección 4.3)
    pixels_lab = pipeline_preparacion.transform(img_original)

    # Búsqueda de K (sección 5.2)
    mejor_k, resultados_k = buscar_mejor_k(pixels_lab, **kwargs_busqueda)

    # Muestrario (sección 6.2)
    colores_rgb, proporciones, orden = generar_muestrario(mejor_k["modelo"], pixels_lab)

    # Reconstrucción en alta resolución (sección 8)
    img_reconstruida = reconstruir_imagen_alta_resolucion(
        mejor_k["modelo"], img_original, lado_max=lado_reconstruccion
    )

    return {
        "pixels_lab": pixels_lab,
        "modelo": mejor_k["modelo"],
        "k": mejor_k["k"],
        "silhouette": mejor_k["score"],
        "resultados_k": resultados_k,
        "colores_rgb": colores_rgb,
        "proporciones": proporciones,
        "centroides_lab_ordenados": mejor_k["modelo"].cluster_centers_[orden],
        "img_reconstruida": img_reconstruida,
    }


def parsear_nombre_archivo(nombre_archivo):
    """Extrae autor y título directamente del nombre del archivo
    (convención: autor-slug_titulo-slug.jpg)."""
    base = os.path.splitext(nombre_archivo)[0]
    partes = base.split("_", 1)
    autor = partes[0].replace("-", " ").title()
    titulo = partes[1].replace("-", " ").title() if len(partes) > 1 else "Sin título"
    return autor, titulo
