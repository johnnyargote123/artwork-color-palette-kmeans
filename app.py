"""
4) INTERFAZ (Streamlit)
Sube una obra, se extrae su paleta con K-means en CIELAB y se muestra cómo la representa.
Toda la lógica está en la carpeta palette/.

Ejecutar:  python -m streamlit run app.py
"""
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

from palette import (
    buscar_mejor_k,
    construir_paleta,
    convertir_a_lab,
    delta_e_promedio,
    reconstruir_imagen,
    redimensionar_proporcional,
)

CARPETA_EJEMPLOS = Path(__file__).parent / "ejemplos"
EXTENSIONES = {".jpg", ".jpeg", ".png", ".webp"}

st.set_page_config(page_title="Paletas de color con K-means", page_icon="🎨", layout="wide")


# ---------- Cálculo (en caché: la misma imagen no se procesa dos veces) ----------
@st.cache_data(show_spinner=False)
def analizar(imagen_bytes: bytes, lado_max: int, k_min: int, k_max: int):
    import io

    img = Image.open(io.BytesIO(imagen_bytes)).convert("RGB")
    img_small = redimensionar_proporcional(img, lado_max=lado_max)   # paso 1: redimensionar
    pixels_lab = convertir_a_lab(img_small)                          # paso 2: RGB -> CIELAB
    mejor, resultados = buscar_mejor_k(pixels_lab, k_range=range(k_min, k_max + 1))  # paso 3

    centroides = mejor["modelo"].cluster_centers_
    labels = mejor["labels"]
    ancho, alto = img_small.size

    return {
        "original": img,
        "paleta": construir_paleta(centroides, labels),
        "reconstruida": reconstruir_imagen(centroides, labels, alto, ancho),
        "delta_e": delta_e_promedio(pixels_lab, centroides, labels),
        "k": mejor["k"],
        "silhouette": mejor["score"],
        "tabla_k": pd.DataFrame(
            [{"K": r["k"], "Silhouette": r["score"],
              "Cluster más pequeño (%)": r["prop_min_cluster"] * 100} for r in resultados]
        ),
        "tamano_trabajo": img_small.size,
    }


def interpretar_delta_e(valor: float) -> str:
    if valor < 2:
        return "casi imperceptible"
    if valor < 10:
        return "perceptible al observar de cerca"
    return "se nota a simple vista"


# ---------- Barra lateral ----------
with st.sidebar:
    st.header("Imagen")
    archivo = st.file_uploader("Sube una obra", type=["jpg", "jpeg", "png", "webp"])

    ejemplos = sorted(p for p in CARPETA_EJEMPLOS.glob("*") if p.suffix.lower() in EXTENSIONES) \
        if CARPETA_EJEMPLOS.exists() else []
    ejemplo = None
    if ejemplos and archivo is None:
        nombre = st.selectbox("…o elige un ejemplo", [p.stem for p in ejemplos])
        ejemplo = next(p for p in ejemplos if p.stem == nombre)

    st.header("Parámetros")
    lado_max = st.slider("Lado máximo al redimensionar (px)", 80, 250, 150, 10,
                         help="Más píxeles = más detalle, pero el cálculo tarda más.")
    k_min, k_max = st.slider("Rango de K a evaluar", 2, 12, (2, 10))
    st.caption("K se elige con Silhouette Score, descartando clusters con menos del 3 % de los píxeles.")

# ---------- Encabezado ----------
st.title("🎨 Paletas de color con K-means")
st.caption("Agrupación de píxeles en espacio CIELAB · K elegido con Silhouette Score · evaluación con ΔE CIEDE2000")

if archivo is not None:
    imagen_bytes = archivo.getvalue()
elif ejemplo is not None:
    imagen_bytes = ejemplo.read_bytes()
else:
    st.info("Sube una imagen en la barra lateral para extraer su paleta.")
    st.stop()

with st.spinner(f"Probando K de {k_min} a {k_max}…"):
    r = analizar(imagen_bytes, lado_max, k_min, k_max)

# ---------- Resumen ----------
c1, c2, c3 = st.columns(3)
c1.metric("K elegido", r["k"])
c2.metric("Silhouette", f"{r['silhouette']:.3f}")
c3.metric("ΔE promedio", f"{r['delta_e']:.1f}", help="Diferencia percibida entre la imagen original y la reconstruida")

# ---------- Paleta ----------
st.subheader("Paleta")
bloques = "".join(
    f'<div style="flex:{max(c.proporcion, 0.04)};background:{c.hex};height:90px;'
    f'display:flex;align-items:flex-end;padding:6px;min-width:52px">'
    f'<span style="background:rgba(255,255,255,.85);color:#111;font:12px monospace;'
    f'padding:2px 5px;border-radius:3px">{c.hex}</span></div>'
    for c in r["paleta"]
)
st.markdown(f'<div style="display:flex;border-radius:8px;overflow:hidden">{bloques}</div>',
            unsafe_allow_html=True)
st.caption("El ancho de cada bloque es proporcional a la cantidad de píxeles de ese color.")

st.dataframe(
    pd.DataFrame([{"HEX": c.hex, "RGB": str(c.rgb), "Lab (L, a, b)": str(c.lab),
                   "Proporción": f"{c.proporcion * 100:.1f} %"} for c in r["paleta"]]),
    hide_index=True, width="stretch",
)

# ---------- Original vs. reconstruida ----------
st.subheader("¿Qué tan bien representa la paleta a la obra?")
izq, der = st.columns(2)
izq.image(r["original"], caption="Original", width="stretch")
der.image(r["reconstruida"], caption=f"Pintada solo con los {r['k']} colores de la paleta",
          width="stretch")
st.write(f"ΔE CIEDE2000 promedio: **{r['delta_e']:.1f}** — la diferencia "
         f"{interpretar_delta_e(r['delta_e'])}. La reconstrucción se hace sobre la imagen "
         f"de trabajo de {r['tamano_trabajo'][0]}×{r['tamano_trabajo'][1]} px.")

# ---------- Elección de K ----------
with st.expander("Ver cómo se eligió K"):
    tabla = r["tabla_k"].copy()
    tabla["Válido (≥ 3 %)"] = tabla["Cluster más pequeño (%)"] >= 3
    st.line_chart(tabla.set_index("K")["Silhouette"])
    st.dataframe(tabla.style.format({"Silhouette": "{:.3f}", "Cluster más pequeño (%)": "{:.1f}"}),
                 hide_index=True, width="stretch")
    st.caption("Se elige el K con mayor silhouette entre los válidos. Si ninguno es válido, se usa el mejor de todos.")
