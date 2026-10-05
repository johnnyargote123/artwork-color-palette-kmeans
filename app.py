"""
Interfaz (Streamlit) del Microproyecto 1 de MLNS.
Sube una obra y obtén su paleta con K-means en CIELAB, la evaluación de la reconstrucción,
la armonía de color, la discriminabilidad y el t-SNE.
Toda la lógica está en la carpeta palette/, copiada del notebook final.

Ejecutar:  python -m streamlit run app.py
"""
import io
from pathlib import Path

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image
from skimage.color import lab2rgb

from palette import (
    CHROMA_MINIMO,
    UMBRAL_DISCRIMINABILIDAD,
    calcular_chroma,
    evaluar_armonia_saturados,
    evaluar_discriminabilidad,
    evaluar_reconstruccion,
    extraer_paleta,
    parsear_nombre_archivo,
    rgb_a_cmyk,
    rgb_a_hex,
    sugerir_color_armonico,
    visualizar_tsne,
)

CARPETA_EJEMPLOS = Path(__file__).parent / "ejemplos"
EXTENSIONES = {".jpg", ".jpeg", ".png", ".webp"}

st.set_page_config(page_title="Paletas de color con K-means", page_icon="🎨", layout="wide")


# ---------- Cálculos en caché: la misma imagen no se procesa dos veces ----------
@st.cache_data(show_spinner=False)
def analizar(imagen_bytes: bytes, k_min: int, k_max: int, tolerancia: float):
    img = Image.open(io.BytesIO(imagen_bytes)).convert("RGB")
    r = extraer_paleta(img, k_range=range(k_min, k_max + 1), tolerancia=tolerancia)
    r["original"] = img
    r["reconstruccion"] = evaluar_reconstruccion(r["modelo"], r["pixels_lab"])
    r["discriminabilidad"] = evaluar_discriminabilidad(r["centroides_lab_ordenados"])
    return r


@st.cache_data(show_spinner=False)
def calcular_tsne(imagen_bytes: bytes, perplexity: int):
    img = Image.open(io.BytesIO(imagen_bytes)).convert("RGB")
    from palette import pipeline_preparacion
    return visualizar_tsne(pipeline_preparacion.transform(img), perplexity=perplexity)


def lab_a_hex(lab):
    rgb = np.clip(lab2rgb(np.asarray(lab).reshape(1, 1, 3)).reshape(3), 0, 1)
    return rgb_a_hex(rgb)


def franjas(hexes, proporciones, bordes=None, alto=90):
    """Muestrario horizontal: el ancho de cada franja es proporcional a su peso en la imagen."""
    bordes = bordes or [""] * len(hexes)
    bloques = "".join(
        f'<div style="flex:{max(p, 0.035)};background:{h};height:{alto}px;{b}'
        f'display:flex;align-items:flex-end;padding:6px;min-width:46px;box-sizing:border-box">'
        f'<span style="background:rgba(255,255,255,.85);color:#111;font:11px monospace;'
        f'padding:2px 4px;border-radius:3px">{h}</span></div>'
        for h, p, b in zip(hexes, proporciones, bordes)
    )
    st.markdown(f'<div style="display:flex;gap:2px;border-radius:8px;overflow:hidden">{bloques}</div>',
                unsafe_allow_html=True)


def muestra(hex_color, etiqueta):
    st.markdown(
        f'<div style="background:{hex_color};height:70px;border-radius:6px"></div>'
        f'<div style="font:12px monospace;margin-top:4px">{etiqueta}<br>{hex_color}</div>',
        unsafe_allow_html=True,
    )


# ---------- Barra lateral ----------
with st.sidebar:
    st.header("Imagen")
    archivo = st.file_uploader("Sube una obra", type=["jpg", "jpeg", "png", "webp"])

    ejemplos = (sorted(p for p in CARPETA_EJEMPLOS.glob("*") if p.suffix.lower() in EXTENSIONES)
                if CARPETA_EJEMPLOS.exists() else [])
    ejemplo = None
    if ejemplos and archivo is None:
        etiquetas = {f"{t} — {a}": p for p in ejemplos for a, t in [parsear_nombre_archivo(p.name)]}
        ejemplo = etiquetas[st.selectbox("…o elige un ejemplo", list(etiquetas))]

    st.header("Parámetros")
    k_min, k_max = st.slider("Rango de K", 2, 12, (4, 9),
                             help="El notebook usa 4 a 9: con menos de 4, las obras de alto contraste "
                                  "quedaban en solo 'zona clara' y 'zona oscura'.")
    tolerancia = st.slider("Tolerancia de desempate", 0.0, 0.05, 0.01, 0.005, format="%.3f",
                           help="Si dos K tienen silhouette a menos de esta distancia, se elige el K más alto.")
    st.caption("Se descartan los K donde el cluster más pequeño tiene menos del 2 % de los píxeles.")

# ---------- Encabezado ----------
st.title("🎨 Paletas de color con K-means")
st.caption("K-means sobre píxeles en CIELAB · K elegido con Silhouette Score · "
           "evaluación con ΔE CIEDE2000, armonía de Itten y discriminabilidad")

if archivo is not None:
    imagen_bytes, nombre = archivo.getvalue(), archivo.name
elif ejemplo is not None:
    imagen_bytes, nombre = ejemplo.read_bytes(), ejemplo.name
else:
    st.info("Sube una imagen en la barra lateral para extraer su paleta.")
    st.stop()

with st.spinner(f"Probando K de {k_min} a {k_max}…"):
    r = analizar(imagen_bytes, k_min, k_max, tolerancia)

colores = r["colores_rgb"]
proporciones = r["proporciones"]
centroides = r["centroides_lab_ordenados"]
hexes = [rgb_a_hex(c) for c in colores]
chromas = calcular_chroma(centroides)
rec = r["reconstruccion"]
disc = r["discriminabilidad"]

# ---------- Resumen ----------
m1, m2, m3, m4 = st.columns(4)
m1.metric("K elegido", r["k"])
m2.metric("Silhouette", f"{r['silhouette']:.3f}", help="Calculado sobre una muestra de 4.000 píxeles")
m3.metric("ΔE00 medio", f"{rec['media']:.2f}", help="Error perceptual entre cada píxel y su color en la paleta")
m4.metric("Distancia mínima entre colores", f"{disc['distancia_minima']:.1f}",
          help=f"Por debajo de {UMBRAL_DISCRIMINABILIDAD}, dos colores de la paleta se confunden")

# ---------- Muestrario ----------
st.subheader("Muestrario")
franjas(hexes, proporciones)
st.caption("Ordenado de más claro a más oscuro. El ancho de cada franja es proporcional a su peso en la imagen.")

tabla = []
for h, c, p, lab, ch in zip(hexes, colores, proporciones, centroides, chromas):
    cc, mm, yy, kk = rgb_a_cmyk(c)
    tabla.append({
        "HEX": h,
        "RGB": str(tuple(int(v) for v in np.round(c * 255))),
        "CMYK": f"{cc*100:.0f}% {mm*100:.0f}% {yy*100:.0f}% {kk*100:.0f}%",
        "Lab": f"{lab[0]:.0f}, {lab[1]:.0f}, {lab[2]:.0f}",
        "Croma": round(float(ch), 1),
        "% de la imagen": round(float(p) * 100, 1),
    })
st.dataframe(pd.DataFrame(tabla), hide_index=True, width="stretch")

# ---------- Original vs. reconstruida ----------
izq, der = st.columns(2)
izq.image(r["original"], caption="Original", width="stretch")
der.image(r["img_reconstruida"], caption=f"Pintada solo con los {r['k']} colores (800 px)", width="stretch")

# ---------- Detalle ----------
tab_k, tab_rec, tab_arm, tab_disc, tab_tsne = st.tabs(
    ["Elección de K", "Reconstrucción", "Armonía", "Discriminabilidad", "t-SNE"]
)

with tab_k:
    df_k = pd.DataFrame([{"K": x["k"], "Silhouette": x["score"],
                          "Cluster más pequeño (%)": x["prop_min_cluster"] * 100}
                         for x in r["resultados_k"]])
    df_k["Válido (≥ 2 %)"] = df_k["Cluster más pequeño (%)"] >= 2
    df_k["Elegido"] = df_k["K"] == r["k"]
    st.line_chart(df_k.set_index("K")["Silhouette"])
    st.dataframe(df_k.style.format({"Silhouette": "{:.3f}", "Cluster más pequeño (%)": "{:.1f}"}),
                 hide_index=True, width="stretch")
    st.caption(f"Entre los K válidos cuyo silhouette está a menos de {tolerancia:.3f} del mejor, "
               "se elige el más alto: a igual calidad estadística, más colores capturan más matices.")

with tab_rec:
    a, b, c = st.columns(3)
    a.metric("ΔE00 medio", f"{rec['media']:.2f}")
    b.metric("ΔE00 mediana", f"{rec['mediana']:.2f}")
    c.metric("ΔE00 p95", f"{rec['p95']:.2f}")
    umbrales = pd.DataFrame([{"Umbral": f"ΔE00 ≤ {u}",
                              "% de píxeles": round(float(np.mean(rec["delta_e00"] <= u) * 100), 1)}
                             for u in [1, 2, 3, 5, 10]])
    st.dataframe(umbrales, hide_index=True)
    hist = pd.DataFrame({"ΔE00": rec["delta_e00"]})
    st.altair_chart(
        alt.Chart(hist).mark_bar().encode(
            x=alt.X("ΔE00:Q", bin=alt.Bin(maxbins=40), title="ΔE00 por píxel"),
            y=alt.Y("count():Q", title="Píxeles"),
        ).properties(height=220),
        width="stretch",
    )
    st.caption("Referencia: ΔE00 < 1 imperceptible · 2-10 perceptible de cerca · > 10 se nota a simple vista. "
               "La fidelidad depende tanto de K como de qué tan homogéneas son las zonas de color de la obra.")

with tab_arm:
    plantilla, error, n_sat = evaluar_armonia_saturados(centroides, chroma_minimo=CHROMA_MINIMO)
    bordes = ["" if ch >= CHROMA_MINIMO else "outline:3px dashed #d33;outline-offset:-3px;" for ch in chromas]
    franjas(hexes, proporciones, bordes=bordes, alto=60)
    st.caption(f"Borde rojo punteado: color con croma < {CHROMA_MINIMO}. Se excluye porque su tono es inestable.")

    if plantilla:
        st.write(f"Con los **{n_sat} colores saturados**, la plantilla de Itten más cercana es "
                 f"**{plantilla}**, con un error promedio de **{error:.1f}°**.")
    else:
        st.warning("No hay suficientes colores saturados (se necesitan al menos 2) para evaluar la armonía.")

    st.markdown("**Sugerencia de colores armónicos**")
    s1, s2 = st.columns(2)
    idx_base = s1.selectbox("Color base", range(len(hexes)),
                            index=int(np.argmax(proporciones)),
                            format_func=lambda i: f"{hexes[i]} ({proporciones[i]*100:.1f} %)")
    tipo = s2.selectbox("Plantilla", ["complementaria", "analoga", "triada", "tetrada"])
    sugeridos = sugerir_color_armonico(centroides[idx_base], plantilla=tipo)
    cols = st.columns(len(sugeridos) + 1)
    with cols[0]:
        muestra(hexes[idx_base], "Base")
    for col, lab in zip(cols[1:], sugeridos):
        with col:
            muestra(lab_a_hex(lab), "Sugerido")
    if chromas[idx_base] < CHROMA_MINIMO:
        st.caption("El color base es casi neutro: las sugerencias serán casi iguales a él, porque solo cambia el tono.")
    else:
        st.caption("Se conserva la luminosidad y la saturación del color base; solo cambia el tono.")

with tab_disc:
    i, j = disc["par_mas_parecido"]
    st.write(f"Distancia mínima entre dos colores de la paleta: **{disc['distancia_minima']:.1f}** "
             f"(promedio entre todos los pares: {disc['distancia_promedio']:.1f}).")
    p1, p2, _ = st.columns([1, 1, 3])
    with p1:
        muestra(hexes[i], "Par más parecido")
    with p2:
        muestra(hexes[j], "")
    if disc["distancia_minima"] < UMBRAL_DISCRIMINABILIDAD:
        st.warning("Estos dos colores son difíciles de distinguir. Si vas a usar la paleta para categorías "
                   "de un gráfico, conviene fusionarlos o reemplazar uno.")
    else:
        st.success("Todos los colores de la paleta se distinguen con facilidad.")

with tab_tsne:
    st.caption("Proyección 2D de 2.000 píxeles en LAB. Cada punto tiene su color real; "
               "se omite PCA porque los datos ya tienen solo 3 dimensiones.")
    perplexity = st.select_slider("Perplexity", options=[5, 15, 30, 50], value=30)
    if st.button("Calcular t-SNE"):
        with st.spinner("Calculando t-SNE…"):
            emb, colores_pixel = calcular_tsne(imagen_bytes, perplexity)
        df_t = pd.DataFrame({"x": emb[:, 0], "y": emb[:, 1],
                             "color": [rgb_a_hex(c) for c in colores_pixel]})
        st.altair_chart(
            alt.Chart(df_t).mark_circle(size=18, opacity=0.85).encode(
                x=alt.X("x:Q", axis=None), y=alt.Y("y:Q", axis=None),
                color=alt.Color("color:N", scale=None),
            ).properties(height=480),
            width="stretch",
        )
